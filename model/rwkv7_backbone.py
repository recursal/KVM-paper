import importlib
import inspect
from dataclasses import dataclass, field, fields, asdict
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from utils.defer import defer
from utils.grad_cp import maybe_ckpt
from utils.opt import set_label
from utils.init import orthogonal_

from model.statesdictcache import StatesDictCache
from model.tokenshift import token_shift_delta

from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin, Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from utils.logger import print0 as print

from pydantic import BaseModel, ConfigDict


@dataclass
class CausalLMOutputWithPastAndAccuracy(CausalLMOutputWithPast):
    acc: torch.FloatTensor | None = None


class RotaryEmbedding(nn.Module):
    angular_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, full_dim: int, partial_dim: int, base=10000.0):
        super().__init__()
        self.full_dim = full_dim
        self.partial_dim = partial_dim
        self.base = base
        self.initialized = False

    def forward(
        self, position_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE - when this was in __init__, it would be NOT-INITIALIZED when using lm-eval via HF!!!
        if not self.initialized:
            angular_freq = (1 / self.base) ** torch.linspace(
                0.0,
                1.0,
                steps=self.partial_dim // 2,
                dtype=torch.float32,
                device=position_ids.device,
            )
            angular_freq = angular_freq.repeat_interleave(2)
            angular_freq = torch.cat(
                [angular_freq, angular_freq.new_zeros(self.full_dim - self.partial_dim)]
            )
            self.register_buffer("angular_freq", angular_freq, persistent=False)
            self.initialized = True
        B, T = position_ids.shape
        theta = position_ids.float().view(B, T, 1, 1) @ self.angular_freq.view(
            1, 1, 1, -1
        )
        cos = (
            theta.cos().bfloat16()
        )  # nn.Buffer(theta.cos().bfloat16(), persistent=False)
        sin = (
            theta.sin().bfloat16()
        )  # nn.Buffer(theta.sin().bfloat16(), persistent=False)
        sin[..., 1::2] *= -1
        return (cos, sin)


def apply_rotary_embeddings(x_BTHD, position_embeddings):
    cos, sin = position_embeddings
    assert cos.size(1) == x_BTHD.size(1), f"{cos.size()} {x_BTHD.size()}"
    # cos, sin = (
    #     cos[:, : x_BTHD.size(-3), :, :],
    #     sin[:, : x_BTHD.size(-3), :, :],
    # )
    x_flip = (
        x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2)
        .flip(-1)
        .view(x_BTHD.shape)
    )
    return cos * x_BTHD + sin * x_flip


def _load_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Unable to import sequence mixer module {module_path!r} for {class_path!r}. "
            "Some mixers require optional dependencies such as `fla`."
        ) from exc
    return getattr(module, class_name)


class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        d_hidden = int(config.ffn_expansion * config.hidden_size) // 32 * 32
        self.c_fc = set_label(
            "matrix_params", nn.Linear(config.hidden_size, d_hidden, bias=False)
        )
        orthogonal_(self.c_fc.weight, gain=1)
        self.c_proj = set_label(
            "matrix_params", nn.Linear(d_hidden, config.hidden_size, bias=False)
        )
        self.c_proj.weight.data.zero_()

        if config.use_tokenshift_ffn:
            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (layer_idx / config.num_hidden_layers)
                ddd = torch.ones(1, 1, config.hidden_size)
                for i in range(config.hidden_size):
                    ddd[0, 0, i] = i / config.hidden_size
                self.x_k = set_label(
                    "scalars", nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))
                )

    def forward(self, x, past_key_values=None, **kwargs):
        if self.config.use_tokenshift_ffn:
            old_final_token = None
            if past_key_values is not None:
                layer_seq_len = past_key_values.get_seq_length(self.layer_idx)
                current_seq_len = int(x.size(1))
                if layer_seq_len < current_seq_len:
                    raise AssertionError(
                        "RWKV7 backbone MLP cache invariant violated: the shared layer cache must "
                        "already include the current attention pass before the MLP runs."
                    )

                # Attention and MLP share a single per-layer cache namespace and sequence counter.
                # The attention mixer writes the current tokens first, so the MLP can only require
                # cached token-shift history for the portion of the layer cache that predates this
                # MLP invocation.
                prior_mlp_seq_len = layer_seq_len - current_seq_len
                old_final_token = past_key_values.get_states(self.layer_idx).get(
                    "mlp_final_token"
                )
                if prior_mlp_seq_len > 0 and old_final_token is None:
                    raise AssertionError(
                        "RWKV7 backbone cached MLP decode requires 'mlp_final_token' from a prior "
                        "MLP pass. The layer cache is shared across attention and MLP, so layer "
                        "sequence length alone is not sufficient."
                    )
                if prior_mlp_seq_len == 0 and old_final_token is not None:
                    raise AssertionError(
                        "RWKV7 backbone MLP cache invariant violated: found 'mlp_final_token' even "
                        "though no prior MLP tokens exist for this layer."
                    )
                if old_final_token is not None and old_final_token.shape != (
                    x.shape[0],
                    1,
                    x.shape[2],
                ):
                    raise AssertionError(
                        "RWKV7 backbone cached MLP decode requires 'mlp_final_token' to have shape "
                        f"{(x.shape[0], 1, x.shape[2])}, got {tuple(old_final_token.shape)}."
                    )
                past_key_values.update(
                    self.layer_idx, offset=0, states_dict={"mlp_final_token": x[:, -1:]}
                )
            xk = x + self.x_k * token_shift_delta(x, old_final_token, fallback="zero")
        else:
            xk = x
        x = self.c_fc(xk)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_scale = self.config.residual_scale or 1.0

        if config.alt_layer_every > 1 and layer_idx % config.alt_layer_every == 0:
            token_mixer_class_path = config.alt_token_mixer_class_path
            if token_mixer_class_path is None:
                token_mixer_class_path = (
                    "model." + config.alt_token_mixer + "_mixer.SequenceMixer"
                )
        else:
            token_mixer_class_path = config.token_mixer_class_path
            if token_mixer_class_path is None:
                token_mixer_class_path = (
                    "model." + config.token_mixer + "_mixer.SequenceMixer"
                )

        token_mixer_class = _load_class(token_mixer_class_path)

        self.ln_attn = set_label("scalars", nn.LayerNorm(config.hidden_size))
        self.attn = token_mixer_class(config, layer_idx)

        self.ln_mlp = set_label("scalars", nn.LayerNorm(config.hidden_size))
        self.mlp = MLP(config, layer_idx)
        if self.config.use_block_lambdas:
            self.lambdas = set_label("scalars", nn.Parameter(torch.tensor([1.0, 0.0])))

    def forward(self, x, x0, **kwargs):
        # NOTE - we separate these so that compilation is faster when changing things in each part
        if self.training:
            x = self.forward_attn_maybe_compiled(x, x0, **kwargs)
        else:
            x = self.forward_attn(x, x0, **kwargs)
        x = self.forward_mlp(x, x0, **kwargs)
        return x

    def forward_attn(self, x, x0, **kwargs):
        if self.config.use_block_lambdas:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        attn_fn = (
            getattr(self.attn, "forward_inference", self.attn.forward)
            if not self.training and kwargs.get("past_key_values") is not None
            else self.attn.forward
        )
        return x + attn_fn(self.ln_attn(x), x0=x0, **kwargs) * self.residual_scale

    @defer(torch.compile)
    def forward_attn_maybe_compiled(self, x, x0, **kwargs):
        return self.forward_attn(x, x0, **kwargs)

    @defer(torch.compile)
    def forward_mlp(self, x, x0, **kwargs):
        return x + self.mlp(self.ln_mlp(x), x0=x0, **kwargs) * self.residual_scale


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return grad_output, gy


class RWKV7BackboneConfigBase(PretrainedConfig):
    model_type = "rwkv7backbone"

    def __init__(
        self,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        tie_worhidden_sizedings: bool = False,
        max_position_embeddings: int = 1024 * 1024 * 1024,
        **kwargs: Any,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_worhidden_sizedings=tie_worhidden_sizedings,
            max_position_embeddings=max_position_embeddings,
            **kwargs,
        )


class RWKV7BackboneConfigDataclass(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_class_path: str = "model.rwkv7_backbone.RWKV7BackboneForCausalLM"

    vocab_size: int = 50304
    num_hidden_layers: int = 12  # was n_layer
    num_attention_heads: int = 6  # putting this here instead of mixer because we need it for rope
    d_qk_head: int | None = None
    d_v_head: int | None = None
    d_head: int | None = None
    hidden_size: int = 768
    mup_base_dim: int = 768

    rope_theta: float = 10_000.0
    rope_partial_dim: int = 64

    alt_rope_theta: float = 10_000.0
    alt_rope_partial_dim: int = 64
    alt_share_v0: int = 0

    ffn_expansion: float = 4.0
    token_mixer: str = "rwkv7"
    token_mixer_class_path: str | None = None
    alt_token_mixer: str = "swa"
    alt_token_mixer_class_path: str | None = None
    alt_layer_every: int = -1
    use_tokenshift_ffn: int = False
    use_block_lambdas: int = False
    use_skip_connections: int = False
    use_l2wrap: int = False
    logit_softcap: float = 0.0
    # this should scale with 1/num_hidden_layers
    residual_scale: float | None = None


class MixerConfigDataclass(RWKV7BackboneConfigDataclass):
    use_value_residual: int = 1
    gptalpha_value_residual_mode: str = "rwkv"
    gptalpha_token_shift_mode: str = "rwkv"
    kvm_value_residual_mode: str = "rwkv"
    kvm_token_shift_mode: str = "rwkv"
    kvm_apply_merge_gate_to_initial_state: int = 0
    kvm_apply_merge_gate_to_appends: int = 1

    kvm_use_merge_gate_keys: int = 1
    kvm_use_merge_gate_values: int = 1
    kvm_use_head_temps: int = 1
    kvm_use_vlens: int = 1

    ovq_value_residual_mode: str = "rwkv"
    ovq_token_shift_mode: str = "rwkv"
    use_tokenshift_att: int = 0

    swa_value_residual_mode: str = "rwkv"
    swa_token_shift_mode: str = "rwkv"
    swa_window_size: int = -1

    ovq_token_shift_mode: str = "rwkv"
    ovq_window_size: int = 128
    ovq_chunk_size: int = 128
    ovq_max_centroids: int = 2048
    ovq_normalize_centroids: int = 0
    ovq_use_min_state_size: int = 0
    ovq_use_counts_for_lr_only: int = 0
    ovq_use_delta_rule: int = 0

    kda_expand_k: float = 1.0
    kda_expand_v: float = 1.0
    kda_conv_size: int = 4
    kda_conv_bias: int = 0
    kda_allow_neg_eigval: int = 0
    kda_token_shift_mode: str = "none"
    kda_value_residual_mode: str = "none"

    gated_deltanet_expand_k: float = 0.75
    gated_deltanet_expand_v: float = 1.5
    gated_deltanet_num_kv_heads: int | None = None
    gated_deltanet_conv_size: int = 4
    gated_deltanet_conv_bias: int = 0
    gated_deltanet_use_mamba_gate: int = 1
    gated_deltanet_token_shift_mode: str = "none"
    gated_deltanet_value_residual_mode: str = "none"

    sink_len: int = 1
    chunk_len: int = 256
    n_max_d_chunks: int = 1
    n_bswa_chunks: int = 2
    state_budget_mode: str = "fixed"
    state_growth_factor: float = 1.0
    state_growth_exponent: float = 0.5
    state_round_down: int = 1
    state_min_len: int = -1
    state_saturation_n: int | None = None
    use_overflow_v_key_weighting: int = 1
    use_state_deltarule: int = 0
    use_state_decay: int = 0

    lact_value_residual_mode: str = "rwkv"
    lact_token_shift_mode: str = "rwkv"
    lact_actual_rope_partial_dim: int = -1
    lact_lact_window_size: int = -1
    lact_chunk_size: int = -1
    lact_num_fw_heads: int = -1
    lact_inter_multi: float = 1.0
    lact_norm_eps: float = 1e-6
    lact_attn_qk_norm: int = 0
    lact_w0_w2_low_rank: int = 32
    lact_fw_init_gain: float = 0.5
    lact_ttt_lag: int = 0
    lact_factor: float = 1.0
    lact_use_muon: int = 0
    lact_learnable_muon_lr: int = 0
    lact_muon_lr_reduce_exp: int = 2
    lact_use_momentum: int = 1
    lact_ttt_prenorm: int = 0
    lact_ttt_nope: int = 0
    lact_no_v_silu: int = 0
    lact_use_bswa: int = 0

    param_dtype: str | None = "bfloat16"

class MixerConfig(RWKV7BackboneConfigBase):
    def __init__(self, **kwargs):
        attributes = MixerConfigDataclass.model_fields.keys()
        applicable = {attr: kwargs[attr] for attr in attributes if attr in kwargs}
        mixer = MixerConfigDataclass(**applicable)
        for field_name, value in mixer.model_dump().items():
            if field_name in kwargs:
                kwargs.pop(field_name)
            setattr(self, field_name, value)
        super().__init__(**kwargs)


class RWKV7BackbonePreTrainedModel(PreTrainedModel):
    config_class = MixerConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block"]

    def _init_weights(self, module):
        pass  # Weights are initialized in place


class RWKV7BackboneModel(RWKV7BackbonePreTrainedModel):
    def __init__(self, config: MixerConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # FIXME - how do we want to integrate with HF on this?
        self.gradient_checkpointing = False

        self.d_qk_head = (
            config.d_qk_head or config.d_head or (config.hidden_size // config.num_attention_heads)
        )
        self.d_v_head = (
            config.d_v_head or config.d_head or (config.hidden_size // config.num_attention_heads)
        )

        # FIXME - huggingface canonically names this config.hidden_size not hidden_size
        self.wte = set_label(
            "wte_embed",
            nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx),
            mup_d_in_ratio=1.0,
        )
        self.h = nn.ModuleList(
            [Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        with torch.no_grad():
            nn.init.uniform_(self.wte.weight, a=-1e-4, b=1e-4)

        if self.config.use_skip_connections:
            self.encoder_layers = config.num_hidden_layers // 2
            self.decoder_layers = config.num_hidden_layers - self.encoder_layers
            self.skip_weights = set_label(
                "scalars", nn.Parameter(torch.ones(self.decoder_layers))
            )
        else:
            self.encoder_layers = config.num_hidden_layers
            self.decoder_layers = 0

        self.ln_emb = set_label("scalars", nn.LayerNorm(config.hidden_size))
        self.ln_head = set_label("scalars", nn.LayerNorm(config.hidden_size))

        self.rotary = RotaryEmbedding(
            full_dim=self.d_qk_head,
            partial_dim=config.rope_partial_dim,
            base=config.rope_theta,
        )
        if config.alt_layer_every > 1:
            self.alt_rotary = RotaryEmbedding(
                full_dim=self.d_qk_head,
                partial_dim=config.alt_rope_partial_dim,
                base=config.alt_rope_theta,
            )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: list[int] | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)

        past_seen_tokens = None
        if cache_position is not None and cache_position.numel() > 0:
            past_seen_tokens = int(cache_position.reshape(-1)[0].item())
        elif position_ids is not None and position_ids.numel() > 0:
            past_seen_tokens = int(position_ids.reshape(-1)[0].item())
        elif (
            attention_mask is not None
            and past_key_values is not None
            and attention_mask.shape[-1] > inputs_embeds.shape[1]
        ):
            past_seen_tokens = int(attention_mask.shape[-1] - inputs_embeds.shape[1])
        elif past_key_values is None:
            past_seen_tokens = 0
        else:
            raise ValueError(
                "Cannot infer cached token count from past_key_values alone. "
                "Pass cache_position or position_ids when using a layer-indexed cache."
            )

        if use_cache and not isinstance(past_key_values, StatesDictCache):
            past_key_values = StatesDictCache()

        if position_ids is None:
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )
            position_ids = position_ids.unsqueeze(0)

        if attention_mask is not None:
            # check for full mask
            mask_full = torch.all(
                attention_mask.sum(dim=-1) == attention_mask.shape[-1]
            ).item()
            if mask_full:
                attention_mask = None
            else:
                raise NotImplementedError(
                    "left or right padding attention_masks are not supported by the model code"
                )

        if past_seen_tokens > 0 and inputs_embeds.shape[1] > 1:
            assert False, (
                "The current implementation does not support multi-token generation with past_key_values. This is because the attention mask handling for such a case is non-trivial and has not been implemented yet. Please use single-token generation when past_key_values are involved."
            )

            def _lower_right_causal_mask(
                q_len: int, kv_len: int, device: torch.device
            ) -> torch.Tensor:
                diagonal_offset = kv_len - q_len
                q_idx = torch.arange(q_len, device=device).unsqueeze(1)
                kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
                return kv_idx <= (q_idx + diagonal_offset)

            print("Creating lower right causal mask")
            attention_mask = _lower_right_causal_mask(
                inputs_embeds.shape[1],
                past_seen_tokens + inputs_embeds.shape[1],
                inputs_embeds.device,
            )

        hidden_states = inputs_embeds

        hidden_states = self._forward_model(
            input_ids,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    @defer(torch.compile)
    def embed(self, input_ids):
        x = self.wte(input_ids)
        x = self.ln_emb(x)
        return x

    def _forward_model(self, input_ids, x, position_ids, **kwargs):
        x0 = x

        position_embeddings = self.rotary(position_ids)
        alt_position_embeddings = (
            self.alt_rotary(position_ids) if hasattr(self, "alt_rotary") else None
        )

        if self.config.alt_layer_every > 0:
            first_main_layer = self.h[1]
            first_alt_layer = self.h[0]
        else:
            first_main_layer = self.h[0]
            first_alt_layer = None
        get_first_main_layer_kwargs_fn = getattr(
            first_main_layer.attn, "get_first_layer_kwargs", lambda x0, x, **kwargs: {}
        )
        first_main_layer_kwargs = get_first_main_layer_kwargs_fn(
            x0=x0, x=first_main_layer.ln_attn(x0), input_ids=input_ids, **kwargs
        )  # pre-compute value for first block, which needs to be accessed by subsequent blocks

        if first_alt_layer is not None:
            get_first_alt_layer_kwargs_fn = getattr(
                first_alt_layer.attn,
                "get_first_layer_kwargs",
                lambda x0, x, **kwargs: {},
            )
            first_alt_layer_kwargs = get_first_alt_layer_kwargs_fn(
                x0=x0, x=first_alt_layer.ln_attn(x0), input_ids=input_ids, **kwargs
            )
        else:
            first_alt_layer_kwargs = {}

        main_layer_kwargs = dict(x0=x0, **first_main_layer_kwargs, **kwargs)
        alt_layer_kwargs = dict(x0=x0, **first_alt_layer_kwargs, **kwargs)

        if self.config.alt_share_v0:
            if (
                first_alt_layer is not None
                and alt_layer_kwargs.get("v_first", None) is not None
            ):
                main_layer_kwargs["v_first"] = alt_layer_kwargs.get("v_first", None)

        # Store outputs for U-Net skip connections
        skip_connections = []

        for i in range(self.encoder_layers):
            is_alt_layer = self.config.alt_layer_every > 0 and (
                i % self.config.alt_layer_every == 0
            )
            layer_kwargs = alt_layer_kwargs if is_alt_layer else main_layer_kwargs
            layer_kwargs["position_embeddings"] = (
                alt_position_embeddings
                if alt_position_embeddings is not None
                and (i % self.config.alt_layer_every == 0)
                else position_embeddings
            )
            # FIXME - how do we want to integrate with HF on this? should we obey self.gradient_checkpointing?
            x = maybe_ckpt(self.h[i], x, **layer_kwargs)
            if self.config.use_skip_connections:
                skip_connections.append(x)

        for i in range(self.decoder_layers):
            is_alt_layer = self.config.alt_layer_every > 0 and (
                i % self.config.alt_layer_every == 0
            )
            layer_kwargs = alt_layer_kwargs if is_alt_layer else main_layer_kwargs
            layer_kwargs["position_embeddings"] = (
                alt_position_embeddings
                if alt_position_embeddings is not None
                and (i % self.config.alt_layer_every == 0)
                else position_embeddings
            )
            skip_connection = skip_connections.pop()
            weighted_skip = self.skip_weights[i] * skip_connection
            # FIXME - how do we want to integrate with HF on this? should we obey self.gradient_checkpointing?
            x = maybe_ckpt(
                self.h[self.encoder_layers + i], x + weighted_skip, **layer_kwargs
            )

        x = self.ln_head(x)

        return x


class RWKV7BackboneForCausalLM(RWKV7BackbonePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)

        self.transformer = RWKV7BackboneModel(config)

        self.lm_head = set_label(
            "lm_head", nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )
        orthogonal_(
            self.lm_head.weight, gain=0.5 * (config.vocab_size / config.hidden_size) ** 0.5
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_logits: bool = True,
        **kwargs,
    ):
        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return self._compute_loss_and_logits(
            outputs,
            labels=labels,
            logits_to_keep=logits_to_keep,
            return_logits=return_logits,
        )

    @defer(torch.compile)
    def _compute_loss_and_logits(
        self,
        outputs: BaseModelOutputWithPast,
        labels: torch.LongTensor | None,
        logits_to_keep: int | torch.Tensor,
        return_logits: bool = True,
    ) -> CausalLMOutputWithPastAndAccuracy:
        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.logit_softcap != 0:
            logits = self.config.logit_softcap * torch.tanh(
                logits / self.config.logit_softcap
            )
        logits = logits.float()

        loss = None
        acc = None
        if labels is not None:
            loss_function = getattr(self, "criterion", None)
            if loss_function is None:
                loss_function = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_function(
                input=logits.view(labels.numel(), -1), target=labels.view(-1)
            )
            if self.config.use_l2wrap:
                loss = L2Wrap.apply(loss, logits)
            loss = loss.float()

        returned_logits = logits if return_logits else None
        return CausalLMOutputWithPastAndAccuracy(
            loss=loss,
            logits=returned_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            acc=acc,
        )


# Register the model with AutoModel APIs for HuggingFace transformers
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register(MixerConfig.model_type, MixerConfig)
AutoModelForCausalLM.register(MixerConfig, RWKV7BackboneForCausalLM)

# Ensure save_pretrained writes the custom code file and auto_map entries in config.json.
MixerConfig.register_for_auto_class()
RWKV7BackboneForCausalLM.register_for_auto_class("AutoModelForCausalLM")
