from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from utils.opt import set_label
from utils.init import ortho_init

from .rwkv7_backbone import StatesDictCache, apply_rotary_embeddings
from .tokenshift import apply_token_shift


class ValueResidualMixin(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        value_residual_mode: str,
        tokenshift_mode: str,
        qk_norm: bool,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d_qk_head = (
            config.d_qk_head or config.d_head or (config.hidden_size // config.num_attention_heads)
        )
        self.d_v_head = (
            config.d_v_head or config.d_head or (config.hidden_size // config.num_attention_heads)
        )
        d_qk = self.num_attention_heads * self.d_qk_head
        d_v = self.num_attention_heads * self.d_v_head
        # assert config.d_head is not None or self.hidden_size % self.num_attention_heads == 0

        # NOTE - if using this base class for actual RWKV, you must override this in rwkv models to get classic RWKV behavior
        self.rwkv_tokenshift_fallback = "first"

        self.tokenshift_mode = tokenshift_mode if config.use_tokenshift_att else "none"

        if self.tokenshift_mode == "rwkv":
            ratio_1_to_almost0 = 1.0 - (layer_idx / config.num_hidden_layers)
            with torch.no_grad():
                ddd_qk = torch.ones(d_qk)
                linear = torch.zeros(d_qk)
                for i in range(d_qk):
                    ddd_qk[i] = i / d_qk
                    linear[i] = i / (d_qk - 1) - 0.5

                ddd_v = torch.ones(d_v)
                linear = torch.zeros(d_v)
                for i in range(d_v):
                    ddd_v[i] = i / d_v
                    linear[i] = i / (d_v - 1) - 0.5

            self.x_q = set_label(
                "scalars",
                nn.Parameter(1.0 - torch.pow(ddd_qk, 0.2 * ratio_1_to_almost0)),
            )
            self.x_k = set_label(
                "scalars",
                nn.Parameter(1.0 - torch.pow(ddd_qk, 0.7 * ratio_1_to_almost0)),
            )
            self.x_v = set_label(
                "scalars",
                nn.Parameter(1.0 - torch.pow(ddd_v, 0.7 * ratio_1_to_almost0)),
            )

        self.c_q = set_label("matrix_params", nn.Linear(self.hidden_size, d_qk, bias=False))
        self.c_k = set_label("matrix_params", nn.Linear(self.hidden_size, d_qk, bias=False))
        self.c_v = set_label("matrix_params", nn.Linear(self.hidden_size, d_v, bias=False))

        if qk_norm:
            self.ln_q = set_label("scalars", nn.LayerNorm(self.d_qk_head))
            self.ln_k = set_label("scalars", nn.LayerNorm(self.d_qk_head))
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()

        self.c_proj = set_label(
            "matrix_params", nn.Linear(d_v, self.hidden_size, bias=False)
        )
        with torch.no_grad():
            self.c_proj.weight.zero_()

        self.lamb = None
        self.value_residual_mode = "none"
        if config.use_value_residual:
            assert value_residual_mode in ["linear", "rwkv", "rwkv_post_tokenshift"], (
                f"Unsupported value residual mode: {value_residual_mode!r}"
            )
            if value_residual_mode == "linear":
                self.value_residual_mode = "linear"
                self.lamb = set_label("scalars", nn.Parameter(torch.tensor(0.5)))

            # NOTE - this isn't really rwkv's method for value residuals, it's just per channel with rwkv's inits
            elif value_residual_mode in ["rwkv", "rwkv_post_tokenshift"]:
                self.value_residual_mode = value_residual_mode

                if not hasattr(self, "x_v"):
                    # RWKV-style value residual uses x_v for gated token-shift input; zero keeps this path identity.
                    self.x_v = set_label("scalars", nn.Parameter(torch.zeros(d_v)))

                def compute_lora_dim(c, base_multiplier, power):
                    return max(32, int(round((base_multiplier * (c**power)) / 32) * 32))

                d_value_residual_lora = compute_lora_dim(self.hidden_size, 1.3, 0.5)
                d_value_residual_lora_base = compute_lora_dim(
                    config.mup_base_dim, 1.3, 0.5
                )
                self.v_gate1 = set_label(
                    "matrix_params",
                    nn.Linear(self.hidden_size, d_value_residual_lora, bias=False),
                )
                self.v_gate2 = set_label(
                    "matrix_params",
                    nn.Linear(d_value_residual_lora, d_v, bias=False),
                    mup_d_in_ratio=d_value_residual_lora / d_value_residual_lora_base,
                )

                with torch.no_grad():
                    ratio_1_to_almost0 = 1.0 - (layer_idx / config.num_hidden_layers)
                    ddd = torch.ones(d_v)
                    linear = torch.zeros(d_v)
                    for i in range(d_v):
                        ddd[i] = i / d_v
                        linear[i] = i / (d_v - 1) - 0.5

                with torch.no_grad():
                    self.v_gate1.weight.zero_()
                    self.v_gate2.weight.copy_(
                        ortho_init(torch.empty(d_value_residual_lora, d_v), 0.1).T
                    )
                self.v_gate0 = set_label(
                    "scalars",
                    nn.Parameter(
                        torch.zeros(1, 1, d_v) + 0.73 - linear.view(1, 1, d_v) * 0.4
                    ),
                )

    _CACHE_Q_FINAL = "att_q_final_token"
    _CACHE_K_FINAL = "att_k_final_token"
    _CACHE_V_FINAL = "att_v_final_token"
    _CACHE_X_FINAL = "att_x_final_token"

    def calc_qkv(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        past_key_values: StatesDictCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()

        cache_states = (
            past_key_values.get_states(self.layer_idx)
            if past_key_values is not None
            else {}
        )
        past_seq_len = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else 0
        )
        cached_decode = past_seq_len > 0
        if cached_decode and x.size(1) != 1:
            raise AssertionError(f"cached decode expects single-token inputs.")

        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        v = self.maybe_apply_value_residual_pre_tokenshift(v, x, v_first)

        q_prior_token = None
        k_prior_token = None
        v_prior_token = None
        x_prior_token = None
        if past_key_values is not None:
            q_prior_token = cache_states.get(self._CACHE_Q_FINAL)
            k_prior_token = cache_states.get(self._CACHE_K_FINAL)
            v_prior_token = cache_states.get(self._CACHE_V_FINAL)
            x_prior_token = cache_states.get(self._CACHE_X_FINAL)
            past_key_values.update(
                self.layer_idx,
                offset=0,
                states_dict={
                    self._CACHE_Q_FINAL: q[:, -1:].detach(),
                    self._CACHE_K_FINAL: k[:, -1:].detach(),
                    self._CACHE_V_FINAL: v[:, -1:].detach(),
                    self._CACHE_X_FINAL: x[:, -1:].detach(),
                },
            )

        if self.tokenshift_mode == "rwkv":
            q = apply_token_shift(q, self.x_q, q_prior_token)
            k = apply_token_shift(k, self.x_k, k_prior_token)
            v = apply_token_shift(v, self.x_v, v_prior_token)

        v = self.maybe_apply_value_residual_post_tokenshift(
            v, x, v_first, x_prior_token
        )

        q = q.reshape(batch_size, seq_len, self.num_attention_heads, -1)
        k = k.reshape(batch_size, seq_len, self.num_attention_heads, -1)
        v = v.reshape(batch_size, seq_len, self.num_attention_heads, -1)

        q = self.ln_q(q)
        k = self.ln_k(k)

        return q, k, v

    def get_first_layer_kwargs(self, x0, x, x_prior_token=None, **kwargs):
        del x0, x_prior_token, kwargs
        if self.value_residual_mode == "none":
            return {"v_first": None}
        # Match the origin/hf mixers that this shared path was extracted from:
        # the shared residual anchor is the plain layer-0 value projection.
        # NOTE: this is technically incorrect for rwkv style stuff but we are using it anyway for sake of parity with older runs.
        return {"v_first": self.c_v(x)}

    def maybe_apply_value_residual_pre_tokenshift(
        self, v: torch.Tensor, x: torch.Tensor, v_first: torch.Tensor | None
    ) -> torch.Tensor:
        if self.value_residual_mode == "linear":
            return (1 - self.lamb) * v + self.lamb * v_first.view_as(v)
        elif self.value_residual_mode == "rwkv":
            value_gate = torch.sigmoid(self.v_gate0 + self.v_gate2(self.v_gate1(x)))
            return v + (v_first.to(v.dtype) - v) * value_gate.to(v.dtype)
        else:
            return v

    def _rwkv_value_gate_input(
        self,
        x: torch.Tensor,
        x_prior_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x_heads = x.reshape(batch_size, seq_len, self.num_attention_heads, self.d_v_head)
        prev_x_heads = None
        if x_prior_token is not None:
            prev_x_heads = x_prior_token.reshape(
                batch_size, 1, self.num_attention_heads, self.d_v_head
            )
        x_heads = apply_token_shift(
            x_heads,
            self.x_v.view(1, 1, self.num_attention_heads, self.d_v_head),
            prev_x_heads,
            fallback=self.rwkv_tokenshift_fallback,
        )
        return x_heads.reshape_as(x)

    def maybe_apply_value_residual_post_tokenshift(
        self,
        v: torch.Tensor,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        x_prior_token: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.value_residual_mode == "rwkv_post_tokenshift":
            xv = self._rwkv_value_gate_input(x, x_prior_token)
            value_gate = torch.sigmoid(self.v_gate0 + self.v_gate2(self.v_gate1(xv)))
            return v + (v_first.to(v.dtype) - v) * value_gate.to(v.dtype)
        else:
            return v
