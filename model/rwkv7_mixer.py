import torch
from torch import nn
import torch.nn.functional as F

from fla.ops.rwkv7.chunk import chunk_rwkv7
from fla.ops.rwkv7.fused_recurrent import fused_mul_recurrent_rwkv7

from utils.opt import set_label
from utils.init import orthogonal_, ortho_init

from .rwkv7_backbone import StatesDictCache
from .tokenshift import token_shift_delta


class SequenceMixer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.d_head or (self.hidden_size // self.num_attention_heads)
        assert config.d_head is not None or self.hidden_size % self.num_attention_heads == 0

        c = config.hidden_size
        h = self.num_attention_heads
        n = self.head_dim

        with torch.no_grad():
            ratio_0_to_1 = layer_idx / (config.num_hidden_layers - 1)
            ratio_1_to_almost0 = 1.0 - (layer_idx / config.num_hidden_layers)
            ddd = torch.ones(1, 1, config.hidden_size)
            for i in range(config.hidden_size):
                ddd[0, 0, i] = i / config.hidden_size

            www = torch.zeros(c)
            zigzag = torch.zeros(c)
            linear = torch.zeros(c)
            for i in range(c):
                linear[i] = i / (c - 1) - 0.5
                zigzag[i] = ((i % n) - ((n - 1) / 2)) / ((n - 1) / 2)
                zigzag[i] = zigzag[i] * abs(zigzag[i])
                www[i] = -6 + 6 * (i / (c - 1)) ** (1 + ratio_0_to_1**0.3)

        if config.use_tokenshift_att:
            self.x_r = set_label(
                "scalars", nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            )
            self.x_w = set_label(
                "scalars", nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            )
            self.x_k = set_label(
                "scalars", nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            )
            self.x_v = set_label(
                "scalars", nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            )
            self.x_a = set_label(
                "scalars", nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            )
            self.x_g = set_label(
                "scalars", nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            )

        def compute_lora_dim(c, base_multiplier, power):
            return max(32, int(round((base_multiplier * (c**power)) / 32) * 32))

        d_decay_lora = compute_lora_dim(c, 1.8, 0.5)
        d_decay_lora_base = compute_lora_dim(config.mup_base_dim, 1.8, 0.5)
        self.w1 = set_label("matrix_params", nn.Linear(c, d_decay_lora, bias=False))
        self.w2 = set_label(
            "matrix_params",
            nn.Linear(d_decay_lora, c, bias=False),
            mup_d_in_ratio=d_decay_lora / d_decay_lora_base,
        )
        self.w0 = set_label(
            "scalars", nn.Parameter(www.reshape(1, 1, c) + 0.5 + zigzag * 2.5)
        )

        d_aaa_lora = compute_lora_dim(c, 1.8, 0.5)
        d_aaa_lora_base = compute_lora_dim(config.mup_base_dim, 1.8, 0.5)
        self.a1 = set_label("matrix_params", nn.Linear(c, d_aaa_lora, bias=False))
        self.a2 = set_label(
            "matrix_params",
            nn.Linear(d_aaa_lora, c, bias=False),
            mup_d_in_ratio=d_aaa_lora / d_aaa_lora_base,
        )
        self.a0 = set_label(
            "scalars",
            nn.Parameter(torch.zeros(1, 1, c) - 0.19 + zigzag * 0.3 + linear * 0.4),
        )

        d_mv_lora = compute_lora_dim(c, 1.3, 0.5)
        d_mv_lora_base = compute_lora_dim(config.mup_base_dim, 1.3, 0.5)
        self.v1 = set_label("matrix_params", nn.Linear(c, d_mv_lora, bias=False))
        self.v2 = set_label(
            "matrix_params",
            nn.Linear(d_mv_lora, c, bias=False),
            mup_d_in_ratio=d_mv_lora / d_mv_lora_base,
        )
        self.v0 = set_label(
            "scalars", nn.Parameter(torch.zeros(1, 1, c) + 0.73 - linear * 0.4)
        )

        d_gate_lora = compute_lora_dim(c, 0.6, 0.8)
        d_gate_lora_base = compute_lora_dim(config.mup_base_dim, 0.6, 0.8)
        self.g1 = set_label("matrix_params", nn.Linear(c, d_gate_lora, bias=False))
        self.g2 = set_label(
            "matrix_params",
            nn.Linear(d_gate_lora, c, bias=False),
            mup_d_in_ratio=d_gate_lora / d_gate_lora_base,
        )
        with torch.no_grad():
            self.w1.weight.zero_()
            self.w2.weight.copy_(ortho_init(torch.empty(d_decay_lora, c), 0.1).T)
            self.a1.weight.zero_()
            self.a2.weight.copy_(ortho_init(torch.empty(d_aaa_lora, c), 0.1).T)
            self.v1.weight.zero_()
            self.v2.weight.copy_(ortho_init(torch.empty(d_mv_lora, c), 0.1).T)
            self.g1.weight.zero_()
            self.g2.weight.copy_(ortho_init(torch.empty(d_gate_lora, c), 0.1).T)

        self.k_k = set_label(
            "scalars", nn.Parameter(torch.zeros(1, 1, c) + 0.71 - linear * 0.1)
        )
        self.k_a = set_label("scalars", nn.Parameter(torch.zeros(1, 1, c) + 1.02))
        self.r_k = set_label("scalars", nn.Parameter(torch.zeros(h, n) - 0.04))

        self.receptance = set_label("matrix_params", nn.Linear(c, c, bias=False))
        orthogonal_(self.receptance.weight, gain=1)
        self.key = set_label("matrix_params", nn.Linear(c, c, bias=False))
        orthogonal_(self.key.weight, gain=0.1)
        self.value = set_label("matrix_params", nn.Linear(c, c, bias=False))
        orthogonal_(self.value.weight, gain=1)
        self.output = set_label("matrix_params", nn.Linear(c, c, bias=False))
        self.ln_x = set_label("scalars", nn.GroupNorm(h, c, eps=64e-5))
        with torch.no_grad():
            self.output.weight.zero_()
            self.ln_x.weight.copy_(((1 + layer_idx) / config.num_hidden_layers) ** 0.7)

    def get_first_layer_kwargs(self, x0, x, **kwargs):
        if self.config.use_tokenshift_att:
            x_prior_token = None
            past_key_values = kwargs.get("past_key_values", None)
            cached_decode = (
                past_key_values is not None
                and past_key_values.get_seq_length(self.layer_idx) > 0
            )
            if cached_decode:
                x_prior_token = past_key_values.get_states(self.layer_idx).get(
                    "att_final_token"
                )
                if x_prior_token is None:
                    raise AssertionError(
                        "RWKV7 cached decode requires 'att_final_token' in the cache."
                    )
            xx = token_shift_delta(x, x_prior_token, fallback="zero")
            xv = x + xx * self.x_v
        else:
            xv = x
        v = self.value(xv)
        return {"v_first": v}

    def forward(
        self,
        x,
        v_first,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: StatesDictCache | None = None,
        **kwargs,
    ):
        b, t, c = x.shape
        h = self.num_attention_heads
        cached_decode = (
            past_key_values is not None
            and past_key_values.get_seq_length(self.layer_idx) > 0
        )

        if self.config.use_tokenshift_att:
            x_prior_token = None
            if past_key_values is not None:
                states_dict = past_key_values.get_states(self.layer_idx)
                x_prior_token = states_dict.get("att_final_token")
                if cached_decode and x_prior_token is None:
                    raise AssertionError(
                        "RWKV7 cached decode requires 'att_final_token' in the cache."
                    )
                past_key_values.update(
                    self.layer_idx,
                    offset=0,
                    states_dict={"att_final_token": x[:, -1:].clone()},
                )
            xx = token_shift_delta(x, x_prior_token, fallback="zero")
            xr = x + xx * self.x_r
            xw = x + xx * self.x_w
            xk = x + xx * self.x_k
            xv = x + xx * self.x_v
            xa = x + xx * self.x_a
            xg = x + xx * self.x_g
        else:
            xr, xw, xk, xv, xa, xg = x, x, x, x, x, x

        r = self.receptance(xr)
        log_neglog_w = -F.softplus(-(self.w0 + self.w2(torch.tanh(self.w1(xw))))) - 0.5
        log_w = -log_neglog_w.exp()
        k = self.key(xk)
        v = self.value(xv)
        v = v + (v_first - v) * torch.sigmoid(self.v0 + self.v2(self.v1(xv)))

        a = torch.sigmoid(self.a0 + self.a2(self.a1(xa)))
        g = self.g2(torch.sigmoid(self.g1(xg)))

        kk = k * self.k_k
        kk = F.normalize(kk.view(b, t, h, -1), dim=-1, p=2.0).view(b, t, c)
        k = k * (1 + (a - 1) * self.k_a)

        r = r.view(b, t, h, -1)
        log_w = log_w.view(b, t, h, -1)
        k = k.view(b, t, h, -1)
        v = v.view(b, t, h, -1)
        aa = -kk.view(b, t, h, -1)
        bb = (kk * a).view(b, t, h, -1)

        s = None
        if past_key_values is not None:
            s = past_key_values.get_states(self.layer_idx).get("rwkv7_time_mix")
            if cached_decode and s is None:
                raise AssertionError(
                    "RWKV7 cached decode requires 'rwkv7_time_mix' in the cache."
                )
        if self.training or t >= 64:
            x, s = chunk_rwkv7(
                r=r,
                w=log_w,
                k=k,
                v=v,
                a=aa,
                b=bb,
                initial_state=s,
                output_final_state=not self.training,
            )
        else:
            x, s = fused_mul_recurrent_rwkv7(
                r=r,
                w=log_w,
                k=k,
                v=v,
                kk=kk,
                a=a,
                initial_state=s,
                output_final_state=not self.training,
            )
        if past_key_values is not None:
            past_key_values.update(
                layer_idx=self.layer_idx, offset=t, states_dict={"rwkv7_time_mix": s}
            )

        x = self.ln_x(x.view(b * t, c)).view(b, t, c)

        x = x + ((r * k * self.r_k.view(h, -1)).sum(dim=-1, keepdim=True) * v).view(
            b, t, c
        )
        x = self.output(x * g)
        return x
