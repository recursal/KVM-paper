from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from utils.defer import defer
from utils.flex_attention import separately_compiled_flex_attention
from utils.opt import set_label
from utils.init import ortho_init

from .rwkv7_backbone import RotaryEmbedding, StatesDictCache, apply_rotary_embeddings

from model.value_residual_mixin import ValueResidualMixin
from model.gptalpha_mixer import GeneralGPTAlphaSequenceMixer


def _sliding_window_causal_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = kv_idx <= q_idx
    swa_mask = kv_idx >= (q_idx - (window_size - 1))
    return swa_mask & causal_mask


class SequenceMixer(GeneralGPTAlphaSequenceMixer):
    def __init__(self, config, layer_idx: int):
        super().__init__(
            config,
            layer_idx,
            config.swa_value_residual_mode,
            config.swa_token_shift_mode,
        )

        swa_window_size = config.swa_window_size
        if swa_window_size <= 0:
            swa_window_size = config.chunk_len
        self.swa_window_size = swa_window_size

        self.prefer_flex_attention = True

    def _mask_mod(self, q_len: int, kv_len: int):
        q_offset = kv_len - q_len
        window_size = self.swa_window_size

        def mask_mod(b, h, q_idx, kv_idx):
            q_abs_idx = q_idx + q_offset
            mask = kv_idx <= q_abs_idx
            mask = mask & (kv_idx >= (q_abs_idx - (window_size - 1)))
            return mask

        return mask_mod

    def forward(
        self,
        x,
        v_first,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: StatesDictCache | None = None,
        block_mask=None,
        **kwargs,
    ):
        batch_size, seq_len, _ = x.size()

        q, k, v = self.calc_qkv(x, v_first, past_key_values)

        q = apply_rotary_embeddings(q, position_embeddings)
        k = apply_rotary_embeddings(k, position_embeddings)

        if block_mask is not None:
            y = separately_compiled_flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=block_mask,
            )
        else:
            attn_mask = _sliding_window_causal_mask(
                seq_len, self.swa_window_size, q.device
            )
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y

    _CACHE_SWA_K = "kvm_alt_swa_swa_k"
    _CACHE_SWA_V = "kvm_alt_swa_swa_v"

    def forward_inference(
        self,
        x,
        v_first,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: StatesDictCache | None = None,
        position_ids: torch.LongTensor | None = None,
        block_mask=None,
        **kwargs,
    ):
        batch_size, seq_len, _ = x.size()
        q, k, v = self.calc_qkv(x, v_first, past_key_values)
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

        q = q.reshape(batch_size, seq_len, self.num_attention_heads, -1)
        k = k.reshape(batch_size, seq_len, self.num_attention_heads, -1)
        v = v.reshape(batch_size, seq_len, self.num_attention_heads, -1)

        q = apply_rotary_embeddings(q, position_embeddings)
        k = apply_rotary_embeddings(k, position_embeddings)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if cached_decode:
            swa_k, swa_v = self._crop_swa_cache(
                torch.cat([cache_states[self._CACHE_SWA_K], k], dim=2),
                torch.cat([cache_states[self._CACHE_SWA_V], v], dim=2),
            )
            y = F.scaled_dot_product_attention(
                q,
                swa_k,
                swa_v,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            attn_mask = _sliding_window_causal_mask(
                seq_len, self.swa_window_size, q.device
            )
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            swa_k, swa_v = self._crop_swa_cache(k, v)

        if past_key_values is not None:
            past_key_values.update(
                self.layer_idx,
                offset=seq_len,
                states_dict={
                    self._CACHE_SWA_K: swa_k,
                    self._CACHE_SWA_V: swa_v,
                },
            )

        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y

    def _crop_swa_cache(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(k.size(2)) <= self.swa_window_size:
            return k, v
        return k[:, :, -self.swa_window_size :, :], v[:, :, -self.swa_window_size :, :]
