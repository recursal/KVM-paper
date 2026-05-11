from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from model.gptalpha_mixer import GeneralGPTAlphaSequenceMixer
from model.value_residual_mixin import ValueResidualMixin
from utils.flex_attention import separately_compiled_flex_attention

from .rwkv7_backbone import StatesDictCache


class SequenceMixer(GeneralGPTAlphaSequenceMixer):
    _CACHE_BSWA_K = "bswa_k"
    _CACHE_BSWA_V = "bswa_v"

    def __init__(self, config, layer_idx: int):
        super().__init__(
            config,
            layer_idx,
            config.gptalpha_value_residual_mode,
            config.gptalpha_token_shift_mode,
        )

        self.chunk_len = int(config.chunk_len)
        self.n_bswa_chunks = int(config.n_bswa_chunks)
        if self.chunk_len <= 0:
            raise ValueError("gptalpha_bswa requires chunk_len > 0")
        if self.n_bswa_chunks <= 0:
            raise ValueError("gptalpha_bswa requires n_bswa_chunks > 0")
        self.bswa_len = self.chunk_len * self.n_bswa_chunks

        self.prefer_flex_attention = True

    def _mask_mod(self, q_len: int, kv_len: int):
        n_bswa_chunks = self.n_bswa_chunks
        chunk_len = self.chunk_len
        q_offset = kv_len - q_len
        sink_len = self.config.sink_len

        def mask_mod(b, h, q_idx, kv_idx):
            q_abs_idx = q_idx + q_offset
            causal_mask = kv_idx <= q_abs_idx
            bswa_mask = (
                kv_idx // chunk_len >= q_abs_idx // chunk_len - n_bswa_chunks + 1
            )
            sink_mask = kv_idx < sink_len
            return (sink_mask | bswa_mask) & causal_mask

        return mask_mod

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_key_values: StatesDictCache | None = None,
        block_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        # assert attn_mask is None, "attn_mask is not supported by gptalpha_bswa"

        q_len = q.size(1)
        kv_len = k.size(1)

        is_prefill = q_len > 1
        if is_prefill and past_key_values is not None:
            assert past_key_values.get_seq_length(self.layer_idx) == 0, (
                f"chunked prefill not yet supported, q_len {q_len}, kv_len {kv_len}, past length {past_key_values.get_seq_length(self.layer_idx)}"
            )

        if self.training or is_prefill:
            assert block_mask is not None, (
                "block_mask must be provided during training or prefill when using gptalpha_bswa"
            )
            output = separately_compiled_flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=block_mask,
            )

        if past_key_values is not None:
            states_update = {
                "sink_size": self.config.sink_len,
                "bswa_block_size": self.chunk_len,
                "bswa_n_blocks": self.n_bswa_chunks,
                self._CACHE_BSWA_K: k,
                self._CACHE_BSWA_V: v,
            }
            states_dict = past_key_values.update(
                self.layer_idx, offset=q_len, states_dict=states_update
            )
            k, v = states_dict["bswa_k"], states_dict["bswa_v"]

        if not self.training and not is_prefill:
            is_causal = False
            if attn_mask is None:
                if q_len == 1:
                    is_causal = False
                else:
                    if q_len == kv_len:
                        is_causal = True
                    else:
                        raise NotImplementedError(
                            f"Unsupported attention mask configuration: attn_mask is None but sequence lengths of q and k do not match: {q.shape} vs {k.shape}"
                        )
            output = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=is_causal,
                attn_mask=attn_mask,
            )
        return output
