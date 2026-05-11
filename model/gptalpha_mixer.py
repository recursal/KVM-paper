import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    _create_sparse_block_from_block_mask,
)

from utils import ortho_init, set_label, separately_compiled_flex_attention

from .rwkv7_backbone import StatesDictCache, apply_rotary_embeddings

from model.value_residual_mixin import ValueResidualMixin

def create_doc_id_block_mask(doc_ids: torch.Tensor, q_len: int, kv_len: int):
    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = doc_ids[q_idx] == doc_ids[kv_idx]
        return causal_mask & document_mask

    BLOCK_SIZE = 128
    assert len(doc_ids.shape) == 1
    block_begin_doc_ids = doc_ids[::BLOCK_SIZE]
    block_end_doc_ids = doc_ids[BLOCK_SIZE - 1 :: BLOCK_SIZE]
    qblock_idx_gpu = torch.arange(
        q_len // BLOCK_SIZE, dtype=torch.int32, device=doc_ids.device
    )[:, None]
    kblock_idx_gpu = torch.arange(
        kv_len // BLOCK_SIZE, dtype=torch.int32, device=doc_ids.device
    )[None, :]
    full_block_mask = (
        (qblock_idx_gpu > kblock_idx_gpu)
        & (block_begin_doc_ids[:, None] == block_end_doc_ids[None, :])
        & (block_end_doc_ids[:, None] == block_begin_doc_ids[None, :])
    )
    partial_block_mask = (
        (qblock_idx_gpu >= kblock_idx_gpu)
        & (block_begin_doc_ids[:, None] <= block_end_doc_ids[None, :])
        & (block_end_doc_ids[:, None] >= block_begin_doc_ids[None, :])
    )

    full_block_mask = full_block_mask & (
        ~partial_block_mask
    )  # be careful to mask off any partial blocks

    return _create_sparse_block_from_block_mask(
        (partial_block_mask[None, None, :, :], full_block_mask[None, None, :, :]),
        document_causal_mask,
        (q_len, kv_len),
        BLOCK_SIZE,
        BLOCK_SIZE,
    )


class GeneralGPTAlphaSequenceMixer(ValueResidualMixin):
    def __init__(
        self, config, layer_idx: int, value_residual_mode: str, tokenshift_mode: str
    ):
        super().__init__(
            config, layer_idx, value_residual_mode, tokenshift_mode, qk_norm=True
        )

        self.prefer_flex_attention = False

        self._block_mask_cache = {}

    def _mask_mod(self, q_len: int, kv_len: int):
        q_offset = kv_len - q_len

        def mask_mod(b, h, q_idx, kv_idx):
            q_abs_idx = q_idx + q_offset
            mask = kv_idx <= q_abs_idx
            return mask

        return mask_mod

    def _block_mask(self, q_len: int, kv_len: int, mask_mod, device: torch.device):
        cache_key = (
            device.type if device.index is None else f"{device.type}:{device.index}",
            q_len,
            kv_len,
        )
        block_mask = self._block_mask_cache.get(cache_key)
        if block_mask is None:
            block_mask = create_block_mask(
                mask_mod=mask_mod(q_len, kv_len),
                B=None,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=device,
                BLOCK_SIZE=128,
            )
            self._block_mask_cache[cache_key] = block_mask
        return block_mask

    def get_first_layer_kwargs(self, x0, x, input_ids, **kwargs):
        rv = super().get_first_layer_kwargs(x0, x, input_ids, **kwargs)

        q_len = x.size(1)
        if (
            past_key_values := kwargs.get("past_key_values", None)
        ) is not None and past_key_values.get_seq_length(self.layer_idx) > 0:
            kv_len = past_key_values.get_seq_length(self.layer_idx) + q_len
        else:
            kv_len = q_len

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if cu_seqlens is not None:
            doc_ids = torch.zeros(kv_len, dtype=torch.int32, device=x.device) - 1
            for i in range(len(cu_seqlens) - 1):
                doc_ids[cu_seqlens[i] : cu_seqlens[i + 1]] = i
            rv["block_mask"] = create_doc_id_block_mask(doc_ids, q_len, kv_len)
        elif self.prefer_flex_attention:
            is_prefill = q_len > 1
            if self.training or is_prefill:
                rv["block_mask"] = self._block_mask(
                    q_len, kv_len, self._mask_mod, x.device
                )

        return rv

    def forward(
        self,
        x,
        v_first,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: StatesDictCache | None = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = x.size()

        q, k, v = self.calc_qkv(x, v_first, past_key_values)

        q = apply_rotary_embeddings(q, position_embeddings)
        k = apply_rotary_embeddings(k, position_embeddings)

        y = self._attention(
            q, k, v, attn_mask=attention_mask, past_key_values=past_key_values, **kwargs
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        y = self.c_proj(y)
        return y


class SequenceMixer(GeneralGPTAlphaSequenceMixer):
    def __init__(self, config, layer_idx: int):
        super().__init__(
            config,
            layer_idx,
            config.gptalpha_value_residual_mode,
            config.gptalpha_token_shift_mode,
        )

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
        if past_key_values is not None:
            states_dict = past_key_values.update(
                self.layer_idx, offset=q.size(1), states_dict={"k": k, "v": v}
            )
            k, v = states_dict["k"], states_dict["v"]

        is_causal = False
        if attn_mask is None:
            if q.size(1) == 1:
                is_causal = False
            else:
                if q.size(1) == k.size(1):
                    is_causal = True
                else:
                    raise NotImplementedError(
                        f"Unsupported attention mask configuration: attn_mask is None but sequence lengths of q and k do not match: {q.shape} vs {k.shape}"
                    )

        if block_mask is not None:
            return separately_compiled_flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=block_mask,
            )
        else:
            return F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
