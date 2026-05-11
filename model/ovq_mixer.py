from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from utils.opt import set_label

from .rwkv7_backbone import StatesDictCache, apply_rotary_embeddings
from .value_residual_mixin import ValueResidualMixin


def _lower_right_causal_mask(
    q_len: int, kv_len: int, device: torch.device
) -> torch.Tensor:
    diagonal_offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    return kv_idx <= (q_idx + diagonal_offset)


def _l2_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_dtype = x.dtype
    return (x / (x.norm(dim=-1, keepdim=True) + eps)).to(x_dtype)


def _plateau_dict_size(t: int, max_centroids: int) -> int:
    if t <= 0:
        return 0
    return min(
        int((float(t) * float(max_centroids)) / (float(t) + float(max_centroids))),
        max_centroids,
    )


def _min_dict_size(t: int, max_centroids: int) -> int:
    if t <= 0:
        return 0
    return min(int(t), max_centroids)


def _compute_dict_sizes(
    seq_len: int,
    chunk_size: int,
    max_centroids: int,
    *,
    use_min_state_size: bool,
) -> list[int]:
    n_chunks = int((seq_len + chunk_size - 1) // chunk_size)
    sizes = [0]
    for chunk_idx in range(1, n_chunks + 1):
        t = min(seq_len, chunk_idx * chunk_size)
        if use_min_state_size:
            sizes.append(_min_dict_size(t, max_centroids))
        else:
            sizes.append(_plateau_dict_size(t, max_centroids))

    for idx in range(1, len(sizes)):
        sizes[idx] = max(sizes[idx], sizes[idx - 1])
        sizes[idx] = min(sizes[idx], max_centroids)
    return sizes


def _get_nn_assignments(
    d_k: torch.Tensor, k_c: torch.Tensor, *, num_new: int
) -> torch.Tensor:
    bsz, num_attention_heads, chunk_len, _ = k_c.shape
    d_sz = int(d_k.size(2))

    if d_sz == 0:
        best_cluster = torch.zeros(
            (bsz, num_attention_heads, chunk_len), device=k_c.device, dtype=torch.long
        )
        if num_new <= 1:
            return best_cluster

        sim0 = (k_c * k_c[:, :, :1, :]).sum(dim=-1)
        sim0 = sim0.clone()
        sim0[:, :, 0] = float("inf")
        low_sim_idx = sim0.topk(k=num_new - 1, dim=-1, largest=False).indices
        new_ids = torch.arange(1, num_new, device=k_c.device, dtype=torch.long).view(
            1, 1, -1
        )
        new_ids = new_ids.expand(bsz, num_attention_heads, -1)
        return best_cluster.scatter(2, low_sim_idx, new_ids)

    sim = torch.matmul(k_c, d_k.transpose(-1, -2))
    best_sim, best_cluster = sim.max(dim=-1)
    if num_new == 0:
        return best_cluster

    low_sim_idx = best_sim.topk(k=num_new, dim=-1, largest=False).indices
    new_ids = torch.arange(
        d_sz, d_sz + num_new, device=k_c.device, dtype=torch.long
    ).view(1, 1, -1)
    new_ids = new_ids.expand(bsz, num_attention_heads, -1)
    return best_cluster.scatter(2, low_sim_idx, new_ids)


def _update_dict_hard(
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    counts: torch.Tensor,
    *,
    k_c: torch.Tensor,
    v_c: torch.Tensor,
    num_new: int,
    d_sz: int,
    normalize_centroids: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, num_attention_heads, chunk_len, d = k_c.shape
    assignments = _get_nn_assignments(d_k[:, :, :d_sz, :], k_c, num_new=num_new)
    idx_counts = assignments.unsqueeze(-1)

    ones = torch.ones(
        (bsz, num_attention_heads, chunk_len, 1), device=k_c.device, dtype=counts.dtype
    )
    counts = counts.scatter_add(2, idx_counts, ones)

    lr = 1.0 / counts.gather(2, idx_counts)
    lr_k = lr.to(d_k.dtype)
    lr_v = lr.to(d_v.dtype)

    idx_d = assignments.unsqueeze(-1).expand(bsz, num_attention_heads, chunk_len, d)
    k_quant = torch.gather(d_k, 2, idx_d)
    v_quant = torch.gather(d_v, 2, idx_d)
    d_k = d_k.scatter_add(2, idx_d, -lr_k * (k_quant - k_c))
    d_v = d_v.scatter_add(2, idx_d, -lr_v * (v_quant - v_c))

    if normalize_centroids:
        d_k = F.normalize(d_k, dim=-1)
        d_v = F.normalize(d_v, dim=-1)

    return d_k, d_v, counts


def _update_dict_sparse_k_delta_v(
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    counts: torch.Tensor,
    *,
    k_c: torch.Tensor,
    delta_c: torch.Tensor,
    num_new: int,
    d_sz: int,
    normalize_centroids: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, num_attention_heads, chunk_len, d = k_c.shape
    assignments = _get_nn_assignments(d_k[:, :, :d_sz, :], k_c, num_new=num_new)
    idx_counts = assignments.unsqueeze(-1)

    ones = torch.ones(
        (bsz, num_attention_heads, chunk_len, 1), device=k_c.device, dtype=counts.dtype
    )
    counts = counts.scatter_add(2, idx_counts, ones)

    lr = 1.0 / counts.gather(2, idx_counts)
    lr_k = lr.to(d_k.dtype)
    lr_v = lr.to(d_v.dtype)

    idx_d = assignments.unsqueeze(-1).expand(bsz, num_attention_heads, chunk_len, d)
    k_quant = torch.gather(d_k, 2, idx_d)
    d_k = d_k.scatter_add(2, idx_d, -lr_k * (k_quant - k_c))
    d_v = d_v.scatter_add(2, idx_d, lr_v * delta_c.to(d_v.dtype))

    if normalize_centroids:
        d_k = F.normalize(d_k, dim=-1)

    return d_k, d_v, counts


def _predict_from_centroids(
    q_c: torch.Tensor,
    d_k_c: torch.Tensor,
    d_v_c: torch.Tensor,
    counts_c: torch.Tensor,
    *,
    use_count_bias: bool,
) -> torch.Tensor:
    if int(d_k_c.size(2)) == 0:
        bsz, num_attention_heads, chunk_len, d = q_c.shape
        return torch.zeros(
            (bsz, num_attention_heads, chunk_len, d), device=q_c.device, dtype=d_v_c.dtype
        )

    attn_bias = None
    if use_count_bias:
        log_counts = torch.where(
            counts_c > 0,
            counts_c.log(),
            torch.full_like(counts_c, float("-inf")),
        ).squeeze(-1)
        attn_bias = log_counts.unsqueeze(2).expand(-1, -1, q_c.size(2), -1)
        if attn_bias.dtype != q_c.dtype:
            attn_bias = attn_bias.to(q_c.dtype)

    return F.scaled_dot_product_attention(
        q_c,
        d_k_c,
        d_v_c,
        attn_mask=attn_bias,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0,
    )


def ovq_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_k: torch.Tensor | None,
    d_v: torch.Tensor | None,
    d_counts: torch.Tensor | None,
    *,
    beta: torch.Tensor,
    chunk_size: int,
    max_centroids: int,
    normalize_centroids: bool,
    use_min_state_size: bool,
    use_counts_for_lr_only: bool,
    use_delta_rule: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, num_attention_heads, tsz, d = q.shape

    q = q * (beta.view(1, num_attention_heads, 1, 1).to(q.dtype) * math.sqrt(float(d)))

    dict_sizes = _compute_dict_sizes(
        tsz,
        chunk_size,
        max_centroids,
        use_min_state_size=use_min_state_size,
    )
    n_chunks = len(dict_sizes) - 1

    if d_k is None:
        d_k = torch.zeros(
            (bsz, num_attention_heads, max_centroids, d), device=q.device, dtype=k.dtype
        )
    if d_v is None:
        d_v = torch.zeros(
            (bsz, num_attention_heads, max_centroids, d), device=q.device, dtype=v.dtype
        )
    if d_counts is None:
        d_counts = torch.zeros(
            (bsz, num_attention_heads, max_centroids, 1), device=q.device, dtype=torch.float32
        )
    counts = d_counts

    outs: list[torch.Tensor] = []
    for chunk_idx in range(n_chunks):
        s = chunk_idx * chunk_size
        e = min(tsz, (chunk_idx + 1) * chunk_size)
        q_c = q[:, :, s:e, :]
        k_c = k[:, :, s:e, :]
        v_c = v[:, :, s:e, :]

        d_sz = int(dict_sizes[chunk_idx])
        k_star = torch.cat([d_k[:, :, :d_sz, :], k_c], dim=2) if d_sz > 0 else k_c
        v_star = torch.cat([d_v[:, :, :d_sz, :], v_c], dim=2) if d_sz > 0 else v_c

        chunk_len = int(e - s)
        kv_len = int(d_sz + chunk_len)
        causal_mask = _lower_right_causal_mask(
            q_len=chunk_len, kv_len=kv_len, device=q.device
        )

        if use_counts_for_lr_only:
            attn_out = F.scaled_dot_product_attention(
                q_c,
                k_star,
                v_star,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
            )
        else:
            ones = torch.ones(
                (bsz, num_attention_heads, chunk_len, 1), device=q.device, dtype=torch.float32
            )
            c_star = torch.cat([counts[:, :, :d_sz, :], ones], dim=2)
            log_c_star = torch.where(
                c_star > 0,
                c_star.log(),
                torch.full_like(c_star, float("-inf")),
            ).squeeze(-1)

            causal_bias = torch.zeros(
                (chunk_len, kv_len), device=q.device, dtype=log_c_star.dtype
            )
            causal_bias = causal_bias.masked_fill(~causal_mask, float("-inf"))
            attn_bias = causal_bias.view(
                1, 1, chunk_len, kv_len
            ) + log_c_star.unsqueeze(2)
            if attn_bias.dtype != q_c.dtype:
                attn_bias = attn_bias.to(q_c.dtype)

            attn_out = F.scaled_dot_product_attention(
                q_c,
                k_star,
                v_star,
                attn_mask=attn_bias,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
            )

        outs.append(attn_out)

        num_new = int(dict_sizes[chunk_idx + 1] - dict_sizes[chunk_idx])
        if use_delta_rule:
            pred_input = k_c * (
                beta.view(1, num_attention_heads, 1, 1).to(k_c.dtype) * math.sqrt(float(d))
            )
            centroid_out = _predict_from_centroids(
                pred_input,
                d_k[:, :, :d_sz, :],
                d_v[:, :, :d_sz, :],
                counts[:, :, :d_sz, :],
                use_count_bias=not use_counts_for_lr_only,
            )
            delta_c = v_c - centroid_out.to(v_c.dtype)
            d_k, d_v, counts = _update_dict_sparse_k_delta_v(
                d_k,
                d_v,
                counts,
                k_c=k_c,
                delta_c=delta_c,
                num_new=num_new,
                d_sz=d_sz,
                normalize_centroids=normalize_centroids,
            )
        else:
            d_k, d_v, counts = _update_dict_hard(
                d_k,
                d_v,
                counts,
                k_c=k_c,
                v_c=v_c,
                num_new=num_new,
                d_sz=d_sz,
                normalize_centroids=normalize_centroids,
            )

    return torch.cat(outs, dim=2), d_k, d_v, counts


def _empty_ovq_dict_state(
    *,
    batch_size: int,
    num_attention_heads: int,
    max_centroids: int,
    d_k_head: int,
    d_v_head: int,
    device: torch.device,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_k = torch.zeros(
        (batch_size, num_attention_heads, max_centroids, d_k_head), device=device, dtype=k_dtype
    )
    d_v = torch.zeros(
        (batch_size, num_attention_heads, max_centroids, d_v_head), device=device, dtype=v_dtype
    )
    d_counts = torch.zeros(
        (batch_size, num_attention_heads, max_centroids, 1), device=device, dtype=torch.float32
    )
    return d_k, d_v, d_counts


def _build_ovq_decode_cache_state(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    beta: torch.Tensor,
    chunk_size: int,
    max_centroids: int,
    normalize_centroids: bool,
    use_min_state_size: bool,
    use_counts_for_lr_only: bool,
    use_delta_rule: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_attention_heads, seq_len, d_k_head = k.shape
    _, _, _, d_v_head = v.shape
    pending_len = seq_len % chunk_size
    committed_len = seq_len - pending_len
    d_k, d_v, d_counts = _empty_ovq_dict_state(
        batch_size=batch_size,
        num_attention_heads=num_attention_heads,
        max_centroids=max_centroids,
        d_k_head=d_k_head,
        d_v_head=d_v_head,
        device=k.device,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
    )

    if committed_len > 0:
        dict_sizes = _compute_dict_sizes(
            committed_len,
            chunk_size,
            max_centroids,
            use_min_state_size=use_min_state_size,
        )
        committed_chunks = committed_len // chunk_size
        for chunk_idx in range(committed_chunks):
            s = chunk_idx * chunk_size
            e = s + chunk_size
            k_c = k[:, :, s:e, :]
            v_c = v[:, :, s:e, :]
            d_sz = int(dict_sizes[chunk_idx])
            num_new = int(dict_sizes[chunk_idx + 1] - dict_sizes[chunk_idx])
            if use_delta_rule:
                pred_input = k_c * (
                    beta.view(1, num_attention_heads, 1, 1).to(k_c.dtype)
                    * math.sqrt(float(d_k_head))
                )
                centroid_out = _predict_from_centroids(
                    pred_input,
                    d_k[:, :, :d_sz, :],
                    d_v[:, :, :d_sz, :],
                    d_counts[:, :, :d_sz, :],
                    use_count_bias=not use_counts_for_lr_only,
                )
                delta_c = v_c - centroid_out.to(v_c.dtype)
                d_k, d_v, d_counts = _update_dict_sparse_k_delta_v(
                    d_k,
                    d_v,
                    d_counts,
                    k_c=k_c,
                    delta_c=delta_c,
                    num_new=num_new,
                    d_sz=d_sz,
                    normalize_centroids=normalize_centroids,
                )
            else:
                d_k, d_v, d_counts = _update_dict_hard(
                    d_k,
                    d_v,
                    d_counts,
                    k_c=k_c,
                    v_c=v_c,
                    num_new=num_new,
                    d_sz=d_sz,
                    normalize_centroids=normalize_centroids,
                )

    pending_k = k[:, :, committed_len:, :]
    pending_v = v[:, :, committed_len:, :]
    return d_k, d_v, d_counts, pending_k, pending_v


class SequenceMixer(ValueResidualMixin):
    _CACHE_D_K = "ovq_d_k"
    _CACHE_D_V = "ovq_d_v"
    _CACHE_D_COUNTS = "ovq_d_counts"
    _CACHE_PENDING_K = "ovq_pending_k"
    _CACHE_PENDING_V = "ovq_pending_v"

    def __init__(self, config, layer_idx: int):
        super().__init__(
            config,
            layer_idx,
            config.ovq_value_residual_mode,
            config.ovq_token_shift_mode,
            qk_norm=False,
        )

        self.config = config
        self.layer_idx = int(layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d_qk_head = (
            config.d_qk_head or config.d_head or (config.hidden_size // config.num_attention_heads)
        )
        self.d_v_head = (
            config.d_v_head or config.d_head or (config.hidden_size // config.num_attention_heads)
        )
        assert config.d_head is not None or self.hidden_size % self.num_attention_heads == 0

        window_size = config.ovq_window_size
        self.window_size = None if window_size <= 0 else window_size
        self.chunk_size = config.ovq_chunk_size
        self.max_centroids = config.ovq_max_centroids
        self.normalize_centroids = config.ovq_normalize_centroids
        self.use_min_state_size = config.ovq_use_min_state_size
        self.use_counts_for_lr_only = config.ovq_use_counts_for_lr_only
        self.use_delta_rule = config.ovq_use_delta_rule

        if self.chunk_size <= 0:
            raise ValueError("ovq_chunk_size must be positive")
        if self.max_centroids <= 0:
            raise ValueError("ovq_max_centroids must be positive")

        self.beta = set_label("scalars", nn.Parameter(torch.ones(self.num_attention_heads)))

    def forward(
        self,
        x,
        v_first,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: StatesDictCache | None = None,
        **kwargs,
    ):
        del attention_mask, kwargs
        bsz, tsz, _ = x.size()

        q, k, v = self.calc_qkv(x, v_first, past_key_values)

        q = _l2_norm(q)
        k = _l2_norm(k)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        y, _, _, _ = ovq_attention(
            q,
            k,
            v,
            None,
            None,
            None,
            beta=self.beta,
            chunk_size=self.chunk_size,
            max_centroids=self.max_centroids,
            normalize_centroids=self.normalize_centroids,
            use_min_state_size=self.use_min_state_size,
            use_counts_for_lr_only=self.use_counts_for_lr_only,
            use_delta_rule=self.use_delta_rule,
        )

        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y

    def forward_inference(
        self,
        x,
        v_first,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: StatesDictCache | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ):
        del attention_mask, position_ids, kwargs
        bsz, tsz, _ = x.size()
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

        q = _l2_norm(q)
        k = _l2_norm(k)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if cached_decode:
            d_k = cache_states[self._CACHE_D_K]
            d_v = cache_states[self._CACHE_D_V]
            d_counts = cache_states[self._CACHE_D_COUNTS]
            pending_k = cache_states[self._CACHE_PENDING_K]
            pending_v = cache_states[self._CACHE_PENDING_V]
            pending_len = int(pending_k.size(2))
            if pending_len != int(pending_v.size(2)):
                raise AssertionError(
                    "OVQ cache pending_k and pending_v must have matching sequence lengths."
                )
            if pending_len >= self.chunk_size:
                raise AssertionError(
                    "OVQ cache pending chunk must stay smaller than ovq_chunk_size."
                )

            committed_len = past_seq_len - pending_len
            if committed_len < 0 or (committed_len % self.chunk_size) != 0:
                raise AssertionError(
                    "OVQ cache committed context must be aligned to full chunks."
                )
            committed_chunks = committed_len // self.chunk_size
            dict_sizes = _compute_dict_sizes(
                past_seq_len + tsz,
                self.chunk_size,
                self.max_centroids,
                use_min_state_size=self.use_min_state_size,
            )
            d_sz = int(dict_sizes[committed_chunks])

            q_scaled = q * (
                self.beta.view(1, self.num_attention_heads, 1, 1).to(q.dtype)
                * math.sqrt(float(self.d_qk_head))
            )
            current_k_star = torch.cat([pending_k, k], dim=2)
            current_v_star = torch.cat([pending_v, v], dim=2)
            if d_sz > 0:
                k_star = torch.cat([d_k[:, :, :d_sz, :], current_k_star], dim=2)
                v_star = torch.cat([d_v[:, :, :d_sz, :], current_v_star], dim=2)
            else:
                k_star = current_k_star
                v_star = current_v_star

            if self.use_counts_for_lr_only:
                attn_mask = _lower_right_causal_mask(
                    q_len=1, kv_len=int(k_star.size(2)), device=q.device
                )
                y = F.scaled_dot_product_attention(
                    q_scaled,
                    k_star,
                    v_star,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=1.0,
                )
            else:
                ones = torch.ones(
                    (bsz, self.num_attention_heads, int(current_k_star.size(2)), 1),
                    device=q.device,
                    dtype=torch.float32,
                )
                c_star = torch.cat([d_counts[:, :, :d_sz, :], ones], dim=2)
                log_c_star = torch.where(
                    c_star > 0,
                    c_star.log(),
                    torch.full_like(c_star, float("-inf")),
                ).squeeze(-1)
                attn_bias = log_c_star.unsqueeze(2)
                if attn_bias.dtype != q_scaled.dtype:
                    attn_bias = attn_bias.to(q_scaled.dtype)
                y = F.scaled_dot_product_attention(
                    q_scaled,
                    k_star,
                    v_star,
                    attn_mask=attn_bias,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=1.0,
                )

            pending_k = current_k_star
            pending_v = current_v_star
            if int(pending_k.size(2)) == self.chunk_size:
                num_new = int(
                    dict_sizes[committed_chunks + 1] - dict_sizes[committed_chunks]
                )
                if self.use_delta_rule:
                    pred_input = pending_k * (
                        self.beta.view(1, self.num_attention_heads, 1, 1).to(pending_k.dtype)
                        * math.sqrt(float(self.d_qk_head))
                    )
                    centroid_out = _predict_from_centroids(
                        pred_input,
                        d_k[:, :, :d_sz, :],
                        d_v[:, :, :d_sz, :],
                        d_counts[:, :, :d_sz, :],
                        use_count_bias=not self.use_counts_for_lr_only,
                    )
                    delta_c = pending_v - centroid_out.to(pending_v.dtype)
                    d_k, d_v, d_counts = _update_dict_sparse_k_delta_v(
                        d_k,
                        d_v,
                        d_counts,
                        k_c=pending_k,
                        delta_c=delta_c,
                        num_new=num_new,
                        d_sz=d_sz,
                        normalize_centroids=self.normalize_centroids,
                    )
                else:
                    d_k, d_v, d_counts = _update_dict_hard(
                        d_k,
                        d_v,
                        d_counts,
                        k_c=pending_k,
                        v_c=pending_v,
                        num_new=num_new,
                        d_sz=d_sz,
                        normalize_centroids=self.normalize_centroids,
                    )
                pending_k = pending_k[:, :, :0, :]
                pending_v = pending_v[:, :, :0, :]
        else:
            y, _, _, _ = ovq_attention(
                q,
                k,
                v,
                None,
                None,
                None,
                beta=self.beta,
                chunk_size=self.chunk_size,
                max_centroids=self.max_centroids,
                normalize_centroids=self.normalize_centroids,
                use_min_state_size=self.use_min_state_size,
                use_counts_for_lr_only=self.use_counts_for_lr_only,
                use_delta_rule=self.use_delta_rule,
            )
            if past_key_values is not None:
                d_k, d_v, d_counts, pending_k, pending_v = (
                    _build_ovq_decode_cache_state(
                        k,
                        v,
                        beta=self.beta,
                        chunk_size=self.chunk_size,
                        max_centroids=self.max_centroids,
                        normalize_centroids=self.normalize_centroids,
                        use_min_state_size=self.use_min_state_size,
                        use_counts_for_lr_only=self.use_counts_for_lr_only,
                        use_delta_rule=self.use_delta_rule,
                    )
                )
                past_key_values.update(
                    layer_idx=self.layer_idx,
                    offset=tsz,
                    states_dict={
                        self._CACHE_D_K: d_k,
                        self._CACHE_D_V: d_v,
                        self._CACHE_D_COUNTS: d_counts,
                        self._CACHE_PENDING_K: pending_k,
                        self._CACHE_PENDING_V: pending_v,
                    },
                )

        if cached_decode and past_key_values is not None:
            past_key_values.update(
                self.layer_idx,
                offset=tsz,
                states_dict={
                    self._CACHE_D_K: d_k,
                    self._CACHE_D_V: d_v,
                    self._CACHE_D_COUNTS: d_counts,
                    self._CACHE_PENDING_K: pending_k,
                    self._CACHE_PENDING_V: pending_v,
                },
            )

        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y
