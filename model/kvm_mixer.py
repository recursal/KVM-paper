from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from utils.opt import set_label

from .rwkv7_backbone import StatesDictCache, apply_rotary_embeddings

from model.value_residual_mixin import ValueResidualMixin


def _lower_right_causal_mask(
    q_len: int, kv_len: int, device: torch.device
) -> torch.Tensor:
    diagonal_offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    return kv_idx <= (q_idx + diagonal_offset)


def _round_down_to_multiple(x: int, multiple: int) -> int:
    x_i = max(int(x), 0)
    multiple_i = max(int(multiple), 1)
    return (x_i // multiple_i) * multiple_i


def _gather_by_idx(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    idx_full = idx.unsqueeze(-1).expand(-1, -1, -1, int(x.size(-1)))
    return x.gather(2, idx_full)


def _all_idx(x: torch.Tensor, block_len: int) -> torch.Tensor:
    return (
        torch.arange(block_len, device=x.device, dtype=torch.long)
        .view(1, 1, block_len)
        .expand(int(x.size(0)), int(x.size(1)), block_len)
    )


def _split_append_merge_idx_by_maxsim(
    k_block: torch.Tensor,
    n_append: int,
    k_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_len = int(k_block.size(2))
    n_append = min(max(int(n_append), 0), block_len)
    n_merge = block_len - n_append
    all_idx = _all_idx(k_block, block_len)
    empty_idx = all_idx[:, :, :0]

    if block_len == 0:
        return empty_idx, empty_idx
    if n_append == 0:
        return empty_idx, all_idx
    if n_merge == 0:
        return all_idx, empty_idx

    if int(k_ref.size(2)) == 0:
        return all_idx[:, :, n_merge:], all_idx[:, :, :n_merge]

    with torch.no_grad():
        scores = (
            torch.matmul(k_block, k_ref.transpose(-1, -2)).float().max(dim=-1).values
        )
        sorted_idx = torch.argsort(scores, dim=-1, descending=False)
        append_idx = sorted_idx[..., :n_append]
        merge_idx = sorted_idx[..., n_append:]
        append_idx, _ = torch.sort(append_idx, dim=-1)
        merge_idx, _ = torch.sort(merge_idx, dim=-1)

    return append_idx, merge_idx


class SequenceMixer(ValueResidualMixin):
    def __init__(self, config, layer_idx: int):
        super().__init__(
            config,
            layer_idx,
            config.kvm_value_residual_mode,
            config.kvm_token_shift_mode,
            qk_norm=True,
        )

        # FIXME - this works for our current configs but won't if kvm is the alt layer
        self.rope_partial_dim = (
            config.rope_partial_dim if config.rope_partial_dim > 0 else self.d_qk_head
        )

        self.ln_s_k = set_label("scalars", nn.LayerNorm(self.d_qk_head))

        self.sink_len = config.sink_len
        self.chunk_len = config.chunk_len
        self.n_max_d_chunks = config.n_max_d_chunks
        self.n_bswa_chunks = config.n_bswa_chunks
        self.max_state_len = self.chunk_len * self.n_max_d_chunks
        self.state_budget_mode = config.state_budget_mode
        self.state_growth_factor = config.state_growth_factor
        self.state_growth_exponent = config.state_growth_exponent
        self.state_round_down = config.state_round_down
        self.state_min_len = config.state_min_len
        self.state_saturation_n = config.state_saturation_n

        if self.config.kvm_use_merge_gate_keys or self.config.kvm_use_merge_gate_values:
            self.key_weighting = set_label(
                "matrix_params", nn.Linear(config.hidden_size, self.num_attention_heads, bias=False)
            )
        if self.config.kvm_use_head_temps:
            self.front_head_temp = set_label(
                "scalars", nn.Parameter(torch.ones(self.config.num_attention_heads))
            )
            self.state_head_temp = set_label(
                "scalars", nn.Parameter(torch.ones(self.config.num_attention_heads))
            )

    def _prepare_state_update_k(self, k_block: torch.Tensor) -> torch.Tensor:
        return self.ln_s_k(
            torch.cat(
                [
                    torch.zeros_like(k_block[..., : self.rope_partial_dim]),
                    k_block[..., self.rope_partial_dim :],
                ],
                dim=-1,
            )
        )

    def _desired_state_len(
        self, ctx_len: int, available_context: int, current_state_len: int
    ) -> int:
        ctx_len_i = max(int(ctx_len), 0)
        available_context_i = max(int(available_context), 0)
        current_state_len_i = max(int(current_state_len), 0)

        if self.state_budget_mode == "fixed":
            target = self.state_min_len
        elif self.state_budget_mode == "power_law":
            target = int(
                math.floor(
                    self.state_growth_factor
                    * (float(ctx_len_i) ** self.state_growth_exponent)
                )
            )
            target = _round_down_to_multiple(target, self.state_round_down)
            target = max(target, self.state_min_len)
        elif self.state_budget_mode == "kvm_saturation":
            t = float(ctx_len_i)
            target = int(
                math.floor(
                    (float(self.state_saturation_n) * t)
                    / (float(self.state_saturation_n) + t)
                )
            )
            target = _round_down_to_multiple(target, self.state_round_down)
            target = max(target, self.state_min_len)
        else:
            raise ValueError(
                f"Unsupported state_budget_mode={self.state_budget_mode!r}"
            )

        target = min(target, available_context_i, self.max_state_len)
        return max(target, current_state_len_i)

    def _bswa_begin_for_total_len(self, total_len: int) -> int:
        # BSWA keeps the last N chunk-aligned blocks; the newest block may be partial.
        bswa_end = _round_down_to_multiple(
            total_len + self.chunk_len - 1, self.chunk_len
        )
        return max(0, bswa_end - (self.n_bswa_chunks * self.chunk_len))

    def _attend_with_state_and_bswa(
        self,
        q: torch.Tensor,
        bswa_k: torch.Tensor,
        bswa_v: torch.Tensor,
        s_k: torch.Tensor,
        s_v: torch.Tensor,
        s_vlen: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.kvm_use_head_temps:
            state_head_temp = self.state_head_temp.view(1, self.num_attention_heads, 1, 1)
            front_head_temp = self.front_head_temp.view(1, self.num_attention_heads, 1, 1)
        else:
            state_head_temp = 1.0
            front_head_temp = 1.0

        s_k_norm = self.ln_s_k(s_k)
        k_star = torch.cat(
            [s_k_norm * state_head_temp, bswa_k * front_head_temp], dim=2
        )

        if self.config.kvm_use_vlens:
            s_v_norm = (F.normalize(s_v.float(), dim=-1) * s_vlen).bfloat16()
        else:
            s_v_norm = s_v / s_vlen
        v_star = torch.cat([s_v_norm, bswa_v], dim=2)

        causal_mask = _lower_right_causal_mask(
            int(q.size(2)), int(k_star.size(2)), device=q.device
        )
        return F.scaled_dot_product_attention(
            q, k_star, v_star, attn_mask=causal_mask, is_causal=False
        )

    def _append_to_state(
        self,
        s_k: torch.Tensor,
        s_v: torch.Tensor,
        s_vlen: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(k_new.size(2)) == 0:
            return s_k, s_v, s_vlen
        if self.config.kvm_use_vlens:
            vlen_new = torch.norm(v_new.float(), dim=-1, keepdim=True)
        else:
            vlen_new = torch.ones_like(v_new[..., :1])
        s_k = torch.cat([s_k, k_new], dim=2)
        s_v = torch.cat([s_v, v_new], dim=2)
        s_vlen = torch.cat([s_vlen, vlen_new], dim=2)
        return s_k, s_v, s_vlen

    def _update_state_from_overflow_tokens(
        self,
        s_k: torch.Tensor,
        s_v: torch.Tensor,
        s_vlen: torch.Tensor,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        merge_gate: torch.Tensor,
        ctx_len: int,
        available_context: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(overflow_k.size(2)) == 0:
            return s_k, s_v, s_vlen

        overflow_k_ungated = self._prepare_state_update_k(overflow_k)
        overflow_v_ungated = overflow_v

        if self.config.kvm_use_merge_gate_keys:
            overflow_k_gated = overflow_k_ungated * merge_gate
        else:
            overflow_k_gated = overflow_k_ungated
        if self.config.kvm_use_merge_gate_values:
            overflow_v_gated = overflow_v_ungated * merge_gate
        else:
            overflow_v_gated = overflow_v_ungated

        if self.config.kvm_apply_merge_gate_to_appends:
            overflow_k_append = overflow_k_gated
            overflow_v_append = overflow_v_gated
        else:
            overflow_k_append = overflow_k_ungated
            overflow_v_append = overflow_v_ungated

        current_state_len = int(s_k.size(2))
        desired_state_len = self._desired_state_len(
            ctx_len, available_context, current_state_len
        )
        overflow_len = int(overflow_k_ungated.size(2))
        n_append = min(max(desired_state_len - current_state_len, 0), overflow_len)
        if n_append > 0:
            append_idx, merge_idx = _split_append_merge_idx_by_maxsim(
                overflow_k_ungated,
                n_append,
                self.ln_s_k(s_k),
            )
            k_append = _gather_by_idx(overflow_k_append, append_idx)
            v_append = _gather_by_idx(overflow_v_append, append_idx)
            k_merge = _gather_by_idx(overflow_k_gated, merge_idx)
            v_merge = _gather_by_idx(overflow_v_gated, merge_idx)
            s_k, s_v, s_vlen = self._append_to_state(
                s_k, s_v, s_vlen, k_append, v_append
            )
        else:
            k_merge = overflow_k_gated
            v_merge = overflow_v_gated

        if int(k_merge.size(2)) == 0:
            return s_k, s_v, s_vlen

        current_state_len = int(s_k.size(2))
        protected_slots = min(self.sink_len, current_state_len)
        if current_state_len <= protected_slots:
            raise AssertionError(
                "KVM state update requires at least one non-sink slot before merging."
            )

        s_k, s_v, s_vlen = self._merge_into_state(
            k_merge, v_merge, s_k, s_v, s_vlen, protected_slots
        )
        return s_k, s_v, s_vlen

    def _merge_into_state(self, k_merge, v_merge, s_k, s_v, s_vlen, protected_slots):
        # obtain normalized state keys
        s_k_norm = self.ln_s_k(s_k)

        # find the most similar key in state for each incoming key to merge
        logits = torch.matmul(k_merge, s_k_norm.transpose(-1, -2))
        logits[..., :protected_slots] = float("-inf")
        best_s_idx = logits.max(dim=-1, keepdim=True).indices
        scores = torch.scatter(
            torch.zeros_like(logits), -1, best_s_idx, torch.ones_like(logits)
        )

        # update state by adding the most similar keys and their values, gated by the merge gate
        s_k = s_k + (scores.mT @ k_merge)
        s_v = s_v + (scores.mT @ v_merge)

        if not self.config.kvm_use_vlens:
            s_vlen = s_vlen + scores.sum(-1, keepdim=True)

        return s_k, s_v, s_vlen

    def forward(
        self,
        x,
        v_first,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: StatesDictCache | None = None,
        **kwargs,
    ):
        q, k, v, merge_gate = self.forward_prefix(
            x, v_first, position_embeddings, attention_mask, past_key_values, **kwargs
        )
        y = self.forward_prefill(
            q,
            k,
            v,
            merge_gate,
            v_first,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )
        return y

    def forward_prefix(
        self,
        x,
        v_first,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        batch_size, q_seq_len, _ = x.size()
        # bswa_len = self.n_bswa_chunks * self.chunk_len

        q, k, v = self.calc_qkv(x, v_first, past_key_values)

        q = apply_rotary_embeddings(q, position_embeddings)
        k = apply_rotary_embeddings(k, position_embeddings)

        if self.config.kvm_use_merge_gate_keys or self.config.kvm_use_merge_gate_values:
            merge_gate = 1.0 + F.elu(
                self.key_weighting(x).view(batch_size, q_seq_len, self.num_attention_heads, 1)
            ).transpose(1, 2)
        else:
            merge_gate = torch.ones(
                (batch_size, self.num_attention_heads, q_seq_len, 1),
                device=x.device,
                dtype=torch.float,
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v, merge_gate

    def forward_prefill(
        self,
        q,
        k,
        v,
        merge_gate,
        v_first,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        batch_size, _, prefill_len, _ = q.size()
        bswa_len = self.n_bswa_chunks * self.chunk_len

        # calc initial attention
        if self.config.kvm_use_head_temps:
            front_head_temp = self.front_head_temp.view(1, self.num_attention_heads, 1, 1)
        else:
            front_head_temp = 1.0
        front_bswa_len = min(prefill_len, bswa_len)
        outs = [
            F.scaled_dot_product_attention(
                q[:, :, :front_bswa_len],
                k[:, :, :front_bswa_len] * front_head_temp,
                v[:, :, :front_bswa_len],
                is_causal=True,
            )
        ]

        # calc initial state
        init_state_len = min(prefill_len, self.chunk_len)
        s_k = self._prepare_state_update_k(k[..., :init_state_len, :])
        s_v = v[..., :init_state_len, :]
        if self.config.kvm_apply_merge_gate_to_initial_state:
            s_k = s_k * merge_gate[..., :init_state_len, :]
            s_v = s_v * merge_gate[..., :init_state_len, :]
        if self.config.kvm_use_vlens:
            s_vlen = torch.norm(
                v[..., :init_state_len, :].float(), dim=-1, keepdim=True
            )
        else:
            s_vlen = torch.ones_like(v[..., :init_state_len, :1])
        state_coverage_len = init_state_len

        # chunkwise processing for attention and state
        for query_begin in range(front_bswa_len, prefill_len, self.chunk_len):
            query_end = min(prefill_len, query_begin + self.chunk_len)
            bswa_begin = self._bswa_begin_for_total_len(query_end)
            if bswa_begin != state_coverage_len:
                raise AssertionError(
                    "KVM prefill state coverage drifted from the active BSWA window."
                )

            # calculate attention across the newly updated state and BSWA window
            out = self._attend_with_state_and_bswa(
                q[:, :, query_begin:query_end, :],
                k[:, :, bswa_begin:query_end, :],
                v[:, :, bswa_begin:query_end, :],
                s_k,
                s_v,
                s_vlen,
            )
            outs.append(out)

            # skip the final output state calculation during training
            if self.training and query_end >= prefill_len:
                break

            # update state
            next_bswa_begin = self._bswa_begin_for_total_len(
                min(prefill_len, query_end + self.chunk_len)
            )
            if next_bswa_begin > bswa_begin:
                s_k, s_v, s_vlen = self._update_state_from_overflow_tokens(
                    s_k,
                    s_v,
                    s_vlen,
                    overflow_k=k[:, :, bswa_begin:next_bswa_begin, :],
                    overflow_v=v[:, :, bswa_begin:next_bswa_begin, :],
                    merge_gate=merge_gate[:, :, bswa_begin:next_bswa_begin, :],
                    ctx_len=query_end,
                    available_context=next_bswa_begin,
                )
                state_coverage_len = next_bswa_begin

        y = torch.cat(outs, dim=-2)
        y = y.transpose(1, 2).contiguous().view(batch_size, prefill_len, -1)
        y = self.c_proj(y)

        if past_key_values is not None:
            bswa_begin = self._bswa_begin_for_total_len(prefill_len)
            expected_state_coverage_len = max(init_state_len, bswa_begin)
            if state_coverage_len != expected_state_coverage_len:
                raise AssertionError(
                    "KVM prefill state progression drifted from the decode-state bookkeeping."
                )
            past_key_values.update(
                self.layer_idx,
                offset=prefill_len,
                states_dict={
                    self._CACHE_S_K: s_k,
                    self._CACHE_S_V: s_v,
                    self._CACHE_S_VLEN: s_vlen,
                    self._CACHE_STATE_COVERAGE_LEN: state_coverage_len,
                    self._CACHE_BSWA_BEGIN: bswa_begin,
                    self._CACHE_BSWA_K: k[:, :, bswa_begin:, :],
                    self._CACHE_BSWA_V: v[:, :, bswa_begin:, :],
                    self._CACHE_BSWA_MERGE_GATE: merge_gate[:, :, bswa_begin:, :],
                },
            )

        return y

    _CACHE_S_K = "kvm_s_k"
    _CACHE_S_V = "kvm_s_v"
    _CACHE_S_VLEN = "kvm_s_vlen"
    _CACHE_STATE_COVERAGE_LEN = "kvm_state_coverage_len"
    _CACHE_BSWA_BEGIN = "kvm_bswa_begin"
    _CACHE_BSWA_K = "kvm_bswa_k"
    _CACHE_BSWA_V = "kvm_bswa_v"
    _CACHE_BSWA_MERGE_GATE = "kvm_bswa_merge_gate"

    def forward_inference(
        self,
        x,
        v_first,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: StatesDictCache | None = None,
        **kwargs,
    ):
        past_seq_len = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else 0
        )
        cached_decode = past_seq_len > 0

        q, k, v, merge_gate = self.forward_prefix(
            x, v_first, position_embeddings, attention_mask, past_key_values, **kwargs
        )

        if not cached_decode:
            return self.forward_prefill(
                q,
                k,
                v,
                merge_gate,
                v_first,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )
        else:
            return self.forward_single(
                q,
                k,
                v,
                merge_gate,
                v_first,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            )

    def forward_single(
        self,
        q,
        k,
        v,
        merge_gate,
        v_first,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        batch_size, _, q_seq_len, _ = q.size()
        bswa_len = self.n_bswa_chunks * self.chunk_len

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

        if q_seq_len != 1:
            raise AssertionError(
                "KVM cached decode expects single-token inputs. Multi-token decode with past_key_values is not supported."
            )

        s_k = cache_states[self._CACHE_S_K]
        s_v = cache_states[self._CACHE_S_V]
        s_vlen = cache_states[self._CACHE_S_VLEN]

        bswa_k = cache_states[self._CACHE_BSWA_K]
        bswa_v = cache_states[self._CACHE_BSWA_V]
        bswa_merge_gate = cache_states[self._CACHE_BSWA_MERGE_GATE]

        old_bswa_begin = int(cache_states[self._CACHE_BSWA_BEGIN])
        state_coverage_len = int(cache_states[self._CACHE_STATE_COVERAGE_LEN])

        full_bswa_k = torch.cat([bswa_k, k], dim=2)
        full_bswa_v = torch.cat([bswa_v, v], dim=2)
        full_bswa_merge_gate = torch.cat([bswa_merge_gate, merge_gate], dim=2)

        new_total_len = past_seq_len + q_seq_len
        # the 'direct' state is the part of the initial chunk that can be copied directly into the state
        direct_state_target = min(new_total_len, self.chunk_len)
        new_bswa_begin = self._bswa_begin_for_total_len(new_total_len)
        new_state_coverage_len = max(direct_state_target, new_bswa_begin)

        # if any of the initial state chunk is still missing, append it
        if state_coverage_len < direct_state_target:
            full_bswa_rel_begin = max(state_coverage_len - old_bswa_begin, 0)
            full_bswa_rel_end = max(direct_state_target - old_bswa_begin, 0)
            direct_k = self._prepare_state_update_k(
                full_bswa_k[:, :, full_bswa_rel_begin:full_bswa_rel_end, :]
            )
            direct_v = full_bswa_v[:, :, full_bswa_rel_begin:full_bswa_rel_end, :]
            s_k, s_v, s_vlen = self._append_to_state(
                s_k,
                s_v,
                s_vlen,
                direct_k,
                direct_v,
            )
            state_coverage_len = direct_state_target

        if state_coverage_len > new_state_coverage_len:
            raise AssertionError(
                "KVM cache state coverage cannot shrink during decoding."
            )

        # if we have transitioned across a chunk boundary and need more state, update the state from this chunk
        if state_coverage_len < new_state_coverage_len:
            full_bswa_rel_begin = max(state_coverage_len - old_bswa_begin, 0)
            full_bswa_rel_end = max(new_state_coverage_len - old_bswa_begin, 0)
            overflow_k = full_bswa_k[:, :, full_bswa_rel_begin:full_bswa_rel_end, :]
            overflow_v = full_bswa_v[:, :, full_bswa_rel_begin:full_bswa_rel_end, :]
            merge_gate = full_bswa_merge_gate[
                :, :, full_bswa_rel_begin:full_bswa_rel_end, :
            ]
            overflow_len = int(overflow_k.size(2))
            if overflow_len != self.chunk_len:
                raise AssertionError(
                    "KVM decode can only materialize one overflow BSWA chunk per step."
                )
            s_k, s_v, s_vlen = self._update_state_from_overflow_tokens(
                s_k,
                s_v,
                s_vlen,
                overflow_k=overflow_k,
                overflow_v=overflow_v,
                merge_gate=merge_gate,
                ctx_len=(self.n_bswa_chunks * self.chunk_len) + int(state_coverage_len),
                available_context=int(state_coverage_len) + self.chunk_len,
            )
            state_coverage_len = new_state_coverage_len
        if state_coverage_len != new_state_coverage_len:
            raise AssertionError(
                "KVM decode state coverage drifted from the BSWA boundary."
            )

        # update the tokens in our bswa window
        full_bswa_rel_begin = max(new_bswa_begin - old_bswa_begin, 0)
        full_bswa_rel_end = max(new_total_len - old_bswa_begin, 0)
        new_bswa_k = full_bswa_k[:, :, full_bswa_rel_begin:full_bswa_rel_end, :]
        new_bswa_v = full_bswa_v[:, :, full_bswa_rel_begin:full_bswa_rel_end, :]
        new_bswa_merge_gate = full_bswa_merge_gate[
            :, :, full_bswa_rel_begin:full_bswa_rel_end, :
        ]

        # if the new total length still fits in BSWA, we can attend to everything with a single efficient call; otherwise we need to include state in attention
        if new_total_len <= bswa_len:
            if self.config.kvm_use_head_temps:
                front_head_temp = self.front_head_temp.view(1, self.num_attention_heads, 1, 1)
            else:
                front_head_temp = 1.0
            causal_mask = _lower_right_causal_mask(
                int(q.size(2)), int(new_bswa_k.size(2)), device=q.device
            )
            out = F.scaled_dot_product_attention(
                q,
                new_bswa_k * front_head_temp,
                new_bswa_v,
                attn_mask=causal_mask,
                is_causal=False,
            )
        else:
            out = self._attend_with_state_and_bswa(
                q, new_bswa_k, new_bswa_v, s_k, s_v, s_vlen
            )

        y = out.transpose(1, 2).contiguous().view(batch_size, q_seq_len, -1)
        y = self.c_proj(y)

        past_key_values.update(
            self.layer_idx,
            offset=q_seq_len,
            states_dict={
                self._CACHE_S_K: s_k,
                self._CACHE_S_V: s_v,
                self._CACHE_S_VLEN: s_vlen,
                self._CACHE_STATE_COVERAGE_LEN: state_coverage_len,
                self._CACHE_BSWA_BEGIN: new_bswa_begin,
                self._CACHE_BSWA_K: new_bswa_k,
                self._CACHE_BSWA_V: new_bswa_v,
                self._CACHE_BSWA_MERGE_GATE: new_bswa_merge_gate,
            },
        )
        return y
