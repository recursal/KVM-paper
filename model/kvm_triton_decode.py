"""Preallocated cached-decode state for the Triton KVM implementation.

This module is shared by fixed and adaptive state schedules.  Schedule policy
only changes the active state length at recurrent-update boundaries; cache
maintenance and state-plus-front attention are otherwise identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from model.kernels.kvm_triton_decode import write_recent_ring


@dataclass(frozen=True)
class DecodeCacheSnapshot:
    """Immutable tensors and scalar bookkeeping used to reset timed trials."""

    state_k: torch.Tensor
    state_v: torch.Tensor
    state_vlen: torch.Tensor
    recent_k: torch.Tensor
    recent_v: torch.Tensor
    recent_gate: torch.Tensor
    total_len: int
    state_coverage_len: int
    recent_begin: int


class TritonKVMDecodeCache:
    """Optimized one-token cache used by the Triton KVM mixer.

    ``state_capacity`` can be a bounded-generation capacity hint. If an
    adaptive schedule later outgrows it, the state and update workspace grow
    at that recurrent-update boundary; ordinary token steps remain
    allocation-free.
    """

    cache_representation = "preallocated double-mapped 512-token recent ring"

    def __init__(
        self,
        mixer: Any,
        snapshot: DecodeCacheSnapshot,
        *,
        state_capacity: int,
    ) -> None:
        if not snapshot.state_k.is_cuda:
            raise RuntimeError("Triton KVM decode requires CUDA/ROCm tensors")
        self.mixer = mixer
        self.batch = int(snapshot.state_k.size(0))
        self.kv_heads = int(snapshot.state_k.size(1))
        self.q_heads = int(mixer.num_attention_heads)
        self.head_dim = int(mixer.d_qk_head)
        self.value_dim = int(mixer.d_v_head)
        self.chunk_len = int(mixer.chunk_len)
        self.ring_capacity = int(mixer.n_bswa_chunks * mixer.chunk_len)
        self.state_capacity = int(state_capacity)
        self.kv_rows = self.batch * self.kv_heads
        self.device = snapshot.state_k.device
        self.dtype = snapshot.state_k.dtype
        initial_state_len = int(snapshot.state_k.size(2))
        if self.state_capacity < initial_state_len:
            raise ValueError("state capacity is smaller than the initial state")
        initial_recent_len = int(snapshot.recent_k.size(2))
        if initial_recent_len > self.ring_capacity:
            raise ValueError("initial recent front exceeds the ring capacity")

        state_shape_k = (
            self.kv_rows,
            self.state_capacity,
            self.head_dim,
        )
        state_shape_v = (
            self.kv_rows,
            self.state_capacity,
            self.value_dim,
        )
        self.state_k = torch.empty(state_shape_k, device=self.device, dtype=self.dtype)
        self.state_v = torch.empty(state_shape_v, device=self.device, dtype=self.dtype)
        self.state_k_attn = torch.empty_like(self.state_k)
        self.state_v_attn = torch.empty_like(self.state_v)
        self.state_vlen = torch.empty(
            self.kv_rows,
            self.state_capacity,
            device=self.device,
            dtype=torch.float32,
        )
        self.recent_k = torch.empty(
            self.kv_rows,
            2 * self.ring_capacity,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.recent_v = torch.empty(
            self.kv_rows,
            2 * self.ring_capacity,
            self.value_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.recent_gate = torch.empty(
            self.kv_rows,
            2 * self.ring_capacity,
            device=self.device,
            dtype=self.dtype,
        )
        self.out = torch.empty(
            self.batch,
            self.q_heads,
            1,
            self.value_dim,
            device=self.device,
            dtype=self.dtype,
        )

        overflow_key_shape = (self.kv_rows, self.chunk_len, self.head_dim)
        overflow_value_shape = (self.kv_rows, self.chunk_len, self.value_dim)
        self.overflow_select_k = torch.empty(
            overflow_key_shape, device=self.device, dtype=self.dtype
        )
        self.overflow_zeroed_k = torch.empty_like(self.overflow_select_k)
        self.overflow_append_k = torch.empty_like(self.overflow_select_k)
        self.overflow_merge_k = torch.empty_like(self.overflow_select_k)
        self.overflow_raw_v = torch.empty(
            overflow_value_shape, device=self.device, dtype=self.dtype
        )
        self.overflow_append_v = torch.empty(
            overflow_value_shape, device=self.device, dtype=self.dtype
        )
        self.overflow_merge_v = torch.empty_like(self.overflow_append_v)
        self.overflow_gate = torch.empty(
            self.kv_rows,
            self.chunk_len,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        norm_stat_shape = (self.kv_rows, self.chunk_len, 1)
        self.overflow_ln_mean = torch.empty(
            norm_stat_shape, device=self.device, dtype=torch.float32
        )
        self.overflow_ln_rstd = torch.empty_like(self.overflow_ln_mean)

        self.update_args = mixer._make_triton_args(self.batch, self.chunk_len)
        self.update_args.max_state_len = self.state_capacity
        self.update_args.initial_state_len = initial_state_len
        self.update_buffers = self._allocate_update_buffers(initial_state_len)
        route_shape = (self.kv_rows, 1, self.chunk_len)
        self.append_pos_by_token = torch.full(
            route_shape, -1, device=self.device, dtype=torch.int32
        )
        self.best_idx_by_token = torch.full_like(self.append_pos_by_token, -1)
        self.undo_k_by_token = torch.empty(
            *route_shape,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.undo_v_by_token = torch.empty(
            *route_shape,
            self.value_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.before_device = torch.empty(1, device=self.device, dtype=torch.int32)
        self.after_device = torch.empty_like(self.before_device)
        self.n_append_device = torch.empty_like(self.before_device)
        self.state_temperature, self.front_temperature = mixer._head_temperatures(
            self.device
        )
        self.ln_weight_fp32 = mixer.ln_s_k.weight.float()
        self.ln_bias_fp32 = mixer.ln_s_k.bias.float()
        self.reset_(snapshot)

    def _allocate_update_buffers(self, current_state_len: int) -> dict[str, Any]:
        # Keep the training-kernel import lazy so the mixer can select the
        # architecture-compatible packaged forward before Triton's JITFunction
        # and compile hook are initialized.
        from model.kernels import kvm_triton_training_kernels as update_kernels

        allocation_state_after = min(
            self.state_capacity,
            int(current_state_len) + self.chunk_len,
        )
        allocation_schedule = self.mixer._make_generation_update_schedule(
            current_state_len=int(current_state_len),
            state_after=allocation_state_after,
            n_append=allocation_state_after - int(current_state_len),
            overflow_len=self.chunk_len,
        )
        self.update_args.max_state_len = self.state_capacity
        return update_kernels.allocate_work_buffers(
            self.update_args, allocation_schedule, self.device
        )

    def _reserve_state_capacity(self, required: int) -> None:
        """Grow adaptive-state storage at a chunk boundary when necessary."""
        required = int(required)
        if required <= self.state_capacity:
            return
        if required > int(self.mixer.max_state_len):
            raise RuntimeError(
                f"Triton KVM decode requires {required} state rows, exceeding "
                f"the configured maximum {self.mixer.max_state_len}"
            )

        lookahead_capacity = self.state_length_after_cycle(
            self.mixer,
            self.total_len + 1,
            required,
            self.ring_capacity,
        )
        new_capacity = min(
            int(self.mixer.max_state_len),
            max(required, lookahead_capacity),
        )

        def grow(source: torch.Tensor, tail: tuple[int, ...]) -> torch.Tensor:
            destination = source.new_empty((self.kv_rows, new_capacity, *tail))
            destination[:, : self.state_len].copy_(source[:, : self.state_len])
            return destination

        self.state_k = grow(self.state_k, (self.head_dim,))
        self.state_v = grow(self.state_v, (self.value_dim,))
        self.state_k_attn = grow(self.state_k_attn, (self.head_dim,))
        self.state_v_attn = grow(self.state_v_attn, (self.value_dim,))
        grown_vlen = self.state_vlen.new_empty(self.kv_rows, new_capacity)
        grown_vlen[:, : self.state_len].copy_(
            self.state_vlen[:, : self.state_len]
        )
        self.state_vlen = grown_vlen
        self.state_capacity = new_capacity
        self.update_buffers = self._allocate_update_buffers(self.state_len)

    @staticmethod
    def state_length_after_cycle(
        mixer: Any, total_len: int, state_len: int, cycle_len: int | None = None
    ) -> int:
        """Return the required state capacity for a bounded decode interval."""
        steps = mixer.chunk_len if cycle_len is None else int(cycle_len)
        coverage = max(
            min(int(total_len), mixer.chunk_len),
            mixer._bswa_begin_for_total_len(int(total_len)),
        )
        active = int(state_len)
        for step in range(1, steps + 1):
            next_begin = mixer._bswa_begin_for_total_len(int(total_len) + step)
            if next_begin <= coverage:
                continue
            active = mixer._desired_state_len(
                (mixer.n_bswa_chunks * mixer.chunk_len) + coverage,
                coverage + mixer.chunk_len,
                active,
            )
            coverage = next_begin
        return active

    def reset_(self, snapshot: DecodeCacheSnapshot) -> None:
        """Restore a snapshot; callers keep this outside measured GPU events."""
        state_len = int(snapshot.state_k.size(2))
        recent_len = int(snapshot.recent_k.size(2))
        if state_len > self.state_capacity or recent_len > self.ring_capacity:
            raise ValueError("snapshot does not fit the preallocated decode cache")
        flat_state_k = snapshot.state_k.reshape(
            self.kv_rows, state_len, self.head_dim
        )
        flat_state_v = snapshot.state_v.reshape(
            self.kv_rows, state_len, self.value_dim
        )
        flat_state_vlen = snapshot.state_vlen.reshape(self.kv_rows, state_len)
        self.state_k[:, :state_len].copy_(flat_state_k)
        self.state_v[:, :state_len].copy_(flat_state_v)
        self.state_vlen[:, :state_len].copy_(flat_state_vlen.float())
        normalized_k = self.mixer.ln_s_k(snapshot.state_k).reshape(
            self.kv_rows, state_len, self.head_dim
        )
        normalized_v = (
            F.normalize(snapshot.state_v.float(), dim=-1)
            * snapshot.state_vlen.float()
        ).to(self.dtype).reshape(self.kv_rows, state_len, self.value_dim)
        self.state_k_attn[:, :state_len].copy_(normalized_k)
        self.state_v_attn[:, :state_len].copy_(normalized_v)

        flat_recent_k = snapshot.recent_k.reshape(
            self.kv_rows, recent_len, self.head_dim
        )
        flat_recent_v = snapshot.recent_v.reshape(
            self.kv_rows, recent_len, self.value_dim
        )
        flat_recent_gate = snapshot.recent_gate.reshape(self.kv_rows, recent_len)
        self.recent_k[:, :recent_len].copy_(flat_recent_k)
        self.recent_k[:, self.ring_capacity : self.ring_capacity + recent_len].copy_(
            flat_recent_k
        )
        self.recent_v[:, :recent_len].copy_(flat_recent_v)
        self.recent_v[:, self.ring_capacity : self.ring_capacity + recent_len].copy_(
            flat_recent_v
        )
        self.recent_gate[:, :recent_len].copy_(flat_recent_gate)
        self.recent_gate[
            :, self.ring_capacity : self.ring_capacity + recent_len
        ].copy_(flat_recent_gate)
        self.total_len = int(snapshot.total_len)
        self.state_len = state_len
        self.state_coverage_len = int(snapshot.state_coverage_len)
        self.recent_begin = int(snapshot.recent_begin)
        self.recent_start = 0
        self.recent_len = recent_len
        self.update_count = 0
        self.append_pos_by_token.fill_(-1)
        self.best_idx_by_token.fill_(-1)

    def append_recent_(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        new_gate: torch.Tensor,
        *,
        total_len: int,
        recent_begin: int,
    ) -> None:
        """Append one token to the recent ring and update cache bookkeeping."""
        evicted = int(recent_begin) - self.recent_begin
        if evicted < 0 or evicted > self.recent_len:
            raise AssertionError("invalid recent-window transition")
        if evicted:
            self.recent_start = (self.recent_start + evicted) % self.ring_capacity
            self.recent_len -= evicted
        if self.recent_len >= self.ring_capacity:
            raise AssertionError("recent ring append would exceed capacity")
        write_pos = (self.recent_start + self.recent_len) % self.ring_capacity
        write_recent_ring(
            new_k,
            new_v,
            new_gate,
            self.recent_k,
            self.recent_v,
            self.recent_gate,
            write_pos,
        )
        self.recent_len += 1
        self.total_len = int(total_len)
        self.recent_begin = int(recent_begin)

    def logical_recent(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return views of the logical recent window for validation only."""
        begin = self.recent_start
        end = begin + self.recent_len
        return (
            self.recent_k[:, begin:end],
            self.recent_v[:, begin:end],
            self.recent_gate[:, begin:end],
        )

    def logical_state_dict(self) -> dict[str, Any]:
        """Expose the logical cache contents for validation and inspection."""
        recent_k, recent_v, recent_gate = self.logical_recent()
        mixer = self.mixer
        return {
            mixer._CACHE_S_K: self.state_k[:, : self.state_len].reshape(
                self.batch, self.kv_heads, self.state_len, self.head_dim
            ),
            mixer._CACHE_S_V: self.state_v[:, : self.state_len].reshape(
                self.batch, self.kv_heads, self.state_len, self.value_dim
            ),
            mixer._CACHE_S_VLEN: self.state_vlen[:, : self.state_len].reshape(
                self.batch, self.kv_heads, self.state_len, 1
            ),
            mixer._CACHE_STATE_COVERAGE_LEN: self.state_coverage_len,
            mixer._CACHE_BSWA_BEGIN: self.recent_begin,
            mixer._CACHE_BSWA_K: recent_k.reshape(
                self.batch, self.kv_heads, self.recent_len, self.head_dim
            ),
            mixer._CACHE_BSWA_V: recent_v.reshape(
                self.batch, self.kv_heads, self.recent_len, self.value_dim
            ),
            mixer._CACHE_BSWA_MERGE_GATE: recent_gate.reshape(
                self.batch, self.kv_heads, self.recent_len, 1
            ),
        }

    def persistent_metadata(self) -> dict[str, Any]:
        return {
            "total_len": self.total_len,
            "active_state_len": self.state_len,
            "state_capacity": self.state_capacity,
            "state_coverage_len": self.state_coverage_len,
            "recent_begin": self.recent_begin,
            "recent_start": self.recent_start,
            "recent_len": self.recent_len,
            "recent_capacity": self.ring_capacity,
            "update_count": self.update_count,
            "cache_representation": self.cache_representation,
        }


__all__ = ["TritonKVMDecodeCache", "DecodeCacheSnapshot"]
