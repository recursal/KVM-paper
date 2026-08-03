from __future__ import annotations

import argparse
import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd import Function

from .kvm_triton_decode import TritonKVMDecodeCache, DecodeCacheSnapshot
from .kvm_mixer import SequenceMixer as TorchKVMSequenceMixer



def _choose_dividing_block(size: int, candidates: tuple[int, ...]) -> int:
    for candidate in candidates:
        if candidate <= size and size % candidate == 0:
            return candidate
    raise ValueError(f"no supported block size divides {size}")


class _KvmTritonTrainingFunction(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_flat: torch.Tensor,
        bswa_k_flat: torch.Tensor,
        bswa_v_flat: torch.Tensor,
        overflow_k_flat: torch.Tensor,
        overflow_v_flat: torch.Tensor,
        overflow_select_k_flat: torch.Tensor,
        overflow_append_k_flat: torch.Tensor,
        overflow_append_v_flat: torch.Tensor,
        overflow_merge_k_flat: torch.Tensor,
        overflow_merge_v_flat: torch.Tensor,
        initial_k_flat: torch.Tensor,
        initial_v_flat: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        state_temperature: torch.Tensor,
        front_temperature: torch.Tensor,
        triton_args: argparse.Namespace,
        schedule: Any,
    ) -> torch.Tensor:
        from model.kernels.kvm_triton_training_kernels import (
            build_prefill_forward,
        )

        if not q_flat.is_cuda:
            raise RuntimeError("kvm_triton_mixer requires CUDA/ROCm tensors")

        forward = build_prefill_forward(
            triton_args,
            schedule,
            q_flat,
            overflow_k_flat,
            overflow_v_flat,
            bswa_k_flat,
            bswa_v_flat,
            ln_weight,
            ln_bias,
            state_temperature,
            front_temperature,
            initial_k_flat=initial_k_flat,
            initial_v_flat=initial_v_flat,
            overflow_select_k_flat=overflow_select_k_flat,
            overflow_append_k_flat=overflow_append_k_flat,
            overflow_append_v_flat=overflow_append_v_flat,
            overflow_merge_k_flat=overflow_merge_k_flat,
            overflow_merge_v_flat=overflow_merge_v_flat,
        )
        ctx.triton_args = triton_args
        ctx.schedule = schedule
        ctx.save_for_backward(
            q_flat,
            bswa_k_flat,
            bswa_v_flat,
            overflow_k_flat,
            overflow_v_flat,
            overflow_select_k_flat,
            overflow_append_k_flat,
            overflow_append_v_flat,
            overflow_merge_k_flat,
            overflow_merge_v_flat,
            initial_k_flat,
            initial_v_flat,
            ln_weight,
            ln_bias,
            state_temperature,
            front_temperature,
            forward["out"],
            forward["lse"],
            forward["state_k"],
            forward["state_v"],
            forward["state_k_attn"],
            forward["state_v_attn"],
            forward["state_vlen"],
            forward["append_pos_by_token"],
            forward["best_idx_by_token"],
            forward["undo_k_by_token"],
            forward["undo_v_by_token"],
        )
        return forward["out"]

    @staticmethod
    def backward(ctx, dout: torch.Tensor):  # type: ignore[override]
        from model.kernels.kvm_triton_training_kernels import (
            run_training_backward_reconstruct_live_state,
        )

        (
            q_flat,
            bswa_k_flat,
            bswa_v_flat,
            overflow_k_flat,
            overflow_v_flat,
            overflow_select_k_flat,
            overflow_append_k_flat,
            overflow_append_v_flat,
            overflow_merge_k_flat,
            overflow_merge_v_flat,
            initial_k_flat,
            initial_v_flat,
            ln_weight,
            ln_bias,
            state_temperature,
            front_temperature,
            out,
            lse,
            state_k,
            state_v,
            state_k_attn,
            state_v_attn,
            state_vlen,
            append_pos_by_token,
            best_idx_by_token,
            undo_k_by_token,
            undo_v_by_token,
        ) = ctx.saved_tensors
        saved_forward = {
            "out": out,
            "lse": lse,
            "state_k": state_k,
            "state_v": state_v,
            "state_k_attn": state_k_attn,
            "state_v_attn": state_v_attn,
            "state_vlen": state_vlen,
            "append_pos_by_token": append_pos_by_token,
            "best_idx_by_token": best_idx_by_token,
            "undo_k_by_token": undo_k_by_token,
            "undo_v_by_token": undo_v_by_token,
        }
        result = run_training_backward_reconstruct_live_state(
            ctx.triton_args,
            ctx.schedule,
            q_flat,
            bswa_k_flat,
            bswa_v_flat,
            overflow_k_flat,
            ln_weight,
            ln_bias,
            dout.contiguous(),
            state_temperature,
            front_temperature,
            initial_k_flat=initial_k_flat,
            initial_v_flat=initial_v_flat,
            overflow_select_k_flat=overflow_select_k_flat,
            overflow_append_k_flat=overflow_append_k_flat,
            overflow_append_v_flat=overflow_append_v_flat,
            overflow_merge_k_flat=overflow_merge_k_flat,
            overflow_merge_v_flat=overflow_merge_v_flat,
            saved_forward=saved_forward,
        )
        return (
            result["dq"].to(q_flat.dtype),
            result["d_bswa_k"].to(bswa_k_flat.dtype),
            result["d_bswa_v"].to(bswa_v_flat.dtype),
            result["d_overflow_k"].to(overflow_k_flat.dtype),
            torch.zeros_like(overflow_v_flat),
            torch.zeros_like(overflow_select_k_flat),
            result["d_append_k"].to(overflow_append_k_flat.dtype),
            result["d_append_v"].to(overflow_append_v_flat.dtype),
            result["d_merge_k"].to(overflow_merge_k_flat.dtype),
            result["d_merge_v"].to(overflow_merge_v_flat.dtype),
            result["d_initial_k"].to(initial_k_flat.dtype),
            result["d_initial_v"].to(initial_v_flat.dtype),
            result["d_ln_weight"].to(ln_weight.dtype),
            result["d_ln_bias"].to(ln_bias.dtype),
            result["d_state_temperature"].to(state_temperature.dtype),
            result["d_front_temperature"].to(front_temperature.dtype),
            None,
            None,
        )


class SequenceMixer(TorchKVMSequenceMixer):
    """KVM backed by the optimized Triton training/prefill kernels.

    Append selection is ranked globally across each overflow chunk, and
    selected tokens are appended before merge targets are chosen. These are
    the eager ``kvm_mixer.py`` routing semantics.

    Remaining intentional differences from the dense PyTorch implementation:
    - `kvm_use_vlens=1` is required;
    - prefill inputs are padded to the next chunk for Triton launch, but the
      schedule and returned outputs use the real ragged length;
    - state updates use FP32 merge accumulation, one rounded BF16 state write,
      and cached normalized state;
    - training treats append/merge routing as fixed non-differentiable metadata.
    """

    _CACHE_TRITON_DECODE = "kvm_triton_decode_cache"

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        if not config.kvm_use_vlens:
            raise ValueError("kvm_triton_mixer currently requires kvm_use_vlens=1")

        self.triton_sub_block = _choose_dividing_block(
            self.chunk_len, (128, 64, 32, 16)
        )
        self.triton_attn_block = _choose_dividing_block(
            self.chunk_len, (64, 128, 32, 16)
        )
        self.triton_state_chunk = 16
        self.triton_attn_state_chunk = 64
        self.triton_group_chunks = 12
        self.triton_update_token_block = min(8, self.triton_sub_block)
        if self.triton_sub_block % self.triton_update_token_block:
            raise ValueError("internal Triton update token block must divide sub_block")

    def _triton_state_schedule_params(
        self,
    ) -> tuple[str, float, float, int, int | None]:
        if self.state_budget_mode == "fixed":
            return "fixed", 0.0, 1.0, self.state_min_len, None
        if self.state_budget_mode == "power_law":
            return (
                "power_law",
                float(self.state_growth_factor),
                float(self.state_growth_exponent),
                int(self.state_min_len),
                None,
            )
        if self.state_budget_mode == "kvm_saturation":
            return (
                "kvm_saturation",
                0.0,
                1.0,
                int(self.state_min_len),
                self.state_saturation_n,
            )
        raise ValueError(f"unsupported state_budget_mode={self.state_budget_mode!r}")

    def _make_triton_args(self, batch_size: int, q_len: int) -> argparse.Namespace:
        padded_q_len = int(math.ceil(q_len / self.chunk_len) * self.chunk_len)
        q_rows = batch_size * self.num_attention_heads
        # MI325X profiles favor two warps once the query-row grid is large
        # enough to fill the device.  Smaller grids need four warps to retain
        # occupancy.  KV-owned backward has a much larger state-chunk grid and
        # remains faster at four warps.
        attn_num_warps = 2 if q_rows >= 128 else 4
        (
            state_budget_mode,
            schedule_factor,
            schedule_exponent,
            state_min_len,
            state_saturation_n,
        ) = self._triton_state_schedule_params()
        wide_state_scan = (
            q_rows >= 128
            and state_budget_mode == "power_law"
            and schedule_factor == 16.0
            and schedule_exponent == 0.5
        )
        return argparse.Namespace(
            batch=batch_size,
            q_heads=self.num_attention_heads,
            kv_heads=self.num_key_value_heads,
            q_len=padded_q_len,
            logical_q_len=q_len,
            initial_state_len=min(q_len, self.chunk_len),
            # Let the Triton helper size temporary buffers from the actual
            # prefill schedule. The config max can be millions of slots and is
            # only a capacity cap, not the number of active state rows.
            max_state_len=0,
            state_capacity=self.max_state_len,
            state_chunk=self.triton_state_chunk,
            scan_state_chunk=32 if wide_state_scan else self.triton_state_chunk,
            attn_state_chunk=self.triton_attn_state_chunk,
            group_chunks=8 if wide_state_scan else (
                16 if q_rows >= 128 else self.triton_group_chunks
            ),
            update_token_block=self.triton_update_token_block,
            split_append_target_scan=True,
            macro_block=self.chunk_len,
            bswa_chunks=self.n_bswa_chunks,
            sub_block=self.triton_sub_block,
            attn_block=self.triton_attn_block,
            schedule_factor=schedule_factor,
            schedule_exponent=schedule_exponent,
            state_budget_mode=state_budget_mode,
            state_min_len=state_min_len,
            state_saturation_n=state_saturation_n,
            state_round_down=self.state_round_down,
            dim=self.d_qk_head,
            value_dim=self.d_v_head,
            sink_len=self.sink_len,
            ln_eps=float(self.ln_s_k.eps),
            scan_num_warps=4 if q_rows >= 128 else 8,
            update_num_warps=8,
            attn_num_warps=attn_num_warps,
            attn_kv_num_warps=4,
            waves_per_eu=1,
            scan_waves_per_eu=0,
            update_waves_per_eu=0,
            attn_waves_per_eu=1,
            q_head_loop_unroll_factor=4,
            skip_temperature_grad=False,
            skip_temperature_atomic=False,
            temperature_grad_backend="atomic",
            update_grad_backend="triton",
            attn_grad_backend="kv-owned",
            undo_mode="stash",
            cache_from_rounded_state=True,
            reconstruct_live_state_backward=True,
            fuse_restore_refresh=True,
            fuse_state_dkdv_raw=False,
            append_score_precision="bf16_rounded",
            route_score_precision="fp32",
            aotriton_forward_attention=bool(
                self.config.kvm_aotriton_forward_attention
            ),
            append_policy="global",
            merge_order="append_before_merge",
        )

    def _make_schedule(self, triton_args: argparse.Namespace):
        from model.kernels.kvm_triton_training_kernels import make_schedule

        return make_schedule(triton_args)

    def _flatten_qkv_for_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        merge_gate: torch.Tensor,
        padded_len: int,
    ) -> dict[str, torch.Tensor]:
        """Pad, prepare, and flatten the streams consumed by Triton kernels.

        Stream preparation uses regular PyTorch when gradients or general tensor
        shapes require it. BF16 inference with equal key/value dimensions uses an
        equivalent fused Triton kernel; the stream selection and layout below are
        shared by both paths.
        """
        needs_grad = torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (k, v, merge_gate, self.ln_s_k.weight, self.ln_s_k.bias)
        )
        use_fused_preparation = (
            not needs_grad
            and q.is_cuda
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and self.d_qk_head == self.d_v_head
        )
        batch_size, q_heads, q_len, _ = q.shape
        kv_heads = int(k.size(1))
        if q_heads != self.num_attention_heads:
            raise AssertionError("KVM Triton query head count mismatch.")
        if kv_heads != self.num_key_value_heads:
            raise AssertionError("KVM Triton key/value head count mismatch.")
        pad_len = padded_len - q_len
        if pad_len < 0:
            raise ValueError("padded_len must be >= q_len")
        if pad_len:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            merge_gate = F.pad(merge_gate, (0, 0, 0, pad_len), value=1.0)

        q_flat = q.reshape(
            batch_size * q_heads, padded_len, self.d_qk_head
        ).contiguous()
        k_flat = k.reshape(
            batch_size * kv_heads, padded_len, self.d_qk_head
        ).contiguous()
        v_flat = v.reshape(
            batch_size * kv_heads, padded_len, self.d_v_head
        ).contiguous()
        gate_flat = merge_gate.reshape(
            batch_size * kv_heads, padded_len, 1
        ).contiguous()

        if use_fused_preparation:
            from model.kernels.kvm_triton_training_kernels import (
                prepare_kvm_streams,
            )

            prepared_k, gated_k, gated_v = prepare_kvm_streams(
                k,
                v,
                merge_gate,
                self.ln_s_k.weight,
                self.ln_s_k.bias,
                rope_partial_dim=self.rope_partial_dim,
                ln_eps=float(self.ln_s_k.eps),
            )
            prepared_k = prepared_k.reshape(
                batch_size * kv_heads, padded_len, self.d_qk_head
            )
            gated_k = gated_k.reshape(
                batch_size * kv_heads, padded_len, self.d_qk_head
            )
            gated_v = gated_v.reshape(
                batch_size * kv_heads, padded_len, self.d_v_head
            )
        else:
            prepared_k = self._prepare_state_update_k(k).reshape(
                batch_size * kv_heads, padded_len, self.d_qk_head
            ).contiguous()
            gated_k = (prepared_k * gate_flat).to(prepared_k.dtype)
            gated_v = (v_flat * gate_flat).to(v_flat.dtype)

        merge_k = gated_k if self.config.kvm_use_merge_gate_keys else prepared_k
        merge_v = gated_v if self.config.kvm_use_merge_gate_values else v_flat
        append_k = merge_k if self.config.kvm_apply_merge_gate_to_appends else prepared_k
        append_v = merge_v if self.config.kvm_apply_merge_gate_to_appends else v_flat
        if self.config.kvm_apply_merge_gate_to_initial_state:
            initial_k = (
                gated_k
                if self.config.kvm_use_merge_gate_keys
                else (prepared_k * gate_flat).to(prepared_k.dtype)
            )
            initial_v = (
                gated_v
                if self.config.kvm_use_merge_gate_values
                else (v_flat * gate_flat).to(v_flat.dtype)
            )
        else:
            initial_k = prepared_k
            initial_v = v_flat

        return {
            "q": q_flat,
            "bswa_k": k_flat,
            "bswa_v": v_flat,
            "select_k": prepared_k,
            "append_k": append_k,
            "append_v": append_v,
            "merge_k": merge_k,
            "merge_v": merge_v,
            "initial_k": initial_k,
            "initial_v": initial_v,
        }

    def _head_temperatures(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.kvm_use_head_temps:
            return (
                self.state_head_temp.float(),
                self.front_head_temp.float(),
            )
        ones = torch.ones(self.num_attention_heads, device=device, dtype=torch.float32)
        return ones, ones

    def _triton_forward_prefill_raw(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        merge_gate: torch.Tensor,
        triton_args: argparse.Namespace,
        schedule,
    ) -> dict[str, torch.Tensor]:
        flat = self._flatten_qkv_for_triton(q, k, v, merge_gate, triton_args.q_len)
        return self._triton_forward_prefill_raw_from_flat(flat, triton_args, schedule)

    def _triton_forward_prefill_raw_from_flat(
        self,
        flat: dict[str, torch.Tensor],
        triton_args: argparse.Namespace,
        schedule,
    ) -> dict[str, torch.Tensor]:
        from model.kernels.kvm_triton_training_kernels import (
            build_prefill_forward,
        )

        noncontiguous = [
            name for name, tensor in flat.items() if not tensor.is_contiguous()
        ]
        if noncontiguous:
            raise ValueError(
                "KVM Triton prefill requires contiguous flattened streams; got "
                + ", ".join(noncontiguous)
            )
        q_flat = flat["q"]
        state_temperature, front_temperature = self._head_temperatures(q_flat.device)
        # Safe AOT is a training option; inference retains the fast Triton path.
        inference_args = argparse.Namespace(**vars(triton_args))
        inference_args.aotriton_forward_attention = False
        return build_prefill_forward(
            inference_args,
            schedule,
            q_flat,
            flat["merge_k"],
            flat["merge_v"],
            flat["bswa_k"],
            flat["bswa_v"],
            self.ln_s_k.weight.float(),
            self.ln_s_k.bias.float(),
            state_temperature,
            front_temperature,
            initial_k_flat=flat["initial_k"],
            initial_v_flat=flat["initial_v"],
            overflow_select_k_flat=flat["select_k"],
            overflow_append_k_flat=flat["append_k"],
            overflow_append_v_flat=flat["append_v"],
            overflow_merge_k_flat=flat["merge_k"],
            overflow_merge_v_flat=flat["merge_v"],
        )

    def _triton_prefill_out(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        merge_gate: torch.Tensor,
        triton_args: argparse.Namespace,
        schedule,
    ) -> torch.Tensor:
        batch_size, q_heads, _, _ = q.shape
        flat = self._flatten_qkv_for_triton(q, k, v, merge_gate, triton_args.q_len)
        state_temperature, front_temperature = self._head_temperatures(q.device)
        out_flat = _KvmTritonTrainingFunction.apply(
            flat["q"],
            flat["bswa_k"],
            flat["bswa_v"],
            flat["select_k"],
            flat["merge_v"],
            flat["select_k"],
            flat["append_k"],
            flat["append_v"],
            flat["merge_k"],
            flat["merge_v"],
            flat["initial_k"],
            flat["initial_v"],
            self.ln_s_k.weight.float(),
            self.ln_s_k.bias.float(),
            state_temperature,
            front_temperature,
            triton_args,
            schedule,
        )
        return out_flat.reshape(batch_size, q_heads, triton_args.q_len, self.d_v_head)

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
        del v_first, position_embeddings, kwargs
        if attention_mask is not None:
            raise ValueError("kvm_triton_mixer does not support attention_mask")

        batch_size, _, prefill_len, _ = q.size()
        triton_args = self._make_triton_args(batch_size, int(prefill_len))
        schedule = self._make_schedule(triton_args)

        needs_grad = torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (
                q,
                k,
                v,
                merge_gate,
                self.ln_s_k.weight,
                self.ln_s_k.bias,
                getattr(self, "state_head_temp", q),
                getattr(self, "front_head_temp", q),
            )
        )
        if past_key_values is not None and needs_grad:
            raise ValueError(
                "kvm_triton_mixer does not support autograd prefill while updating "
                "past_key_values"
            )
        if needs_grad:
            out = self._triton_prefill_out(q, k, v, merge_gate, triton_args, schedule)
            out = out[:, :, :prefill_len, :]
            y = out.transpose(1, 2).contiguous().view(batch_size, prefill_len, -1)
            return self.c_proj(y)

        forward = self._triton_forward_prefill_raw(
            q, k, v, merge_gate, triton_args, schedule
        )
        out = forward["out"].reshape(
            batch_size, self.num_attention_heads, triton_args.q_len, self.d_v_head
        )[:, :, :prefill_len, :]
        y = out.transpose(1, 2).contiguous().view(batch_size, prefill_len, -1)
        y = self.c_proj(y)

        if past_key_values is not None:
            final_state_len = int(schedule.final_state_len)
            bswa_begin = self._bswa_begin_for_total_len(prefill_len)
            state_coverage_len = int(schedule.final_state_coverage_len)
            expected_state_coverage_len = max(triton_args.initial_state_len, bswa_begin)
            if state_coverage_len != expected_state_coverage_len:
                raise AssertionError(
                    "KVM Triton prefill state progression drifted from cache bookkeeping."
                )
            snapshot = DecodeCacheSnapshot(
                state_k=forward["state_k"][:, :final_state_len].reshape(
                    batch_size,
                    self.num_key_value_heads,
                    final_state_len,
                    self.d_qk_head,
                ),
                state_v=forward["state_v"][:, :final_state_len].reshape(
                    batch_size,
                    self.num_key_value_heads,
                    final_state_len,
                    self.d_v_head,
                ),
                state_vlen=forward["state_vlen"][:, :final_state_len].reshape(
                    batch_size, self.num_key_value_heads, final_state_len, 1
                ),
                recent_k=k[:, :, bswa_begin:, :],
                recent_v=v[:, :, bswa_begin:, :],
                recent_gate=merge_gate[:, :, bswa_begin:, :],
                total_len=prefill_len,
                state_coverage_len=state_coverage_len,
                recent_begin=bswa_begin,
            )
            state_capacity = max(
                final_state_len,
                min(self.chunk_len, self.max_state_len),
                TritonKVMDecodeCache.state_length_after_cycle(
                    self,
                    prefill_len,
                    final_state_len,
                    self.n_bswa_chunks * self.chunk_len,
                ),
            )
            decode_cache = TritonKVMDecodeCache(
                self,
                snapshot,
                state_capacity=state_capacity,
            )
            past_key_values.update(
                self.layer_idx,
                offset=prefill_len,
                states_dict={self._CACHE_TRITON_DECODE: decode_cache},
            )

        return y

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
        """Decode one token through the integrated optimized Triton cache."""
        del v_first, position_embeddings, kwargs
        if attention_mask is not None:
            raise ValueError("kvm_triton_mixer does not support attention_mask")
        if past_key_values is None:
            raise ValueError("Triton KVM cached decode requires past_key_values")
        if int(q.size(2)) != 1:
            raise AssertionError(
                "Triton KVM cached decode expects a single-token input"
            )

        cache_states = past_key_values.get_states(self.layer_idx)
        decode_cache = cache_states.get(self._CACHE_TRITON_DECODE)
        if not isinstance(decode_cache, TritonKVMDecodeCache):
            raise RuntimeError(
                "Triton KVM cache is missing its optimized decode state; run a "
                "Triton prefill with this cache before decoding"
            )
        if decode_cache.mixer is not self:
            raise RuntimeError("Triton KVM decode cache belongs to another mixer")
        past_seq_len = past_key_values.get_seq_length(self.layer_idx)
        if decode_cache.total_len != past_seq_len:
            raise AssertionError(
                "Triton KVM decode cache length disagrees with the layer cache: "
                f"{decode_cache.total_len} != {past_seq_len}"
            )

        out = self._decode_one_token(q, k, v, merge_gate, decode_cache)
        past_key_values.update(
            self.layer_idx,
            offset=1,
            states_dict={self._CACHE_TRITON_DECODE: decode_cache},
        )
        batch_size = int(q.size(0))
        y = out.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        return self.c_proj(y)

    def _decode_one_token(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        merge_gate: torch.Tensor,
        decode_cache: TritonKVMDecodeCache,
    ) -> torch.Tensor:
        """Advance ``decode_cache`` and attend for one post-QKV token."""
        if int(k.size(2)) != 1:
            raise ValueError("Triton KVM decode cache update accepts one token")
        new_total_len = decode_cache.total_len + 1
        new_recent_begin = self._bswa_begin_for_total_len(new_total_len)
        direct_state_target = min(new_total_len, decode_cache.chunk_len)
        new_state_coverage = max(direct_state_target, new_recent_begin)
        if new_state_coverage < decode_cache.state_coverage_len:
            raise AssertionError("KVM decode state coverage cannot shrink")
        if direct_state_target > decode_cache.state_coverage_len:
            if direct_state_target - decode_cache.state_coverage_len != 1:
                raise AssertionError("decode crossed more than one direct-state token")
            self._append_decode_direct_state(
                decode_cache,
                k,
                v,
                merge_gate,
            )
            decode_cache.state_coverage_len = direct_state_target
        if new_state_coverage > decode_cache.state_coverage_len:
            if new_state_coverage - decode_cache.state_coverage_len != self.chunk_len:
                raise AssertionError("decode crossed more than one state boundary")
            self._run_decode_state_update(decode_cache)
            decode_cache.state_coverage_len = new_state_coverage

        decode_cache.append_recent_(
            k,
            v,
            merge_gate,
            total_len=new_total_len,
            recent_begin=new_recent_begin,
        )
        return self._decode_attention(q, decode_cache)

    def _run_decode_state_update(
        self,
        decode_cache: TritonKVMDecodeCache,
    ) -> None:
        from model.kernels import (
            kvm_triton_training_kernels as update_kernels,
        )
        from model.kernels.kvm_triton_decode import gather_overflow

        current_state_len = decode_cache.state_len
        desired_state_len = self._desired_state_len(
            (self.n_bswa_chunks * decode_cache.chunk_len)
            + decode_cache.state_coverage_len,
            decode_cache.state_coverage_len + decode_cache.chunk_len,
            current_state_len,
        )
        n_append = min(
            max(desired_state_len - current_state_len, 0),
            decode_cache.chunk_len,
        )
        state_after = current_state_len + n_append
        decode_cache._reserve_state_capacity(state_after)
        gather_overflow(
            decode_cache.recent_k,
            decode_cache.recent_v,
            decode_cache.recent_gate,
            ring_start=decode_cache.recent_start,
            overflow_len=decode_cache.chunk_len,
            zeroed_k=decode_cache.overflow_zeroed_k,
            raw_v=decode_cache.overflow_raw_v,
            gate_out=decode_cache.overflow_gate,
            rope_partial_dim=int(self.rope_partial_dim),
        )
        torch.ops.aten.native_layer_norm.out(
            decode_cache.overflow_zeroed_k,
            [decode_cache.head_dim],
            self.ln_s_k.weight,
            self.ln_s_k.bias,
            float(self.ln_s_k.eps),
            out0=decode_cache.overflow_select_k,
            out1=decode_cache.overflow_ln_mean,
            out2=decode_cache.overflow_ln_rstd,
        )
        if self.config.kvm_use_merge_gate_keys:
            torch.mul(
                decode_cache.overflow_select_k,
                decode_cache.overflow_gate,
                out=decode_cache.overflow_merge_k,
            )
        else:
            decode_cache.overflow_merge_k.copy_(decode_cache.overflow_select_k)
        if self.config.kvm_use_merge_gate_values:
            torch.mul(
                decode_cache.overflow_raw_v,
                decode_cache.overflow_gate,
                out=decode_cache.overflow_merge_v,
            )
        else:
            decode_cache.overflow_merge_v.copy_(decode_cache.overflow_raw_v)
        if self.config.kvm_apply_merge_gate_to_appends:
            decode_cache.overflow_append_k.copy_(decode_cache.overflow_merge_k)
            decode_cache.overflow_append_v.copy_(decode_cache.overflow_merge_v)
        else:
            decode_cache.overflow_append_k.copy_(decode_cache.overflow_select_k)
            decode_cache.overflow_append_v.copy_(decode_cache.overflow_raw_v)
        schedule = self._make_generation_update_schedule(
            current_state_len=current_state_len,
            state_after=state_after,
            n_append=n_append,
            overflow_len=decode_cache.chunk_len,
        )
        decode_cache.update_args.initial_state_len = current_state_len
        decode_cache.before_device.fill_(current_state_len)
        decode_cache.after_device.fill_(state_after)
        decode_cache.n_append_device.fill_(n_append)
        update_kernels.run_forward_state_update(
            decode_cache.update_args,
            schedule,
            decode_cache.overflow_merge_k,
            decode_cache.overflow_merge_v,
            decode_cache.ln_weight_fp32,
            decode_cache.ln_bias_fp32,
            decode_cache.update_buffers,
            decode_cache.state_k,
            decode_cache.state_v,
            decode_cache.state_k_attn,
            decode_cache.state_v_attn,
            decode_cache.state_vlen,
            decode_cache.append_pos_by_token,
            decode_cache.best_idx_by_token,
            decode_cache.undo_k_by_token,
            decode_cache.undo_v_by_token,
            0,
            False,
            overflow_select_k_flat=decode_cache.overflow_select_k,
            overflow_append_k_flat=decode_cache.overflow_append_k,
            overflow_append_v_flat=decode_cache.overflow_append_v,
            overflow_merge_k_flat=decode_cache.overflow_merge_k,
            overflow_merge_v_flat=decode_cache.overflow_merge_v,
            has_appends=n_append > 0,
            before_by_macro_device=decode_cache.before_device,
            after_by_macro_device=decode_cache.after_device,
            n_append_by_macro_device=decode_cache.n_append_device,
        )
        decode_cache.state_len = state_after
        decode_cache.update_count += 1

    def _append_decode_direct_state(
        self,
        decode_cache: TritonKVMDecodeCache,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        new_gate: torch.Tensor,
    ) -> None:
        """Extend the initial direct state while the prompt is below one chunk."""
        required = decode_cache.state_len + 1
        decode_cache._reserve_state_capacity(required)
        prepared_k = self._prepare_state_update_k(new_k)
        stored_v = new_v
        if self.config.kvm_apply_merge_gate_to_initial_state:
            prepared_k = (prepared_k * new_gate).to(prepared_k.dtype)
            stored_v = (stored_v * new_gate).to(stored_v.dtype)

        flat_k = prepared_k.reshape(decode_cache.kv_rows, decode_cache.head_dim)
        flat_v = stored_v.reshape(decode_cache.kv_rows, decode_cache.value_dim)
        raw_v = new_v.reshape(decode_cache.kv_rows, decode_cache.value_dim)
        vlen = torch.linalg.vector_norm(raw_v.float(), dim=-1)
        state_position = decode_cache.state_len
        decode_cache.state_k[:, state_position].copy_(flat_k)
        decode_cache.state_v[:, state_position].copy_(flat_v)
        decode_cache.state_vlen[:, state_position].copy_(vlen)
        decode_cache.state_k_attn[:, state_position].copy_(
            self.ln_s_k(prepared_k).reshape(
                decode_cache.kv_rows,
                decode_cache.head_dim,
            )
        )
        normalized_v = (
            F.normalize(flat_v.float(), dim=-1) * vlen[:, None]
        ).to(decode_cache.dtype)
        decode_cache.state_v_attn[:, state_position].copy_(normalized_v)
        decode_cache.state_len = required

    def _decode_attention(
        self,
        q: torch.Tensor,
        decode_cache: TritonKVMDecodeCache,
    ) -> torch.Tensor:
        """Attend over the recurrent and recent state owned by ``decode_cache``."""
        from model.kernels.kvm_triton_decode import kvm_decode_attention

        active_state_len = (
            decode_cache.state_len
            if decode_cache.total_len > decode_cache.ring_capacity
            else 0
        )
        return kvm_decode_attention(
            q,
            decode_cache.state_k_attn,
            decode_cache.state_v_attn,
            decode_cache.recent_k,
            decode_cache.recent_v,
            active_state_len=active_state_len,
            recent_start=decode_cache.recent_start,
            recent_len=decode_cache.recent_len,
            state_temperature=decode_cache.state_temperature,
            front_temperature=decode_cache.front_temperature,
            q_heads=self.num_attention_heads,
            kv_heads=self.num_key_value_heads,
            out=decode_cache.out,
        )

    def _make_generation_update_schedule(
        self,
        current_state_len: int,
        state_after: int,
        n_append: int,
        overflow_len: int,
    ):
        from model.kernels.kvm_triton_training_kernels import MixerPrefillSchedule

        if overflow_len != self.chunk_len:
            raise AssertionError(
                "KVM Triton decode can only materialize one full overflow chunk per update."
            )
        return MixerPrefillSchedule(
            before_by_macro=torch.tensor([current_state_len], dtype=torch.int32),
            after_by_macro=torch.tensor([state_after], dtype=torch.int32),
            n_append_by_macro=torch.tensor([n_append], dtype=torch.int32),
            valid_update_by_macro=torch.tensor([1], dtype=torch.int32),
            attention_state_len_by_macro=torch.tensor(
                [current_state_len], dtype=torch.int32
            ),
            front_len=overflow_len,
            initial_state_len=current_state_len,
            final_state_len=state_after,
            final_state_coverage_len=0,
        )
