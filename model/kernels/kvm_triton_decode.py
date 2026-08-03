"""Triton kernels for allocation-free KVM single-token decode.

The attention kernel consumes the normalized recurrent state and the recent
front as separate logical segments.  It deliberately does not materialize a
combined K/V tensor.  The recent front is stored twice in a fixed-size ring so
that its logical contents are always a contiguous slice.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _write_recent_ring_kernel(
    new_k,
    new_v,
    new_gate,
    ring_k,
    ring_v,
    ring_gate,
    write_pos,
    NEW_K_BATCH_STRIDE: tl.constexpr,
    NEW_K_HEAD_STRIDE: tl.constexpr,
    NEW_V_BATCH_STRIDE: tl.constexpr,
    NEW_V_HEAD_STRIDE: tl.constexpr,
    NEW_GATE_BATCH_STRIDE: tl.constexpr,
    NEW_GATE_HEAD_STRIDE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch_id = row // KV_HEADS
    kv_head = row - batch_id * KV_HEADS
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = tl.arange(0, VALUE_DIM)
    k = tl.load(
        new_k
        + batch_id * NEW_K_BATCH_STRIDE
        + kv_head * NEW_K_HEAD_STRIDE
        + key_offsets
    )
    v = tl.load(
        new_v
        + batch_id * NEW_V_BATCH_STRIDE
        + kv_head * NEW_V_HEAD_STRIDE
        + value_offsets
    )
    gate = tl.load(
        new_gate
        + batch_id * NEW_GATE_BATCH_STRIDE
        + kv_head * NEW_GATE_HEAD_STRIDE
    )
    k_base = ring_k + row * (2 * RING_CAPACITY) * HEAD_DIM
    v_base = ring_v + row * (2 * RING_CAPACITY) * VALUE_DIM
    gate_base = ring_gate + row * (2 * RING_CAPACITY)
    first = write_pos
    second = write_pos + RING_CAPACITY
    tl.store(k_base + first * HEAD_DIM + key_offsets, k)
    tl.store(k_base + second * HEAD_DIM + key_offsets, k)
    tl.store(v_base + first * VALUE_DIM + value_offsets, v)
    tl.store(v_base + second * VALUE_DIM + value_offsets, v)
    tl.store(gate_base + first, gate)
    tl.store(gate_base + second, gate)


@triton.jit
def _prepare_overflow_kernel(
    ring_k,
    ring_v,
    ring_gate,
    select_k,
    append_k,
    append_v,
    merge_k,
    merge_v,
    ln_weight,
    ln_bias,
    ring_start,
    RING_CAPACITY: tl.constexpr,
    OVERFLOW_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    ROPE_PARTIAL_DIM: tl.constexpr,
    LN_EPS: tl.constexpr,
    GATE_KEYS: tl.constexpr,
    GATE_VALUES: tl.constexpr,
    GATE_APPENDS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1).to(tl.int64)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = tl.arange(0, VALUE_DIM)
    source_token = ring_start + token
    ring_k_base = ring_k + row * (2 * RING_CAPACITY) * HEAD_DIM
    ring_v_base = ring_v + row * (2 * RING_CAPACITY) * VALUE_DIM
    ring_gate_base = ring_gate + row * (2 * RING_CAPACITY)

    raw_k = tl.load(ring_k_base + source_token * HEAD_DIM + key_offsets)
    raw_k = tl.where(key_offsets < ROPE_PARTIAL_DIM, 0.0, raw_k).to(tl.float32)
    mean = tl.sum(raw_k, axis=0) / HEAD_DIM
    centered = raw_k - mean
    variance = tl.sum(centered * centered, axis=0) / HEAD_DIM
    weight = tl.load(ln_weight + key_offsets).to(tl.float32)
    bias = tl.load(ln_bias + key_offsets).to(tl.float32)
    prepared_k = (
        centered * tl.rsqrt(variance + LN_EPS) * weight + bias
    ).to(tl.bfloat16)

    gate = tl.load(ring_gate_base + source_token).to(tl.float32)
    raw_v = tl.load(
        ring_v_base + source_token * VALUE_DIM + value_offsets
    ).to(tl.float32)
    gated_k = prepared_k.to(tl.float32) * gate if GATE_KEYS else prepared_k
    gated_v = raw_v * gate if GATE_VALUES else raw_v
    appended_k = gated_k if GATE_APPENDS else prepared_k
    appended_v = gated_v if GATE_APPENDS else raw_v

    key_target = (row * OVERFLOW_LEN + token) * HEAD_DIM + key_offsets
    value_target = (row * OVERFLOW_LEN + token) * VALUE_DIM + value_offsets
    tl.store(select_k + key_target, prepared_k)
    tl.store(append_k + key_target, appended_k)
    tl.store(merge_k + key_target, gated_k)
    tl.store(append_v + value_target, appended_v)
    tl.store(merge_v + value_target, gated_v)


@triton.jit
def _gather_overflow_kernel(
    ring_k,
    ring_v,
    ring_gate,
    zeroed_k,
    raw_v,
    gate_out,
    ring_start,
    RING_CAPACITY: tl.constexpr,
    OVERFLOW_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    ROPE_PARTIAL_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1).to(tl.int64)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = tl.arange(0, VALUE_DIM)
    source_token = ring_start + token
    source_k = (
        ring_k
        + row * (2 * RING_CAPACITY) * HEAD_DIM
        + source_token * HEAD_DIM
        + key_offsets
    )
    source_v = (
        ring_v
        + row * (2 * RING_CAPACITY) * VALUE_DIM
        + source_token * VALUE_DIM
        + value_offsets
    )
    source_gate = ring_gate + row * (2 * RING_CAPACITY) + source_token
    target_k = (row * OVERFLOW_LEN + token) * HEAD_DIM + key_offsets
    target_v = (row * OVERFLOW_LEN + token) * VALUE_DIM + value_offsets
    k = tl.load(source_k)
    tl.store(zeroed_k + target_k, tl.where(key_offsets < ROPE_PARTIAL_DIM, 0.0, k))
    tl.store(raw_v + target_v, tl.load(source_v))
    tl.store(gate_out + row * OVERFLOW_LEN + token, tl.load(source_gate))


@triton.jit
def _kvm_decode_attention_kernel(
    q,
    state_k,
    state_v,
    recent_k,
    recent_v,
    out,
    state_temperature,
    front_temperature,
    active_state_len,
    recent_start,
    recent_len,
    Q_BATCH_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
    FRONT_BLOCK: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    q_row = tl.program_id(0).to(tl.int64)
    group_size: tl.constexpr = Q_HEADS // KV_HEADS
    batch_id = q_row // Q_HEADS
    q_head = q_row - batch_id * Q_HEADS
    kv_head = q_head // group_size
    kv_row = batch_id * KV_HEADS + kv_head
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = tl.arange(0, VALUE_DIM)
    state_offsets = tl.arange(0, STATE_BLOCK)
    front_offsets = tl.arange(0, FRONT_BLOCK)

    q_vec = tl.load(
        q + batch_id * Q_BATCH_STRIDE + q_head * Q_HEAD_STRIDE + key_offsets
    )
    state_temp = tl.load(state_temperature + q_head).to(tl.float32)
    front_temp = tl.load(front_temperature + q_head).to(tl.float32)
    state_k_base = state_k + kv_row * STATE_CAPACITY * HEAD_DIM
    state_v_base = state_v + kv_row * STATE_CAPACITY * VALUE_DIM
    recent_k_base = recent_k + kv_row * (2 * RING_CAPACITY) * HEAD_DIM
    recent_v_base = recent_v + kv_row * (2 * RING_CAPACITY) * VALUE_DIM

    maximum = -float("inf")
    denominator = 0.0
    accumulator = tl.zeros((VALUE_DIM,), tl.float32)

    for block_begin in tl.range(0, active_state_len, STATE_BLOCK, num_stages=1):
        idx = block_begin + state_offsets
        valid = idx < active_state_len
        keys = tl.load(
            state_k_base + idx[:, None] * HEAD_DIM + key_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        effective_keys = (keys.to(tl.float32) * state_temp).to(tl.bfloat16)
        scores = tl.sum(
            effective_keys.to(tl.float32) * q_vec[None, :].to(tl.float32), axis=1
        ) * SCALE_LOG2
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_maximum = tl.maximum(maximum, block_max)
        correction = tl.exp2(maximum - new_maximum)
        probabilities = tl.exp2(scores - new_maximum)
        probabilities = tl.where(valid, probabilities, 0.0)
        values = tl.load(
            state_v_base + idx[:, None] * VALUE_DIM + value_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=0)
        maximum = new_maximum

    for block_begin in tl.range(0, recent_len, FRONT_BLOCK, num_stages=1):
        logical_idx = block_begin + front_offsets
        valid = logical_idx < recent_len
        idx = recent_start + logical_idx
        keys = tl.load(
            recent_k_base + idx[:, None] * HEAD_DIM + key_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        effective_keys = (keys.to(tl.float32) * front_temp).to(tl.bfloat16)
        scores = tl.sum(
            effective_keys.to(tl.float32) * q_vec[None, :].to(tl.float32), axis=1
        ) * SCALE_LOG2
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_maximum = tl.maximum(maximum, block_max)
        correction = tl.exp2(maximum - new_maximum)
        probabilities = tl.exp2(scores - new_maximum)
        probabilities = tl.where(valid, probabilities, 0.0)
        values = tl.load(
            recent_v_base + idx[:, None] * VALUE_DIM + value_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=0)
        maximum = new_maximum

    tl.store(out + q_row * VALUE_DIM + value_offsets, accumulator / denominator)


def write_recent_ring(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    new_gate: torch.Tensor,
    ring_k: torch.Tensor,
    ring_v: torch.Tensor,
    ring_gate: torch.Tensor,
    write_pos: int,
) -> None:
    """Write one K/V/gate token to both halves of a double-mapped ring."""
    kv_rows, doubled_capacity, head_dim = ring_k.shape
    capacity = doubled_capacity // 2
    value_dim = int(ring_v.size(-1))
    kv_heads = int(new_k.size(1))
    if not 0 <= int(write_pos) < capacity:
        raise ValueError("ring write position is outside its capacity")
    _write_recent_ring_kernel[(kv_rows,)](
        new_k,
        new_v,
        new_gate,
        ring_k,
        ring_v,
        ring_gate,
        int(write_pos),
        NEW_K_BATCH_STRIDE=int(new_k.stride(0)),
        NEW_K_HEAD_STRIDE=int(new_k.stride(1)),
        NEW_V_BATCH_STRIDE=int(new_v.stride(0)),
        NEW_V_HEAD_STRIDE=int(new_v.stride(1)),
        NEW_GATE_BATCH_STRIDE=int(new_gate.stride(0)),
        NEW_GATE_HEAD_STRIDE=int(new_gate.stride(1)),
        KV_HEADS=kv_heads,
        RING_CAPACITY=capacity,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        num_warps=4,
    )


def prepare_overflow(
    ring_k: torch.Tensor,
    ring_v: torch.Tensor,
    ring_gate: torch.Tensor,
    *,
    ring_start: int,
    overflow_len: int,
    select_k: torch.Tensor,
    append_k: torch.Tensor,
    append_v: torch.Tensor,
    merge_k: torch.Tensor,
    merge_v: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    ln_eps: float,
    rope_partial_dim: int,
    gate_keys: bool,
    gate_values: bool,
    gate_appends: bool,
) -> None:
    """Prepare one logical overflow chunk into reusable state-update buffers."""
    kv_rows, doubled_capacity, head_dim = ring_k.shape
    capacity = doubled_capacity // 2
    value_dim = int(ring_v.size(-1))
    if ring_start + overflow_len > doubled_capacity:
        raise ValueError("double-mapped overflow slice is not contiguous")
    _prepare_overflow_kernel[(kv_rows, overflow_len)](
        ring_k,
        ring_v,
        ring_gate,
        select_k,
        append_k,
        append_v,
        merge_k,
        merge_v,
        ln_weight,
        ln_bias,
        int(ring_start),
        RING_CAPACITY=capacity,
        OVERFLOW_LEN=overflow_len,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        ROPE_PARTIAL_DIM=int(rope_partial_dim),
        LN_EPS=float(ln_eps),
        GATE_KEYS=bool(gate_keys),
        GATE_VALUES=bool(gate_values),
        GATE_APPENDS=bool(gate_appends),
        num_warps=4,
    )


def gather_overflow(
    ring_k: torch.Tensor,
    ring_v: torch.Tensor,
    ring_gate: torch.Tensor,
    *,
    ring_start: int,
    overflow_len: int,
    zeroed_k: torch.Tensor,
    raw_v: torch.Tensor,
    gate_out: torch.Tensor,
    rope_partial_dim: int,
) -> None:
    """Gather one ring chunk into preallocated eager-equivalent update inputs."""
    kv_rows, doubled_capacity, head_dim = ring_k.shape
    capacity = doubled_capacity // 2
    value_dim = int(ring_v.size(-1))
    _gather_overflow_kernel[(kv_rows, overflow_len)](
        ring_k,
        ring_v,
        ring_gate,
        zeroed_k,
        raw_v,
        gate_out,
        int(ring_start),
        RING_CAPACITY=capacity,
        OVERFLOW_LEN=overflow_len,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        ROPE_PARTIAL_DIM=int(rope_partial_dim),
        num_warps=4,
    )


def kvm_decode_attention(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    *,
    active_state_len: int,
    recent_start: int,
    recent_len: int,
    state_temperature: torch.Tensor,
    front_temperature: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one online-softmax attention over separate state and front banks."""
    batch = int(q.size(0))
    head_dim = int(q.size(-1))
    value_dim = int(state_v.size(-1))
    state_capacity = int(state_k.size(1))
    ring_capacity = int(recent_k.size(1)) // 2
    if int(q.size(2)) != 1:
        raise ValueError("Triton KVM decode attention requires q_len=1")
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if out is None:
        out = torch.empty(
            batch, q_heads, 1, value_dim, device=q.device, dtype=q.dtype
        )
    q_rows = batch * q_heads
    state_block = 64
    front_block = 64
    _kvm_decode_attention_kernel[(q_rows,)](
        q,
        state_k,
        state_v,
        recent_k,
        recent_v,
        out,
        state_temperature,
        front_temperature,
        int(active_state_len),
        int(recent_start),
        int(recent_len),
        Q_BATCH_STRIDE=int(q.stride(0)),
        Q_HEAD_STRIDE=int(q.stride(1)),
        STATE_CAPACITY=state_capacity,
        RING_CAPACITY=ring_capacity,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        STATE_BLOCK=state_block,
        FRONT_BLOCK=front_block,
        SCALE_LOG2=math.log2(math.e) / math.sqrt(float(head_dim)),
        num_warps=4,
    )
    return out


__all__ = [
    "kvm_decode_attention",
    "gather_overflow",
    "prepare_overflow",
    "write_recent_ring",
    "_kvm_decode_attention_kernel",
    "_gather_overflow_kernel",
    "_prepare_overflow_kernel",
    "_write_recent_ring_kernel",
]
