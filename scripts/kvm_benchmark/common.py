from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any, Callable, ContextManager

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_kvm.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.kvm_triton_decode import (  # noqa: E402
    TritonKVMDecodeCache,
    DecodeCacheSnapshot,
)
from model.kvm_triton_mixer import SequenceMixer  # noqa: E402

BATCH = 8
HEADS = 32
DIM = 128
CHUNK = 256
RECENT_CHUNKS = 2
DECODE_EVICTION_BYTES = 512 * 1024 * 1024
CONTEXTS = (512, 1024, 2048, 4096, 8192, 16384, 32768)
SCHEDULES = ("fixed256", "sqrt16")
PHASES = ("decode", "prefill", "backward")
ARMS = ("kvm", "full_attention")
SEEDS = tuple(range(10))
DTYPE = torch.bfloat16

# Result and artifact utilities.


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        position = q * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p10": percentile(0.1),
        "p90": percentile(0.9),
        "stdev": stdev,
        "cv": stdev / mean,
        "cv_percent": stdev / mean * 100.0,
    }


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def concatenate_heads(output: torch.Tensor) -> torch.Tensor:
    """Return the shared post-QKV output layout for either implementation."""
    return output.transpose(1, 2).contiguous().flatten(2)


def require_post_qkv_output(output: torch.Tensor, tokens: int) -> torch.Tensor:
    expected = (BATCH, tokens, HEADS * DIM)
    if tuple(output.shape) != expected:
        raise AssertionError(
            f"post-QKV output mismatch: got {tuple(output.shape)}, expected {expected}"
        )
    if output.dtype != DTYPE:
        raise AssertionError(
            f"post-QKV output dtype mismatch: got {output.dtype}, expected {DTYPE}"
        )
    return output


def build_mixer(schedule: str, context: int) -> SequenceMixer:
    mixer = SequenceMixer.__new__(SequenceMixer)
    nn.Module.__init__(mixer)
    mixer.layer_idx = 0
    mixer.config = SimpleNamespace(
        kvm_use_vlens=True,
        kvm_use_merge_gate_keys=True,
        kvm_use_merge_gate_values=True,
        kvm_apply_merge_gate_to_appends=False,
        kvm_apply_merge_gate_to_initial_state=False,
        kvm_use_head_temps=True,
        kvm_aotriton_forward_attention=True,
    )
    mixer.num_attention_heads = HEADS
    mixer.num_key_value_heads = HEADS
    mixer.kv_group_size = 1
    mixer.d_qk_head = DIM
    mixer.d_v_head = DIM
    mixer.rope_partial_dim = 64
    mixer.sink_len = 1
    mixer.chunk_len = CHUNK
    mixer.n_bswa_chunks = RECENT_CHUNKS
    mixer.state_budget_mode = "fixed" if schedule == "fixed256" else "power_law"
    mixer.state_growth_factor = 1.0 if schedule == "fixed256" else 16.0
    mixer.state_growth_exponent = 0.5
    mixer.state_min_len = CHUNK
    mixer.state_round_down = 1
    mixer.state_saturation_n = None
    target = CHUNK if schedule == "fixed256" else int(16.0 * math.sqrt(context + CHUNK))
    mixer.max_state_len = max(CHUNK, target)
    mixer.triton_sub_block = 128
    mixer.triton_attn_block = 64
    mixer.triton_state_chunk = 16
    mixer.triton_attn_state_chunk = 64
    mixer.triton_group_chunks = 12
    mixer.triton_update_token_block = 8
    mixer.ln_s_k = nn.LayerNorm(DIM, eps=1.0e-5, device="cuda", dtype=DTYPE)
    mixer.front_head_temp = nn.Parameter(
        torch.ones(HEADS, device="cuda", dtype=DTYPE), requires_grad=True
    )
    mixer.state_head_temp = nn.Parameter(
        torch.ones(HEADS, device="cuda", dtype=DTYPE), requires_grad=True
    )
    mixer.c_proj = nn.Identity()
    return mixer


def state_len_for_context(mixer: SequenceMixer, context: int) -> int:
    args = mixer._make_triton_args(BATCH, context)
    return int(mixer._make_schedule(args).final_state_len)


def make_snapshot(
    mixer: SequenceMixer, context: int, generator: torch.Generator
) -> DecodeCacheSnapshot:
    state_len = state_len_for_context(mixer, context)
    recent_begin = mixer._bswa_begin_for_total_len(context)
    recent_len = context - recent_begin
    state_k = torch.randn(
        BATCH, HEADS, state_len, DIM,
        generator=generator, device="cuda", dtype=DTYPE,
    )
    state_v = torch.randn(
        state_k.shape, generator=generator, device="cuda", dtype=DTYPE
    )
    state_vlen = torch.linalg.vector_norm(state_v.float(), dim=-1, keepdim=True)
    recent_k = torch.randn(
        BATCH, HEADS, recent_len, DIM,
        generator=generator, device="cuda", dtype=DTYPE,
    )
    recent_v = torch.randn(
        recent_k.shape, generator=generator, device="cuda", dtype=DTYPE
    )
    gate_logits = torch.randn(
        BATCH, HEADS, recent_len, 1,
        generator=generator, device="cuda", dtype=DTYPE,
    )
    return DecodeCacheSnapshot(
        state_k=state_k,
        state_v=state_v,
        state_vlen=state_vlen,
        recent_k=recent_k,
        recent_v=recent_v,
        recent_gate=1.0 + F.elu(gate_logits),
        total_len=context,
        state_coverage_len=max(min(context, CHUNK), recent_begin),
        recent_begin=recent_begin,
    )


# Prepared calls separate untimed application setup from timed work.


Backend = Callable[[], ContextManager[None]]
EventPair = tuple[torch.cuda.Event, torch.cuda.Event]


@dataclass
class OperatorInvocation:
    """One operator call with application-realistic untimed preparation."""

    prepare: Callable[[], None]
    run: Callable[[], Any]


@dataclass
class DecodeTrajectory:
    """One decode cycle measured as individually timed cold-cache calls."""

    evict_hardware_cache: Callable[[], None]
    stage_token: Callable[[int], None]
    step: Callable[[], Any]
    update_count: Callable[[], int] | None = None
    updates_before_timing: int = 0
    expected_update_tokens: tuple[int, ...] = ()

    def assert_update_tokens(self, observed: list[int]) -> None:
        if self.update_count is None:
            return
        expected = list(self.expected_update_tokens)
        if observed != expected:
            raise AssertionError(
                f"decode benchmark expected KVM updates at tokens {expected}, "
                f"got {observed}"
            )
        timed_updates = self.update_count() - self.updates_before_timing
        if timed_updates != len(expected):
            raise AssertionError(
                f"decode update counter changed by {timed_updates}, expected "
                f"{len(expected)}"
            )


@dataclass(frozen=True)
class TimingResult:
    total_ms: float

PreparedCall = OperatorInvocation | DecodeTrajectory
CallFactory = Callable[[], PreparedCall]


@dataclass
class Arm:
    factory: CallFactory
    backend: Backend
    events: tuple[EventPair, ...]


def allocate_event_pairs(count: int) -> tuple[EventPair, ...]:
    return tuple(
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(count)
    )


def make_arm(
    factory: CallFactory,
    backend: Backend,
    event_count: int,
) -> Arm:
    return Arm(
        factory,
        backend,
        allocate_event_pairs(event_count),
    )


# Backend selection and benchmark input/state preparation.


def flash_backend() -> ContextManager[None]:
    return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])


def allocate_hardware_cache_eviction_buffer() -> torch.Tensor:
    return torch.zeros(
        DECODE_EVICTION_BYTES // 4,
        device="cuda",
        dtype=torch.float32,
    )


def stream_hardware_cache_eviction(buffer: torch.Tensor) -> None:
    # The MI325X has a 256 MiB Infinity Cache. Streaming through twice that
    # capacity models unrelated layer traffic between visits to this layer.
    buffer.add_(1.0)


def fill_gate_(tensor: torch.Tensor, generator: torch.Generator) -> None:
    """Populate a positive merge gate without allocating replacement storage."""
    tensor.normal_(generator=generator)
    F.elu(tensor, inplace=True)
    tensor.add_(1.0)


def refresh_decode_snapshot_(
    snapshot: DecodeCacheSnapshot, generator: torch.Generator
) -> None:
    """Repopulate a preallocated KVM cache snapshot."""
    snapshot.state_k.normal_(generator=generator)
    snapshot.state_v.normal_(generator=generator)
    snapshot.state_vlen.copy_(
        torch.linalg.vector_norm(
            snapshot.state_v.float(), dim=-1, keepdim=True
        )
    )
    snapshot.recent_k.normal_(generator=generator)
    snapshot.recent_v.normal_(generator=generator)
    fill_gate_(snapshot.recent_gate, generator)


def refresh_decode_tokens_(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    new_gate: torch.Tensor,
    generator: torch.Generator,
) -> None:
    q.normal_(generator=generator)
    new_k.normal_(generator=generator)
    new_v.normal_(generator=generator)
    fill_gate_(new_gate, generator)


def refresh_training_tensors_(
    tensors: dict[str, torch.Tensor], generator: torch.Generator
) -> None:
    for name in ("q", "k", "v", "dout"):
        tensors[name].normal_(generator=generator)
    fill_gate_(tensors["gate"], generator)


def prepare_decode(
    schedule: str,
    context: int,
    seed: int,
    *,
    evict_before_each_call: bool = True,
) -> dict[str, Arm]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kvm_generator = torch.Generator(device="cuda").manual_seed(
        seed * 2 + 10_000
    )
    dense_generator = torch.Generator(device="cuda").manual_seed(
        seed * 2 + 10_001
    )
    mixer = build_mixer(schedule, context).eval()
    snapshot = make_snapshot(mixer, context, generator)
    capacity = TritonKVMDecodeCache.state_length_after_cycle(
        mixer, context, int(snapshot.state_k.size(2)), CHUNK + 1
    )
    cache = TritonKVMDecodeCache(mixer, snapshot, state_capacity=capacity)
    q = torch.randn(
        BATCH, HEADS, CHUNK + 1, DIM,
        generator=generator, device="cuda", dtype=DTYPE,
    )
    new_k = torch.randn(
        q.shape, generator=generator, device="cuda", dtype=DTYPE
    )
    new_v = torch.randn(
        q.shape, generator=generator, device="cuda", dtype=DTYPE
    )
    new_gate = 1.0 + F.elu(
        torch.randn(
            BATCH, HEADS, CHUNK + 1, 1,
            generator=generator, device="cuda", dtype=DTYPE,
        )
    )
    dense_k = torch.randn(
        BATCH, HEADS, context + CHUNK + 1, DIM,
        generator=generator, device="cuda", dtype=DTYPE,
    )
    dense_v = torch.randn(
        dense_k.shape, generator=generator, device="cuda", dtype=DTYPE
    )
    eviction_buffer = (
        allocate_hardware_cache_eviction_buffer()
        if evict_before_each_call
        else None
    )
    current_q = torch.empty(
        BATCH, HEADS, 1, DIM, device="cuda", dtype=DTYPE
    )
    current_k = torch.empty_like(current_q)
    current_v = torch.empty_like(current_q)
    current_gate = torch.empty(
        BATCH, HEADS, 1, 1, device="cuda", dtype=DTYPE
    )
    kvm_output = torch.empty(
        BATCH, 1, HEADS * DIM, device="cuda", dtype=DTYPE
    )
    dense_output = torch.empty_like(kvm_output)

    def write_output(
        source: torch.Tensor, destination: torch.Tensor
    ) -> torch.Tensor:
        destination.copy_(source.transpose(1, 2).flatten(2))
        return destination

    def evict_hardware_cache() -> None:
        # This work is ordered before, but excluded from, every timing event.
        if eviction_buffer is not None:
            stream_hardware_cache_eviction(eviction_buffer)

    def stage_token(step: int) -> None:
        # Projections are outside the benchmark boundary. Re-stage their small
        # outputs after eviction so Q/K/V and the merge gate are fresh, as they
        # would be immediately after the current layer's projections.
        current_q.copy_(q[:, :, step : step + 1])
        current_k.copy_(new_k[:, :, step : step + 1])
        current_v.copy_(new_v[:, :, step : step + 1])
        current_gate.copy_(new_gate[:, :, step : step + 1])

    def kvm_factory() -> PreparedCall:
        refresh_decode_snapshot_(snapshot, kvm_generator)
        refresh_decode_tokens_(q, new_k, new_v, new_gate, kvm_generator)
        with torch.no_grad():
            cache.reset_(snapshot)
            write_output(
                mixer._decode_one_token(
                    q[:, :, :1],
                    new_k[:, :, :1],
                    new_v[:, :, :1],
                    new_gate[:, :, :1],
                    cache,
                ),
                kvm_output,
            )
        updates_before_timing = cache.update_count

        def step() -> Any:
            with torch.no_grad():
                return write_output(
                    mixer._decode_one_token(
                        current_q,
                        current_k,
                        current_v,
                        current_gate,
                        cache,
                    ),
                    kvm_output,
                )

        return DecodeTrajectory(
            evict_hardware_cache=evict_hardware_cache,
            stage_token=stage_token,
            step=step,
            update_count=lambda: cache.update_count,
            updates_before_timing=updates_before_timing,
            expected_update_tokens=(CHUNK,),
        )

    def dense_factory() -> PreparedCall:
        # Reinitialize the complete transformer prompt cache, not only the
        # generated-token suffix.
        dense_k.normal_(generator=dense_generator)
        dense_v.normal_(generator=dense_generator)
        refresh_decode_tokens_(q, new_k, new_v, new_gate, dense_generator)
        with torch.no_grad():
            dense_k[:, :, context : context + 1].copy_(new_k[:, :, :1])
            dense_v[:, :, context : context + 1].copy_(new_v[:, :, :1])
            write_output(
                F.scaled_dot_product_attention(
                    q[:, :, :1],
                    dense_k[:, :, : context + 1],
                    dense_v[:, :, : context + 1],
                    dropout_p=0.0,
                    is_causal=False,
                ),
                dense_output,
            )

        position = context

        def stage_dense_token(step: int) -> None:
            nonlocal position
            stage_token(step)
            position = context + step

        def step() -> Any:
            with torch.no_grad():
                dense_k[:, :, position : position + 1].copy_(current_k)
                dense_v[:, :, position : position + 1].copy_(current_v)
                return write_output(
                    F.scaled_dot_product_attention(
                        current_q,
                        dense_k[:, :, : position + 1],
                        dense_v[:, :, : position + 1],
                        dropout_p=0.0,
                        is_causal=False,
                    ),
                    dense_output,
                )

        return DecodeTrajectory(
            evict_hardware_cache=evict_hardware_cache,
            stage_token=stage_dense_token,
            step=step,
        )

    return {
        "kvm": make_arm(kvm_factory, nullcontext, CHUNK),
        "full_attention": make_arm(dense_factory, flash_backend, CHUNK),
    }


def base_training_tensors(seed: int, context: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return {
        "q": torch.randn(
            BATCH, HEADS, context, DIM,
            generator=generator, device="cuda", dtype=DTYPE,
        ),
        "k": torch.randn(
            BATCH, HEADS, context, DIM,
            generator=generator, device="cuda", dtype=DTYPE,
        ),
        "v": torch.randn(
            BATCH, HEADS, context, DIM,
            generator=generator, device="cuda", dtype=DTYPE,
        ),
        "gate": 1.0 + F.elu(
            torch.randn(
                BATCH, HEADS, context, 1,
                generator=generator, device="cuda", dtype=DTYPE,
            )
        ),
        "dout": torch.randn(
            BATCH, HEADS, context, DIM,
            generator=generator, device="cuda", dtype=DTYPE,
        ),
    }


def prepare_prefill_or_backward(
    schedule: str, context: int, seed: int, phase: str
) -> dict[str, Arm]:
    base = base_training_tensors(seed, context)
    mixer = build_mixer(schedule, context)
    eviction_buffer = allocate_hardware_cache_eviction_buffer()
    staged = {
        name: torch.empty_like(base[name]) for name in ("q", "k", "v", "gate")
    }
    kvm_generator = torch.Generator(device="cuda").manual_seed(
        seed * 2 + 20_000
    )
    dense_generator = torch.Generator(device="cuda").manual_seed(
        seed * 2 + 20_001
    )

    def stage_forward_inputs() -> None:
        for name, destination in staged.items():
            destination.copy_(base[name])

    if phase == "prefill":
        mixer.eval()

        def prepare_prefill() -> None:
            stream_hardware_cache_eviction(eviction_buffer)
            stage_forward_inputs()

        def kvm_factory() -> PreparedCall:
            refresh_training_tensors_(base, kvm_generator)

            def run() -> Any:
                with torch.no_grad():
                    return mixer.forward_prefill(
                        staged["q"], staged["k"], staged["v"], staged["gate"],
                        None, None, None,
                    )

            return OperatorInvocation(prepare_prefill, run)

        def dense_factory() -> PreparedCall:
            refresh_training_tensors_(base, dense_generator)

            def run() -> Any:
                with torch.no_grad():
                    return concatenate_heads(
                        F.scaled_dot_product_attention(
                            staged["q"], staged["k"], staged["v"],
                            dropout_p=0.0, is_causal=True,
                        )
                    )

            return OperatorInvocation(prepare_prefill, run)

        return {
            "kvm": make_arm(kvm_factory, nullcontext, 1),
            "full_attention": make_arm(dense_factory, flash_backend, 1),
        }

    mixer.train()

    for tensor in staged.values():
        tensor.requires_grad_(True)
    upstream_source = concatenate_heads(base["dout"])
    upstream = torch.empty_like(upstream_source)
    require_post_qkv_output(upstream_source, context)

    def prepare_backward() -> None:
        stream_hardware_cache_eviction(eviction_buffer)
        upstream.copy_(upstream_source)

    def refresh_backward_inputs(generator: torch.Generator) -> None:
        with torch.no_grad():
            refresh_training_tensors_(base, generator)
            stage_forward_inputs()
            upstream_source.copy_(concatenate_heads(base["dout"]))

    def kvm_backward_factory() -> PreparedCall:
        refresh_backward_inputs(kvm_generator)
        out = mixer.forward_prefill(
            staged["q"], staged["k"], staged["v"], staged["gate"],
            None, None, None,
        )
        require_post_qkv_output(out, context)
        inputs = (
            staged["q"],
            staged["k"],
            staged["v"],
            staged["gate"],
            mixer.ln_s_k.weight,
            mixer.ln_s_k.bias,
            mixer.state_head_temp,
            mixer.front_head_temp,
        )
        return OperatorInvocation(
            prepare_backward,
            lambda: torch.autograd.grad(out, inputs, upstream),
        )

    def dense_backward_factory() -> PreparedCall:
        refresh_backward_inputs(dense_generator)
        out = concatenate_heads(
            F.scaled_dot_product_attention(
                staged["q"], staged["k"], staged["v"],
                dropout_p=0.0, is_causal=True,
            )
        )
        require_post_qkv_output(out, context)
        return OperatorInvocation(
            prepare_backward,
            lambda: torch.autograd.grad(
                out, (staged["q"], staged["k"], staged["v"]), upstream
            ),
        )

    return {
        "kvm": make_arm(kvm_backward_factory, nullcontext, 1),
        "full_attention": make_arm(dense_backward_factory, flash_backend, 1),
    }


# Timing and backend verification.


def run_decode_for_backend_proof(trajectory: DecodeTrajectory) -> Any:
    result = None
    update_tokens = []
    for token in range(1, CHUNK + 1):
        trajectory.stage_token(token)
        updates_before_step = (
            trajectory.update_count() if trajectory.update_count is not None else 0
        )
        result = trajectory.step()
        if (
            trajectory.update_count is not None
            and trajectory.update_count() > updates_before_step
        ):
            update_tokens.append(token)
    trajectory.assert_update_tokens(update_tokens)
    return result


def time_decode_calls(
    trajectory: DecodeTrajectory, event_pairs: tuple[EventPair, ...]
) -> tuple[TimingResult, Any]:
    if len(event_pairs) != CHUNK:
        raise AssertionError(f"decode requires {CHUNK} preallocated event pairs")
    result = None
    update_tokens = []
    for token in range(1, CHUNK + 1):
        trajectory.evict_hardware_cache()
        trajectory.stage_token(token)
        updates_before_step = (
            trajectory.update_count() if trajectory.update_count is not None else 0
        )
        start, end = event_pairs[token - 1]
        start.record()
        result = trajectory.step()
        end.record()
        if (
            trajectory.update_count is not None
            and trajectory.update_count() > updates_before_step
        ):
            update_tokens.append(token)
    torch.cuda.synchronize()
    trajectory.assert_update_tokens(update_tokens)
    invocation_ms = tuple(start.elapsed_time(end) for start, end in event_pairs)
    return TimingResult(sum(invocation_ms)), result


def time_once(arm: Arm) -> TimingResult:
    with arm.backend():
        call = arm.factory()
        torch.cuda.synchronize()
        if isinstance(call, DecodeTrajectory):
            timing, result = time_decode_calls(call, arm.events)
        else:
            if len(arm.events) != 1:
                raise AssertionError("one operator invocation requires one event pair")
            call.prepare()
            start, end = arm.events[0]
            start.record()
            result = call.run()
            end.record()
            end.synchronize()
            elapsed = float(start.elapsed_time(end))
            timing = TimingResult(elapsed)
    del result, call
    return timing


def measure_arm(
    arm: Arm, *, warmup: int, rounds: int, target_ms: float, unit_scale: float
) -> dict[str, Any]:
    for _ in range(warmup):
        time_once(arm)
    pilot = time_once(arm)
    iterations = min(
        50, max(1, math.ceil(target_ms / max(pilot.total_ms, 1.0e-6)))
    )
    samples = []
    for _round in range(rounds):
        elapsed_values = []
        for _iteration in range(iterations):
            timing = time_once(arm)
            elapsed_values.append(timing.total_ms)
        samples.append(statistics.fmean(elapsed_values) / unit_scale)
    return {
        "iterations_per_round": iterations,
        "samples": samples,
        "summary": distribution(samples),
    }


def backend_proof(
    arm_name: str, arm: Arm, phase: str, output_tokens: int
) -> dict[str, Any]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        with arm.backend():
            call = arm.factory()
            if isinstance(call, DecodeTrajectory):
                result = run_decode_for_backend_proof(call)
            else:
                call.prepare()
                result = call.run()
        if phase == "backward":
            expected_gradients = 8 if arm_name == "kvm" else 3
            if not isinstance(result, tuple) or len(result) != expected_gradients:
                raise AssertionError(
                    f"{arm_name} returned an invalid backward result"
                )
            if any(gradient is None for gradient in result):
                raise AssertionError(f"{arm_name} omitted a required gradient")
        else:
            require_post_qkv_output(result, output_tokens)
        torch.cuda.synchronize()
    names = [event.key for event in profiler.key_averages()]
    if arm_name == "kvm":
        fragments = {
            "decode": ("_kvm_decode_attention_kernel",),
            "prefill": ("_kvm_attn_live_state_fwd_kernel",),
            "backward": (
                "_kvm_attn_live_state_dq_kernel",
                "_kvm_attn_snapshot_bswa_dkdv_kernel",
            ),
        }[phase]
    elif arm_name == "full_attention":
        fragments = (
            (
                "attn_bwd",
                "flash_attention_backward",
                "fmha_backward",
                "_bwd_kernel_fuse",
                "bwd_kernel",
            )
            if phase == "backward"
            else (
                "attn_fwd",
                "flash_attention_forward",
                "fmha_forward",
                "flash_fwd",
            )
        )
    else:
        raise ValueError(f"unsupported arm {arm_name!r}")
    matched = [
        name
        for name in names
        if any(fragment in name.lower() for fragment in fragments)
    ]
    low_level = (
        matched
        if arm_name == "kvm"
        else [name for name in matched if not name.startswith("aten::")]
    )
    return {
        "passed": bool(low_level),
        "matched_symbols": matched,
        "low_level_symbols": low_level,
    }
