"""The one-token-to-32K decode time series."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gc
import json
import math
from pathlib import Path
from queue import Queue
import random
import statistics
import subprocess
import sys
from typing import Any

import torch
import torch.nn.functional as F

from .common import (
    BATCH,
    BENCHMARK_SCRIPT,
    CHUNK,
    DIM,
    DTYPE,
    HEADS,
    REPO_ROOT,
    allocate_event_pairs,
    allocate_hardware_cache_eviction_buffer,
    backend_proof,
    build_mixer,
    distribution,
    fill_gate_,
    flash_backend,
    package_version,
    prepare_decode,
    stream_hardware_cache_eviction,
    write_json_exclusive,
)
from model.kvm_triton_decode import TritonKVMDecodeCache, DecodeCacheSnapshot

DECODE_SERIES_START = 1
DECODE_SERIES_END = 32768
DECODE_SERIES_ARMS = ("kvm_fixed256", "kvm_sqrt16", "full_attention")


# Full-horizon trajectory implementations.


def decode_series_schedule(arm: str) -> str | None:
    if arm == "kvm_fixed256":
        return "fixed256"
    if arm == "kvm_sqrt16":
        return "sqrt16"
    if arm == "full_attention":
        return None
    raise ValueError(f"unsupported decode-series arm {arm!r}")


def allocate_decode_series_sources(
    end_position: int,
    generator: torch.Generator,
    *,
    include_gate: bool,
) -> dict[str, torch.Tensor]:
    shape = (BATCH, HEADS, end_position, DIM)
    sources = {
        name: torch.empty(shape, device="cuda", dtype=DTYPE)
        for name in ("q", "k", "v")
    }
    if include_gate:
        sources["gate"] = torch.empty(
            BATCH, HEADS, end_position, 1, device="cuda", dtype=DTYPE
        )
    refresh_decode_series_sources_(sources, generator)
    return sources


def refresh_decode_series_sources_(
    sources: dict[str, torch.Tensor], generator: torch.Generator
) -> None:
    for name in ("q", "k", "v"):
        sources[name].normal_(generator=generator)
    if "gate" in sources:
        fill_gate_(sources["gate"], generator)


def copy_single_token_output(
    source: torch.Tensor, destination: torch.Tensor
) -> torch.Tensor:
    destination.copy_(source.transpose(1, 2).flatten(2))
    return destination


def run_kvm_decode_series(
    schedule: str, end_position: int, seed: int
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed + 30_000)
    mixer = build_mixer(schedule, end_position).eval()
    sources = allocate_decode_series_sources(
        end_position, generator, include_gate=True
    )
    snapshot = DecodeCacheSnapshot(
        state_k=torch.empty(
            BATCH, HEADS, 1, DIM, device="cuda", dtype=DTYPE
        ),
        state_v=torch.empty(
            BATCH, HEADS, 1, DIM, device="cuda", dtype=DTYPE
        ),
        state_vlen=torch.empty(
            BATCH, HEADS, 1, 1, device="cuda", dtype=torch.float32
        ),
        recent_k=torch.empty(
            BATCH, HEADS, 1, DIM, device="cuda", dtype=DTYPE
        ),
        recent_v=torch.empty(
            BATCH, HEADS, 1, DIM, device="cuda", dtype=DTYPE
        ),
        recent_gate=torch.empty(
            BATCH, HEADS, 1, 1, device="cuda", dtype=DTYPE
        ),
        total_len=DECODE_SERIES_START,
        state_coverage_len=DECODE_SERIES_START,
        recent_begin=0,
    )

    def refresh_snapshot() -> None:
        with torch.no_grad():
            snapshot.state_k.copy_(
                mixer._prepare_state_update_k(sources["k"][:, :, :1])
            )
            snapshot.state_v.copy_(sources["v"][:, :, :1])
            snapshot.state_vlen.copy_(
                torch.linalg.vector_norm(
                    sources["v"][:, :, :1].float(), dim=-1, keepdim=True
                )
            )
            snapshot.recent_k.copy_(sources["k"][:, :, :1])
            snapshot.recent_v.copy_(sources["v"][:, :, :1])
            snapshot.recent_gate.copy_(sources["gate"][:, :, :1])

    refresh_snapshot()
    required_capacity = TritonKVMDecodeCache.state_length_after_cycle(
        mixer,
        DECODE_SERIES_START,
        int(snapshot.state_k.size(2)),
        end_position - DECODE_SERIES_START,
    )
    if required_capacity > int(mixer.max_state_len):
        raise AssertionError(
            f"decode series requires state {required_capacity}, configured "
            f"maximum is {mixer.max_state_len}"
        )
    cache = TritonKVMDecodeCache(
        mixer,
        snapshot,
        state_capacity=max(required_capacity, int(snapshot.state_k.size(2))),
    )
    current_q = torch.empty(BATCH, HEADS, 1, DIM, device="cuda", dtype=DTYPE)
    current_k = torch.empty_like(current_q)
    current_v = torch.empty_like(current_q)
    current_gate = torch.empty(
        BATCH, HEADS, 1, 1, device="cuda", dtype=DTYPE
    )
    output = torch.empty(BATCH, 1, HEADS * DIM, device="cuda", dtype=DTYPE)
    eviction_buffer = allocate_hardware_cache_eviction_buffer()
    event_pairs = allocate_event_pairs(end_position - DECODE_SERIES_START)

    def stage(position: int) -> None:
        index = position - 1
        current_q.copy_(sources["q"][:, :, index : index + 1])
        current_k.copy_(sources["k"][:, :, index : index + 1])
        current_v.copy_(sources["v"][:, :, index : index + 1])
        current_gate.copy_(sources["gate"][:, :, index : index + 1])

    def step() -> torch.Tensor:
        with torch.no_grad():
            return copy_single_token_output(
                mixer._decode_one_token(
                    current_q,
                    current_k,
                    current_v,
                    current_gate,
                    cache,
                ),
                output,
            )

    # Traverse the complete horizon once without timing. This compiles and
    # autotunes every position-dependent path, including direct-state appends,
    # recurrent updates, and all active state lengths.
    with torch.no_grad():
        cache.reset_(snapshot)
        for position in range(DECODE_SERIES_START + 1, end_position + 1):
            stage(position)
            step()
    torch.cuda.synchronize()

    # The measured trajectory receives different numerical inputs and returns
    # to its one-token starting point while retaining warmed kernels.
    refresh_decode_series_sources_(sources, generator)
    refresh_snapshot()
    with torch.no_grad():
        cache.reset_(snapshot)
    torch.cuda.synchronize()

    kinds = []
    state_lengths = []
    update_positions = []
    for invocation, position in enumerate(
        range(DECODE_SERIES_START + 1, end_position + 1)
    ):
        stream_hardware_cache_eviction(eviction_buffer)
        stage(position)
        updates_before = cache.update_count
        state_len_before = cache.state_len
        start, end = event_pairs[invocation]
        start.record()
        step()
        end.record()
        if cache.update_count > updates_before:
            kind = "update"
            update_positions.append(position)
        elif cache.state_len > state_len_before:
            kind = "direct_state_append"
        else:
            kind = "ordinary"
        kinds.append(kind)
        state_lengths.append(cache.state_len)
    torch.cuda.synchronize()
    invocation_ms = [start.elapsed_time(end) for start, end in event_pairs]
    if cache.total_len != end_position:
        raise AssertionError(
            f"KVM trajectory ended at {cache.total_len}, expected {end_position}"
        )
    if cache.update_count != len(update_positions):
        raise AssertionError("KVM update count disagrees with observed positions")
    if any(
        right - left != CHUNK
        for left, right in zip(update_positions, update_positions[1:])
    ):
        raise AssertionError("KVM recurrent updates are not 256 tokens apart")
    return {
        "positions": list(range(DECODE_SERIES_START + 1, end_position + 1)),
        "milliseconds": invocation_ms,
        "kinds": kinds,
        "state_lengths": state_lengths,
        "update_positions": update_positions,
        "final_update_count": cache.update_count,
        "final_state_len": cache.state_len,
    }


def run_attention_decode_series(end_position: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed + 40_000)
    sources = allocate_decode_series_sources(
        end_position, generator, include_gate=False
    )
    dense_k = torch.empty(
        BATCH, HEADS, end_position, DIM, device="cuda", dtype=DTYPE
    )
    dense_v = torch.empty_like(dense_k)
    current_q = torch.empty(BATCH, HEADS, 1, DIM, device="cuda", dtype=DTYPE)
    current_k = torch.empty_like(current_q)
    current_v = torch.empty_like(current_q)
    output = torch.empty(BATCH, 1, HEADS * DIM, device="cuda", dtype=DTYPE)
    eviction_buffer = allocate_hardware_cache_eviction_buffer()
    event_pairs = allocate_event_pairs(end_position - DECODE_SERIES_START)

    def reset() -> None:
        dense_k[:, :, :1].copy_(sources["k"][:, :, :1])
        dense_v[:, :, :1].copy_(sources["v"][:, :, :1])

    def stage(position: int) -> None:
        index = position - 1
        current_q.copy_(sources["q"][:, :, index : index + 1])
        current_k.copy_(sources["k"][:, :, index : index + 1])
        current_v.copy_(sources["v"][:, :, index : index + 1])

    def step(position: int) -> torch.Tensor:
        index = position - 1
        with torch.no_grad():
            dense_k[:, :, index : index + 1].copy_(current_k)
            dense_v[:, :, index : index + 1].copy_(current_v)
            return copy_single_token_output(
                F.scaled_dot_product_attention(
                    current_q,
                    dense_k[:, :, : position],
                    dense_v[:, :, : position],
                    dropout_p=0.0,
                    is_causal=False,
                ),
                output,
            )

    with flash_backend(), torch.no_grad():
        reset()
        for position in range(DECODE_SERIES_START + 1, end_position + 1):
            stage(position)
            step(position)
    torch.cuda.synchronize()

    refresh_decode_series_sources_(sources, generator)
    with torch.no_grad():
        reset()
    torch.cuda.synchronize()
    with flash_backend(), torch.no_grad():
        for invocation, position in enumerate(
            range(DECODE_SERIES_START + 1, end_position + 1)
        ):
            stream_hardware_cache_eviction(eviction_buffer)
            stage(position)
            start, end = event_pairs[invocation]
            start.record()
            step(position)
            end.record()
    torch.cuda.synchronize()
    invocation_ms = [start.elapsed_time(end) for start, end in event_pairs]
    return {
        "positions": list(range(DECODE_SERIES_START + 1, end_position + 1)),
        "milliseconds": invocation_ms,
        "kinds": ["ordinary"] * (end_position - DECODE_SERIES_START),
        "state_lengths": list(
            range(DECODE_SERIES_START + 1, end_position + 1)
        ),
        "update_positions": [],
        "final_update_count": 0,
        "final_state_len": end_position,
    }


# One arm/run GPU worker.


def run_decode_series_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("worker requires a CUDA/ROCm-visible GPU")
    if args.device >= torch.cuda.device_count():
        raise ValueError(
            f"device {args.device} is unavailable; "
            f"found {torch.cuda.device_count()} visible GPUs"
        )
    torch.cuda.set_device(args.device)
    environment_checks = {
        "device_is_mi325x": "MI325X" in torch.cuda.get_device_name(),
        "torch_is_2_9_1": torch.__version__.startswith("2.9.1"),
        "rocm_is_7_2": (torch.version.hip or "").startswith("7.2"),
    }
    if not args.smoke and not all(environment_checks.values()):
        raise RuntimeError(
            f"paper benchmark environment mismatch: {environment_checks}"
        )
    schedule = decode_series_schedule(args.arm)
    proof_schedule = schedule or "fixed256"
    proof_arms = prepare_decode(proof_schedule, 512, args.seed)
    proof_arm_name = "kvm" if schedule is not None else "full_attention"
    proof = backend_proof(proof_arm_name, proof_arms[proof_arm_name], "decode", 1)
    if not proof["passed"]:
        raise RuntimeError(f"decode-series backend proof failed: {proof}")
    del proof_arms
    gc.collect()
    torch.cuda.empty_cache()

    trajectory = (
        run_kvm_decode_series(schedule, args.end_position, args.seed)
        if schedule is not None
        else run_attention_decode_series(args.end_position, args.seed)
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in trajectory["milliseconds"]):
        raise RuntimeError("decode series contains an invalid timing")
    payload = {
        "arm": args.arm,
        "schedule": schedule,
        "run": args.run,
        "seed": args.seed,
        "workload": {
            "batch": BATCH,
            "heads": HEADS,
            "head_dim": DIM,
            "dtype": str(DTYPE),
            "initial_cached_tokens": DECODE_SERIES_START,
            "final_position": args.end_position,
            "timed_decode_calls": args.end_position - DECODE_SERIES_START,
        },
        "backend_proof": proof,
        "trajectory": trajectory,
        "environment": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": package_version("triton"),
        },
    }
    write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "run": args.run,
                "positions": len(trajectory["positions"]),
                "mean_ms": statistics.fmean(trajectory["milliseconds"]),
                "updates": trajectory["final_update_count"],
            },
            sort_keys=True,
        )
    )


# Shuffled local GPU execution.


def decode_series_worker_path(root: Path, arm: str, run: int) -> Path:
    return root / "raw" / f"{arm}_run{run}.json"


def submit_decode_series(args: argparse.Namespace) -> None:
    tasks = [
        (arm, run)
        for arm in DECODE_SERIES_ARMS
        for run in range(args.runs)
    ]
    random.Random(args.shuffle_seed).shuffle(tasks)
    args.root.mkdir(parents=True, exist_ok=True)
    commands = []
    for arm, run in tasks:
        output = decode_series_worker_path(args.root.resolve(), arm, run)
        if output.exists():
            raise FileExistsError(output)
        command = [
            str(Path(sys.executable).absolute()),
            str(BENCHMARK_SCRIPT),
            "decode-series-worker",
            "--arm",
            arm,
            "--run",
            str(run),
            "--seed",
            str(run),
            "--end-position",
            str(DECODE_SERIES_END),
            "--output",
            str(output),
        ]
        commands.append(command)
    if args.dry_run:
        print(json.dumps({"commands": commands}, indent=2))
        return
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError(
            "decode-series-submit requires at least one CUDA/ROCm-visible GPU"
        )
    gpu_slots = min(args.max_parallel, device_count)
    available_devices: Queue[int] = Queue()
    for device in range(gpu_slots):
        available_devices.put(device)

    def run_command(index: int) -> tuple[str, int, int]:
        arm, run = tasks[index]
        device = available_devices.get()
        try:
            command = [*commands[index], "--device", str(device)]
            result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        finally:
            available_devices.put(device)
        return arm, run, result.returncode

    with ThreadPoolExecutor(max_workers=gpu_slots) as executor:
        futures = [executor.submit(run_command, index) for index in range(len(tasks))]
        completed = [future.result() for future in as_completed(futures)]
    failures = [row for row in completed if row[-1] != 0]
    if failures:
        raise RuntimeError(f"decode-series workers failed: {failures}")
    print(json.dumps({"jobs": len(completed), "gpu_slots": gpu_slots}))


# Plotting and analysis.


def render_decode_series_plot(
    summaries: dict[str, list[dict[str, float | int]]],
    speedups: dict[str, list[float]],
    output_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "decode_1_to_32768_timeseries.png"
    if output.exists():
        raise FileExistsError(output)
    styles = {
        "kvm_sqrt16": ("KVM ($16\\sqrt{n}$)", "#0072B2"),
        "kvm_fixed256": ("KVM (fixed 256)", "#009E73"),
        "full_attention": ("Attention", "#D55E00"),
    }
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    positions = list(range(DECODE_SERIES_START + 1, DECODE_SERIES_END + 1))
    for arm in DECODE_SERIES_ARMS:
        label, color = styles[arm]
        rows = summaries[arm]
        means = [float(row["mean"]) for row in rows]
        axes[0].plot(positions, means, color=color, linewidth=0.75, label=label)
        axes[0].fill_between(
            positions,
            [float(row["p10"]) for row in rows],
            [float(row["p90"]) for row in rows],
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("ms/token")
    axes[0].set_title("(a) Per-position decode time")
    axes[0].legend(frameon=False, loc="upper left")

    for arm, label, color in (
        ("kvm_fixed256", "Attention / KVM-256", "#009E73"),
        ("kvm_sqrt16", "Attention / KVM-sqrt", "#0072B2"),
    ):
        axes[1].plot(
            positions,
            speedups[arm],
            color=color,
            linewidth=0.75,
            label=label,
        )
    axes[1].axhline(1.0, color="#555555", linewidth=0.7, linestyle=":")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("speedup")
    axes[1].set_title("(b) Full attention / KVM")
    axes[1].legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.set_xlim(DECODE_SERIES_START, DECODE_SERIES_END)
        ax.set_xticks((0, 8192, 16384, 24576, 32768))
        ax.set_xticklabels(("0", "8", "16", "24", "32"))
        ax.set_xlabel(r"position ($\times 1024$)")
        ax.grid(axis="y", color="#B0B0B0", linewidth=0.4, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(output, dpi=600, metadata={"Creator": BENCHMARK_SCRIPT.name})
    plt.close(fig)
    return output


def analyze_decode_series(args: argparse.Namespace) -> None:
    runs_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in DECODE_SERIES_ARMS
    }
    expected_positions = list(
        range(DECODE_SERIES_START + 1, DECODE_SERIES_END + 1)
    )
    for arm in DECODE_SERIES_ARMS:
        for run in range(args.runs):
            path = decode_series_worker_path(args.root, arm, run)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("arm") != arm or payload.get("run") != run:
                raise RuntimeError(f"decode-series identity mismatch: {path}")
            if not payload.get("backend_proof", {}).get("passed"):
                raise RuntimeError(f"decode-series backend proof failed: {path}")
            trajectory = payload["trajectory"]
            if trajectory["positions"] != expected_positions:
                raise RuntimeError(f"decode-series position mismatch: {path}")
            for values_name in ("milliseconds", "kinds", "state_lengths"):
                if len(trajectory[values_name]) != len(expected_positions):
                    raise RuntimeError(
                        f"decode-series {values_name} length mismatch: {path}"
                    )
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in trajectory["milliseconds"]
            ):
                raise RuntimeError(f"invalid decode-series timing: {path}")
            runs_by_arm[arm].append(payload)

    args.root.mkdir(parents=True, exist_ok=True)
    raw_path = args.root / "decode_series_runs.csv"
    with raw_path.open("x", encoding="utf-8", newline="") as stream:
        fields = (
            "arm",
            "run",
            "seed",
            "position",
            "milliseconds",
            "kind",
            "state_len",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm in DECODE_SERIES_ARMS:
            for payload in runs_by_arm[arm]:
                trajectory = payload["trajectory"]
                for position, elapsed, kind, state_len in zip(
                    trajectory["positions"],
                    trajectory["milliseconds"],
                    trajectory["kinds"],
                    trajectory["state_lengths"],
                    strict=True,
                ):
                    writer.writerow(
                        {
                            "arm": arm,
                            "run": payload["run"],
                            "seed": payload["seed"],
                            "position": position,
                            "milliseconds": elapsed,
                            "kind": kind,
                            "state_len": state_len,
                        }
                    )

    summaries: dict[str, list[dict[str, float | int]]] = {}
    summary_path = args.root / "decode_series_mean.csv"
    with summary_path.open("x", encoding="utf-8", newline="") as stream:
        fields = (
            "arm",
            "position",
            "count",
            "mean",
            "median",
            "min",
            "max",
            "p10",
            "p90",
            "stdev",
            "cv",
            "cv_percent",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm in DECODE_SERIES_ARMS:
            arm_rows = []
            arrays = [payload["trajectory"]["milliseconds"] for payload in runs_by_arm[arm]]
            for index, position in enumerate(expected_positions):
                row = {
                    "arm": arm,
                    "position": position,
                    **distribution([float(values[index]) for values in arrays]),
                }
                writer.writerow(row)
                arm_rows.append(row)
            summaries[arm] = arm_rows

    attention_means = [
        float(row["mean"]) for row in summaries["full_attention"]
    ]
    speedups = {
        arm: [
            attention / float(row["mean"])
            for attention, row in zip(attention_means, summaries[arm], strict=True)
        ]
        for arm in ("kvm_fixed256", "kvm_sqrt16")
    }
    speedup_path = args.root / "decode_series_speedup.csv"
    with speedup_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("position", "attention_over_fixed256", "attention_over_sqrt16"),
        )
        writer.writeheader()
        for index, position in enumerate(expected_positions):
            writer.writerow(
                {
                    "position": position,
                    "attention_over_fixed256": speedups["kvm_fixed256"][index],
                    "attention_over_sqrt16": speedups["kvm_sqrt16"][index],
                }
            )

    render_decode_series_plot(summaries, speedups, args.root / "plots")
    print(
        json.dumps(
            {
                "raw_rows": len(expected_positions) * args.runs * len(DECODE_SERIES_ARMS),
                "mean_rows": len(expected_positions) * len(DECODE_SERIES_ARMS),
                "plots": 1,
            },
            sort_keys=True,
        )
    )
