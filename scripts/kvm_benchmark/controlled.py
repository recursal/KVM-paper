"""The benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gc
import json
import math
from pathlib import Path
from queue import Queue
import random
import subprocess
import sys
from typing import Any

import torch

from .common import (
    ARMS,
    BATCH,
    BENCHMARK_SCRIPT,
    CHUNK,
    CONTEXTS,
    DIM,
    HEADS,
    PHASES,
    REPO_ROOT,
    SCHEDULES,
    backend_proof,
    distribution,
    measure_arm,
    package_version,
    prepare_decode,
    prepare_prefill_or_backward,
    write_json_exclusive,
)

PLOT_FILES = {
    "prefill": "b8_h32_forward.png",
    "backward": "b8_h32_backward.png",
    "decode": "b8_h32_ar_decode.png",
}
MASTER_PLOT_FILE = "b8_h32_all_phases.png"
MASTER_PHASES = ("decode", "prefill", "backward")
PHASE_TITLES = {
    "decode": "Decode",
    "prefill": "Prefill",
    "backward": "Backward",
}
PLOT_SERIES = (
    ("kvm_sqrt16", "KVM ($16\\sqrt{n}$)", "#0072B2", "o", "-"),
    ("kvm_fixed256", "KVM (fixed 256)", "#009E73", "^", "-."),
    ("full_attention", "Attention", "#D55E00", "s", "--"),
)


# Plotting.


def combined_plot_data(
    raw_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, float | int]]:
    grouped: defaultdict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in raw_rows:
        if row["arm"] == "full_attention":
            series = "full_attention"
        else:
            series = f"kvm_{row['schedule']}"
        grouped[(row["phase"], series, int(row["context"]))].append(
            float(row["median"])
        )

    expected = {
        (phase, series, context)
        for phase in PLOT_FILES
        for series, *_ in PLOT_SERIES
        for context in CONTEXTS
    }
    if set(grouped) != expected:
        raise ValueError(
            "combined plot grid mismatch; "
            f"missing={sorted(expected - set(grouped))}, "
            f"unexpected={sorted(set(grouped) - expected)}"
        )
    indexed = {}
    for key, values in grouped.items():
        indexed[key] = distribution(values)
    return indexed


def render_combined_plots(
    raw_rows: list[dict[str, Any]], output_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indexed = combined_plot_data(raw_rows)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.25,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.75,
            "ytick.labelsize": 8.75,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
        }
    )
    output_dir.mkdir(exist_ok=True)
    outputs = []

    def draw_phase(axis: Any, phase: str, *, legend: bool) -> None:
        ymax = max(
            float(indexed[(phase, series, context)]["max"])
            for series, *_ in PLOT_SERIES
            for context in CONTEXTS
        ) * 1.06
        for series, label, color, marker, linestyle in PLOT_SERIES:
            rows = [indexed[(phase, series, context)] for context in CONTEXTS]
            axis.fill_between(
                CONTEXTS,
                [float(row["min"]) for row in rows],
                [float(row["max"]) for row in rows],
                color=color,
                alpha=0.14,
                linewidth=0,
            )
            axis.plot(
                CONTEXTS,
                [float(row["median"]) for row in rows],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.65,
                markersize=4.1,
                markeredgewidth=0.75,
                label=label,
            )
        axis.set_xscale("linear")
        axis.set_yscale("linear")
        axis.set_xlim(0, CONTEXTS[-1] * 1.04)
        axis.set_ylim(0, ymax)
        axis.set_xticks((0, 8192, 16384, 24576, 32768))
        axis.set_xticklabels(("0", "8", "16", "24", "32"))
        axis.set_xticks(CONTEXTS, minor=True)
        axis.set_xlabel(r"$T$ ($\times 1024$)", labelpad=2)
        axis.set_ylabel("ms/token" if phase == "decode" else "ms", labelpad=2)
        axis.grid(
            axis="y", which="major", color="#A8A8A8", linewidth=0.4, alpha=0.5
        )
        axis.tick_params(axis="both", which="major", length=3.2, pad=2.0)
        axis.tick_params(axis="x", which="minor", length=2.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if legend:
            axis.legend(
                loc="upper left",
                frameon=False,
                handlelength=2.15,
                borderaxespad=0.15,
                labelspacing=0.25,
            )

    for phase, filename in PLOT_FILES.items():
        output = output_dir / filename
        if output.exists():
            raise FileExistsError(output)
        fig, axis = plt.subplots(figsize=(3.5, 2.65), layout="constrained")
        draw_phase(axis, phase, legend=True)
        fig.savefig(output, dpi=600, metadata={"Creator": BENCHMARK_SCRIPT.name})
        plt.close(fig)
        outputs.append(output)

    master_output = output_dir / MASTER_PLOT_FILE
    if master_output.exists():
        raise FileExistsError(master_output)
    fig, axes = plt.subplots(
        1,
        len(MASTER_PHASES),
        figsize=(7.2, 2.75),
        layout="constrained",
    )
    for panel_index, (axis, phase) in enumerate(zip(axes, MASTER_PHASES)):
        draw_phase(axis, phase, legend=False)
        axis.set_title(
            f"({chr(ord('a') + panel_index)}) {PHASE_TITLES[phase]}", pad=3.0
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=len(PLOT_SERIES),
        frameon=False,
        handlelength=2.15,
        columnspacing=1.5,
    )
    fig.savefig(
        master_output,
        dpi=600,
        metadata={"Creator": BENCHMARK_SCRIPT.name},
    )
    plt.close(fig)
    outputs.append(master_output)
    return outputs


# One schedule/context/phase GPU worker.


def run_worker(args: argparse.Namespace) -> None:
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
    records = []
    proofs = None
    for seed in args.seeds:
        arms = (
            prepare_decode(args.schedule, args.context, seed)
            if args.phase == "decode"
            else prepare_prefill_or_backward(
                args.schedule, args.context, seed, args.phase
            )
        )
        if proofs is None:
            proofs = {
                arm_name: backend_proof(
                    arm_name,
                    arms[arm_name],
                    args.phase,
                    1 if args.phase == "decode" else args.context,
                )
                for arm_name in ARMS
            }
            failed_proofs = {
                arm_name: proof
                for arm_name, proof in proofs.items()
                if not proof["passed"]
            }
            if failed_proofs:
                raise RuntimeError(f"backend proof failed: {failed_proofs}")
        order = ARMS if seed % 2 == 0 else tuple(reversed(ARMS))
        measured = {}
        for arm_name in order:
            measured[arm_name] = measure_arm(
                arms[arm_name],
                warmup=args.warmup,
                rounds=args.rounds,
                target_ms=args.target_ms,
                unit_scale=CHUNK if args.phase == "decode" else 1.0,
            )
        records.append({"seed": seed, "arms": measured})
        del arms
        gc.collect()
        torch.cuda.empty_cache()

    aggregate = {
        arm: distribution(
            [record["arms"][arm]["summary"]["median"] for record in records]
        )
        for arm in ARMS
    }
    payload = {
        "workload": {
            "schedule": args.schedule,
            "phase": args.phase,
            "batch": BATCH,
            "heads": HEADS,
            "head_dim": DIM,
            "context": args.context,
            "dtype": "torch.bfloat16",
        },
        "backend_proofs": proofs,
        "records": records,
        "aggregate": aggregate,
        "environment": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": package_version("triton"),
        },
    }
    write_json_exclusive(args.output, payload)
    print(json.dumps({"aggregate": aggregate}, sort_keys=True))


# Shuffled local GPU execution.


def worker_path(root: Path, schedule: str, context: int, phase: str) -> Path:
    return root / "raw" / f"{schedule}_b8h32_t{context}_{phase}.json"


def submit(args: argparse.Namespace) -> None:
    tasks = [
        (schedule, context, phase)
        for schedule in SCHEDULES
        for context in CONTEXTS
        for phase in PHASES
    ]
    random.Random(args.shuffle_seed).shuffle(tasks)
    args.root.mkdir(parents=True, exist_ok=True)
    commands = []
    for schedule, context, phase in tasks:
        output = worker_path(args.root.resolve(), schedule, context, phase)
        if output.exists():
            raise FileExistsError(output)
        command = [
            # Do not resolve this path: the virtualenv interpreter is a symlink,
            # and resolving it bypasses the environment's installed packages.
            str(Path(sys.executable).absolute()), str(BENCHMARK_SCRIPT),
            "worker", "--schedule", schedule, "--context", str(context),
            "--phase", phase, "--output", str(output),
        ]
        commands.append(command)
    if args.dry_run:
        print(json.dumps({"commands": commands}, indent=2))
        return
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("submit requires at least one CUDA/ROCm-visible GPU")
    gpu_slots = min(args.max_parallel, device_count)
    available_devices: Queue[int] = Queue()
    for device in range(gpu_slots):
        available_devices.put(device)

    def run_command(index: int) -> tuple[str, int, str, int]:
        schedule, context, phase = tasks[index]
        device = available_devices.get()
        try:
            command = [*commands[index], "--device", str(device)]
            result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        finally:
            available_devices.put(device)
        return schedule, context, phase, result.returncode

    with ThreadPoolExecutor(max_workers=gpu_slots) as executor:
        futures = [executor.submit(run_command, index) for index in range(len(tasks))]
        completed = [future.result() for future in as_completed(futures)]
    failures = [row for row in completed if row[-1] != 0]
    if failures:
        raise RuntimeError(f"benchmark workers failed: {failures}")
    print(json.dumps({"jobs": len(completed), "gpu_slots": gpu_slots}))


# Analysis.


def analyze(args: argparse.Namespace) -> None:
    raw_rows = []
    summaries = []
    for schedule in SCHEDULES:
        for phase in PHASES:
            for context in CONTEXTS:
                path = worker_path(args.root, schedule, context, phase)
                payload = json.loads(path.read_text(encoding="utf-8"))
                expected = {
                    "schedule": schedule,
                    "phase": phase,
                    "batch": BATCH,
                    "heads": HEADS,
                    "head_dim": DIM,
                    "context": context,
                    "dtype": "torch.bfloat16",
                }
                if payload.get("workload") != expected:
                    raise RuntimeError(f"workload metadata mismatch: {path}")
                records = payload.get("records", [])
                seeds = [record["seed"] for record in records]
                if not records or len(seeds) != len(set(seeds)):
                    raise RuntimeError(f"invalid seed records: {path}")
                backend_proofs = payload.get("backend_proofs", {})
                if set(backend_proofs) != set(ARMS):
                    raise RuntimeError(f"backend proof coverage mismatch: {path}")
                if not all(proof.get("passed") for proof in backend_proofs.values()):
                    raise RuntimeError(f"backend proof failed: {path}")
                for record in records:
                    for arm in ARMS:
                        arm_result = record["arms"][arm]
                        samples = arm_result.get("samples", [])
                        if not samples or any(
                            not math.isfinite(value) or value <= 0.0
                            for value in samples
                        ):
                            raise RuntimeError(f"invalid timing samples: {path}")
                        raw_rows.append(
                            {
                                "schedule": schedule,
                                "phase": phase,
                                "context": context,
                                "batch": BATCH,
                                "heads": HEADS,
                                "arm": arm,
                                "seed": record["seed"],
                                "median": arm_result["summary"]["median"],
                            }
                        )
                for arm in ARMS:
                    summaries.append(
                        {
                            "schedule": schedule,
                            "phase": phase,
                            "context": context,
                            "batch": BATCH,
                            "heads": HEADS,
                            "arm": arm,
                            **payload["aggregate"][arm],
                        }
                    )
                speedups = [
                    record["arms"]["full_attention"]["summary"]["median"]
                    / record["arms"]["kvm"]["summary"]["median"]
                    for record in records
                ]
                summaries.append(
                    {
                        "schedule": schedule,
                        "phase": phase,
                        "context": context,
                        "batch": BATCH,
                        "heads": HEADS,
                        "arm": "full_attention_over_kvm_speedup",
                        **distribution(speedups),
                    }
                )
    args.root.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("runs.csv", raw_rows),
        ("summary.csv", summaries),
    ):
        path = args.root / name
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plots = render_combined_plots(raw_rows, args.root / "plots")
    print(json.dumps({"rows": len(raw_rows), "plots": len(plots)}))
