#!/usr/bin/env python3
"""Run the KVM paper benchmarks.

The ``worker``/``submit``/``analyze`` commands reproduce the B8/H32/D128
fixed-context benchmark. The ``decode-series-*`` commands reproduce the full
one-token-to-32K decode trajectory. Both compare Triton KVM with PyTorch Flash
SDPA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kvm_benchmark.common import CONTEXTS, PHASES, REPO_ROOT, SCHEDULES, SEEDS
from kvm_benchmark.controlled import analyze, run_worker, submit
from kvm_benchmark.decode_series import (
    DECODE_SERIES_ARMS,
    DECODE_SERIES_END,
    analyze_decode_series,
    run_decode_series_worker,
    submit_decode_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker", help="run one GPU benchmark coordinate")
    worker.add_argument("--schedule", choices=SCHEDULES, required=True)
    worker.add_argument("--context", type=int, choices=CONTEXTS, required=True)
    worker.add_argument("--phase", choices=PHASES, required=True)
    worker.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    worker.add_argument("--warmup", type=int, default=5)
    worker.add_argument("--rounds", type=int, default=4)
    worker.add_argument("--target-ms", type=float, default=250.0)
    worker.add_argument("--device", type=int, default=0)
    worker.add_argument("--smoke", action="store_true")
    worker.add_argument("--output", type=Path, required=True)

    submit_parser = subparsers.add_parser(
        "submit", help="run all 42 benchmark jobs on local GPUs"
    )
    submit_parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "experiments" / "kvm_benchmark_b8h32",
    )
    submit_parser.add_argument("--shuffle-seed", type=int, default=20260722)
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument("--max-parallel", type=int, default=8)

    analyze_parser = subparsers.add_parser(
        "analyze", help="check artifacts and plot benchmark results"
    )
    analyze_parser.add_argument("--root", type=Path, required=True)

    series_worker = subparsers.add_parser(
        "decode-series-worker",
        help="time every decode position in one trajectory",
    )
    series_worker.add_argument("--arm", choices=DECODE_SERIES_ARMS, required=True)
    series_worker.add_argument("--run", type=int, required=True)
    series_worker.add_argument("--seed", type=int, required=True)
    series_worker.add_argument("--device", type=int, default=0)
    series_worker.add_argument(
        "--end-position", type=int, default=DECODE_SERIES_END
    )
    series_worker.add_argument("--smoke", action="store_true")
    series_worker.add_argument("--output", type=Path, required=True)

    series_submit = subparsers.add_parser(
        "decode-series-submit",
        help="run the shuffled three-arm, ten-run decode time series on local GPUs",
    )
    series_submit.add_argument("--root", type=Path, required=True)
    series_submit.add_argument("--runs", type=int, default=10)
    series_submit.add_argument("--shuffle-seed", type=int, default=20260722)
    series_submit.add_argument("--dry-run", action="store_true")
    series_submit.add_argument("--max-parallel", type=int, default=8)

    series_analyze = subparsers.add_parser(
        "decode-series-analyze",
        help="average and plot the complete per-position decode series",
    )
    series_analyze.add_argument("--root", type=Path, required=True)
    series_analyze.add_argument("--runs", type=int, default=10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.command in ("worker", "decode-series-worker") and args.output.exists():
        raise FileExistsError(args.output)
    if args.command in ("worker", "decode-series-worker") and args.device < 0:
        raise ValueError("device must be nonnegative")
    if args.command == "worker":
        if args.warmup < 0 or args.rounds <= 0 or args.target_ms <= 0:
            raise ValueError("warmup must be nonnegative; rounds and target-ms positive")
        if not args.seeds or len(args.seeds) != len(set(args.seeds)):
            raise ValueError("seeds must be nonempty and unique")
    elif args.command == "decode-series-worker":
        if args.run < 0 or args.seed < 0:
            raise ValueError("run and seed must be nonnegative")
        if not (2 <= args.end_position <= DECODE_SERIES_END):
            raise ValueError("end-position must be between 2 and 32768")
    if args.command in ("submit", "decode-series-submit") and args.max_parallel <= 0:
        raise ValueError("max-parallel must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    handlers = {
        "worker": run_worker,
        "submit": submit,
        "analyze": analyze,
        "decode-series-worker": run_decode_series_worker,
        "decode-series-submit": submit_decode_series,
        "decode-series-analyze": analyze_decode_series,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
