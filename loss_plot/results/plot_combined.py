#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss_plot.results.plot_block_losses_seaborn import (
    build_series,
    plot_stacked_series_groups,
)


LABELS = [
    "BSWA",
    "OVQ/SWA (sat.)",
    "RWKV-7",
    "KVM (fixed size)",
    "KVM/SWA (sat.)",
    "KVM (sqrt)",
    "GPTAHalfRoPE",
    "GPTAHalfRoPE/SWA",

    # "GPTAHalfRoPE/SWA 7Gtok",
    # "GPTANoPE/SWA 7Gtok",
    # "GPTANoPE/SWA 3Gtok",
]

MODEL_DIR_PATTERNS = [
    "bswa_{size}",
    "ovqsat_swa256_{size}",
    "rwkv7_ffn4_{size}",
    "kvm256_{size}",
    "kvmsat_swa256_{size}",
    "kvmsqrt16_{size}",
    "gptalpha_{size}",
    "gptalphaprope_swa256_{size}",

    # "gptalphaprope_swa256_{size}",
    # "gptalpha_swa256_{size}", 
    # "gptalpha_swa256_3Gtok_{size}",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a single stacked figure with one panel per model size and a shared legend "
            "on the right."
        )
    )
    parser.add_argument(
        "--logs-root",
        default="results_block_losses/new_logs",
        help="Root directory containing per-size eval subdirectories. Default: %(default)s.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["120M", "350M"],
        help="Model sizes to plot in top-to-bottom order. Default: %(default)s.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Number of consecutive positions per block. Default: %(default)s.",
    )
    parser.add_argument(
        "--nickname",
        type=str,
        default="textbook_chapters",
        help="Nickname of the evaluation, used in the legend. Default: %(default)s.",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=16384,
        help="Number of consecutive positions per block. Default: %(default)s.",
    )
    parser.add_argument(
        "--metric",
        choices=("loss", "perplexity"),
        default="loss",
        help="Metric to plot in the stacked figure. Default: %(default)s.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI. Default: %(default)s.",
    )
    parser.add_argument(
        "--output-path",
        default="results_textbook_chapters",
        help="Output image path. Default: %(default)s.",
    )
    return parser.parse_args()


def json_paths_for_size(
    logs_root: Path, size: str, nickname: str, context_length: int
) -> list[str]:
    return [
        str(
            (
                logs_root
                / pattern.format(size=size)
                / f"{nickname}_eval_{context_length}.json"
            ).resolve()
        )
        for pattern in MODEL_DIR_PATTERNS
    ]


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")

    logs_root = Path(args.logs_root).expanduser().resolve()
    output_path = (
        (
            Path(args.output_path)
            / (args.nickname + "_" + str(args.context_length) + ".png")
        )
        .expanduser()
        .resolve()
    )

    series_groups = [
        build_series(
            json_paths_for_size(logs_root, size, args.nickname, args.context_length),
            LABELS,
            args.block_size,
        )
        for size in args.sizes
    ]
    panel_titles = [f"{args.nickname} {size}" for size in args.sizes]

    plot_stacked_series_groups(
        series_groups,
        panel_titles,
        block_size=args.block_size,
        output_path=output_path,
        dpi=args.dpi,
        metric=args.metric,
    )

    print(f"saved_plot : {output_path}")
    print(f"logs_root  : {logs_root}")
    print(f"block_size : {args.block_size}")
    print(f"metric     : {args.metric}")
    print(f"sizes      : {', '.join(args.sizes)}")


if __name__ == "__main__":
    main()
