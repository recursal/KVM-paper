#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "speedtrain-mpl"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ImportError as exc:
    raise SystemExit(
        "This script requires matplotlib, pandas, and seaborn in the active environment."
    ) from exc


@dataclass
class BlockLoss:
    block_index: int
    start: int
    end: int
    token_count: int
    mean_loss: float | None

    @property
    def perplexity(self) -> float | None:
        if self.mean_loss is None:
            return None
        return math.exp(self.mean_loss)


@dataclass
class LossSeries:
    path: Path
    label: str
    context_length: int
    blocks: list[BlockLoss]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot token-weighted block averages of per-position losses and perplexities "
            "from one or more eval JSON files using seaborn."
        )
    )
    parser.add_argument(
        "json_paths", nargs="+", help="One or more eval JSON files."
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Number of consecutive positions per block. Default: %(default)s.",
    )
    parser.add_argument(
        "--metric",
        choices=("loss", "perplexity", "both"),
        default="both",
        help="Which metric panels to plot. Default: %(default)s.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional legend labels in the same order as json_paths.",
    )
    parser.add_argument(
        "--baseline",
        help=(
            "Optional baseline series. Match by label, input path, basename, or stem. "
            "Use 'mean' to compare against the mean across all input JSONs."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to a PNG in the current directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI. Default: %(default)s.",
    )
    return parser.parse_args()


def sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "baseline"


def default_label(path: Path) -> str:
    return path.stem


def load_json(path: Path) -> dict:
    with path.open("rt", encoding="utf-8") as f:
        return json.load(f)


def mean_optional(values: list[float | None]) -> float | None:
    non_null = [float(value) for value in values if value is not None]
    if not non_null:
        return None
    return sum(non_null) / len(non_null)


def compute_block_losses(data: dict, block_size: int) -> list[BlockLoss]:
    per_token_position = data.get("per_token_position")
    if not isinstance(per_token_position, dict):
        raise ValueError("Missing 'per_token_position' section.")

    token_counts = per_token_position.get("tokens")
    avg_nll = per_token_position.get("avg_nll")
    if not isinstance(token_counts, list) or not isinstance(avg_nll, list):
        raise ValueError(
            "'per_token_position' must contain list fields 'tokens' and 'avg_nll'."
        )
    if len(token_counts) != len(avg_nll):
        raise ValueError(
            f"'per_token_position.tokens' length {len(token_counts)} does not match "
            f"'per_token_position.avg_nll' length {len(avg_nll)}."
        )

    blocks: list[BlockLoss] = []
    for start in range(0, len(token_counts), block_size):
        stop = min(start + block_size, len(token_counts))
        weighted_loss_sum = 0.0
        block_token_count = 0
        for count, loss in zip(token_counts[start:stop], avg_nll[start:stop]):
            count_int = int(count)
            if count_int <= 0:
                continue
            if loss is None:
                raise ValueError(
                    f"Found a null loss for positions with tokens in block {start}:{stop}."
                )
            weighted_loss_sum += count_int * float(loss)
            block_token_count += count_int
        mean_loss = None
        if block_token_count > 0:
            mean_loss = weighted_loss_sum / block_token_count
        blocks.append(
            BlockLoss(
                block_index=start // block_size,
                start=start,
                end=stop - 1,
                token_count=block_token_count,
                mean_loss=mean_loss,
            )
        )
    return blocks


def build_series(
    json_paths: list[str], labels: list[str] | None, block_size: int
) -> list[LossSeries]:
    if labels is not None and len(labels) not in {0, len(json_paths)}:
        raise ValueError(
            "--labels must be omitted or have exactly one entry per input JSON."
        )

    series_list: list[LossSeries] = []
    for index, json_path in enumerate(json_paths):
        path = Path(json_path).expanduser().resolve()
        data = load_json(path)
        label = labels[index] if labels else default_label(path)
        context_length = int(
            data.get("context_length", len(data["per_token_position"]["tokens"]))
        )
        series_list.append(
            LossSeries(
                path=path,
                label=label,
                context_length=context_length,
                blocks=compute_block_losses(data, block_size),
            )
        )
    return series_list


def get_reference_blocks(series_list: list[LossSeries]) -> list[BlockLoss]:
    return max(
        series_list, key=lambda item: (len(item.blocks), item.context_length)
    ).blocks


def build_mean_baseline(series_list: list[LossSeries]) -> LossSeries:
    if not series_list:
        raise ValueError("Need at least one series to compute a mean baseline.")

    reference_blocks = get_reference_blocks(series_list)
    mean_blocks: list[BlockLoss] = []
    for ref_block in reference_blocks:
        block_losses: list[float | None] = []
        block_token_count = 0
        for series in series_list:
            if ref_block.block_index >= len(series.blocks):
                continue
            block = series.blocks[ref_block.block_index]
            block_losses.append(block.mean_loss)
            block_token_count += block.token_count
        mean_blocks.append(
            BlockLoss(
                block_index=ref_block.block_index,
                start=ref_block.start,
                end=ref_block.end,
                token_count=block_token_count,
                mean_loss=mean_optional(block_losses),
            )
        )

    return LossSeries(
        path=Path("<mean>"),
        label="mean",
        context_length=max(series.context_length for series in series_list),
        blocks=mean_blocks,
    )


def resolve_baseline(baseline: str, series_list: list[LossSeries]) -> LossSeries:
    if baseline.strip().lower() == "mean":
        return build_mean_baseline(series_list)

    baseline_path = Path(baseline).expanduser()
    normalized_inputs = {
        baseline,
        baseline_path.name,
        baseline_path.stem,
    }
    try:
        normalized_inputs.add(str(baseline_path.resolve()))
    except OSError:
        pass

    matches = [
        series
        for series in series_list
        if series.label == baseline
        or str(series.path) in normalized_inputs
        or series.path.name in normalized_inputs
        or series.path.stem in normalized_inputs
    ]
    if not matches:
        available = ", ".join(series.label for series in series_list)
        raise ValueError(
            f"Could not find baseline {baseline!r}. Available labels: {available}"
        )
    if len(matches) > 1:
        options = ", ".join(
            f"{series.label} ({series.path.name})" for series in matches
        )
        raise ValueError(f"Baseline {baseline!r} is ambiguous. Matches: {options}")
    return matches[0]


def block_tick_values(
    blocks: list[BlockLoss], max_ticks: int = 12
) -> tuple[list[int], list[str]]:
    if not blocks:
        return [], []
    stride = max(1, math.ceil(len(blocks) / max_ticks))
    tick_positions = [block.block_index for block in blocks[::stride]]
    tick_labels = [f"{block.start}-{block.end}" for block in blocks[::stride]]
    return tick_positions, tick_labels


def endpoint_tick_values(blocks: list[BlockLoss]) -> tuple[list[int], list[str]]:
    if not blocks:
        return [], []
    first_block = blocks[0]
    last_block = blocks[-1]
    return [first_block.block_index, last_block.block_index], [
        str(first_block.end + 1),
        str(last_block.end + 1),
    ]


def make_output_path(output: str | None, baseline: LossSeries | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    base_name = "block_losses_seaborn"
    if baseline is not None:
        base_name += f"_vs_{sanitize_label(baseline.label)}"
    return Path.cwd() / f"{base_name}.png"


def metric_value(block: BlockLoss, metric_name: str) -> float | None:
    if metric_name == "loss":
        return block.mean_loss
    return block.perplexity


def build_plot_dataframe(
    series_list: list[LossSeries],
    baseline: LossSeries | None,
    metric: str,
) -> tuple[pd.DataFrame, list[str], dict[str, str], set[str]]:
    records: list[dict[str, object]] = []
    panel_order: list[str] = []
    panel_to_ylabel: dict[str, str] = {}
    relative_panels: set[str] = set()

    if metric in {"loss", "both"}:
        absolute_panel = "Mean loss"
        panel_order.append(absolute_panel)
        panel_to_ylabel[absolute_panel] = "Mean loss"
        for series in series_list:
            for block in series.blocks:
                value = metric_value(block, "loss")
                if value is None:
                    continue
                records.append(
                    {
                        "panel": absolute_panel,
                        "series": series.label,
                        "block_index": block.block_index,
                        "value": value,
                    }
                )
        if baseline is not None:
            relative_panel = f"Loss delta vs {baseline.label}"
            panel_order.append(relative_panel)
            panel_to_ylabel[relative_panel] = relative_panel
            relative_panels.add(relative_panel)
            for series in series_list:
                for series_block, baseline_block in zip(series.blocks, baseline.blocks):
                    series_value = metric_value(series_block, "loss")
                    baseline_value = metric_value(baseline_block, "loss")
                    if series_value is None or baseline_value is None:
                        continue
                    records.append(
                        {
                            "panel": relative_panel,
                            "series": series.label,
                            "block_index": series_block.block_index,
                            "value": series_value - baseline_value,
                        }
                    )

    if metric in {"perplexity", "both"}:
        absolute_panel = "Perplexity"
        panel_order.append(absolute_panel)
        panel_to_ylabel[absolute_panel] = "Perplexity"
        for series in series_list:
            for block in series.blocks:
                value = metric_value(block, "perplexity")
                if value is None:
                    continue
                records.append(
                    {
                        "panel": absolute_panel,
                        "series": series.label,
                        "block_index": block.block_index,
                        "value": value,
                    }
                )
        if baseline is not None:
            relative_panel = f"PPL delta vs {baseline.label}"
            panel_order.append(relative_panel)
            panel_to_ylabel[relative_panel] = relative_panel
            relative_panels.add(relative_panel)
            for series in series_list:
                for series_block, baseline_block in zip(series.blocks, baseline.blocks):
                    series_value = metric_value(series_block, "perplexity")
                    baseline_value = metric_value(baseline_block, "perplexity")
                    if series_value is None or baseline_value is None:
                        continue
                    records.append(
                        {
                            "panel": relative_panel,
                            "series": series.label,
                            "block_index": series_block.block_index,
                            "value": series_value - baseline_value,
                        }
                    )

    return (
        pd.DataFrame.from_records(records),
        panel_order,
        panel_to_ylabel,
        relative_panels,
    )


def default_title(metric: str) -> str:
    if metric == "loss":
        return "Block-Averaged Losses"
    if metric == "perplexity":
        return "Block-Averaged Perplexities"
    return "Block-Averaged Losses And Perplexities"


def build_palette(series_list: list[LossSeries]) -> dict[str, str]:
    colors = [
        "#0f766e",
        "#2563eb",
        "#dc2626",
        "#ea580c",
        "#0891b2",
        "#65a30d",
        "#b45309",
        "#475569",
    ]
    return {
        series.label: colors[index % len(colors)]
        for index, series in enumerate(series_list)
    }


def set_plot_theme(font_scale: float = 1.0) -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=font_scale,
        rc={
            "axes.facecolor": "#f8fafc",
            "figure.facecolor": "#f4f7fb",
            "grid.color": "#d7e0ea",
            "axes.edgecolor": "#c3cfdb",
            "axes.linewidth": 1.1,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#d7e0ea",
        },
    )


def plot_group_on_axis(
    ax: plt.Axes,
    series_list: list[LossSeries],
    *,
    metric: str,
    ylabel: str,
    title: str,
    palette: dict[str, str],
    tick_positions: list[int],
    tick_labels: list[str],
    block_size: int,
    show_xlabel: bool,
    legend_mode: str | bool,
) -> None:
    records: list[dict[str, object]] = []
    for series in series_list:
        for block in series.blocks:
            value = metric_value(block, metric)
            if value is None:
                continue
            records.append(
                {
                    "series": series.label,
                    "block_index": block.block_index,
                    "value": value,
                }
            )
    plot_df = pd.DataFrame.from_records(records)
    if plot_df.empty:
        raise ValueError(f"No values were available to plot for {title!r}.")

    sns.lineplot(
        data=plot_df,
        x="block_index",
        y="value",
        hue="series",
        style="series",
        markers=True,
        dashes=False,
        linewidth=2.4,
        palette=palette,
        legend=legend_mode,
        ax=ax,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(
        f"Position ({block_size} consecutive tokens per block)" if show_xlabel else ""
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.tick_params(axis="x", pad=5)
    ax.grid(True, axis="y", alpha=0.42)
    ax.grid(True, axis="x", alpha=0.16)
    ax.margins(x=0.015)
    ax.text(
        0.985,
        0.97,
        title,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#1f2937",
    )
    sns.despine(ax=ax, left=False, bottom=False)


def plot_stacked_series_groups(
    series_groups: list[list[LossSeries]],
    panel_titles: list[str],
    *,
    block_size: int,
    output_path: Path,
    dpi: int,
    metric: str = "loss",
) -> None:
    if len(series_groups) != len(panel_titles):
        raise ValueError("series_groups and panel_titles must have the same length.")
    if not series_groups:
        raise ValueError("Need at least one series group to plot.")

    reference_blocks = get_reference_blocks(
        [series for group in series_groups for series in group]
    )
    tick_positions, tick_labels = endpoint_tick_values(reference_blocks)
    palette = build_palette(series_groups[0])
    ylabel = "Mean loss" if metric == "loss" else "Perplexity"

    set_plot_theme(font_scale=0.82)
    figure, axes = plt.subplots(
        len(series_groups),
        1,
        figsize=(11.6, 6.9),
        sharex=True,
    )
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes)
        except TypeError:
            axes = [axes]

    ordered_labels = [series.label for series in series_groups[0]]
    legend_handles_by_label: dict[str, object] = {}
    for index, (ax, series_list, panel_title) in enumerate(
        zip(axes, series_groups, panel_titles)
    ):
        plot_group_on_axis(
            ax,
            series_list,
            metric=metric,
            ylabel=ylabel,
            title=panel_title,
            palette=palette,
            tick_positions=tick_positions,
            tick_labels=tick_labels,
            block_size=block_size,
            show_xlabel=index == len(series_groups) - 1,
            legend_mode="full" if index == 0 else False,
        )
        if index == 0:
            raw_handles, raw_labels = ax.get_legend_handles_labels()
            legend_handles_by_label = {
                label: handle
                for handle, label in zip(raw_handles, raw_labels)
                if label in ordered_labels
            }
            if ax.legend_ is not None:
                ax.legend_.remove()

    legend_handles = [
        legend_handles_by_label[label]
        for label in ordered_labels
        if label in legend_handles_by_label
    ]
    figure.legend(
        legend_handles,
        [label for label in ordered_labels if label in legend_handles_by_label],
        loc="center left",
        bbox_to_anchor=(0.79, 0.5),
        ncol=1,
        frameon=True,
        columnspacing=1.1,
        handletextpad=0.7,
        borderaxespad=0.0,
        fontsize=11,
    )
    figure.subplots_adjust(top=0.97, bottom=0.11, left=0.09, right=0.76, hspace=0.12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(figure)


def plot_series(
    series_list: list[LossSeries],
    block_size: int,
    output_path: Path,
    title: str | None,
    baseline: LossSeries | None,
    dpi: int,
    metric: str,
) -> None:
    plot_df, panel_order, panel_to_ylabel, relative_panels = build_plot_dataframe(
        series_list=series_list,
        baseline=baseline,
        metric=metric,
    )
    if plot_df.empty:
        raise ValueError("No block values were available to plot.")

    set_plot_theme()

    row_height = 3.2 if len(panel_order) > 1 else 4.3
    aspect = 2.55 if len(panel_order) > 1 else 2.35
    grid = sns.relplot(
        data=plot_df,
        kind="line",
        x="block_index",
        y="value",
        hue="series",
        style="series",
        row="panel",
        row_order=panel_order,
        estimator=None,
        errorbar=None,
        markers=True,
        dashes=False,
        linewidth=2.4,
        palette=build_palette(series_list),
        facet_kws={"sharex": True, "sharey": False, "legend_out": False},
        height=row_height,
        aspect=aspect,
    )
    grid.set_titles("")

    figure = grid.figure
    figure.subplots_adjust(
        top=0.88 if title is None else 0.84,
        bottom=0.28 if len(panel_order) == 1 else 0.18,
        hspace=0.40,
    )
    figure.suptitle(
        title or default_title(metric),
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )

    baseline_text = baseline.label if baseline is not None else "none"
    figure.text(
        0.055,
        0.948 if title is None else 0.924,
        f"Block size: {block_size} tokens | Baseline: {baseline_text}",
        ha="left",
        va="top",
        fontsize=11,
        color="#475569",
    )

    reference_blocks = get_reference_blocks(series_list)
    tick_positions, tick_labels = block_tick_values(reference_blocks)

    show_panel_titles = len(panel_order) > 1
    for ax, panel_name in zip(grid.axes.flat, panel_order):
        if show_panel_titles:
            ax.set_title(panel_name, loc="left", pad=12, fontsize=14, fontweight="bold")
        else:
            ax.set_title("")
        ax.set_ylabel(panel_to_ylabel[panel_name])
        ax.set_xlabel(f"Position ({block_size} consecutive tokens per block)")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=38, ha="right")
        ax.grid(True, axis="y", alpha=0.42)
        ax.grid(True, axis="x", alpha=0.16)
        ax.margins(x=0.015)
        if panel_name in relative_panels:
            ax.axhline(
                0.0,
                color="#475569",
                linewidth=1.3,
                linestyle="--",
                alpha=0.85,
                zorder=1,
            )
        sns.despine(ax=ax, left=False, bottom=False)

    if grid.legend is not None:
        grid.legend.set_title(None)
        sns.move_legend(
            grid,
            "lower center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=max(1, min(len(series_list), 4)),
            title=None,
            frameon=True,
            borderaxespad=0.0,
            columnspacing=1.4,
            handletextpad=0.7,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")

    series_list = build_series(args.json_paths, args.labels, args.block_size)
    baseline = resolve_baseline(args.baseline, series_list) if args.baseline else None
    output_path = make_output_path(args.output, baseline)
    plot_series(
        series_list=series_list,
        block_size=args.block_size,
        output_path=output_path,
        title=args.title,
        baseline=baseline,
        dpi=args.dpi,
        metric=args.metric,
    )

    print(f"saved_plot : {output_path}")
    print(f"block_size : {args.block_size}")
    print(f"metric     : {args.metric}")
    for series in series_list:
        print(f"series     : {series.label} ({series.path})")
    if baseline is not None:
        print(f"baseline   : {baseline.label}")


if __name__ == "__main__":
    main()
