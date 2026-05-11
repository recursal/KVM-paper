#!/usr/bin/env python3
"""Extract compact tables from lm-eval result JSONs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


NIAH_TASKS = ("niah_single_1", "niah_single_2", "niah_single_3")
NIAH_LENGTHS = (4096, 8192, 16384, 32768)

SHORT_METRICS = (
    ("arc_challenge", "acc_norm,none", "arc_challenge_acc_norm"),
    ("arc_easy", "acc,none", "arc_easy_acc"),
    ("hellaswag", "acc_norm,none", "hellaswag_acc_norm"),
    ("lambada_openai", "acc,none", "lambada_acc"),
    ("lambada_openai", "perplexity,none", "lambada_perplexity"),
    ("piqa", "acc,none", "piqa_acc"),
    ("winogrande", "acc,none", "winogrande_acc"),
)

LONG_FIELDNAMES = (
    ["model"]
    + [f"{task}_{length}" for task in NIAH_TASKS for length in NIAH_LENGTHS]
    + ["longbench_fewshot"]
)

SHORT_FIELDNAMES = [
    "model",
    "lambada_acc",
    "lambada_perplexity",
    "arc_challenge_acc_norm",
    "arc_easy_acc",
    "hellaswag_acc_norm",
    "piqa_acc",
    "winogrande_acc",
]

ALL_FIELDNAMES = LONG_FIELDNAMES + [column_name for _, _, column_name in SHORT_METRICS]

MODEL_LAYOUT: list[tuple[str, str] | None] = [
    # ("bswa_120M", "120M BSWA"),
    # ("rwkv7_ffn4_120M", "120M RWKV-7"),
    # ("gptalpha_120M", "120M GPTAlpha-7"),
    # ("kvm256_120M", "120M KVM 256"),
    # ("kvmsqrt16_120M", "120M KVM sqrt"),
    # None,
    # ("ovqsat_swa256_120M", "120M OVQ/SWA hybrid"),
    # # ("gptalpha_swa256_120M", "120M GPTAlpha-7 NoPE/SWA hybrid"),
    # ("gptalphaprope_swa256_120M", "120M GPTAlpha-7 PRoPE/SWA hybrid"),
    # ("kvmsat_swa256_120M", "120M KVM/SWA hybrid"),
    # None,
    # ("bswa_350M", "350M BSWA"),
    # ("rwkv7_ffn4_350M", "350M RWKV-7"),
    # ("gptalpha_350M", "350M GPTAlpha-7"),
    # ("kvm256_350M", "350M KVM 256"),
    # ("kvmsqrt16_350M", "350M KVM sqrt"),
    # None,
    # ("ovqsat_swa256_350M", "350M OVQ/SWA hybrid"),
    # ("gptalphaprope_swa256_350M", "350M GPTAlpha-7 PRoPE/SWA hybrid"),
    # ("kvmsat_swa256_350M", "350M KVM/SWA hybrid"),
    # None,
    # ("gptalphaprope_swa256_350M", "350M GPTAlpha-7 PRoPE/SWA hybrid 7Gtok"),
    # ("gptalpha_swa256_350M", "350M GPTAlpha-7 NoPE/SWA hybrid 7Gtok"),
    # ("gptalpha_swa256_3Gtok_350M", "350M GPTAlpha-7 NoPE/SWA hybrid 3GTok"),
    # None,
    ("kvm_no_sink_120M", "120M KVM256 no sink"),
    ("kvm_no_head_temps_120M", "120M KVM256 no head temps"),
    ("kvm_no_vlens_120M", "120M KVM256 no v-len normalization"),
    ("kvm_no_merge_gate_120M", "120M KVM256 no merge gate"),
    #("kvm_sqrt16_no_merge_gate_120M", "120M KVMsqrt16 no merge gate"),
    #("kvm_swa_saturate1024_no_merge_gate_120M", "120M KVMsat no merge gate"),
    ("kvm_no_merge_gate_350M", "350M KVM256 no merge gate"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected eval metrics into flat files or LaTeX tables."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results",
        help="Directory containing one subdirectory per model (default: results).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write the output to this path instead of stdout.",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "csv", "latex"),
        default="tsv",
        help="Output format (default: tsv).",
    )
    parser.add_argument(
        "--table",
        choices=("all", "long", "short"),
        default="all",
        help="Which table layout to emit (default: all).",
    )
    parser.add_argument(
        "--missing",
        default="",
        help="Placeholder for missing values (default: empty string).",
    )
    return parser.parse_args()


def latest_json(eval_dir: Path) -> Path | None:
    json_files = sorted(eval_dir.rglob("*.json"))
    return json_files[-1] if json_files else None


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open() as handle:
        return json.load(handle)


def extract_value(
    data: dict[str, Any],
    task_name: str,
    metric_name: str,
) -> float | int | None:
    results = data.get("results", {})
    task = results.get(task_name, {})
    return task.get(metric_name)


def build_empty_row(model_name: str) -> dict[str, Any]:
    row = {field: None for field in ALL_FIELDNAMES}
    row["model"] = model_name
    return row


def build_row(model_name: str, model_dir: Path | None) -> dict[str, Any]:
    row = build_empty_row(model_name)
    if model_dir is None:
        return row

    ruler_data = load_json(latest_json(model_dir / "evals-ruler"))
    niah_data = load_json(latest_json(model_dir / "evals-niah"))
    long_niah_data = load_json(latest_json(model_dir / "evals-long-niah"))
    longbench_data = load_json(latest_json(model_dir / "evals-longbench-fewshot"))
    short_data = load_json(latest_json(model_dir / "evals-short"))

    row["ruler_4096"] = extract_value(
        ruler_data,
        "ruler",
        "4096,none",
    )

    for task_name in NIAH_TASKS:
        for length in (4096, 8192, 16384):
            metric_name = f"{length},none"
            row[f"{task_name}_{length}"] = extract_value(
                niah_data,
                task_name,
                metric_name,
            )

        row[f"{task_name}_{32768}"] = extract_value(
            long_niah_data,
            task_name,
            "32768,none",
        )

    row["longbench_fewshot"] = extract_value(
        longbench_data,
        "longbench_fewshot",
        "score,none",
    )

    for task_name, metric_name, column_name in SHORT_METRICS:
        row[column_name] = extract_value(short_data, task_name, metric_name)

    return row


def fieldnames_for(table_name: str) -> list[str]:
    if table_name == "long":
        return LONG_FIELDNAMES
    if table_name == "short":
        return SHORT_FIELDNAMES
    return ALL_FIELDNAMES


def format_decimal(value: Any, missing: str) -> str:
    if value is None:
        return missing
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.3f}"
    return str(value)


def format_percent(value: Any, missing: str) -> str:
    if value is None:
        return missing
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value * 100:.1f}"
    return str(value)


def format_number(value: Any, missing: str) -> str:
    if value is None:
        return missing
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.1f}"
    return str(value)


def formatted_flat_rows(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    missing: str,
) -> list[dict[str, str]]:
    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        formatted = {}
        for field in fieldnames:
            if field == "model":
                formatted[field] = str(row[field])
            else:
                formatted[field] = format_decimal(row.get(field), missing)
        formatted_rows.append(formatted)
    return formatted_rows


def writer_for(format_name: str, handle: Any, fieldnames: list[str]) -> csv.DictWriter:
    delimiter = "\t" if format_name == "tsv" else ","
    return csv.DictWriter(
        handle,
        fieldnames=fieldnames,
        delimiter=delimiter,
        extrasaction="ignore",
    )


def latex_long_table(rows_by_model: dict[str, dict[str, Any]], missing: str) -> str:
    lines = [
        r"\begin{table}[htb]",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\linewidth}",
        r"        \begin{tabular}{l*{14}{r}}",
        r"            \toprule",
        r"            & \multicolumn{4}{c}{NIAH-S1$\uparrow$}",
        r"            & \multicolumn{4}{c}{NIAH-S2$\uparrow$}",
        r"            & \multicolumn{4}{c}{NIAH-S3$\uparrow$} & LB$\uparrow$ & RULER$\uparrow$ \\",
        r"            \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}",
        r"            Architecture & 4K & 8K & 16K & 32K & 4K & 8K & 16K & 32K & 4K & 8K & 16K & 32K & avg. & avg. \\",
        r"            \midrule",
    ]

    for item in MODEL_LAYOUT:
        if item is None:
            lines.append(r"            \midrule")
            continue

        model_key, label = item
        row = rows_by_model.get(model_key, build_empty_row(model_key))
        values = [
            format_percent(row.get(f"{task}_{length}"), missing)
            for task in NIAH_TASKS
            for length in NIAH_LENGTHS
        ]
        values.append(format_percent(row.get("longbench_fewshot"), missing))
        values.append(format_percent(row.get("ruler_4096"), missing))
        lines.append(f"            {' & '.join([label] + values)} \\\\")

    lines.extend(
        [
            r"            \bottomrule",
            r"        \end{tabular}",
            r"    \end{adjustbox}",
            r"    \caption{NIAH, RULER-4096 and average of LongBench (\"LB\") few-shot evaluations}",
            r"    \label{tab:evals-long}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def latex_short_table(rows_by_model: dict[str, dict[str, Any]], missing: str) -> str:
    lines = [
        r"\begin{table}[htb]",
        r"    \centering",
        r"    \begin{adjustbox}{max width=\linewidth}",
        r"        \begin{tabular}{lrrrrrrrr}",
        r"            \toprule",
        r"            Architecture & val loss$\downarrow$ & lmbda ppl$\downarrow$ & lmbda$\uparrow$ & arc\_c$\uparrow$ & arc\_e$\uparrow$ & hella$\uparrow$ & piqa$\uparrow$ & winog$\uparrow$ \\",
        r"            \midrule",
    ]

    for item in MODEL_LAYOUT:
        if item is None:
            lines.append(r"            \midrule")
            continue

        model_key, label = item
        row = rows_by_model.get(model_key, build_empty_row(model_key))
        values = [
            "",
            format_number(row.get("lambada_perplexity"), missing),
            format_percent(row.get("lambada_acc"), missing),
            format_percent(row.get("arc_challenge_acc_norm"), missing),
            format_percent(row.get("arc_easy_acc"), missing),
            format_percent(row.get("hellaswag_acc_norm"), missing),
            format_percent(row.get("piqa_acc"), missing),
            format_percent(row.get("winogrande_acc"), missing),
        ]
        lines.append(f"            {' & '.join([label] + values)} \\\\")

    lines.extend(
        [
            r"            \bottomrule",
            r"        \end{tabular}",
            r"    \end{adjustbox}",
            r"    \caption{Standard short context language modeling evaluations}",
            r"    \label{tab:evals-short}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def latex_output(
    rows_by_model: dict[str, dict[str, Any]],
    table_name: str,
    missing: str,
) -> str:
    if table_name == "long":
        return latex_long_table(rows_by_model, missing)
    if table_name == "short":
        return latex_short_table(rows_by_model, missing)
    return (
        latex_long_table(rows_by_model, missing)
        + "\n\n"
        + latex_short_table(
            rows_by_model,
            missing,
        )
    )


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.is_dir():
        print(f"results directory not found: {results_dir}", file=sys.stderr)
        return 1

    model_dirs = {path.name: path for path in results_dir.iterdir() if path.is_dir()}
    model_names = sorted(set(model_dirs) | {item[0] for item in MODEL_LAYOUT if item})
    rows = [
        build_row(model_name, model_dirs.get(model_name)) for model_name in model_names
    ]
    rows_by_model = {row["model"]: row for row in rows}

    if args.format == "latex":
        output_text = latex_output(rows_by_model, args.table, args.missing)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n")
        else:
            print(output_text)
        return 0

    fieldnames = fieldnames_for(args.table)
    formatted_rows = formatted_flat_rows(rows, fieldnames, args.missing)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = writer_for(args.format, handle, fieldnames)
            writer.writeheader()
            writer.writerows(formatted_rows)
    else:
        writer = writer_for(args.format, sys.stdout, fieldnames)
        writer.writeheader()
        writer.writerows(formatted_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
