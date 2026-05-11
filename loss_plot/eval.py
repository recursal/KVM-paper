from utils.env import setup_env

if __name__ == "__main__":
    setup_env()

import argparse
import importlib
import json
import math
import os
import sys
import typing
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from pydantic import BaseModel, ConfigDict, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_cmdline_configs
from .common import (
    get_tokenizer_adapter,
    preprocess_dataset,
    resolve_dataset,
)


HF_WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")
RAW_CHECKPOINT_PATTERNS = ("state_step*.pt", "model_step*.pt")
TOKENIZER_SENTINELS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
LEGACY_MODEL_CLASS_PATH_ALIASES = {
    "model.rwkv7_backbone.GPT": "model.rwkv7_backbone.RWKV7BackboneForCausalLM",
}


# @dataclass(kw_only=True)
class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 1
    device: str | None = None
    dtype: str = "bfloat16"
    compile: int = 0
    output_path: str | None = None
    context_length: int = 32768
    num_workers: int = 4
    loss_chunk_length: int = 256
    nickname: str = "textbook_chapters"
    dataset: str | None = None
    dataset_name: str | None = None
    split: str = "test"
    text_column: str = "text"
    tokenizer: str = "auto"
    tokenizer_name: str | None = None
    tokenizer_trust_remote_code: int = 0
    skip_tokens: int = 0
    min_tokens: int = 0
    preprocess_batch_size: int = 64
    max_doc_count: int | None = 1000
    seed: int | None = 0


# @dataclass(kw_only=True)
class CLI_Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: typing.Any = None
    model: typing.Any = None
    eval: EvalConfig = field(default_factory=EvalConfig)


@dataclass(kw_only=True)
class RuntimeContext:
    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    master_process: bool
    distributed: bool


def print_cli_help():
    # from config import format_config_override_help

    script_name = Path(sys.argv[0]).name
    lines = [
        f"usage: {script_name} [-h] --logs_path LOGS_PATH [-c CONFIG] [--section.option VALUE ...]",
        "",
        "Evaluate a top-level Hugging Face model export on a dataset.",
        "",
        "options:",
        "  -h, --help            show this help message and exit",
        "  --logs_path LOGS_PATH HF model dir, HF weights file, or run log path",
        "  -c CONFIG             YAML or JSON config file. Repeatable; later files override earlier ones.",
        "",
        # format_config_override_help(CLI_Config),
    ]
    print("\n".join(lines))


def class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def normalize_model_class_path(model_class_path: str) -> str:
    return LEGACY_MODEL_CLASS_PATH_ALIASES.get(model_class_path, model_class_path)


def is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    return any((path / weight_name).is_file() for weight_name in HF_WEIGHT_FILENAMES)


def has_raw_checkpoints(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        next(path.glob(pattern), None) is not None
        for pattern in RAW_CHECKPOINT_PATTERNS
    )


def has_tokenizer_assets(path: Path) -> bool:
    return any((path / name).is_file() for name in TOKENIZER_SENTINELS)


def resolve_hf_model_dir(logs_path: str) -> Path:
    path = Path(logs_path).expanduser().resolve()
    if path.is_file() and path.suffix == ".txt":
        candidate = path.with_suffix("")
        if candidate.is_dir():
            path = candidate
        else:
            raise FileNotFoundError(
                f"Could not derive a run directory from log file {path}."
            )

    if path.is_dir():
        if is_hf_model_dir(path):
            return path
        if has_raw_checkpoints(path):
            raise ValueError(
                "Raw checkpoints are not supported by eval.py anymore. "
                "Point --logs_path at the saved HF model directory containing config.json "
                "and model.safetensors or pytorch_model.bin."
            )
        raise FileNotFoundError(
            f"Could not find a Hugging Face model export under {path}. "
            "Expected config.json and model.safetensors or pytorch_model.bin."
        )

    if not path.is_file():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.suffix == ".pt":
        raise ValueError(
            "Raw checkpoints are not supported by eval.py anymore. "
            "Point --logs_path at the saved HF model directory instead."
        )

    if path.name in HF_WEIGHT_FILENAMES or path.suffix in {".bin", ".safetensors"}:
        candidate = path.parent
        if is_hf_model_dir(candidate):
            return candidate
        raise FileNotFoundError(
            f"Found HF weight file {path}, but its parent directory does not look like a complete HF export."
        )

    if path.name == "config.json" and is_hf_model_dir(path.parent):
        return path.parent

    raise FileNotFoundError(
        f"Unsupported input path {path}. "
        "Use a HF model directory, a HF weights file, or a run log path whose sibling directory is a HF export."
    )


def resolve_output_path(model_dir: Path, eval_config: EvalConfig) -> Path:
    if eval_config.output_path:
        return Path(eval_config.output_path).expanduser().resolve()
    return (
        model_dir
        / f"{eval_config.nickname}_eval_{int(eval_config.context_length)}.json"
    )


def parse_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def init_runtime(eval_config: EvalConfig) -> RuntimeContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))

    if distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if eval_config.device is not None and str(
            torch.device(eval_config.device)
        ) != str(device):
            if rank == 0:
                print(
                    f"Overriding eval.device={eval_config.device} with local DDP device {device}."
                )
    else:
        device = parse_device(eval_config.device)

    return RuntimeContext(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
        master_process=rank == 0,
        distributed=distributed,
    )


def parse_dtype(dtype_name: str | None) -> torch.dtype:
    if dtype_name is None:
        return torch.float32
    normalized = dtype_name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}.")
    return mapping[normalized]


def resolve_runtime_dtype(eval_config: EvalConfig, device: torch.device) -> torch.dtype:
    dtype = parse_dtype(eval_config.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        print(
            f"Using float32 instead of {eval_config.dtype} because CPU eval does not use low-precision reliably."
        )
        return torch.float32
    return dtype


def maybe_apply_compile(model: torch.nn.Module, eval_config: EvalConfig) -> None:
    if not bool(eval_config.compile):
        return
    from utils.defer import apply_deferred

    apply_deferred(model)


def build_runtime_model_config(model_dir: Path):
    from model.rwkv7_backbone import MixerConfig

    model_config = MixerConfig.from_pretrained(str(model_dir), local_files_only=True)
    model_config.model_class_path = normalize_model_class_path(
        model_config.model_class_path
    )
    return model_config


def build_model(
    model_dir: Path, model_config, eval_config: EvalConfig, device: torch.device
):
    runtime_dtype = resolve_runtime_dtype(eval_config, device)
    model_class = class_from_path(model_config.model_class_path)
    model = model_class.from_pretrained(
        str(model_dir),
        config=model_config,
        local_files_only=True,
        torch_dtype=runtime_dtype,
    )
    model.eval()
    model.to(device=device, dtype=runtime_dtype)
    maybe_apply_compile(model, eval_config)
    return model


def resolve_tokenizer_settings(
    eval_config: EvalConfig, model_dir: Path
) -> tuple[str, str | None]:
    tokenizer = str(eval_config.tokenizer)
    tokenizer_name = eval_config.tokenizer_name
    if tokenizer_name is not None:
        return tokenizer, tokenizer_name
    if tokenizer.lower() in {"auto", "hf"} and has_tokenizer_assets(model_dir):
        return "hf", str(model_dir)
    return tokenizer, tokenizer_name


def maybe_apply_logit_softcap(logits: torch.Tensor, softcap: float) -> torch.Tensor:
    if softcap <= 0:
        return logits
    return softcap * torch.tanh(logits / softcap)


def _token_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)
    return token_loss.masked_fill(labels == -100, 0.0).float()


def build_loss_series(
    loss_sum: torch.Tensor, token_count: torch.Tensor
) -> dict[str, list[float | int | None] | bool | int]:
    avg_nll: list[float | None] = []
    perplexity: list[float | None] = []
    counts = [int(count) for count in token_count.tolist()]
    for summed_loss, count in zip(loss_sum.tolist(), counts):
        if count <= 0:
            avg_nll.append(None)
            perplexity.append(None)
            continue
        mean_loss = float(summed_loss) / count
        avg_nll.append(mean_loss)
        perplexity.append(math.exp(mean_loss))
    return {
        "position_is_zero_indexed": True,
        "tokens": counts,
        "avg_nll": avg_nll,
        "perplexity": perplexity,
    }


def evaluate_batch_loss_by_position(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_chunk_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss_chunk_length <= 0:
        raise ValueError("eval.loss_chunk_length must be positive.")
    if not hasattr(model, "transformer") or not isinstance(
        getattr(model, "lm_head", None), torch.nn.Module
    ):
        raise TypeError(
            "eval.py expects a top-level HF export backed by model.rwkv7_backbone.RWKV7BackboneForCausalLM."
        )

    with torch.inference_mode():
        hidden_states = model.transformer(
            input_ids=input_ids, use_cache=False
        ).last_hidden_state

    seq_len = int(labels.shape[1])
    valid_mask = labels != -100
    loss_sum_by_position = torch.zeros(seq_len, dtype=torch.float64)
    token_count_by_position = valid_mask.sum(dim=0).to(device="cpu", dtype=torch.int64)
    softcap = float(getattr(model.config, "logit_softcap", 0.0) or 0.0)

    with torch.inference_mode():
        for start in range(0, seq_len, loss_chunk_length):
            end = min(start + loss_chunk_length, seq_len)
            logits = model.lm_head(hidden_states[:, start:end, :])
            logits = maybe_apply_logit_softcap(logits, softcap)
            token_loss = _token_loss_from_logits(logits, labels[:, start:end])
            loss_sum_by_position[start:end] += (
                token_loss.mul(valid_mask[:, start:end])
                .sum(dim=0)
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
            )

    return loss_sum_by_position, token_count_by_position


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print_cli_help()
    raise SystemExit(0)


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Evaluate a dataset from a top-level HF model export."
    )
    parser.add_argument(
        "--logs_path",
        required=True,
        help="HF model dir, HF weights file, or run log path.",
    )
    args, remaining = parser.parse_known_args()
    cli_config = load_cmdline_configs(remaining)
    # cli_config, errors = parse_cmdline_configs(remaining, CLI_Config)
    # if errors != "":
    #     print(errors)
    #     raise SystemExit(-1)
    try:
        cli_config = CLI_Config(**cli_config)
    except ValidationError as e:
        print(e.errors())
        exit(-1)

    return args, cli_config


def build_local_eval_subset(dataset, runtime: RuntimeContext):
    if not runtime.distributed:
        return dataset
    local_indices = list(range(runtime.rank, len(dataset), runtime.world_size))
    return Subset(dataset, local_indices)


def reduce_eval_statistics(
    runtime: RuntimeContext,
    *,
    per_position_loss_sum: torch.Tensor,
    per_position_token_count: torch.Tensor,
    per_chunk_position_loss_sum: torch.Tensor,
    per_chunk_position_token_count: torch.Tensor,
    total_loss_sum: float,
    total_tokens: int,
):
    if not runtime.distributed:
        return (
            per_position_loss_sum,
            per_position_token_count,
            per_chunk_position_loss_sum,
            per_chunk_position_token_count,
            total_loss_sum,
            total_tokens,
        )

    device = runtime.device
    reduced_per_position_loss_sum = per_position_loss_sum.to(device=device)
    reduced_per_position_token_count = per_position_token_count.to(device=device)
    reduced_per_chunk_position_loss_sum = per_chunk_position_loss_sum.to(device=device)
    reduced_per_chunk_position_token_count = per_chunk_position_token_count.to(
        device=device
    )
    reduced_total_loss_sum = torch.tensor(
        total_loss_sum, device=device, dtype=torch.float64
    )
    reduced_total_tokens = torch.tensor(total_tokens, device=device, dtype=torch.int64)

    dist.all_reduce(reduced_per_position_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(reduced_per_position_token_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(reduced_per_chunk_position_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(reduced_per_chunk_position_token_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(reduced_total_loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(reduced_total_tokens, op=dist.ReduceOp.SUM)

    return (
        reduced_per_position_loss_sum.to(device="cpu", dtype=torch.float64),
        reduced_per_position_token_count.to(device="cpu", dtype=torch.int64),
        reduced_per_chunk_position_loss_sum.to(device="cpu", dtype=torch.float64),
        reduced_per_chunk_position_token_count.to(device="cpu", dtype=torch.int64),
        float(reduced_total_loss_sum.item()),
        int(reduced_total_tokens.item()),
    )


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args, cli_config = parse_cli()
    eval_config = cli_config.eval
    runtime = init_runtime(eval_config)

    try:
        model_dir = resolve_hf_model_dir(args.logs_path)
        model_config = build_runtime_model_config(model_dir)

        # if model_config.sequence_length not in {None, -1, EVAL_CONTEXT_LENGTH} and runtime.master_process:
        #     print(
        #         f"Overriding model.sequence_length={model_config.sequence_length} "
        #         f"with dataset eval context {EVAL_CONTEXT_LENGTH}."
        #     )
        # model_config.sequence_length = EVAL_CONTEXT_LENGTH

        tokenizer_mode, tokenizer_name = resolve_tokenizer_settings(
            eval_config, model_dir
        )
        dataset_id, raw_dataset = resolve_dataset(
            dataset=eval_config.dataset,
            dataset_name=eval_config.dataset_name,
            split=eval_config.split,
        )
        if eval_config.batch_size <= 0:
            raise ValueError("eval.batch_size must be positive.")
        if eval_config.num_workers < 0:
            raise ValueError("eval.num_workers must be non-negative.")

        dataset = preprocess_dataset(
            raw_dataset,
            text_column=eval_config.text_column,
            skip_tokens=eval_config.skip_tokens,
            min_tokens=eval_config.min_tokens,
            context_length=eval_config.context_length,
            tokenizer=tokenizer_mode,
            tokenizer_name=tokenizer_name,
            tokenizer_trust_remote_code=bool(
                eval_config.tokenizer_trust_remote_code
            ),
            model_vocab_size=model_config.vocab_size,
            preprocess_batch_size=eval_config.preprocess_batch_size,
            desc=f"Preprocessing {dataset_id}:{eval_config.split} for eval",
            max_doc_count=eval_config.max_doc_count,
            seed=eval_config.seed,
        ).with_format("torch")

        if len(dataset) == 0:
            raise RuntimeError(
                "No evaluation examples remain after skip/filter/truncate preprocessing."
            )

        local_dataset = build_local_eval_subset(dataset, runtime)
        model = build_model(model_dir, model_config, eval_config, runtime.device)
        tokenizer_adapter = get_tokenizer_adapter(
            tokenizer=tokenizer_mode,
            tokenizer_name=tokenizer_name,
            tokenizer_trust_remote_code=bool(
                eval_config.tokenizer_trust_remote_code
            ),
            model_vocab_size=model_config.vocab_size,
        )

        loader = DataLoader(
            local_dataset,
            batch_size=eval_config.batch_size,
            shuffle=False,
            num_workers=eval_config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        report_chunk_length = 256
        report_position_count = min(
            report_chunk_length, int(eval_config.context_length)
        )
        per_position_loss_sum = torch.zeros(
            eval_config.context_length, dtype=torch.float64
        )
        per_position_token_count = torch.zeros(
            eval_config.context_length, dtype=torch.int64
        )
        per_chunk_position_loss_sum = torch.zeros(
            report_position_count, dtype=torch.float64
        )
        per_chunk_position_token_count = torch.zeros(
            report_position_count, dtype=torch.int64
        )
        report_chunk_positions = (
            torch.arange(eval_config.context_length, dtype=torch.long)
            % report_chunk_length
        )

        total_loss_sum = 0.0
        total_tokens = 0
        for batch in loader:
            input_ids = (
                batch["input_ids"].to(runtime.device, non_blocking=True).contiguous()
            )
            labels = batch["labels"].to(runtime.device, non_blocking=True).contiguous()
            batch_loss_by_position, batch_token_count_by_position = (
                evaluate_batch_loss_by_position(
                    model,
                    input_ids,
                    labels,
                    loss_chunk_length=eval_config.loss_chunk_length,
                )
            )
            if batch_loss_by_position.shape[0] != eval_config.context_length:
                raise ValueError(
                    f"Expected eval sequence length {eval_config.context_length}, got {batch_loss_by_position.shape[0]}."
                )

            per_position_loss_sum += batch_loss_by_position
            per_position_token_count += batch_token_count_by_position
            per_chunk_position_loss_sum.scatter_add_(
                0, report_chunk_positions, batch_loss_by_position
            )
            per_chunk_position_token_count.scatter_add_(
                0, report_chunk_positions, batch_token_count_by_position
            )

            total_loss_sum += float(batch_loss_by_position.sum().item())
            total_tokens += int(batch_token_count_by_position.sum().item())

        (
            per_position_loss_sum,
            per_position_token_count,
            per_chunk_position_loss_sum,
            per_chunk_position_token_count,
            total_loss_sum,
            total_tokens,
        ) = reduce_eval_statistics(
            runtime,
            per_position_loss_sum=per_position_loss_sum,
            per_position_token_count=per_position_token_count,
            per_chunk_position_loss_sum=per_chunk_position_loss_sum,
            per_chunk_position_token_count=per_chunk_position_token_count,
            total_loss_sum=total_loss_sum,
            total_tokens=total_tokens,
        )

        if total_tokens == 0:
            raise RuntimeError("No tokens were evaluated.")

        if runtime.master_process:
            avg_nll = total_loss_sum / total_tokens
            result = {
                "checkpoint_path": str(model_dir),
                "checkpoint_step": -1,
                "dataset": dataset_id,
                "split": eval_config.split,
                "tokenizer": tokenizer_adapter.name,
                "context_length": eval_config.context_length,
                "examples": len(dataset),
                "tokens": total_tokens,
                "avg_nll": avg_nll,
                "perplexity": math.exp(avg_nll),
                "per_token_position": build_loss_series(
                    per_position_loss_sum, per_position_token_count
                ),
                "per_token_chunk_position": {
                    "chunk_length": report_chunk_length,
                    **build_loss_series(
                        per_chunk_position_loss_sum, per_chunk_position_token_count
                    ),
                },
            }

            output_path = resolve_output_path(model_dir, eval_config)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wt", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                f.write("\n")

            print(f"model_dir   : {model_dir}")
            print(f"dataset     : {dataset_id} [{eval_config.split}]")
            print(f"tokenizer   : {tokenizer_adapter.name}")
            print(f"context     : {eval_config.context_length}")
            print(f"examples    : {len(dataset)}")
            print(f"tokens      : {total_tokens}")
            print(f"world_size  : {runtime.world_size}")
            print(f"ppl         : {result['perplexity']:.4f}")
            print(
                f"per_pos     : {eval_config.context_length} abs, "
                f"{report_position_count} chunk-pos (chunk_len={report_chunk_length})"
            )
            print(f"saved_json  : {output_path}")
    finally:
        if runtime.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
