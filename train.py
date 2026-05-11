from utils.env import setup_env

if __name__ == "__main__":
    setup_env()

from utils.logger import print0 as print

import importlib
import math
import os
import sys
import time
import uuid
from typing import Any
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from contextlib import contextmanager

from model.rwkv7_backbone import MixerConfig, MixerConfigDataclass
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from pydantic import BaseModel, ConfigDict, ValidationError

from data.dataset_details import DatasetDetails


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any] = field(default_factory=dict)


class OptimizerLabelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opt: str
    # MuP keeps learning rate * decay constant as matrix in_features change by scaling lr as 1/d_in and wd as d_in.
    # You should NOT use it on embeddings (because d_in is constant) or scalar parameters
    # Non-lm_head init variance should also be scaled as 1/d_in, but this is automatic with default or orthogonal inits in our case
    # Orthogonal init with appropriate gain for lm_head ensures appropriate scaling (spectral muP)
    mup: bool

    lr_ratio: float = 1.0
    wd_ratio: float = 1.0
    args: dict[str, Any] = field(default_factory=dict)


class Hyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trainer: str = "default"
    init_seed: int | None = 0
    tokenizer: str = "gpt2"
    train_data: DatasetDetails
    val_data: DatasetDetails = field(
        default_factory=lambda: DatasetDetails(
            data_packing="pad",
            dataset="robbiegwaldd/dclm-10B",
            device_batch_size=16,
            min_document_chars=4096,
            sequence_length=4096,
            shuffle_seed=0,
        )
    )

    min_document_tokens: int | None = (
        None  # if not None, documents with fewer tokens than this are filtered out from the dataset
    )
    preprocess_batch_size: int = 8
    # optimization hyperparams
    batch_size: int = -1  # batch size, in sequences, across all devices
    num_iterations: int = 3000  # number of iterations to run
    warmup_iters: int = 0
    warmdown_iters: int = 900  # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    warmdown_type: str = "linear"  # ['linear', 'cos', 'sqrt']
    warmdown_min_ratio: float = 0.0
    optimizer_defaults: dict[
        str, OptimizerConfig
    ]  # but our config parser didn't like this type declaration
    # [str, OptimizerLabelConfig] but our config parser didn't like this type declaration
    optimization_labels: dict[str, OptimizerLabelConfig] = field(
        default_factory=lambda: {
            "wte_embed": OptimizerLabelConfig(opt="adam", mup=False),
            "matrix_params": OptimizerLabelConfig(opt="adam", mup=True),
            "lm_head": OptimizerLabelConfig(opt="adam", mup=True),
            "scalars": OptimizerLabelConfig(opt="adam", mup=False, wd_ratio=0),
        }
    )
    # evaluation and logging hyperparams
    val_loss_every: int = (
        125  # every how many steps to evaluate val loss? 0 for only at the end
    )
    val_tokens: int = 10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every: int = (
        0  # every how many steps to save the checkpoint? 0 for only at the end
    )
    grad_cp: int = 0
    compile: int = 1
    debug_cuda_memory: int = 0
    debug_cuda_memory_after_step: int = -1
    wandb: str = "speedtrain"
    wandb_name_suffix: str = ""
    strategy: str = "ddp"


class CLI_Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: Hyperparameters = field(default_factory=Hyperparameters)
    model: MixerConfigDataclass = field(default_factory=MixerConfigDataclass)


def print_cli_help():
    script_name = os.path.basename(sys.argv[0])
    lines = [
        f"usage: {script_name} [-h] [-c CONFIG] [-n SUFFIX] [--section.option VALUE ...]",
        "",
        "Train with the main DDP entrypoint. Config files are optional and",
        "can be overridden with `--train.*` / `--model.*` command-line pairs.",
        "",
        "options:",
        "  -h, --help            show this help message and exit",
        "  -c CONFIG             YAML or JSON config file. Repeatable; later files override earlier ones.",
        "  -n SUFFIX             append SUFFIX to the wandb run name after a space",
        "",
        # format_config_override_help(CLI_Config),
    ]
    print("\n".join(lines))


def extract_wandb_name_suffix(argv: list[str]) -> tuple[list[str], str | None]:
    parsed_argv = []
    suffix = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-n":
            if i + 1 >= len(argv):
                from config import CLIError

                raise CLIError("-n requires a suffix value")
            suffix = argv[i + 1]
            i += 2
        elif arg.startswith("-n="):
            suffix = arg[3:]
            i += 1
        else:
            parsed_argv.append(arg)
            i += 1
    return parsed_argv, suffix


@contextmanager
def timed(text: str):
    t0 = time.time()
    print(f"{text}... ", end="")
    yield
    print(f"Done. {int(1000 * (time.time() - t0))}ms")


def class_name_and_module_from_path(class_path):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    target_class = getattr(module, class_name)
    return target_class, class_name, module


def build_optimizer(
    opt_type: str,
    opt_args: dict,
    mup_d_in_ratio: float,
    model_params: list[torch.Tensor],
    master_params: list[torch.Tensor],
):
    assert opt_type in {"muon", "adam"}, f"Unsupported optimizer type: {opt_type}"

    # adjust learning rate and any weight decay for MuP scaling
    lr = opt_args["lr"] / mup_d_in_ratio
    wd = opt_args["weight_decay"] * mup_d_in_ratio
    print(" lr: ", lr, " wd: ", wd)

    if len(master_params) == 0:
        from optimizer.empty_optimizer import EmptyOptimizer

        optimizer = EmptyOptimizer()
    elif opt_type == "muon":
        assert wd == 0, (
            "Weight decay is not currently supported for Muon optimizer. Please remove it from the config."
        )

        from optimizer.muon import Muon

        optimizer = Muon(master_params, lr=lr, momentum=0.95)
    else:
        param_group = {
            "params": master_params,
            "lr": lr,
            "weight_decay": wd,
            "lr_peak": lr,  # used in update_weight_decay_for_current_lr
            "weight_decay_peak": wd,  # used in update_weight_decay_for_current_lr
        }
        optimizer = torch.optim.AdamW(
            [param_group],
            lr=lr,
            betas=(opt_args["beta1"], opt_args["beta2"]),
            weight_decay=wd,
            fused=True,
        )

    optimizer.model_params = model_params
    optimizer.master_params = master_params

    return optimizer


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print_cli_help()
        raise SystemExit(0)

    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    old_default_device = torch.get_default_device()
    torch.set_default_device(device)
    master_process = ddp_rank == 0
    local_rank = ddp_local_rank

    with open(sys.argv[0]) as f:
        code = f.read()

    argv = sys.argv[1:]
    while len(argv) > 0 and "=" in argv[0]:
        argv = argv[1:]
    from config import CLIError, load_cmdline_configs

    try:
        argv, wandb_name_suffix = extract_wandb_name_suffix(argv)
        cli_config = load_cmdline_configs(argv)
    except CLIError as e:
        print(e)
        exit(-1)

    try:
        cli_config = CLI_Config(**cli_config)
    except ValidationError as e:
        print(e.errors())
        exit(-1)

    if wandb_name_suffix is not None:
        cli_config.train.wandb_name_suffix = wandb_name_suffix

    args = cli_config.train
    args.batch_size = ddp_world_size * args.train_data.device_batch_size
    model_config = cli_config.model
    model_config = MixerConfig(
        **model_config.model_dump()
    )  # convert from dataclass to our custom HF config class, which has standard HF superclasses
    init_seed = None if args.init_seed is None else int(args.init_seed)

    logged_cli_config = cli_config.model_dump()

    import utils.grad_cp

    utils.grad_cp.use_grad_cp = bool(args.grad_cp)

    student_model_class, _, student_model_module = class_name_and_module_from_path(
        model_config.model_class_path
    )
    # model_code = build_model_code(student_model_module, model_config)

    model_dtype = torch.bfloat16

    if init_seed is not None:
        print(
            f"Using parameter init seed {init_seed}. Dataset ordering seeds remain fixed."
        )
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed_all(init_seed)

    with timed("Instantiating model"):
        old_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(
            model_dtype
        )  # this way things default to the right dtype but we can also specifically set e.g. fp8 or float weights
        with torch.device(device):
            model = student_model_class(model_config)
        torch.set_default_dtype(old_default_dtype)

    with timed("Testing model"):
        cu_seqlens = None
        if args.train_data.data_packing == "varlen":
            best_doc_len = min(
                args.train_data.max_document_tokens,
                args.train_data.sequence_length // 2,
            )
            cu_seqlens = [0] + torch.full(
                [args.train_data.sequence_length // best_doc_len],
                fill_value=best_doc_len,
                dtype=int,
                device="cpu",
            ).cumsum(dim=0).tolist()
            fake_test_data = torch.zeros(
                [1, args.train_data.sequence_length], dtype=torch.long
            )
        else:
            fake_test_data = torch.zeros(
                [2, args.train_data.sequence_length], dtype=torch.long
            )
        test_results = model(
            input_ids=fake_test_data,
            labels=fake_test_data,
            cu_seqlens=cu_seqlens,
            return_logits=False,
        )
        loss = test_results["loss"]
        del test_results
        loss.backward()

    if args.compile:
        with timed("Reloading model class with torch.compile allowed"):
            from utils.defer import apply_deferred

            apply_deferred(model)

    with timed("Wrapping model with DDP"):
        raw_model = model
        model = DDP(
            model,
            device_ids=[ddp_local_rank],
            find_unused_parameters=args.strategy == "ddp_find_unused_parameters",
        )
        raw_model = model.module  # always contains the "raw" unwrapped model

    with timed("Initializing optimizers"):
        # gather all parameters for each optimizer based on their labels
        labeled_param_sets = {n: [] for n in args.optimization_labels.keys()}
        for n, p in raw_model.named_parameters():
            label = getattr(p, "label", None)
            assert label is not None, f"Parameter found with missing label: {n}"
            assert label in labeled_param_sets, (
                f"Label not found in optimizer args: {label}"
            )
            labeled_param_sets[label].append(p)

        optimizers = []

        default_mup_d_in_ratio = float(model_config.hidden_size) / float(
            model_config.mup_base_dim
        )
        # named_param_groups = {}
        for label, labeled_param_set in labeled_param_sets.items():
            print("label: ", label)
            if len(labeled_param_set) == 0:
                continue
            label_info = args.optimization_labels[label]
            opt_name = label_info.opt
            opt_defaults = args.optimizer_defaults[opt_name]
            updated_lr_wd = {
                "lr": opt_defaults.args["lr"] * label_info.lr_ratio,
                "weight_decay": opt_defaults.args.get("weight_decay", 0.0)
                * label_info.wd_ratio,
            }
            label_args = opt_defaults.args | updated_lr_wd | label_info.args

            ratio_grouped_label_params = {}
            for p in labeled_param_set:
                use_mup = (
                    label_info.mup and p.ndim > 1
                )  # ndim test skips MuP scaling for biases within Linear modules
                if use_mup:
                    mup_d_in_ratio = (
                        getattr(p, "mup_d_in_ratio", None) or default_mup_d_in_ratio
                    )
                else:
                    mup_d_in_ratio = 1.0
                ratio_grouped_label_params.setdefault(float(mup_d_in_ratio), []).append(
                    p
                )

            # pull each separate ratio back apart into separate parameter groups, so that each one gets its own hyperparams in its own optimizer
            for mup_d_in_ratio, group_params in ratio_grouped_label_params.items():
                group_master_params = [
                    p.detach().clone().float() for p in group_params
                ]

                optimizer = build_optimizer(
                    opt_defaults.name,
                    label_args,
                    mup_d_in_ratio,
                    group_params,
                    group_master_params,
                )
                optimizers += [optimizer]

        # learning rate decay scheduler (linear warmup and warmdown)
        def get_lr_ratio(it):
            def _get_lr_ratio(it):
                assert it <= args.num_iterations
                # 1) linear warmup for warmup_iters steps
                if it < args.warmup_iters:
                    return (it + 1) / args.warmup_iters
                # 2) constant lr for a while
                elif it < args.num_iterations - args.warmdown_iters:
                    return 1.0
                # 3) warmdown
                else:
                    decay_ratio = (args.num_iterations - it) / args.warmdown_iters
                    if args.warmdown_type == "cos":
                        return 0.5 - 0.5 * math.cos(decay_ratio * math.pi)
                    elif args.warmdown_type == "sqrt":
                        return decay_ratio**0.5
                    else:
                        return decay_ratio

            return args.warmdown_min_ratio + (
                1.0 - args.warmdown_min_ratio
            ) * _get_lr_ratio(it)

        schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt, get_lr_ratio) for opt in optimizers
        ]

    assert (
        args.val_tokens
        % (
            args.val_data.device_batch_size
            * min(args.val_data.max_document_tokens, args.val_data.sequence_length)
            * ddp_world_size
        )
        == 0
    )
    val_steps = args.val_tokens // (
        args.val_data.device_batch_size
        * min(args.val_data.max_document_tokens, args.val_data.sequence_length)
        * ddp_world_size
    )

    assert args.batch_size % (args.train_data.device_batch_size * ddp_world_size) == 0

    from data.preprocess import preprocess_dataset_and_get_dataloader

    with timed("Loading datasets"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.model_max_length = None  # ensure tokenizer does not truncate our sequences, which would cause silent data bugs. we will handle truncation ourselves if needed. the +1 is for the labels that are shifted by 1 token relative to the inputs.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 0

        # make sure that the validation set is always disjoint from the train set
        if args.val_data is not None and args.train_data is not None:
            if (
                args.val_data.dataset == args.train_data.dataset
                and args.val_data.dataset_name == args.train_data.dataset_name
                and args.val_data.split == args.train_data.split
            ):
                assert args.val_data.data_packing != "prepacked", (
                    f"Validation data cannot be prepacked when using the same dataset split as training data."
                )
                # train/val set overlap - since they are same dataset, make a split after shuffling the same way!
                args.val_data.shuffle_seed = args.train_data.shuffle_seed
                num_val_documents = 1024
                args.val_data.range_begin = -num_val_documents
                args.val_data.range_end = None
                args.train_data.range_begin = 0
                args.train_data.range_end = -num_val_documents
                pass

        val_data_loader = preprocess_dataset_and_get_dataloader(
            ddp_rank,
            ddp_world_size,
            args.val_data,
            tokenizer=tokenizer,
            preprocess_batch_size=args.preprocess_batch_size,
        )
        train_data_loader = preprocess_dataset_and_get_dataloader(
            ddp_rank,
            ddp_world_size,
            args.train_data,
            tokenizer=tokenizer,
            preprocess_batch_size=args.preprocess_batch_size,
        )

    torch.set_default_device(old_default_device)
    val_loader = iter(val_data_loader)
    train_loader = iter(train_data_loader)

    run_id = str(uuid.uuid4())
    logdir = f"logs/{run_id}/"
    logfile_path = f"logs/{run_id}.txt"
    memory_logfile_path = None  # logfile_path

    from trainer import Trainer

    trainer = Trainer()
    trainer.train(
        args,
        model,
        tokenizer,
        optimizers,
        schedulers,
        train_data_loader,
        val_data_loader,
        val_steps,
        device,
        run_id,
        logdir,
        logfile_path,
        memory_logfile_path,
        logged_cli_config,
    )

    dist.barrier()

    dist.destroy_process_group()
