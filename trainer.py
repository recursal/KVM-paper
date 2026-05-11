import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from optimizer.muon import Muon

from train import timed
from utils.logger import print0 as print


def update_weight_decay_for_current_lr(optimizer: torch.optim.Optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        return

    for param_group in optimizer.param_groups:
        weight_decay_peak = float(
            param_group.get("weight_decay_peak", param_group.get("weight_decay", 0.0))
            or 0.0
        )
        if weight_decay_peak == 0.0:
            param_group["weight_decay"] = 0.0
            continue

        lr_peak = float(
            param_group.get("lr_peak", param_group.get("initial_lr", param_group["lr"]))
            or 0.0
        )
        if lr_peak == 0.0:
            raise ValueError(
                "AdamW weight_decay scheduling requires a non-zero peak lr."
            )

        param_group["weight_decay"] = (
            weight_decay_peak * float(param_group["lr"]) / lr_peak
        )


class Trainer:
    def train(
        self,
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
    ):
        def should_log_cuda_memory(current_step: int) -> bool:
            if not bool(args.debug_cuda_memory):
                return False
            after_step = int(args.debug_cuda_memory_after_step)
            return after_step < 0 or current_step >= after_step

        def maybe_log_cuda_memory(tag: str, current_step: int) -> None:
            if not should_log_cuda_memory(current_step):
                return
            torch.cuda.synchronize()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            mib = 1024 * 1024
            line = (
                f"rank:{ddp_rank} step:{current_step} cuda_mem[{tag}] "
                f"alloc:{torch.cuda.memory_allocated() // mib}MiB "
                f"reserved:{torch.cuda.memory_reserved() // mib}MiB "
                f"max_alloc:{torch.cuda.max_memory_allocated() // mib}MiB "
                f"max_reserved:{torch.cuda.max_memory_reserved() // mib}MiB "
                f"free:{free_bytes // mib}MiB total:{total_bytes // mib}MiB"
            )
            print(line)
            if memory_logfile_path is not None:
                with open(logfile_path, "a") as f:
                    f.write(line + "\n")

        master_process = int(os.environ.get("RANK", 0)) == 0

        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        abs_logdir = os.path.abspath(logdir)
        abs_logfile_path = os.path.abspath(logfile_path)

        if isinstance(model, DDP):
            raw_model = model.module  # always contains the "raw" unwrapped model
        else:
            raw_model = model
        model_config = raw_model.config

        with timed("Loading first datum"):
            train_loader = iter(train_data_loader)
            datum = next(train_loader)
            val_loader = iter(val_data_loader)
            val_datum = next(val_loader)

        if len(args.wandb) > 0:
            with timed(
                "Running test batch before logging in to wandb to force compile"
            ):
                test_results = model(
                    input_ids=datum["input_ids"],
                    labels=datum["labels"],
                    cu_seqlens=datum.get("cu_seqlens"),
                    return_logits=False,
                )
                loss = test_results["loss"]
                del test_results
                with model.no_sync():
                    loss.backward()
                model.zero_grad(set_to_none=True)

        wandb_instance = None
        if master_process:
            if len(args.wandb) > 0:
                with timed("Login to wandb"):
                    import wandb
                    import datetime

                    timestamp_str = datetime.datetime.today().strftime(
                        "%Y-%m-%d-%H-%M-%S"
                    )
                    run_name = (
                        f"{model_config.model_class_path} L{model_config.num_hidden_layers} D{model_config.hidden_size} "
                        f"v{model_config.vocab_size} ctx{args.train_data.sequence_length} seed{args.init_seed}"
                    )
                    wandb_run_name = run_name + " " + timestamp_str
                    if args.wandb_name_suffix:
                        wandb_run_name += " " + args.wandb_name_suffix
                    wandb.init(
                        project=args.wandb,
                        name=wandb_run_name,
                        config=logged_cli_config,
                        save_code=False,
                    )
                    wandb_instance = wandb
                    local_run_metadata = {
                        "local_run_id": run_id,
                        "local_logdir": abs_logdir,
                        "local_logfile": abs_logfile_path,
                    }
                    wandb.config.update(local_run_metadata, allow_val_change=True)
                    if wandb.run is not None:
                        wandb.run.summary.update(local_run_metadata)

        # begin logging
        if master_process:
            with timed("Beginning log"):
                os.makedirs(logdir, exist_ok=True)

                from utils.gpu import (
                    collect_accelerator_smi_output,
                    torch_runtime_label,
                )

                # create the log file
                with open(logfile_path, "w") as f:
                    f.write(f"local_run_id: {run_id}\n")
                    f.write(f"local_logdir: {abs_logdir}\n")
                    f.write(f"local_logfile: {abs_logfile_path}\n")
                    f.write("=" * 100 + "\n")
                    f.write("=" * 100 + "\n")
                    # log information about the hardware/software environment this is running on
                    # and print the full smi cmdline output to file
                    smi_name, smi_output = collect_accelerator_smi_output()
                    f.write(
                        f"Running pytorch {torch.__version__} compiled for {torch_runtime_label()}\n{smi_name}:\n{smi_output}\n"
                    )
                    f.write("=" * 100 + "\n")
                    f.write("Config")
                    f.write(str(logged_cli_config))
                    f.write("=" * 100 + "\n")

        dist.barrier()

        print("Starting training...")

        train_accumulation_steps = args.batch_size // (
            args.train_data.device_batch_size * ddp_world_size
        )

        model.train()

        real_tokens = 0
        timed_steps = float("nan")
        training_time_ms = 0
        torch.cuda.synchronize()
        t0 = time.time()
        t_step_end = t0
        last_step = False
        for step in range(args.num_iterations + 1):
            last_step = last_step or (step == args.num_iterations)
            t_step_start = t_step_end

            if last_step or (
                args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0
            ):
                maybe_log_cuda_memory("pre_val", step)
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.time() - t0)
                model.eval()
                val_loader = iter(val_data_loader)
                val_loss = torch.zeros((), device=device, dtype=torch.float32)
                val_acc = torch.zeros((), device=device, dtype=torch.float32)
                for _ in range(val_steps):
                    with torch.no_grad():
                        val_datum = next(val_loader)
                        # Drop the full validation output immediately so the last batch's
                        # logits do not survive into the next training iteration.
                        val_results = model(
                            input_ids=val_datum["input_ids"],
                            labels=val_datum["labels"],
                            cu_seqlens=val_datum.get("cu_seqlens"),
                            return_logits=True,
                        )
                        val_loss += val_results["loss"].detach()

                        val_labels = val_datum["labels"].to(device)
                        attention_mask = val_labels != -100
                        preds = val_results["logits"].argmax(dim=-1)
                        acc = preds.eq(
                            val_labels
                        ).sum() / attention_mask.sum().clamp_min(1)
                        val_acc += acc.float()
                        del val_labels, attention_mask, preds, acc

                        del val_results
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(val_acc, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
                val_acc /= val_steps
                val_loss_item = float(val_loss.item())
                val_acc_item = float(val_acc.item())
                if master_process:
                    print(
                        f"step:{step}/{args.num_iterations} val_loss:{val_loss_item:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / timed_steps:.2f}ms"
                    )
                    if wandb_instance is not None:
                        wandb_instance.log(
                            {
                                "val/loss": val_loss_item,
                                "val/acc": val_acc_item,
                                "tokens": real_tokens,
                            },
                            step=step,
                        )
                    with open(logfile_path, "a") as f:
                        f.write(
                            f"step:{step}/{args.num_iterations} val_loss:{val_loss_item:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / timed_steps:.2f}ms\n"
                        )
                del val_datum
                del val_loader
                maybe_log_cuda_memory("post_val", step)
                torch.cuda.synchronize()
                t0 = time.time()

            if master_process and (
                last_step or (args.save_every > 0 and step % args.save_every == 0)
            ):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.time() - t0)
                if not last_step:
                    log = dict(
                        step=step,
                        # code=code,
                        # init_seed=init_seed,
                        model=raw_model.state_dict(),
                        optimizers=[opt.state_dict() for opt in optimizers],
                    )
                    save_filename = f"model_step{step:06d}.pt"
                    print(
                        f"Saving model checkpoint to {logdir + save_filename}... ",
                        end="",
                    )
                    torch.save(log, logdir + save_filename)
                    print("Done.")

                if last_step:
                    print(f"Saving final HF model to {logdir}... ")
                    raw_model.save_pretrained(logdir)
                    tokenizer.save_pretrained(logdir)
                    # save_model_safetensors(raw_model, logdir + save_filename)

                    with open(os.path.join(logdir, "rwkv7_backbone.py"), "w") as f:
                        model_code = """
import os, sys
hf_wrapper_base_path = os.environ.get('HF_WRAPPER_BASE_PATH')
if hf_wrapper_base_path is not None:
    sys.path.insert(0, hf_wrapper_base_path)

from model.rwkv7_backbone import RWKV7BackboneForCausalLM, RWKV7BackboneModel, MixerConfig
RWKV7BackboneForCausalLM = RWKV7BackboneForCausalLM
RWKV7BackboneModel = RWKV7BackboneModel
RWKV7BackboneConfig = MixerConfig
"""
                        f.write(model_code)

                    print("Done.\n")

                torch.cuda.synchronize()
                t0 = time.time()

            # Run one extra loop iteration so validation/checkpointing also happen after the final train step.
            if last_step:
                break

            model.train()
            maybe_log_cuda_memory("pre_train", step + 1)

            for i in range(1, train_accumulation_steps + 1):
                train_results = model(
                    input_ids=datum["input_ids"],
                    labels=datum["labels"],
                    cu_seqlens=datum.get("cu_seqlens"),
                    return_logits=False,
                )
                loss = train_results["loss"]
                del train_results
                train_loss = loss.detach()
                maybe_log_cuda_memory(f"post_forward_accum{i}", step + 1)

                try:
                    datum = next(train_loader)
                except StopIteration:
                    print("Reached end of dataset. Stopping early.")
                    last_step = True
                    break
                if i < train_accumulation_steps:
                    with model.no_sync():
                        loss.backward()
                else:
                    loss.backward()
                maybe_log_cuda_memory(f"post_backward_accum{i}", step + 1)
                del loss

                real_tokens += args.batch_size * args.train_data.sequence_length

            if not last_step:
                if train_accumulation_steps > 1:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad /= train_accumulation_steps

                for optimizer in optimizers:
                    if isinstance(optimizer, Muon):
                        # momentum warmup for Muon
                        frac = min(step / 500, 1)
                        momentum = (1 - frac) * 0.85 + frac * 0.95
                        for param_group in optimizer.param_groups:
                            param_group["momentum"] = momentum

                for opt, sched in zip(optimizers, schedulers):
                    if len(opt.param_groups) > 0:
                        model_params, master_params = (
                            opt.model_params,
                            opt.master_params,
                        )
                        for p, mp in zip(model_params, master_params):
                            if p.grad is not None:
                                mp.grad = p.grad.to(mp.dtype)

                    update_weight_decay_for_current_lr(opt)
                    opt.step()

                    sched.step()
                    # FIXME - why are we calling this twice?
                    update_weight_decay_for_current_lr(opt)

                    if len(opt.param_groups) > 0:
                        with torch.no_grad():
                            for p, mp in zip(model_params, master_params):
                                p.copy_(mp)
                                p.grad = None
                                mp.grad = None
                maybe_log_cuda_memory("post_optimizer", step + 1)

            if master_process:
                torch.cuda.synchronize()

                if step <= 9:
                    # Ignore the first few warmup steps in throughput timing.
                    training_time_ms = 0
                    t0 = time.time()
                timed_steps = float("nan") if step <= 9 else (step - 9)

                training_time_ms += 1000 * (time.time() - t0)
                t_step_end = time.time()
                step_time = 1000 * (t_step_end - t_step_start)
                step_mtok_per_sec = (
                    args.batch_size
                    * args.train_data.sequence_length
                    / (t_step_end - t_step_start)
                    / 1e6
                )

                lr = schedulers[0].get_lr()[0]
                print(
                    f"step:{step + 1}/{args.num_iterations} loss:{train_loss.item():.4f} time:{training_time_ms / 1000.0:.0f}s {step_mtok_per_sec:.2f}mtok/s step_time:{step_time:.2f} lr:{lr:0.2e}"
                )
                if wandb_instance is not None:
                    log_dict = {
                        "loss": float(train_loss.item()),
                        "lr": lr,
                        "tokens": real_tokens,
                    }
                    kt_s = (
                        args.batch_size
                        * args.train_data.sequence_length
                        / (training_time_ms / timed_steps / 1000.0)
                        / 1000
                    )
                    if kt_s > 0:
                        log_dict["kt/s"] = kt_s
                    wandb_instance.log(log_dict, step=step)
                with open(logfile_path, "a") as f:
                    f.write(
                        f"step:{step + 1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / timed_steps:.2f}ms\n"
                    )
                t0 = time.time()

        if wandb_instance is not None:
            wandb_instance.finish()
        print(
            f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
        )
