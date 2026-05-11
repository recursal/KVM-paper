use_grad_cp = 1


def maybe_ckpt(module, *args, **kwargs):
    if use_grad_cp:
        import torch._dynamo

        torch._dynamo.config.optimize_ddp = False
        import torch.utils.checkpoint

        return torch.utils.checkpoint.checkpoint(
            module, *args, **kwargs, use_reentrant=False
        )
    else:
        return module(*args, **kwargs)
