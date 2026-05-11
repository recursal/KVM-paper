import torch
from torch.nn.attention.flex_attention import flex_attention


@torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)
def compiled_flex_attention(*args, **kwargs):
    return flex_attention(*args, **kwargs)


@torch.compiler.disable
def separately_compiled_flex_attention(*args, **kwargs):
    return compiled_flex_attention(*args, **kwargs)


def causal_mask_mod(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx
