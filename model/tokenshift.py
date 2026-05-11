from __future__ import annotations

from typing import Literal

import torch


TokenShiftFallback = Literal["first", "zero"]


def token_shift_delta(
    x: torch.Tensor,
    prev_token: torch.Tensor | None = None,
    *,
    fallback: TokenShiftFallback = "first",
) -> torch.Tensor:
    if prev_token is None:
        if fallback == "first":
            prev_token = x[:, :1]
        elif fallback == "zero":
            prev_token = torch.zeros_like(x[:, :1])
        else:
            raise ValueError(f"Unsupported token-shift fallback: {fallback!r}")
    elif prev_token.dtype != x.dtype or prev_token.device != x.device:
        prev_token = prev_token.to(device=x.device, dtype=x.dtype)

    return torch.cat([prev_token, x[:, :-1]], dim=1) - x


def apply_token_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    prev_token: torch.Tensor | None = None,
    *,
    fallback: TokenShiftFallback = "first",
) -> torch.Tensor:
    return x + scale * token_shift_delta(x, prev_token, fallback=fallback)
