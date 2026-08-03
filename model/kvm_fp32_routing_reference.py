"""
This module is an experimental semantic reference, not the historical golden.
The historical golden remains the untouched :mod:`model.kvm_mixer` module.

The derived mixer preserves the golden implementation except that merge-route
similarities are computed in FP32 before ``argmax``.  The resulting one-hot
route tensor uses the original key dtype so state-update arithmetic retains
the golden behavior.  It is scoped to the fixed-state KVM system,
where the initial chunk fills the state before any overflow merge; ranked
append selection therefore never occurs.  It does not redefine append-ranking
semantics for growing-state configurations.
"""

from __future__ import annotations

import torch

from .kvm_mixer import SequenceMixer as HistoricalGoldenSequenceMixer


class SequenceMixer(HistoricalGoldenSequenceMixer):
    """KVM golden derivative with FP32 merge-route similarity scoring."""

    def _merge_into_state(self, k_merge, v_merge, s_k, s_v, s_vlen, protected_slots):
        # obtain normalized state keys
        s_k_norm = self.ln_s_k(s_k)

        # find the most similar key in state for each incoming key to merge
        # delta from the historical golden: score routes in FP32.
        logits = torch.matmul(
            k_merge.float(), s_k_norm.float().transpose(-1, -2)
        )
        logits[..., :protected_slots] = float("-inf")
        best_s_idx = logits.max(dim=-1, keepdim=True).indices
        scores = torch.scatter(
            torch.zeros_like(logits, dtype=k_merge.dtype),
            -1,
            best_s_idx,
            torch.ones_like(logits, dtype=k_merge.dtype),
        )

        # update state by adding the most similar keys and their values, gated by the merge gate
        s_k = s_k + (scores.mT @ k_merge)
        s_v = s_v + (scores.mT @ v_merge)

        if not self.config.kvm_use_vlens:
            s_vlen = s_vlen + scores.sum(-1, keepdim=True)

        return s_k, s_v, s_vlen
