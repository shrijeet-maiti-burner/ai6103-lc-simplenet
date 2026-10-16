"""Proposal objective on aligned normal patch scores (optional PyTorch runtime)."""

import torch
from torch.nn import functional as F


def patch_score_consistency(original: torch.Tensor, relit: torch.Tensor) -> torch.Tensor:
    """Mean squared raw-score difference; gradients flow through both branches."""
    if original.shape != relit.shape or original.numel() == 0:
        raise ValueError("Score tensors must have identical, nonempty shapes")
    if not original.is_floating_point() or not relit.is_floating_point():
        raise ValueError("Score tensors must be floating point")
    if not torch.isfinite(original).all() or not torch.isfinite(relit).all():
        raise ValueError("Score tensors must be finite")
    return F.mse_loss(original, relit, reduction="mean")
