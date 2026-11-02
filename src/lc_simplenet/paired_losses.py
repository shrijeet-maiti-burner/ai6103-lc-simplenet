"""Proposal objective on aligned normal patch scores (optional PyTorch runtime)."""

import math

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
    if original.dtype in (torch.float16, torch.bfloat16):
        original = original.float()
    if relit.dtype in (torch.float16, torch.bfloat16):
        relit = relit.float()
    return F.mse_loss(original, relit, reduction="mean")


def validate_objective_settings(*, margin: float, lambda_score: float) -> None:
    """Reject invalid settings before any stateful discriminator forward."""
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    if not math.isfinite(lambda_score) or lambda_score < 0:
        raise ValueError("lambda_score must be finite and nonnegative")


def paired_objective(
    normal_original: torch.Tensor,
    normal_relit: torch.Tensor,
    synthetic_original: torch.Tensor,
    synthetic_relit: torch.Tensor,
    *,
    margin: float,
    lambda_score: float,
) -> dict[str, torch.Tensor]:
    """Average two-view hinge discrimination plus aligned normal-score MSE.

    Inputs are raw discriminator scores (higher means normal). Caller must make
    one joint discriminator forward and split its scores in documented order.
    Synthetic scores participate in discrimination only. Returned tensors keep
    their graphs; detach them only when logging.
    """
    validate_objective_settings(margin=margin, lambda_score=lambda_score)
    scores = (normal_original, normal_relit, synthetic_original, synthetic_relit)
    for score in scores:
        if score.shape != normal_original.shape or score.numel() == 0:
            raise ValueError("All four raw score tensors must have identical nonempty shapes")
        if score.device != normal_original.device or score.dtype != normal_original.dtype:
            raise ValueError("All scores must share device and floating point dtype")
        if not score.is_floating_point() or not torch.isfinite(score).all():
            raise ValueError("Scores must be finite floating point tensors")
    if normal_original.dtype in (torch.float16, torch.bfloat16):
        normal_original, normal_relit, synthetic_original, synthetic_relit = (
            score.float() for score in scores
        )
    discrimination = (
        F.relu(margin - normal_original).mean()
        + F.relu(margin + synthetic_original).mean()
        + F.relu(margin - normal_relit).mean()
        + F.relu(margin + synthetic_relit).mean()
    ) / 2
    consistency = patch_score_consistency(normal_original, normal_relit)
    return {
        "total": discrimination
        if lambda_score == 0
        else discrimination + lambda_score * consistency,
        "discrimination": discrimination,
        "consistency": consistency,
    }
