"""Experimental fixed-teacher protection for the paired SimpleNet objective.

This does not establish defect preservation. The teacher sees clean normal
features and the same post-adapter Gaussian corruption as the student.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from lc_simplenet.paired_losses import paired_objective, validate_objective_settings


def frozen_teacher(adapter: nn.Module, discriminator: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Copy parameters and buffers; leave the supplied trainable modules alone."""
    return tuple(
        copy.deepcopy(module).eval().requires_grad_(False) for module in (adapter, discriminator)
    )


def adaptation_mode(adapter: nn.Module, discriminator: nn.Module) -> None:
    """Train affine weights while fixing every BatchNorm running statistic.

    All four adaptation arms use this mode. It is an explicit difference from
    the original full-study training loop, not a silent compatibility change.
    """
    for network in (adapter, discriminator):
        network.train()
        for module in network.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def validate_anchored_settings(*, margin: float, lambda_score: float, lambda_anchor: float) -> None:
    validate_objective_settings(margin=margin, lambda_score=lambda_score)
    if not math.isfinite(lambda_anchor) or lambda_anchor < 0:
        raise ValueError("lambda_anchor must be finite and nonnegative")


def anchored_objective(
    normal_original: torch.Tensor,
    normal_relit: torch.Tensor,
    synthetic_original: torch.Tensor,
    synthetic_relit: torch.Tensor,
    teacher_normal: torch.Tensor,
    teacher_synthetic: torch.Tensor,
    *,
    margin: float,
    lambda_score: float,
    lambda_anchor: float,
) -> dict[str, torch.Tensor]:
    """Original paired loss plus clean-view normal/corruption response anchoring.

    Raw discriminator scores are higher for normality. Teacher tensors are
    detached even if a caller accidentally passes a differentiable target.
    Zero anchor weight returns the original total tensor without an added zero
    term, retaining its exact update path.
    """
    validate_anchored_settings(
        margin=margin, lambda_score=lambda_score, lambda_anchor=lambda_anchor
    )
    terms = paired_objective(
        normal_original,
        normal_relit,
        synthetic_original,
        synthetic_relit,
        margin=margin,
        lambda_score=lambda_score,
    )
    for target in (teacher_normal, teacher_synthetic):
        if (
            target.shape != normal_original.shape
            or target.device != normal_original.device
            or target.dtype != normal_original.dtype
            or not target.is_floating_point()
            or not torch.isfinite(target).all()
        ):
            raise ValueError("teacher scores must match the finite raw student score tensors")
    # Cloning also permits targets produced by inference_mode without saving an
    # inference tensor for autograd. This never connects the teacher's graph.
    target_normal, target_synthetic = (
        target.detach().clone() for target in (teacher_normal, teacher_synthetic)
    )
    clean, synthetic = normal_original, synthetic_original
    if clean.dtype in (torch.float16, torch.bfloat16):
        clean, synthetic, target_normal, target_synthetic = (
            value.float() for value in (clean, synthetic, target_normal, target_synthetic)
        )
    anchor = (F.mse_loss(clean, target_normal) + F.mse_loss(synthetic, target_synthetic)) / 2
    return {
        **terms,
        "total": terms["total"] if lambda_anchor == 0 else terms["total"] + lambda_anchor * anchor,
        "anchor": anchor,
    }


def anchored_forward(
    adapter: nn.Module,
    discriminator: nn.Module,
    teacher_adapter: nn.Module,
    teacher_discriminator: nn.Module,
    original: torch.Tensor,
    relit: torch.Tensor,
    noise: torch.Tensor,
    *,
    margin: float,
    lambda_score: float,
    lambda_anchor: float,
) -> dict[str, torch.Tensor]:
    """One student joint forward; matching frozen teacher forward under no_grad.

    Student order: normal clean, normal relit, Gaussian clean, Gaussian relit.
    Gaussian noise is added after each model's adapter; its clean-view values
    are shared with the teacher. Both networks process the same four groups to
    avoid a numerical batch-shape confound; only clean-view targets are anchored.
    Inputs are cached frozen-encoder patch vectors.
    Caller sets student adaptation mode explicitly before calling this helper.
    """
    validate_anchored_settings(
        margin=margin, lambda_score=lambda_score, lambda_anchor=lambda_anchor
    )
    teachers = (teacher_adapter, teacher_discriminator)
    if any(module.training for network in teachers for module in network.modules()) or any(
        parameter.requires_grad for network in teachers for parameter in network.parameters()
    ):
        raise ValueError("teacher parameters and all forward modes must be frozen")
    if (
        original.ndim != 2
        or original.numel() == 0
        or relit.shape != original.shape
        or noise.ndim != 2
        or len(noise) != 2 * len(original)
    ):
        raise ValueError("expected aligned nonempty patch matrices and one noise row per view")
    for value in (original, relit, noise):
        if (
            value.device != original.device
            or value.dtype != original.dtype
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError("features and noise must share finite floating point dtype/device")
    inputs = torch.cat((original, relit))
    adapted = adapter(inputs)
    if adapted.shape != noise.shape:
        raise ValueError("noise must match the post-adapter patch matrices")
    with torch.no_grad():
        reference = teacher_adapter(inputs)
        if reference.shape != adapted.shape:
            raise ValueError("teacher and student adapter coordinates differ")
        teacher_scores = teacher_discriminator(torch.cat((reference, reference + noise)))
    scores = discriminator(torch.cat((adapted, adapted + noise)))
    clean, light, fake_clean, fake_light = scores.chunk(4)
    reference_normal, _, reference_fake, _ = teacher_scores.chunk(4)
    return anchored_objective(
        clean,
        light,
        fake_clean,
        fake_light,
        reference_normal,
        reference_fake,
        margin=margin,
        lambda_score=lambda_score,
        lambda_anchor=lambda_anchor,
    )
