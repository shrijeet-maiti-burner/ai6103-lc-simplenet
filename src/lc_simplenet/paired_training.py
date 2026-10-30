"""Paired-view interface for the shared trainer; no optimizer or noise-policy changes."""

import numpy as np
import torch

from lc_simplenet.losses import paired_objective, validate_objective_settings
from lc_simplenet.pairs import LightingPair


def normalized_pair(pair: LightingPair) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an already spatially aligned RGB pair to ImageNet-normalized CHW."""
    if pair.original.shape != pair.relit.shape:
        raise ValueError("Pair views must have identical HWC shapes")
    for image in (pair.original, pair.relit):
        if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
            raise ValueError("Expected nonempty HWC RGB views")
        if not np.issubdtype(image.dtype, np.floating):
            raise ValueError("Expected floating point RGB views")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Expected finite RGB pixels in [0, 1]")
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    views = []
    for image in (pair.original, pair.relit):
        tensor = torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).permute(2, 0, 1)
        views.append((tensor - mean) / std)
    return views[0], views[1]


def paired_discriminator_loss(
    discriminator: torch.nn.Module,
    original_features: torch.Tensor,
    relit_features: torch.Tensor,
    original_noise: torch.Tensor,
    relit_noise: torch.Tensor,
    *,
    margin: float,
    lambda_score: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """One joint forward, in original/relit normal then original/relit fake order.

    Features must be adapted, aligned [patches, channels] tensors. The caller
    supplies Gaussian feature noise under the shared trainer's policy. No branch
    is detached. Diagnostics are detached tensors; loss terms retain gradients.
    """
    validate_objective_settings(margin=margin, lambda_score=lambda_score)
    reference = original_features
    if reference.ndim != 2 or reference.numel() == 0:
        raise ValueError("Expected nonempty [patches, channels] features")
    for value in (reference, relit_features, original_noise, relit_noise):
        if value.shape != reference.shape:
            raise ValueError("Views and noise must have identical feature shapes")
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError("Views and noise must share device and dtype")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("Features and noise must be finite floating point tensors")
    joint = torch.cat(
        [reference, relit_features, reference + original_noise, relit_features + relit_noise]
    )
    scores = discriminator(joint)
    if scores.shape not in ((4 * len(reference),), (4 * len(reference), 1)):
        raise ValueError("Discriminator must return one raw score per patch")
    normal_original, normal_relit, fake_original, fake_relit = scores.chunk(4)
    terms = paired_objective(
        normal_original,
        normal_relit,
        fake_original,
        fake_relit,
        margin=margin,
        lambda_score=lambda_score,
    )
    with torch.no_grad():
        if scores.dtype in (torch.float16, torch.bfloat16):
            normal_original, normal_relit, fake_original, fake_relit = (
                group.float() for group in scores.chunk(4)
            )
        normal = torch.cat([normal_original, normal_relit])
        fake = torch.cat([fake_original, fake_relit])
        diagnostics = {
            "normal_score_mean": normal.mean(),
            "normal_score_std": normal.std(unbiased=False),
            "synthetic_score_mean": fake.mean(),
            "synthetic_score_std": fake.std(unbiased=False),
            "normal_synthetic_gap": normal.mean() - fake.mean(),
            "paired_score_mae": (normal_original - normal_relit).abs().mean(),
        }
    return terms, diagnostics
