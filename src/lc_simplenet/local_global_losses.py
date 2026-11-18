"""experimental local/global supervision, adapted from glass (eccv 2024).

see third_party/licenses/GLASS-MIT.txt and docs/routed-consistency.md.
the discriminator outputs raw normality, so anomaly logits have the opposite sign.
"""

import math

import torch
from torch.nn import functional as F

from lc_simplenet.routed_consistency import adapter_only_score_consistency


def mine_feature_noise(
    discriminator,
    features,
    noise,
    *,
    steps=20,
    step_size=0.001,
    project_every=5,
    inner_radius=0.5,
    outer_radius=1.0,
):
    """ascend fake-class bce, periodically projecting synthetic offsets only."""
    if (
        features.ndim != 2
        or not features.is_floating_point()
        or features.shape != noise.shape
        or features.device != noise.device
        or features.dtype != noise.dtype
        or not torch.isfinite(features).all()
        or not torch.isfinite(noise).all()
    ):
        raise ValueError("aligned finite feature/noise matrices required")
    if (
        type(steps) is not int
        or steps < 0
        or type(project_every) is not int
        or project_every < 1
        or not all(math.isfinite(x) and x > 0 for x in (step_size, inner_radius, outer_radius))
        or inner_radius > outer_radius
    ):
        raise ValueError("invalid mining settings")
    modes = [(m, m.training) for m in discriminator.modules()]
    discriminator.eval()
    base = features.detach()
    delta = noise.detach().clone()
    try:
        for step in range(steps):
            fake = (base + delta).detach().requires_grad_(True)
            loss = F.softplus(discriminator(fake)).mean()
            (gradient,) = torch.autograd.grad(loss, fake, only_inputs=True)
            with torch.no_grad():
                delta = delta + step_size * F.normalize(gradient, dim=1, eps=1e-10)
                if (step + 1) % project_every == 0:
                    norms = delta.norm(dim=1, keepdim=True)
                    delta = delta * norms.clamp(inner_radius, outer_radius) / norms.clamp_min(1e-10)
        if not torch.isfinite(delta).all():
            raise FloatingPointError("nonfinite mined perturbation")
        return delta.detach()
    finally:
        for module, mode in modes:
            module.training = mode


def local_focal_loss(normality, labels):
    """focal gamma 2 on the hardest half by squared anomaly-probability error.

    the empirical median cutoff includes all ties. visible synthetic support
    has label one; unchanged patches have label zero. this is not a defect mask.
    """
    if labels.dtype != torch.bool or labels.numel() != normality.numel():
        raise ValueError("one boolean synthetic-support label per score required")
    logits = -normality.reshape(-1)
    target = labels.to(logits.dtype).reshape(-1)
    probability = logits.sigmoid()
    error = (probability - target).square().detach()
    keep = error >= torch.quantile(error, 0.5)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.where(target > 0, probability, 1 - probability)
    return (((1 - pt) ** 2) * bce)[keep].mean()


def local_global_objective(
    adapter,
    discriminator,
    normal,
    relit,
    changed_original,
    changed_relit,
    noise,
    masks,
    *,
    lambda_score=0.1,
    route_consistency=True,
):
    """compute r0, or its matched no-consistency/full-gradient ablations.

    inputs are aligned frozen patch embeddings, with two boolean visible-support
    masks. the caller supplies local synthesis, paired transforms, gaussian
    noise, optimizer state and data order. no defect-lift term is used.
    """
    if not math.isfinite(lambda_score) or lambda_score < 0:
        raise ValueError("lambda_score must be finite and nonnegative")
    if type(route_consistency) is not bool:
        raise ValueError("route_consistency must be boolean")
    if (
        normal.ndim != 2
        or normal.numel() == 0
        or any(x.shape != normal.shape for x in (relit, changed_original, changed_relit))
    ):
        raise ValueError("aligned nonempty normal and changed features required")
    if masks.dtype != torch.bool or masks.shape != (2, len(normal)):
        raise ValueError("paired visible synthetic supports required")
    z = adapter(torch.cat((normal, relit)))
    delta = mine_feature_noise(discriminator, z, noise)
    n0, n1, f0, f1 = discriminator(torch.cat((z, z + delta))).chunk(4)
    discrimination = (
        F.softplus(-n0).mean()
        + F.softplus(-n1).mean()
        + F.softplus(f0).mean()
        + F.softplus(f1).mean()
    ) / 2
    modes = [(m, m.training) for m in discriminator.modules()]
    discriminator.eval()
    try:
        auxiliary = adapter(torch.cat((normal, relit, changed_original, changed_relit)))
        a0, a1, d0, d1 = discriminator(auxiliary).chunk(4)
        consistency = (
            adapter_only_score_consistency(discriminator, auxiliary, patches_per_view=len(normal))
            if route_consistency
            else F.mse_loss(a0, a1)
        )
    finally:
        for module, mode in modes:
            module.training = mode
    local = local_focal_loss(torch.cat((d0, d1)), masks.reshape(-1))
    return {
        "total": discrimination + local + lambda_score * consistency,
        "discrimination": discrimination,
        "local_focal": local,
        "normal_consistency": consistency,
    }
