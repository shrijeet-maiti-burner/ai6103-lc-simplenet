"""normal-score consistency with gradients restricted to the input features."""

import torch
from torch.func import functional_call

from lc_simplenet.paired_losses import patch_score_consistency


def adapter_only_score_consistency(
    discriminator: torch.nn.Module,
    adapted_features: torch.Tensor,
    *,
    patches_per_view: int,
) -> torch.Tensor:
    """compare the first aligned normal pair through a constant classifier.

    the input starts with original patches, then relit patches. additional
    adapted patches may follow to preserve a caller's auxiliary forward shape.
    classifier parameters are detached, buffers cloned, and running batch
    normalization used. gradients still reach the adapter through its features.
    every module's previous training mode is restored, including on failure.
    classification losses and inference remain the caller's responsibility.
    """
    if type(patches_per_view) is not int or patches_per_view < 1:
        raise ValueError("patches_per_view must be a positive integer")
    if adapted_features.ndim != 2 or adapted_features.shape[0] < 2 * patches_per_view:
        raise ValueError("aligned original and relit patch features are required")
    if not adapted_features.is_floating_point() or not torch.isfinite(adapted_features).all():
        raise ValueError("adapted features must be finite floating point values")
    modes = [(module, module.training) for module in discriminator.modules()]
    state = {name: p.detach() for name, p in discriminator.named_parameters()}
    state.update({name: b.detach().clone() for name, b in discriminator.named_buffers()})
    try:
        discriminator.eval()
        scores = functional_call(discriminator, state, (adapted_features,), strict=True)
        if scores.shape[0] != len(adapted_features) or scores.numel() != len(adapted_features):
            raise ValueError("one raw scalar score per patch is required")
        original = scores[:patches_per_view]
        relit = scores[patches_per_view : 2 * patches_per_view]
        return patch_score_consistency(original, relit)
    finally:
        for module, mode in modes:
            module.training = mode
