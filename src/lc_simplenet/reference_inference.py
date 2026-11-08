"""Single-view upstream score/map computation without its unused dense features."""

import numpy as np
import torch


@torch.no_grad()
def predict_reference_batch(model, images):
    """Keep E/A/D, max-patch scores, interpolation and Gaussian smoothing.

    Upstream interpolates a dense 1,536-channel feature visualization that is
    independent of the mask branch. Supply one zero auxiliary channel instead.
    The original source stays unchanged; this is an explicit memory adaptation.
    """
    if images.ndim != 4 or len(images) == 0 or images.shape[1] != 3:
        raise ValueError("expected a nonempty NCHW RGB batch")
    images = images.to(dtype=torch.float32, device=model.device)
    model.forward_modules.eval()
    if model.pre_proj > 0:
        model.pre_projection.eval()
    model.discriminator.eval()
    features, shapes = model._embed(images, provide_patch_shapes=True, evaluation=True)
    if model.pre_proj > 0:
        features = model.pre_projection(features)
    raw = (-model.discriminator(features)).cpu().numpy()
    scores = model.patch_maker.unpatch_scores(raw, batchsize=len(images))
    scores = model.patch_maker.score(scores.reshape(*scores.shape[:2], -1))
    height, width = shapes[0]
    patches = model.patch_maker.unpatch_scores(raw, batchsize=len(images))
    patches = patches.reshape(len(images), height, width)
    auxiliary = features.new_zeros((len(images), height, width, 1))
    masks, _ = model.anomaly_segmentor.convert_to_segmentation(patches, auxiliary)
    scores, masks = np.asarray(scores), np.stack(masks)
    if not np.isfinite(scores).all() or not np.isfinite(masks).all():
        raise FloatingPointError("nonfinite original-reference score or map")
    return scores, masks
