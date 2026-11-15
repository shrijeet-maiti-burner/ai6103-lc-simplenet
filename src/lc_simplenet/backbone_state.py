"""Preserve the frozen backbone buffers changed by upstream's shape probe."""

import re

import torch

from lc_simplenet.provenance import state_sha256


def _identity(image_shape, pretrained_sha256):
    shape = tuple(image_shape)
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("image_shape must contain three positive integers")
    if not isinstance(pretrained_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", pretrained_sha256
    ):
        raise ValueError("pretrained_sha256 must identify the actual weight file")
    return shape


def capture_backbone_state(backbone, *, image_shape, pretrained_sha256):
    """Capture independent CPU buffers; parameters remain in the pinned weight file.

    Capture after the original shape probe on the original runtime. A freshly
    reconstructed snapshot is evidence only after strict saved-normal replay.
    This does not alter modes, parameters, legacy checkpoints or the upstream.
    """
    shape = _identity(image_shape, pretrained_sha256)
    buffers = {name: value.detach().cpu().clone() for name, value in backbone.named_buffers()}
    if any(
        value.is_floating_point() and not torch.isfinite(value).all() for value in buffers.values()
    ):
        raise ValueError("cannot capture nonfinite backbone buffers")
    return {
        "schema": 1,
        "architecture": f"{type(backbone).__module__}.{type(backbone).__qualname__}",
        "image_shape": list(shape),
        "pretrained_sha256": pretrained_sha256,
        "parameters_sha256": state_sha256(dict(backbone.named_parameters())),
        "buffers_sha256": state_sha256(buffers),
        "backbone_sha256": state_sha256(backbone.state_dict()),
        "buffers": buffers,
    }


@torch.no_grad()
def restore_backbone_state(backbone, snapshot, *, image_shape, pretrained_sha256):
    """Validate the complete snapshot before copying any buffer into the model.

    Restoring state does not promise bit-exact kernels across hardware. Saved
    normal-score and complete-map guards remain mandatory before test scoring.
    """
    shape = _identity(image_shape, pretrained_sha256)
    architecture = f"{type(backbone).__module__}.{type(backbone).__qualname__}"
    if snapshot.get("schema") != 1 or snapshot.get("architecture") != architecture:
        raise ValueError("unsupported snapshot schema or backbone architecture")
    if (
        snapshot.get("image_shape") != list(shape)
        or snapshot.get("pretrained_sha256") != pretrained_sha256
    ):
        raise ValueError("snapshot probe geometry or pretrained identity differs")
    if state_sha256(dict(backbone.named_parameters())) != snapshot["parameters_sha256"]:
        raise ValueError("backbone parameters differ; buffers cannot repair a different model")
    actual = dict(backbone.named_buffers())
    supplied = snapshot["buffers"]
    if actual.keys() != supplied.keys():
        raise ValueError("snapshot buffer names differ")
    for name, expected in actual.items():
        value = supplied[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise ValueError("snapshot buffer shape or dtype differs")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("snapshot contains nonfinite buffers")
    if state_sha256(supplied) != snapshot["buffers_sha256"]:
        raise ValueError("snapshot buffer bytes changed")
    candidate = {name: supplied.get(name, value) for name, value in backbone.state_dict().items()}
    if state_sha256(candidate) != snapshot["backbone_sha256"]:
        raise ValueError("snapshot complete-state fingerprint does not reconstruct")
    for name, value in supplied.items():
        actual[name].copy_(value.to(device=actual[name].device))
    if state_sha256(backbone.state_dict()) != snapshot["backbone_sha256"]:
        raise AssertionError("restored backbone fingerprint differs")
    return {"buffers_restored": len(actual), "backbone_sha256": snapshot["backbone_sha256"]}
