"""Normal-only paired-score diagnostics; not a defect-recall proxy."""

import numpy as np
import torch
from torch.nn import functional as F

from lc_simplenet.pilot import validate_input_geometry


def paired_normal_metrics(trainer, validation, *, noise_seed: int = 8103) -> dict:
    config = trainer.config
    if (validation.dataset, validation.category, validation.split, validation.manifest_sha256) != (
        config.dataset,
        config.category,
        "validation",
        config.manifest_sha256,
    ):
        raise ValueError("Diagnostics require the pinned normal-validation source")
    validate_input_geometry(config, validation)
    trainer.backend.adapter.eval()
    trainer.backend.discriminator.eval()
    generator = torch.Generator().manual_seed(noise_seed)
    lighting = [(b, g) for b in config.brightness for g in config.gamma]
    normal_scores, mses, hinges, gaps, clipped = [], [], [], [], []
    with torch.inference_mode():
        for index in range(len(validation)):
            pixels = trainer.prepare_pixels([validation[index]["image"]])
            transformed = [b * pixels.pow(g) for b, g in lighting]
            views = torch.cat([pixels, *[image.clamp(0, 1) for image in transformed]])
            features, _ = trainer.backend.embed(trainer.normalize(views))
            adapted = trainer.backend.adapter(features)
            scores = trainer.backend.discriminator(adapted).reshape(len(views), -1)
            original = scores[0]
            noise = torch.randn(adapted[: len(original)].shape, generator=generator)
            fake = trainer.backend.discriminator(
                adapted[: len(original)] + config.noise_std * noise.to(trainer.device)
            ).reshape(-1)
            mses.append((scores[1:] - original).square().mean().item())
            hinges.append(
                (
                    F.relu(config.margin - original).mean() + F.relu(config.margin + fake).mean()
                ).item()
            )
            gaps.append((original.mean() - fake.mean()).item())
            normal_scores.append(original.cpu().numpy())
            clipped.extend(
                ((image > 1) & (pixels < 1)).float().mean().item() for image in transformed
            )
    values = np.concatenate(normal_scores)
    result = {
        "scope": "normal-validation diagnostics; no defect sensitivity is established",
        "images": len(validation),
        "manifest_sha256": config.manifest_sha256,
        "lighting_corners": [list(pair) for pair in lighting],
        "pair_mse": float(np.mean(mses)),
        "patch_score_std": float(values.std()),
        "synthetic_hinge": float(np.mean(hinges)),
        "synthetic_gap": float(np.mean(gaps)),
        "newly_clipped_fraction": float(np.mean(clipped)),
        "noise_seed": noise_seed,
    }
    if not all(
        np.isfinite(result[key])
        for key in (
            "pair_mse",
            "patch_score_std",
            "synthetic_hinge",
            "synthetic_gap",
            "newly_clipped_fraction",
        )
    ):
        raise FloatingPointError("Non-finite normal-validation diagnostic")
    return result
