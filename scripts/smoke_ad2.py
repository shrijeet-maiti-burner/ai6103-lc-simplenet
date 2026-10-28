"""Load real AD 2 normal images and run one CPU forward/backward. Not training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from lc_simplenet.data import NormalManifestDataset
from lc_simplenet.pilot import (
    PilotConfig,
    PilotTrainer,
    prediction_geometry,
    validate_input_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def aspect_ratio_changed(original_hw, target_hw) -> bool:
    """True when a full-frame resize from original (h, w) to target (h, w) stretches."""
    height, width = int(original_hw[0]), int(original_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    return height * target_w != width * target_h


def _dataset(manifest, root, sha256, split, output_size):
    return NormalManifestDataset(
        manifest,
        root,
        dataset="mvtec_ad2",
        category="fabric",
        split=split,
        expected_sha256=sha256,
        output_size=output_size,
    )


def _summarize(sample, tag):
    image = sample["image"]
    return {
        "split": tag,
        "image_id": sample["image_id"],
        "original_size_hw": sample["original_size"].tolist(),
        "image_size_hw": sample["image_size"].tolist(),
        "dtype": str(image.dtype),
        "shape": list(image.shape),
        "min": float(image.min()),
        "max": float(image.max()),
        "finite": bool(np.isfinite(image).all()),
        "in_unit_interval": bool(image.min() >= 0 and image.max() <= 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="Overrides MVTEC_AD2_ROOT")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--weights", choices=["random", "imagenet_v1"], default="random")
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--image-width", type=int, default=288)
    parser.add_argument("--train-limit", type=int, default=4)
    parser.add_argument("--val-limit", type=int, default=2)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    if args.weights == "imagenet_v1" and args.device != "cuda":
        raise SystemExit("imagenet_v1 weights are reserved for the CUDA cluster smoke")
    root = args.root or (
        Path(os.environ["MVTEC_AD2_ROOT"]) if os.environ.get("MVTEC_AD2_ROOT") else None
    )
    if root is None:
        raise SystemExit("Set --root or MVTEC_AD2_ROOT")
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    native_train = _dataset(args.manifest, root, digest, "train", None)
    native_val = _dataset(args.manifest, root, digest, "validation", None)
    for source in (native_train, native_val):
        if any("/test_" in record.image_id for record in source.records):
            raise SystemExit("Refusing a source that includes test paths")
        if any(record.source_mode not in {"RGB", "L"} for record in source.records):
            raise SystemExit("Unexpected source mode in the pinned manifest")
    native_samples = [
        _summarize(native_train[i], "train")
        for i in range(min(args.train_limit, len(native_train)))
    ]
    native_samples += [
        _summarize(native_val[i], "validation") for i in range(min(args.val_limit, len(native_val)))
    ]
    load_size = (args.image_height, args.image_width)
    aspect_reports = [
        {
            "image_id": sample["image_id"],
            "original_size_hw": [
                int(sample["original_size_hw"][0]),
                int(sample["original_size_hw"][1]),
            ],
            "target_size_hw": [load_size[0], load_size[1]],
            "changes_aspect_ratio": aspect_ratio_changed(sample["original_size_hw"], load_size),
        }
        for sample in native_samples
    ]
    aspect_change = any(item["changes_aspect_ratio"] for item in aspect_reports)
    config = PilotConfig(
        dataset="mvtec_ad2",
        category="fabric",
        manifest_sha256=digest,
        variant="B0",
        lambda_score=0.0,
        seed=0,
        batch_size=min(2, len(native_train)),
        image_size=load_size,
        ad_resize=None,
        weights=args.weights,
    )
    train = _dataset(args.manifest, root, digest, "train", config.load_size)
    validation = _dataset(args.manifest, root, digest, "validation", config.load_size)
    validate_input_geometry(config, train)
    validate_input_geometry(config, validation)
    resized = [_summarize(train[i], "train") for i in range(min(args.train_limit, len(train)))]
    val_batch = [validation[i] for i in range(min(args.val_limit, len(validation)))]
    resized += [_summarize(sample, "validation") for sample in val_batch]
    trainer = PilotTrainer(config, train, device=args.device)
    pixels = trainer.prepare_pixels([sample["image"] for sample in val_batch])
    normalized = trainer.normalize(pixels)
    if not torch.isfinite(normalized).all():
        raise SystemExit("Non-finite ImageNet-normalized validation batch")
    before = trainer.predict([val_batch[0]["image"]])
    record = trainer.train_step()
    after_step = trainer.predict([val_batch[0]["image"]])
    with tempfile.TemporaryDirectory(
        dir=str(ROOT / "results") if (ROOT / "results").exists() else None
    ) as tmp:
        checkpoint = Path(tmp) / "smoke.pt"
        trainer.save(checkpoint)
        resumed = PilotTrainer(config, train, device=args.device)
        resumed.load(checkpoint)
        reloaded = resumed.predict([val_batch[0]["image"]])
    if not np.allclose(after_step, reloaded, rtol=0, atol=args.atol):
        raise SystemExit(f"Reloaded prediction exceeded atol={args.atol}")
    geometry = prediction_geometry(config, val_batch[0])
    print(
        json.dumps(
            {
                "scope": "real-image smoke; one B0 step; not a training or benchmark result",
                "device": args.device,
                "weights": args.weights,
                "manifest_sha256": digest,
                "train_count": len(train),
                "validation_count": len(validation),
                "native_samples": native_samples,
                "resized_samples": resized,
                "transform": {
                    "reader": "PIL bilinear full-frame resize",
                    "load_size_hw": list(load_size),
                    "original_size_hw": sorted(
                        {tuple(item["original_size_hw"]) for item in aspect_reports}
                    ),
                    "target_size_hw": list(load_size),
                    "crop": None,
                    "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
                    "changes_aspect_ratio": aspect_change,
                    "stretches_aspect_ratio": aspect_change,
                    "samples": aspect_reports,
                    "geometry_frozen": False,
                    "note": (
                        "CPU smoke is a 288x288 pipeline check, not a paper or training result. "
                        "Proposed AD 2 protocol is 256x256 full-frame stretch; not frozen."
                    ),
                },
                "validation_geometry": geometry,
                "normalized_finite": True,
                "normalized_shape": list(normalized.shape),
                "train_step": {k: record[k] for k in ("step", "loss", "image_ids") if k in record},
                "prediction_changed_after_step": bool(not np.array_equal(before, after_step)),
                "reload_atol": args.atol,
                "reload_match": True,
                "test_paths_read": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
