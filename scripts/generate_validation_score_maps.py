#!/usr/bin/env python3
"""Write regular-light and seven-relight anomaly maps for one released head.

Reads ``heads.pt``, ``run.json`` and ``validation-scores.json`` from a formal
256 study B0 or B1 run. Images come only from the ids in that score file, and
each id must be ``<category>/validation/good/<file>``. The npz is written
outside the run directory. Released files are not modified.

    python scripts/generate_validation_score_maps.py \
        --run-dir results/study-artifacts/paired/can-B0-seed0 \
        --root data/mvtec_ad_2 --out results/score-field-maps/can-B0-seed0.npz
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lc_simplenet.pilot import PilotConfig, UpstreamBackend, seed_runtime
from lc_simplenet.score_field import refuse_test_path, validation_image_id
from lc_simplenet.validation_maps import (
    assert_formal_study_head,
    assert_output_outside_run,
    generate_validation_maps,
    save_validation_maps,
)


def config_from(raw: dict) -> PilotConfig:
    raw = dict(raw)
    for key in ("image_size", "brightness", "gamma"):
        raw[key] = tuple(raw[key])
    return PilotConfig(**raw)


def load_study_head(run_dir: Path, device: torch.device):
    """Load ``heads.pt`` only. A local ``final.pt`` is not a released study head."""
    refuse_test_path(run_dir.as_posix())
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = config_from(run["config"])
    assert_formal_study_head(config)
    head = run_dir / "heads.pt"
    if not head.is_file():
        raise FileNotFoundError(f"Formal study heads.pt is required in {run_dir}")
    seed_runtime(config.seed)
    backend = UpstreamBackend(config, device)
    state = torch.load(head, map_location=device, weights_only=True)
    if config_from(state["config"]) != config:
        raise ValueError("heads.pt configuration differs from run.json")
    assert_formal_study_head(config_from(state["config"]))
    if state["encoder_hash"] != backend.encoder_hash:
        raise ValueError("Frozen encoder differs from the checkpoint")
    backend.adapter.load_state_dict(state["adapter"])
    backend.discriminator.load_state_dict(state["discriminator"])
    backend.adapter.eval()
    backend.discriminator.eval()

    def predict(images):
        pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
        pixels = pixels.to(device)
        mean = pixels.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = pixels.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        with torch.inference_mode():
            features, grids = backend.embed((pixels - mean) / std)
            scores = -backend.discriminator(backend.adapter(features))
            return scores.reshape(len(images), *grids[0]).cpu().numpy()

    return config, predict


def read_validation_image(root: Path, image_id: str) -> np.ndarray:
    """One normal-validation image, full-frame bilinear resize to 256×256."""
    category = image_id.replace("\\", "/").split("/", 1)[0]
    validation_image_id(category, image_id)
    root = Path(root).resolve(strict=True)
    path = (root / image_id).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Image escapes the dataset root")
    expected = (root / category / "validation" / "good").resolve()
    if path.parent != expected:
        raise ValueError("Image is not in the normal-validation directory")
    with Image.open(io.BytesIO(path.read_bytes())) as source:
        if getattr(source, "n_frames", 1) != 1 or source.getexif().get(274, 1) != 1:
            raise ValueError(f"{image_id}: multi-frame or oriented image needs an explicit rule")
        image = source.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / np.float32(255)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="MVTec AD 2 dataset root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    assert_output_outside_run(run_dir, args.out)
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite a validation-map file")
    rows = json.loads((run_dir / "validation-scores.json").read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    config, predict = load_study_head(run_dir, torch.device(args.device))
    images = [read_validation_image(args.root, row["image_id"]) for row in rows]
    payload = generate_validation_maps(
        config.category, config.seed, config.variant, rows, images, predict
    )
    save_validation_maps(args.out, payload)
    print(
        f"wrote {len(rows)} validation maps for "
        f"{config.category}/{config.variant}/seed-{config.seed}"
    )


if __name__ == "__main__":
    main()
