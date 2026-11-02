"""Score an AD 2 category's public test images with a finished run's final head.

Writes `test-scores.json`, `test-maps.npz` and `test-scores.meta.json` into the run directory
(which must hold `run.json` and `final.pt`, or the release's `heads.pt`); existing files are
never overwritten. The run's saved validation scores are re-derived for a few images first, as a
guard against a wrong geometry, score direction or checkpoint.

    python scripts/score_public_test.py --run-dir results/ad2-full-study/can/B1/seed-0 \
        --root data/mvtec_ad_2 --device cuda
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from lc_simplenet.pilot import PilotConfig, UpstreamBackend
from lc_simplenet.public_scoring import (
    condition_from_name,
    list_public_test,
    read_resized,
    score_records,
    smooth_map,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAMES = ("final.pt", "heads.pt")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_from(raw: dict) -> PilotConfig:
    raw = dict(raw)
    for key in ("image_size", "brightness", "gamma"):
        raw[key] = tuple(raw[key])
    return PilotConfig(**raw)


def checkpoint_path(run_dir: Path) -> Path:
    """`final.pt` from our own training runs, `heads.pt` from the released study."""
    for name in CHECKPOINT_NAMES:
        if (run_dir / name).exists():
            return run_dir / name
    raise FileNotFoundError(f"No {' or '.join(CHECKPOINT_NAMES)} in {run_dir}")


def load_head(run_dir: Path, device: torch.device):
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = config_from(run["config"])
    if config.dataset != "mvtec_ad2" or config.ad_resize is not None:
        raise ValueError("Public-test scoring supports full-frame AD 2 runs only")
    backend = UpstreamBackend(config, device)
    checkpoint = checkpoint_path(run_dir)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if config_from(state["config"]) != config:
        raise ValueError(f"{checkpoint.name} configuration differs from run.json")
    if state["encoder_hash"] != backend.encoder_hash:
        raise ValueError("Frozen encoder differs from the checkpoint")
    backend.adapter.load_state_dict(state["adapter"])
    backend.discriminator.load_state_dict(state["discriminator"])
    backend.adapter.eval()
    backend.discriminator.eval()

    def predict(images):
        pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().to(device)
        mean = pixels.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = pixels.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        with torch.inference_mode():
            features, grids = backend.embed((pixels - mean) / std)
            scores = -backend.discriminator(backend.adapter(features))
            return scores.reshape(len(images), *grids[0]).cpu().numpy()

    return config, run, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="MVTec AD 2 dataset root")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-validation", type=int, default=3)
    parser.add_argument("--conditions-out", type=Path, help="write image_id -> condition JSON")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    outputs = [run_dir / n for n in ("test-scores.json", "test-maps.npz", "test-scores.meta.json")]
    if any(path.exists() for path in outputs):
        parser.error("refusing to overwrite existing test-score outputs")
    torch.set_num_threads(args.threads)
    config, run, predict = load_head(run_dir, torch.device(args.device))

    saved = json.loads((run_dir / "validation-scores.json").read_text(encoding="utf-8"))
    for row in saved[: args.check_validation]:
        pixels, _ = read_resized(args.root / row["image_id"], config.load_size)
        again = float(predict([pixels])[0].max())
        if not np.isclose(again, row["score"], atol=1e-3):
            raise SystemExit(f"validation re-score mismatch for {row['image_id']}: {again}")

    images = list_public_test(args.root, config.category)
    rows, patch_maps = score_records(images, predict, config.load_size, config.image_size)
    maps = np.stack([smooth_map(m, config.image_size) for m in patch_maps])
    ids_hash = hashlib.sha256("\n".join(i.image_id for i in images).encode()).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()

    (run_dir / "test-scores.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        run_dir / "test-maps.npz", maps=maps, image_ids=[r["image_id"] for r in rows]
    )
    meta = {
        "scope": "AD 2 test_public scoring at the fixed-final head; no selection from these scores",
        "category": config.category,
        "variant": config.variant,
        "lambda_score": config.lambda_score,
        "seed": config.seed,
        "images": len(rows),
        "normal": sum(r["label"] == 0 for r in rows),
        "defective": sum(r["label"] == 1 for r in rows),
        "image_ids_sha256": ids_hash,
        "checkpoint_file": checkpoint_path(run_dir).name,
        "checkpoint_sha256": sha256(checkpoint_path(run_dir)),
        "run_config_sha256": run.get("config_sha256"),
        "validation_rescored": args.check_validation,
        "map": "bilinear to input size, Gaussian sigma 4; coordinates full_frame at image_size",
        "score": "max raw -D patch score",
        "scoring_git_commit": commit,
        "torch": torch.__version__,
        "device": args.device,
    }
    (run_dir / "test-scores.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.conditions_out:
        conditions = {r["image_id"]: condition_from_name(r["image_id"]) for r in rows}
        args.conditions_out.write_text(json.dumps(conditions, indent=2) + "\n")
    print(f"scored {len(rows)} public test images for {config.category}/{config.variant}")


if __name__ == "__main__":
    main()
