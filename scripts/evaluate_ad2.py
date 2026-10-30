import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from lc_simplenet.data import NormalManifestDataset  # noqa: E402
from lc_simplenet.pilot import PilotConfig, PilotTrainer  # noqa: E402

DATA = ROOT / "data/mvtec_ad2"
MANIFEST = ROOT / "data/manifests/ad2-fabric.jsonl"
PUBLIC = DATA / "fabric/test_public"
OUT = ROOT / "results/ad2_fabric_public_evaluation"

RUNS = {
    "B0": ("results/ad2_fabric_b0_config.json", "results/ad2_fabric_b0_100steps"),
    "B1": ("results/ad2_fabric_b1_config.json", "results/ad2_fabric_b1_100steps"),
    "B2": ("results/ad2_fabric_b2_config.json", "results/ad2_fabric_b2_100steps"),
}
SIZE = (256, 256)


def read_image(path):
    with Image.open(path) as im:
        native_size = (im.height, im.width)
        im = im.convert("RGB").resize(
            (SIZE[1], SIZE[0]), Image.Resampling.BILINEAR
        )
        return np.asarray(im, dtype=np.float32) / 255.0, native_size


def read_mask(path):
    with Image.open(path) as im:
        im = im.convert("L").resize(
            (SIZE[1], SIZE[0]), Image.Resampling.NEAREST
        )
        return np.asarray(im) > 0


def evaluate(variant, config_file, run_dir):
    run = ROOT / run_dir
    checkpoint = run / "final.pt"
    result_file = run / "result.json"

    # Build the model from the exact configuration stored in the checkpoint.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = PilotConfig(**saved["config"])
    if config.variant != variant or tuple(config.image_size) != SIZE:
        raise ValueError(f"{variant}: unexpected variant or input size")

    train = NormalManifestDataset(
        MANIFEST, DATA, dataset="mvtec_ad2", category="fabric",
        split="train", expected_sha256=config.manifest_sha256,
        output_size=config.load_size,
    )
    model = PilotTrainer(config, train, device="cuda")
    saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
    if saved["schema"] != 1 or saved["config"] != model.checkpoint()["config"]:
        raise ValueError(f"{variant}: checkpoint configuration mismatch")
    if saved["encoder_hash"] != model.backend.encoder_hash:
        raise ValueError(f"{variant}: frozen encoder hash mismatch")
    model.backend.adapter.load_state_dict(saved["adapter"])
    model.backend.discriminator.load_state_dict(saved["discriminator"])
    model.step = saved["step"]

    threshold = float(
        json.loads(result_file.read_text())["calibration"]["threshold"]
    )

    images = [(p, 0) for p in sorted((PUBLIC / "good").glob("*.png"))]
    images += [(p, 1) for p in sorted((PUBLIC / "bad").glob("*.png"))]
    if len(images) != 156:
        raise ValueError(f"Expected 156 public images, found {len(images)}")

    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []
    rows = []

    for path, label in images:
        pixels, _ = read_image(path)
        patch_map = model.predict([pixels])[0]
        image_score = float(np.max(patch_map))

        # Upsample patch scores to 256x256, then smooth consistently.
        tensor = torch.as_tensor(patch_map, dtype=torch.float32)[None, None]
        upsampled = F.interpolate(
            tensor, size=SIZE, mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        score_map = gaussian_filter(upsampled, sigma=4.0)

        if label:
            mask_path = PUBLIC / "ground_truth" / "bad" / f"{path.stem}_mask.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {path.name}")
            mask = read_mask(mask_path)
        else:
            mask = np.zeros(SIZE, dtype=bool)

        pixel_scores.append(score_map.reshape(-1))
        pixel_labels.append(mask.astype(np.uint8).reshape(-1))
        image_scores.append(image_score)
        image_labels.append(label)

        rows.append({
            "image": path.name,
            "label": label,
            "image_score": image_score,
            "threshold": threshold,
            "predicted_anomaly": int(image_score > threshold),
            "mask": f"{path.stem}_mask.png" if label else "",
        })

    scores = np.asarray(image_scores)
    labels = np.asarray(image_labels)
    predicted = scores > threshold
    normal, defect = labels == 0, labels == 1
    py = np.concatenate(pixel_labels)
    ps = np.concatenate(pixel_scores)

    summary = {
        "scope": "Local public-test metrics; not official AD2 benchmark scores",
        "variant": variant,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "input_size": list(SIZE),
        "counts": {
            "normal": int(normal.sum()),
            "defective": int(defect.sum()),
            "false_positives": int((predicted & normal).sum()),
            "true_positives": int((predicted & defect).sum()),
        },
        "metrics": {
            "image_auroc": float(roc_auc_score(labels, scores)),
            "pixel_auroc_at_256": float(roc_auc_score(py, ps)),
            "validation_threshold": threshold,
            "image_fpr_at_threshold": float((predicted & normal).sum() / normal.sum()),
            "image_recall_at_threshold": float((predicted & defect).sum() / defect.sum()),
        },
    }

    folder = OUT / variant
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (folder / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{variant}: image AUROC={summary['metrics']['image_auroc']:.4f}, "
        f"pixel AUROC@256={summary['metrics']['pixel_auroc_at_256']:.4f}, "
        f"FPR={summary['metrics']['image_fpr_at_threshold']:.4f}, "
        f"recall={summary['metrics']['image_recall_at_threshold']:.4f}"
    )
    return summary


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this on your allocated GPU node.")

    OUT.mkdir(parents=True, exist_ok=True)
    results = [
        evaluate(name, config, run)
        for name, (config, run) in RUNS.items()
    ]
    comparison = [
        {
            "variant": r["variant"],
            **r["metrics"],
        }
        for r in results
    ]
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"\nSaved results to: {OUT}")
    print("Private test folders were not accessed.")


if __name__ == "__main__":
    main()
