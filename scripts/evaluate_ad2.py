"""Local public-test scoring for exploratory AD 2 fabric checkpoints.

`fabric/test_public` contains labelled public test data. The previous 100-step
B0/B1/B2 exploratory checkpoints were evaluated on this split. Pooled
lighting-condition results cannot establish lighting robustness.

This script records image AUROC and pixel AUROC at 256×256. Pixel AUROC is not
AU-PRO. These are not official AD 2 benchmark results. Do not use the scores to
select models, thresholds, or hyperparameters. Further formal evaluation is
coordinated with Aryaman.

Default paths follow `scripts/run_ad2_fabric_batch.sh`: the 5,000-step
exploratory recovery runs. The 100-step exploratory runs are a separate cohort
and must not be confused with those checkpoints or with the formal AD 2 study
policy in PR #8.
"""

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from lc_simplenet.data import NormalManifestDataset
from lc_simplenet.pilot import ROOT, PilotConfig, PilotTrainer, config_hash

DATA = ROOT / "data/mvtec_ad2"
MANIFEST = ROOT / "data/manifests/ad2-fabric.jsonl"
PUBLIC = DATA / "fabric/test_public"

PUBLIC_TEST_DISCLOSURE = (
    "test_public contains labelled public test data. "
    "The previous 100-step B0/B1/B2 exploratory checkpoints were evaluated on "
    "this data. Pooled lighting-condition results cannot establish lighting "
    "robustness. pixel_auroc_at_256 is pixel AUROC on the 256x256 resized maps; "
    "it is not AU-PRO. These are not official AD 2 benchmark results. Further "
    "formal evaluation is coordinated with Aryaman."
)

EXPLORATORY_100_STEP_RUNS = {
    "B0": ("results/ad2_fabric_b0_config.json", "results/ad2_fabric_b0_100steps"),
    "B1": ("results/ad2_fabric_b1_config.json", "results/ad2_fabric_b1_100steps"),
    "B2": ("results/ad2_fabric_b2_config.json", "results/ad2_fabric_b2_100steps"),
}

EXPLORATORY_5000_STEP_RUNS = {
    "B0": ("results/ad2_fabric_b0_config.json", "results/ad2_fabric_b0_5000steps_recovery"),
    "B1": ("results/ad2_fabric_b1_config.json", "results/ad2_fabric_b1_5000steps_recovery"),
    "B2": ("results/ad2_fabric_b2_config.json", "results/ad2_fabric_b2_5000steps_recovery"),
}

COHORTS = {
    "100steps": {
        "kind": "exploratory_100_step",
        "steps": 100,
        "runs": EXPLORATORY_100_STEP_RUNS,
        "output": "results/ad2_fabric_public_evaluation",
        "note": (
            "Previous 100-step exploratory fabric pilots. Distinct from the "
            "5,000-step recovery runs and from the formal AD 2 study in PR #8."
        ),
    },
    "5000steps_recovery": {
        "kind": "exploratory_5000_step",
        "steps": 5000,
        "runs": EXPLORATORY_5000_STEP_RUNS,
        "output": "results/ad2_fabric_5000steps_recovery_public_evaluation",
        "note": (
            "Exploratory 5,000-step fabric recovery runs from "
            "scripts/run_ad2_fabric_batch.sh. Distinct from the 100-step "
            "exploratory checkpoints and from the formal AD 2 study in PR #8."
        ),
    },
}

DEFAULT_COHORT = "5000steps_recovery"
SIZE = (256, 256)


def jsonable(value):
    return json.loads(json.dumps(value))


def cohort_metadata(cohort_id: str) -> dict:
    if cohort_id not in COHORTS:
        raise KeyError(f"unknown evaluation cohort: {cohort_id}")
    cohort = COHORTS[cohort_id]
    return {
        "scope": "Local public-test metrics; not official AD2 benchmark scores",
        "exploratory": True,
        "public_test_disclosure": PUBLIC_TEST_DISCLOSURE,
        "exploratory_kind": cohort["kind"],
        "exploratory_steps": cohort["steps"],
        "exploratory_note": cohort["note"],
        "official_ad2_benchmark": False,
        "metric_note": "pixel_auroc_at_256 is pixel AUROC, not AU-PRO",
        "lighting_note": (
            "Pooled lighting-condition results cannot establish lighting robustness."
        ),
        "formal_evaluation": "Further formal evaluation is coordinated with Aryaman.",
    }


def coerce_pilot_config(raw: dict) -> PilotConfig:
    if type(raw) is not dict:
        raise TypeError("checkpoint configuration must be an object")
    data = dict(raw)
    for key in ("image_size", "brightness", "gamma"):
        if key in data and data[key] is not None:
            data[key] = tuple(data[key])
    return PilotConfig(**data)


def config_file_matches_checkpoint(config_file: Path, config: PilotConfig) -> bool:
    if not config_file.is_file():
        return False
    try:
        raw = json.loads(config_file.read_text(encoding="utf-8"))
        file_config = coerce_pilot_config(raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return jsonable(asdict(file_config)) == jsonable(asdict(config))


def checkpoint_provenance(saved_config: dict, *, variant: str, config_file: Path) -> dict:
    config = coerce_pilot_config(saved_config)
    if config.variant != variant:
        raise ValueError(f"{variant}: checkpoint variant is {config.variant}")
    return {
        "checkpoint_config": jsonable(asdict(config)),
        "checkpoint_config_sha256": config_hash(config),
        "config_file": str(config_file),
        "config_file_matches_checkpoint": config_file_matches_checkpoint(config_file, config),
    }


def resolve_run_inputs(variant: str, config_file: str, run_dir: str, *, root: Path) -> dict:
    config_path = (root / config_file).resolve()
    checkpoint = (root / run_dir / "final.pt").resolve()
    result_file = (root / run_dir / "result.json").resolve()
    missing = []
    if not config_path.is_file():
        missing.append(f"config {config_path}")
    if not checkpoint.is_file():
        missing.append(f"checkpoint {checkpoint}")
    if not result_file.is_file():
        missing.append(f"result {result_file}")
    if missing:
        raise FileNotFoundError(f"{variant}: missing {', '.join(missing)}")
    return {
        "variant": variant,
        "config_file": config_file,
        "run_dir": run_dir,
        "config_path": config_path,
        "checkpoint": checkpoint,
        "result_file": result_file,
    }


def existing_evaluation_outputs(output: Path, variants: list[str]) -> list[Path]:
    existing = []
    comparison = output / "comparison.json"
    if comparison.exists():
        existing.append(comparison)
    for variant in variants:
        for name in ("evaluation.json", "per_image.csv"):
            path = output / variant / name
            if path.exists():
                existing.append(path)
    return existing


def refuse_overwrite(output: Path, variants: list[str]) -> None:
    existing = existing_evaluation_outputs(output, variants)
    if existing:
        raise FileExistsError(f"refusing to overwrite existing evaluation output: {existing[0]}")


def write_variant_outputs(folder: Path, summary: dict, rows: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (folder / "per_image.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_row(summary: dict) -> dict:
    return {
        "variant": summary["variant"],
        "exploratory": summary.get("exploratory", True),
        "exploratory_kind": summary["exploratory_kind"],
        "exploratory_steps": summary["exploratory_steps"],
        "official_ad2_benchmark": summary["official_ad2_benchmark"],
        **summary["metrics"],
    }


def publish_directory(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        else:
            raise FileExistsError(f"refusing to overwrite existing evaluation output: {output}")
    staging.replace(output)


def evaluate_cohort(
    cohort_id: str,
    *,
    output: Path | None = None,
    root: Path | None = None,
    evaluate_one=None,
) -> list[dict]:
    if cohort_id not in COHORTS:
        raise KeyError(f"unknown evaluation cohort: {cohort_id}")
    cohort = COHORTS[cohort_id]
    root = ROOT if root is None else root
    output = (root / cohort["output"]).resolve() if output is None else output.resolve()
    variants = list(cohort["runs"])
    refuse_overwrite(output, variants)
    resolved_runs = [
        resolve_run_inputs(variant, config_file, run_dir, root=root)
        for variant, (config_file, run_dir) in cohort["runs"].items()
    ]
    evaluate_one = score_public_test if evaluate_one is None else evaluate_one
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
    )
    published = False
    try:
        results = []
        for resolved in resolved_runs:
            summary, rows = evaluate_one(resolved, cohort_id=cohort_id)
            write_variant_outputs(staging / resolved["variant"], summary, rows)
            results.append(summary)
        (staging / "comparison.json").write_text(
            json.dumps([comparison_row(row) for row in results], indent=2) + "\n",
            encoding="utf-8",
        )
        publish_directory(staging, output)
        published = True
        return results
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def read_image(path):
    with Image.open(path) as im:
        native_size = (im.height, im.width)
        im = im.convert("RGB").resize((SIZE[1], SIZE[0]), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0, native_size


def read_mask(path):
    with Image.open(path) as im:
        im = im.convert("L").resize((SIZE[1], SIZE[0]), Image.Resampling.NEAREST)
        return np.asarray(im) > 0


def score_public_test(resolved: dict, *, cohort_id: str) -> tuple[dict, list[dict]]:
    variant = resolved["variant"]
    checkpoint = resolved["checkpoint"]
    result_file = resolved["result_file"]
    config_path = resolved["config_path"]

    # Build the model from the exact configuration stored in the checkpoint.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = checkpoint_provenance(
        saved["config"],
        variant=variant,
        config_file=config_path,
    )
    config = coerce_pilot_config(saved["config"])
    if tuple(config.image_size) != SIZE:
        raise ValueError(f"{variant}: unexpected input size")

    train = NormalManifestDataset(
        MANIFEST,
        DATA,
        dataset="mvtec_ad2",
        category="fabric",
        split="train",
        expected_sha256=config.manifest_sha256,
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

    threshold = float(json.loads(result_file.read_text())["calibration"]["threshold"])

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
        upsampled = (
            F.interpolate(tensor, size=SIZE, mode="bilinear", align_corners=False)[0, 0]
            .cpu()
            .numpy()
        )
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

        rows.append(
            {
                "image": path.name,
                "label": label,
                "image_score": image_score,
                "threshold": threshold,
                "predicted_anomaly": int(image_score > threshold),
                "mask": f"{path.stem}_mask.png" if label else "",
            }
        )

    scores = np.asarray(image_scores)
    labels = np.asarray(image_labels)
    predicted = scores > threshold
    normal, defect = labels == 0, labels == 1
    py = np.concatenate(pixel_labels)
    ps = np.concatenate(pixel_scores)

    summary = {
        **cohort_metadata(cohort_id),
        "variant": variant,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        **provenance,
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
    print(
        f"{variant}: image AUROC={summary['metrics']['image_auroc']:.4f}, "
        f"pixel AUROC@256={summary['metrics']['pixel_auroc_at_256']:.4f}, "
        f"FPR={summary['metrics']['image_fpr_at_threshold']:.4f}, "
        f"recall={summary['metrics']['image_recall_at_threshold']:.4f}"
    )
    return summary, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        choices=sorted(COHORTS),
        default=DEFAULT_COHORT,
        help="Exploratory run set to evaluate; default is the 5,000-step recovery runs",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this on your allocated GPU node.")

    output = ROOT / COHORTS[args.cohort]["output"]
    results = evaluate_cohort(args.cohort, output=output)
    print(f"\nSaved results to: {output}")
    print("Private test folders were not accessed.")
    return results


if __name__ == "__main__":
    main()
