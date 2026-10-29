"""Fixed-final single-view SimpleNet on one MVTec AD category, scored after every meta-epoch.

Training repeats scripts/train_baseline.py step for step and keeps the final state. After
each meta-epoch the model scores normal validation and the test set; all RNG states are
saved before scoring and restored after it, so training is unchanged. The final epoch is
our fixed-final result; the per-epoch test trajectory shows what upstream's
best-test-epoch rule would report. It selects nothing and changes no setting.
Scoring reproduces upstream _predict without its unused full-size feature resize.
Usage: python scripts/reproduce_ad.py CATEGORY SEED OUT_DIR [ORIGINAL_RUN]
LC_DATA (default data/mvtec_ad), LC_MANIFESTS (default data/manifests, reads
ad-CATEGORY.jsonl) and LC_DEVICE (default cuda) set the inputs. LC_META_EPOCHS
overrides the 40 meta-epochs for smoke checks only.
"""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from lc_simplenet.baseline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ROOT,
    UPSTREAM_AD,
    UPSTREAM_AD_INPUT,
    UpstreamInput,
    build_model,
    center_crop_box,
    load_trained_state,
    trained_state,
    upstream_module,
)
from lc_simplenet.data import NormalManifestDataset

CATEGORY, SEED, OUT = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
ORIGINAL = Path(sys.argv[4]) if len(sys.argv) > 4 else None
DATA = Path(os.environ.get("LC_DATA", "data/mvtec_ad"))
MANIFEST = Path(os.environ.get("LC_MANIFESTS", "data/manifests")) / f"ad-{CATEGORY}.jsonl"
SHA = json.loads(MANIFEST.with_suffix(".audit.json").read_text())["manifest_sha256"]
SETTINGS = {
    **UPSTREAM_AD,
    "meta_epochs": int(os.environ.get("LC_META_EPOCHS", UPSTREAM_AD["meta_epochs"])),
}
RESIZE, CROP = UPSTREAM_AD_INPUT["resize"], UPSTREAM_AD_INPUT["imagesize"]
BOX = center_crop_box(RESIZE, CROP)
MEAN = torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
device = torch.device(os.environ.get("LC_DEVICE", "cuda"))
cuda = device.type == "cuda"


def git(path, *args):
    return subprocess.run(
        ["git", *args], cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_batches():
    """Same preprocessing as score_ad_test.py in the study release."""
    root = DATA / CATEGORY
    images, labels, masks = [], [], []
    for defect in sorted(p for p in (root / "test").iterdir() if p.is_dir()):
        for path in sorted(defect.glob("*.png")):
            with Image.open(path) as source:
                image = source.convert("RGB").resize((RESIZE, RESIZE), Image.Resampling.BILINEAR)
                images.append(np.asarray(image, dtype=np.float32) / np.float32(255))
            if defect.name == "good":
                mask = np.zeros((RESIZE, RESIZE), bool)
            else:
                with Image.open(root / "ground_truth" / defect.name / f"{path.stem}_mask.png") as m:
                    mask = (
                        np.asarray(
                            m.convert("L").resize((RESIZE, RESIZE), Image.Resampling.NEAREST)
                        )
                        > 0
                    )
            masks.append(mask[BOX[0] : BOX[2], BOX[1] : BOX[3]])
            labels.append(int(defect.name != "good"))
    pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    pixels = (pixels[:, :, BOX[0] : BOX[2], BOX[1] : BOX[3]] - MEAN) / STD
    batches = [
        {
            "image": pixels[s : s + 8],
            "image_id": [str(i) for i in range(s, min(s + 8, len(pixels)))],
        }
        for s in range(0, len(pixels), 8)
    ]
    return batches, np.array(labels), np.stack(masks)


def manifest_loader(split, shuffle):
    source = NormalManifestDataset(
        MANIFEST,
        DATA,
        dataset="mvtec_ad",
        category=CATEGORY,
        split=split,
        expected_sha256=SHA,
        output_size=(RESIZE, RESIZE),
    )
    return torch.utils.data.DataLoader(
        UpstreamInput(source, CROP),
        batch_size=UPSTREAM_AD_INPUT["batch_size"],
        shuffle=shuffle,
        num_workers=2,
    )


def predict(model, batches):
    """Upstream _predict's image scores and maps, without its unused full-size feature resize."""
    model.forward_modules.eval()
    model.pre_projection.eval()
    model.discriminator.eval()
    ids, scores, maps = [], [], []
    with torch.no_grad():
        for batch in batches:
            images = batch["image"].to(torch.float).to(device)
            size = len(images)
            features, shapes = model._embed(images, provide_patch_shapes=True, evaluation=True)
            patch = (-model.discriminator(model.pre_projection(features))).cpu().numpy()
            image_scores = model.patch_maker.unpatch_scores(patch, batchsize=size)
            image_scores = model.patch_maker.score(
                image_scores.reshape(*image_scores.shape[:2], -1)
            )
            patch = model.patch_maker.unpatch_scores(patch, batchsize=size).reshape(
                size, *shapes[0]
            )
            resized = (
                F.interpolate(
                    torch.from_numpy(patch).to(device).unsqueeze(1),
                    size=(CROP, CROP),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .cpu()
                .numpy()
            )
            ids.extend(batch["image_id"])
            scores.extend(image_scores)
            maps.extend(ndimage.gaussian_filter(m, sigma=4) for m in resized)
    return {
        "image_id": np.array(ids),
        "image_score": np.asarray(scores, dtype=np.float32),
        "map": np.stack(maps).astype(np.float32),
    }


def rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if cuda else None,
    )


def set_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if cuda:
        torch.cuda.set_rng_state_all(state[3])


def new_model():
    return build_model(upstream_module("backbones").load("wideresnet50"), device, CROP, SETTINGS)


def main():
    # Fixed inputs are prepared before seeding; the order after fix_seeds is train_baseline.py's.
    tests, labels, masks = test_batches()
    validation_batches = list(manifest_loader("validation", False))
    OUT.mkdir(parents=True, exist_ok=False)
    upstream_module("utils").fix_seeds(SEED, with_cuda=cuda)
    train, _ = manifest_loader("train", True), manifest_loader("validation", False)
    model = new_model()
    model.set_model_dir(str(OUT / "logs"), "upstream")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    rows, val_scores, test_scores, train_seconds = [], [], [], 0.0
    for epoch in range(1, model.meta_epochs + 1):
        start = time.perf_counter()
        model._train_discriminator(train)
        if cuda:
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - start
        state = rng_state()
        val = predict(model, validation_batches)
        test = predict(model, tests)
        set_rng(state)
        rows.append(
            {
                "meta_epoch": epoch,
                "val_mean": float(val["image_score"].mean()),
                "val_std": float(val["image_score"].std()),
                "val_max": float(val["image_score"].max()),
                "test_normal_mean": float(test["image_score"][labels == 0].mean()),
                "image_auroc": float(roc_auc_score(labels, test["image_score"])),
                "pixel_auroc": float(roc_auc_score(masks.ravel(), test["map"].ravel())),
            }
        )
        val_scores.append(val["image_score"])
        test_scores.append(test["image_score"])
        print(json.dumps(rows[-1]), flush=True)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None

    checkpoint = OUT / "checkpoint.pt"
    torch.save({"state": trained_state(model), "settings": SETTINGS}, checkpoint)
    geometry = {"resize_hw": [RESIZE, RESIZE], "crop_box_tlbr": BOX, "map_hw": [CROP, CROP]}
    np.savez_compressed(
        OUT / "validation_normal.npz", **val, **{k: np.asarray(v) for k, v in geometry.items()}
    )
    upstream_scores, upstream_maps = [], []
    for batch in validation_batches:
        image_scores, image_maps, _ = model._predict(batch["image"])
        upstream_scores.extend(image_scores)
        upstream_maps.extend(image_maps)
    reloaded = new_model()
    load_trained_state(reloaded, torch.load(checkpoint, map_location="cpu")["state"])
    again = predict(reloaded, validation_batches)

    best = max(rows, key=lambda r: r["image_auroc"])
    summary = {
        "scope": (
            "fixed-final upstream single-view baseline with a per-meta-epoch test trajectory; "
            "selects nothing"
        ),
        "category": CATEGORY,
        "seed": SEED,
        "settings": SETTINGS,
        "input": UPSTREAM_AD_INPUT,
        "geometry": geometry,
        "test_scoring": (
            "image AUROC of max patch score; pixel AUROC inside the crop, masks resized nearest"
        ),
        "train_images": len(train.dataset),
        "validation_images": len(val["image_id"]),
        "test_images": int(len(labels)),
        "test_defective": int(labels.sum()),
        "manifest_sha256": SHA,
        "checkpoint_sha256": sha256(checkpoint),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "git_commit": git(ROOT, "rev-parse", "HEAD"),
        "git_dirty": bool(git(ROOT, "status", "--porcelain")),
        "upstream_commit": git(ROOT / "third_party/SimpleNet", "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if cuda else "cpu",
        "train_seconds": round(train_seconds, 1),
        "peak_vram_gib_including_scoring": None if peak is None else round(peak, 2),
        "reload_max_abs_score_diff": float(np.abs(again["image_score"] - val["image_score"]).max()),
        "reload_max_abs_map_diff": float(np.abs(again["map"] - val["map"]).max()),
        "scorer_vs_upstream_predict_max_abs_diff": {
            "score": float(
                np.abs(val["image_score"] - np.asarray(upstream_scores, dtype=np.float32)).max()
            ),
            "map": float(np.abs(val["map"] - np.stack(upstream_maps).astype(np.float32)).max()),
        },
        "final": rows[-1],
        "max_test_image_auroc_epoch": best["meta_epoch"],
        "max_test_image_auroc": best["image_auroc"],
    }
    if ORIGINAL is not None:
        original = np.load(ORIGINAL / "validation_normal.npz")
        assert list(original["image_id"]) == list(val["image_id"]), "validation order differs"
        summary["final_val_max_abs_diff_vs_original"] = float(
            np.abs(val["image_score"] - original["image_score"]).max()
        )
    (OUT / "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        OUT / "scores.npz",
        validation=np.stack(val_scores),
        test=np.stack(test_scores),
        test_labels=labels,
        validation_ids=val["image_id"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
