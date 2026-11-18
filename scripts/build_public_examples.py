"""Draw fixed qualitative cases, never choose examples by a model's success."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lc_simplenet.evaluate_runs import discover_runs
from lc_simplenet.evaluation import calibrate, load_scores


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(runs, root, protocol, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frozen = json.loads(protocol.read_text(encoding="utf-8-sig"))
    categories, methods = sorted(frozen["categories"]), frozen["methods"]
    seed = min(frozen["seeds"])
    entries = {(r.category, r.method): r for r in discover_runs(runs) if r.seed == seed}
    if any(
        (c, m) not in entries or not (entries[c, m].path / "test-scores.json").exists()
        for c in categories
        for m in methods
    ):
        raise ValueError(
            "fixed examples require every declared category and method at the first seed"
        )
    manifest = {
        "scope": "post-training qualitative inspection; no example chosen by prediction outcome",
        "rule": "first sorted regular-light defective public image in every category; first seed",
        "display": "native mask bounding box plus 10% padding; all maps use the same crop",
        "scale": "fixed per-head validation-patch 2nd/98th percentiles; no per-test normalization",
        "decision": "whole-image raw score > frozen normal-validation threshold",
        "seed": seed,
        "protocol_sha256": digest(protocol),
        "cases": [],
    }
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18})
    panels = []
    for category in categories:
        reference_rows = load_scores(entries[category, methods[0]].path / "test-scores.json")
        chosen = next(
            r
            for r in sorted(reference_rows, key=lambda r: r["image_id"])
            if r["label"] == 1 and Path(r["image_id"]).stem.endswith("_regular")
        )
        image_path = root / chosen["image_id"]
        mask_path = (
            root
            / category
            / "test_public"
            / "ground_truth"
            / "bad"
            / (image_path.stem + "_mask.png")
        )
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        with Image.open(mask_path) as image:
            if image.size != rgb.size:
                raise ValueError("example mask and original image have different native geometry")
            mask = np.asarray(image)
        if not np.isin(mask, [0, 255]).all() or not (mask == 255).any():
            raise ValueError("expected unchanged nonempty native 0/255 defect mask")
        yy, xx = np.nonzero(mask)
        padding = max(16, int(0.1 * max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1)))
        box = [
            max(0, int(xx.min()) - padding),
            max(0, int(yy.min()) - padding),
            min(rgb.width, int(xx.max()) + padding + 1),
            min(rgb.height, int(yy.max()) + padding + 1),
        ]
        thumb = np.asarray(rgb.crop(box).resize((256, 256), Image.Resampling.BILINEAR))
        mask_view = np.asarray(
            Image.fromarray(mask).crop(box).resize((256, 256), Image.Resampling.NEAREST)
        )
        displays = [(thumb, "input crop", None), (mask_view, "ground truth", (0, 255))]
        case = {
            "category": category,
            "image_id": chosen["image_id"],
            "crop_xyxy": box,
            "image_sha256": digest(image_path),
            "mask_sha256": digest(mask_path),
            "models": [],
        }
        for method in methods:
            entry = entries[category, method]
            rows = load_scores(entry.path / "test-scores.json")
            index = next(i for i, r in enumerate(rows) if r["image_id"] == chosen["image_id"])
            with np.load(entry.path / "test-maps.npz", allow_pickle=False) as archive:
                if archive["image_ids"].tolist() != [r["image_id"] for r in rows]:
                    raise ValueError("example score and map IDs differ")
                score_map = archive["maps"][index]
            native = Image.fromarray(score_map).resize(rgb.size, Image.Resampling.BILINEAR)
            display = np.asarray(native.crop(box).resize((256, 256), Image.Resampling.BILINEAR))
            with np.load(entry.path / "validation-patch-maps.npz", allow_pickle=False) as archive:
                limits = np.percentile(archive["scores"], [2, 98]).tolist()
            if limits[1] <= limits[0]:
                limits[1] = limits[0] + 1e-6
            threshold = calibrate(load_scores(entry.path / "validation-scores.json"), 0.05)[
                "threshold"
            ]
            flagged = rows[index]["score"] > threshold
            displays.append((display, method + ("\ndetected" if flagged else "\nmissed"), limits))
            case["models"].append(
                {
                    "method": method,
                    "score": rows[index]["score"],
                    "threshold": threshold,
                    "flagged": flagged,
                    "heatmap_limits": limits,
                    "maps_sha256": digest(entry.path / "test-maps.npz"),
                    "scores_sha256": digest(entry.path / "test-scores.json"),
                    "validation_patches_sha256": digest(entry.path / "validation-patch-maps.npz"),
                }
            )
        panels.append(displays)
        manifest["cases"].append(case)
    for count, suffix in [(len(categories), "all"), (min(2, len(categories)), "first-two")]:
        figure, axes = plt.subplots(
            count, len(methods) + 2, figsize=(12.5, 2.25 * count), squeeze=False
        )
        for i in range(count):
            for j, (data, title, limits) in enumerate(panels[i]):
                axis = axes[i, j]
                axis.imshow(
                    data,
                    cmap="gray" if j == 1 else "magma",
                    vmin=limits[0] if limits else None,
                    vmax=limits[1] if limits else None,
                )
                axis.set_title(title, fontsize=16, pad=5)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                if j == 0:
                    axis.set_ylabel(categories[i].replace("_", " "), fontsize=18)
        figure.tight_layout(pad=0.25, h_pad=0.8, w_pad=0.2)
        figure.savefig(
            out / f"ad2-public-examples-{suffix}.png", dpi=180, bbox_inches="tight", pad_inches=0.1
        )
        plt.close(figure)
    (out / "public-examples.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"built {len(categories)} fixed examples, all {len(methods)} methods; no outcome selection"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/report-evidence"))
    args = parser.parse_args()
    build(args.runs, args.root, args.protocol, args.out)


if __name__ == "__main__":
    main()
