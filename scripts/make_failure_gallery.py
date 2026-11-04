"""Failure gallery from a rule-based selection (score rank), one PNG per category and method.

Each panel: the image, the saved input-size map shown with a per-panel display scale (display only,
never used for any metric), and for defective images the ground-truth mask outline when found.
Selection comes from select_failure_gallery.py (predictions.csv, seed 0, score rank). Usage:
python make_failure_gallery.py <dataset root> <runs dir> <out dir> <selection json>
where <runs dir> holds <cat>-<method>-seed0/test-maps.npz.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root = Path(sys.argv[1])
release = Path(sys.argv[2])
out = Path(sys.argv[3])
selection = json.loads(Path(sys.argv[4]).read_text())
out.mkdir(parents=True, exist_ok=True)

groups = {}
for item in selection:
    groups.setdefault((item["category"], item["method"]), []).append(item)

for (category, method), items in sorted(groups.items()):
    data = np.load(release / f"{category}-{method}-seed0" / "test-maps.npz")
    index = {str(i): k for k, i in enumerate(data["image_ids"])}
    maps = data["maps"]
    fig, axes = plt.subplots(2, len(items), figsize=(3.2 * len(items), 6.6), squeeze=False)
    for col, item in enumerate(items):
        path = root / item["image_id"]
        image = Image.open(path).convert("RGB")
        amap = maps[index[item["image_id"]]]
        axes[0][col].imshow(image)
        axes[0][col].set_title(
            f"{item['kind'].replace('_', ' ')} #{item['rank']}\n{item['condition']}", fontsize=8
        )
        shown = axes[1][col].imshow(amap, cmap="inferno")
        axes[1][col].set_title(
            f"score {item['score']:.2f} vs threshold {item['threshold']:.2f}", fontsize=8
        )
        parts = path.parts
        mask_path = Path(*parts[:-2]) / "ground_truth" / parts[-2] / (path.stem + "_mask.png")
        if item["kind"] == "false_negative" and mask_path.exists():
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            axes[0][col].contour(mask, levels=[0.5], colors="cyan", linewidths=0.8)
        for row in (0, 1):
            axes[row][col].axis("off")
    fig.suptitle(
        f"{category}, {method}, seed 0 (threshold from regular-light normal validation)", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out / f"{category}-{method}.png", dpi=110)
    plt.close(fig)
    print("wrote", category, method)
