"""Preview range corners on a training image, or an explicitly synthetic illustration."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from lc_simplenet.pairs import make_pair  # noqa: E402


def load_training_preview(path: Path) -> np.ndarray:
    """Match AD 2 training: RGB, full-frame bilinear 256x256, then [0,1]."""
    with Image.open(path) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError("Non-default EXIF orientation needs an explicit geometry rule")
        resized = source.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / np.float32(255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, help="Training image only; omit for synthetic texture"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/lighting-preview"))
    args = parser.parse_args()
    if args.image:
        image = load_training_preview(args.image)
        label = (
            "Training resize: 256x256 full-frame bilinear; inspect clipping and detail visibility"
        )
    else:
        y, x = np.mgrid[:256, :256]
        texture = 0.15 + 0.6 * x / 255 + 0.08 * np.sin(y / 3)
        texture[70:120, 93:96] *= 0.65
        image = np.repeat(texture[..., None], 3, axis=2).astype(np.float32)
        label = "Synthetic texture illustration only; not a dataset or defect-detection result"
    settings = [(1, 1), (0.8, 0.8), (0.8, 1.2), (1.2, 0.8), (1.2, 1.2)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    records = []
    for ax, (brightness, gamma) in zip(axes, settings, strict=True):
        pair = make_pair(image, brightness=brightness, gamma=gamma)
        ax.imshow(pair.relit)
        title = "Original" if (brightness, gamma) == (1, 1) else f"b={brightness}, gamma={gamma}"
        ax.set_title(f"{title}\nClipped channels: {pair.clipped_fraction:.1%}")
        ax.axis("off")
        records.append(
            {"brightness": brightness, "gamma": gamma, "clipped_fraction": pair.clipped_fraction}
        )
    fig.suptitle(label)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "lighting_preview.png", dpi=150)
    plt.close(fig)
    (args.output_dir / "parameters.json").write_text(
        json.dumps(
            {
                "scope": label,
                "input_size": list(image.shape[:2]),
                "preprocessing": "RGB, full-frame bilinear resize, no crop"
                if args.image
                else "synthetic texture generated at 256x256",
                "settings": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output_dir / "lighting_preview.png")


if __name__ == "__main__":
    main()
