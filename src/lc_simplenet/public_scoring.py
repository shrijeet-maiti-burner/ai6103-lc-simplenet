"""Score AD 2 `test_public` images with a frozen trained head, for role D's evaluation.

Only `<category>/test_public/{good,bad}` is read; private folders are never listed. Labels come
from the official folder names. Image scores use the same rule as validation scoring: the maximum
of the raw `-D` patch map. Nothing here chooses a checkpoint, weight or threshold: the head is the
fixed-final checkpoint of an already finished run, and the threshold is applied later by
`evaluate_runs` from normal validation scores.
"""

import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PUBLIC_DIR = "test_public"
LABEL_FOLDERS = {"good": 0, "bad": 1}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
CONDITION_PATTERN = re.compile(r"\d+_(regular|overexposed|underexposed|shift_\d+)")


@dataclass(frozen=True)
class PublicImage:
    image_id: str
    label: int
    path: Path


def list_public_test(root: Path, category: str) -> list[PublicImage]:
    """Official public test images in sorted order; no other test folder is touched."""
    root = Path(root).resolve(strict=True)
    base = root / category / PUBLIC_DIR
    images = []
    for folder, label in LABEL_FOLDERS.items():
        directory = base / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {directory}")
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError("Test image escapes the dataset root")
            images.append(PublicImage(resolved.relative_to(root).as_posix(), label, resolved))
    ids = [image.image_id for image in images]
    if not images or len(set(ids)) != len(ids):
        raise ValueError(f"{category}: no public test images or duplicate ids")
    return images


def condition_from_name(image_id: str) -> str:
    """Lighting condition from the AD 2 file-name suffix, e.g. `003_shift_1.png`.

    This relies on the naming seen in the audited archives (regular, overexposed, underexposed,
    shift_N). Verify it against the dataset documentation before reporting per-condition
    results; an unrecognised name raises instead of guessing a group.
    """
    match = CONDITION_PATTERN.fullmatch(Path(image_id).stem)
    if not match:
        raise ValueError(f"No documented lighting condition in file name: {image_id}")
    return match.group(1)


def read_resized(path: Path, load_size: tuple[int, int]) -> tuple[np.ndarray, list[int]]:
    """RGB float32 in [0, 1], full-frame PIL bilinear resize, as for training and validation."""
    with Image.open(io.BytesIO(Path(path).read_bytes())) as source:
        if getattr(source, "n_frames", 1) != 1 or source.getexif().get(274, 1) != 1:
            raise ValueError(f"{path}: multi-frame or oriented image needs an explicit rule")
        original = [source.height, source.width]
        image = source.convert("RGB").resize(
            (load_size[1], load_size[0]), Image.Resampling.BILINEAR
        )
    return np.asarray(image, dtype=np.float32) / np.float32(255), original


def score_records(
    images: Sequence[PublicImage],
    predict: Callable[[list[np.ndarray]], np.ndarray],
    load_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[list[dict], np.ndarray]:
    """One image at a time (as validation did): image score is the max raw patch score."""
    rows, maps = [], []
    for image in images:
        pixels, original = read_resized(image.path, load_size)
        patch_map = np.asarray(predict([pixels])[0], dtype=np.float32)
        if not np.isfinite(patch_map).all():
            raise FloatingPointError(f"Non-finite prediction for {image.image_id}")
        maps.append(patch_map)
        rows.append(
            {
                "image_id": image.image_id,
                "label": image.label,
                "score": float(patch_map.max()),
                "original_size": original,
                "resized_size": list(load_size),
                "image_size": list(image_size),
                "crop_box_xyxy": None,
                "map_coordinates": "full_frame",
            }
        )
    return rows, np.stack(maps)


def smooth_map(patch_map: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Validation postprocessing: bilinear to input size, then Gaussian sigma 4 input pixels."""
    import torch
    from scipy.ndimage import gaussian_filter
    from torch.nn import functional as F

    resized = F.interpolate(
        torch.from_numpy(patch_map)[None, None],
        size=tuple(image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    return gaussian_filter(resized, sigma=4)
