"""Normal-validation anomaly maps for the frozen score-field selector.

The caller supplies already loaded regular-light validation images and a
predict function. This module does not list a test directory, does not read a
ground-truth mask, and does not write into a released run.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from lc_simplenet.pilot import PilotConfig
from lc_simplenet.score_field import (
    RELIGHT_NAMES,
    refuse_test_path,
    relight_validation_image,
    validation_image_id,
)

REGULAR_ATOL = 1e-3
STUDY_IMAGE_SIZE = (256, 256)


def assert_formal_study_head(config: PilotConfig) -> None:
    """B0 or B1 from the locked full-frame 256 study. B2 and 288 are refused."""
    if config.dataset != "mvtec_ad2" or config.ad_resize is not None:
        raise ValueError("Only full-frame MVTec AD 2 heads are accepted")
    if config.variant not in {"B0", "B1"} or config.lambda_score != 0:
        raise ValueError("Only formal B0 and B1 heads are accepted")
    if tuple(config.image_size) != STUDY_IMAGE_SIZE:
        raise ValueError("Only the formal 256x256 study heads are accepted")


def assert_output_outside_run(run_dir: Path, out: Path) -> None:
    """Released run files are inputs. The npz goes somewhere else."""
    refuse_test_path(out.as_posix())
    resolved_out = out.resolve()
    resolved_run = run_dir.resolve()
    if resolved_out == resolved_run or resolved_out.is_relative_to(resolved_run):
        raise ValueError("Refusing to write inside the released run")


def check_validation_rows(category: str, rows: list[dict]) -> list[str]:
    """Saved regular-light rows must be normal validation ids with a score."""
    if not rows:
        raise ValueError("No normal-validation scores")
    image_ids = []
    for row in rows:
        if row.get("label") != 0:
            raise ValueError("Validation rows must be normal images")
        image_id = row["image_id"]
        validation_image_id(category, image_id)
        resized = row.get("resized_size")
        if resized is not None and [int(value) for value in resized] != [256, 256]:
            raise ValueError("Saved geometry is not the formal 256 study")
        if "score" not in row or not np.isfinite(row["score"]):
            raise ValueError("Saved regular-light score is missing")
        image_ids.append(image_id)
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate validation image id")
    return image_ids


def _one_map(image: np.ndarray, predict) -> np.ndarray:
    mapped = np.asarray(predict([image]), dtype=np.float32)
    if mapped.ndim == 3 and mapped.shape[0] == 1:
        mapped = mapped[0]
    if mapped.ndim != 2 or mapped.size == 0 or not np.isfinite(mapped).all():
        raise ValueError("Prediction must be one finite 2D anomaly map")
    return mapped


def check_regular_reproduction(
    saved_scores, maps: np.ndarray, *, atol: float = REGULAR_ATOL
) -> None:
    """Every regular-light image score must match the released score within ``atol``."""
    got = np.asarray(maps, dtype=np.float64).reshape(len(maps), -1).max(axis=1)
    saved = np.asarray(saved_scores, dtype=np.float64)
    if saved.shape != got.shape or not np.isfinite(saved).all():
        raise ValueError("Saved scores do not match the validation images")
    if float(np.max(np.abs(got - saved))) > atol:
        raise ValueError("Regular-light score reproduction failed the 1e-3 guard")


def generate_validation_maps(
    category: str,
    seed: int,
    variant: str,
    rows: list[dict],
    images: list[np.ndarray],
    predict,
) -> dict:
    """Score regular light, enforce the guard, then score the seven frozen relights.

    ``images`` are HWC RGB arrays in ``[0, 1]``, in the same order as ``rows``.
    A guard failure raises before any relight is scored.
    """
    if variant not in {"B0", "B1"} or type(seed) is not int:
        raise ValueError("Only formal B0 and B1 heads are accepted")
    image_ids = check_validation_rows(category, rows)
    if len(images) != len(rows):
        raise ValueError("Images and validation rows differ in length")
    regular = np.stack([_one_map(image, predict) for image in images])
    check_regular_reproduction([row["score"] for row in rows], regular)
    relit = {}
    for name in RELIGHT_NAMES:
        views = []
        for image in images:
            changed = relight_validation_image(np.asarray(image, dtype=np.float32))
            views.append(changed[name])
        relit[name] = np.stack([_one_map(view, predict) for view in views])
        if relit[name].shape != regular.shape:
            raise ValueError("Relight map shape differs from the regular-light map")
    return {
        "category": category,
        "seed": seed,
        "variant": variant,
        "image_ids": image_ids,
        "regular": regular,
        "relit": relit,
    }


def save_validation_maps(path: Path, payload: dict) -> None:
    """Write one run npz. An existing file is left untouched."""
    path = Path(path)
    refuse_test_path(path.as_posix())
    if path.exists():
        raise FileExistsError("Refusing to overwrite a validation-map file")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "category": np.array(payload["category"]),
        "seed": np.int64(payload["seed"]),
        "variant": np.array(payload["variant"]),
        "image_ids": np.array(payload["image_ids"]),
        "regular": np.asarray(payload["regular"], dtype=np.float32),
    }
    for name in RELIGHT_NAMES:
        arrays[name] = np.asarray(payload["relit"][name], dtype=np.float32)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
