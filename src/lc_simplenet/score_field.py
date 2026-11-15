"""Low-frequency residual of a frozen anomaly map.

The released B0 and B1 checkpoints, their raw scores and their thresholds are
not inputs that this module writes. A caller subtracts a smooth field from an
already computed anomaly map, then fits a new threshold on regular-light
normal validation scores of that residual.

Downsampling is ``adaptive_avg_pool2d`` to a ``k`` by ``k`` grid. PyTorch
forms each coarse cell from input rows ``[start_h, end_h)`` and columns
``[start_w, end_w)``, where the starts and ends are the usual integer
breakpoints ``floor(i * H / k)`` and ``ceil((i + 1) * H / k)``. Upsampling is
bilinear with ``align_corners=False`` and ``antialias=False``: output pixel
``(y, x)`` is placed at the half-pixel source coordinate
``((y + 0.5) * k / H - 0.5, (x + 0.5) * k / W - 0.5)`` and clamped to the
coarse grid, so no sample is read from outside that grid.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from lc_simplenet.metrics import normal_validation_threshold
from lc_simplenet.photometric import relight

DOWNSAMPLE_SIZES = (4, 8, 16)
TARGET_FPR = 0.05
STD_RATIO_FLOOR = 0.5
GAIN_LOW = 0.8
GAIN_HIGH = 1.2
SPOTLIGHT_SIGMA = 0.25
BRIGHTNESS_GAMMA_CORNERS = (
    (0.8, 0.8),
    (0.8, 1.2),
    (1.2, 0.8),
    (1.2, 1.2),
)
SPATIAL_MAPS = ("horizontal", "vertical", "spotlight")
RELIGHT_NAMES = (
    "corner_0.8_0.8",
    "corner_0.8_1.2",
    "corner_1.2_0.8",
    "corner_1.2_1.2",
    *SPATIAL_MAPS,
)
FORBIDDEN_SPLIT_NAMES = frozenset(
    {"test", "test_public", "test_private", "test_private_mixed", "ground_truth"}
)
CATEGORIES = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)
STUDY_SEEDS = (0, 1, 2)
STUDY_VARIANTS = ("B0", "B1")


def _as_batch(anomaly_map: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if anomaly_map.ndim == 2:
        return anomaly_map.unsqueeze(0), True
    if anomaly_map.ndim == 3:
        return anomaly_map, False
    raise ValueError("Anomaly map must have shape (H, W) or (N, H, W)")


def low_frequency_field(anomaly_map: torch.Tensor, downsample_size: int) -> torch.Tensor:
    """Smooth field matched to ``anomaly_map``.

    ``downsample_size`` is the frozen grid side ``k`` in ``{4, 8, 16}``.
    """
    if downsample_size not in DOWNSAMPLE_SIZES:
        raise ValueError("downsample_size must be 4, 8 or 16")
    if not anomaly_map.is_floating_point() or not torch.isfinite(anomaly_map).all():
        raise ValueError("Anomaly map must be finite and floating point")
    batch, squeeze = _as_batch(anomaly_map)
    _, height, width = batch.shape
    if height < downsample_size or width < downsample_size:
        raise ValueError("Anomaly map is smaller than the downsampling grid")
    coarse = F.adaptive_avg_pool2d(batch.unsqueeze(1), (downsample_size, downsample_size))
    field = F.interpolate(
        coarse,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    ).squeeze(1)
    return field.squeeze(0) if squeeze else field


def residual_map(anomaly_map: torch.Tensor, downsample_size: int) -> torch.Tensor:
    """Anomaly map after its low-frequency field is removed."""
    return anomaly_map - low_frequency_field(anomaly_map, downsample_size)


def image_scores(anomaly_maps: torch.Tensor, downsample_size: int) -> torch.Tensor:
    """One score per map: the maximum residual patch. Shape ``(N,)``."""
    residual = residual_map(anomaly_maps, downsample_size)
    if residual.ndim != 3:
        raise ValueError("Expected a batch of anomaly maps")
    return residual.flatten(1).amax(dim=1)


def pixel_centers(height: int, width: int, *, dtype: torch.dtype, device: torch.device):
    """Half-pixel centers ``(y + 0.5) / H`` and ``(x + 0.5) / W`` in ``[0, 1]``."""
    if height < 1 or width < 1:
        raise ValueError("Image size must be positive")
    vertical = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    horizontal = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    return vertical[:, None], horizontal[None, :]


def spatial_gain(
    height: int,
    width: int,
    kind: str,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Frozen gain in ``[0.8, 1.2]``.

    ``horizontal`` ramps with ``x`` only, ``vertical`` with ``y`` only.
    ``spotlight`` is ``0.8 + 0.4 * exp(-((u-0.5)^2 + (v-0.5)^2) / (2 * 0.25^2))``.
    """
    if kind not in SPATIAL_MAPS:
        raise ValueError("Unknown spatial illumination map")
    vertical, horizontal = pixel_centers(
        height, width, dtype=dtype, device=device or torch.device("cpu")
    )
    if kind == "horizontal":
        gain = GAIN_LOW + (GAIN_HIGH - GAIN_LOW) * horizontal
    elif kind == "vertical":
        gain = GAIN_LOW + (GAIN_HIGH - GAIN_LOW) * vertical
    else:
        radius = (horizontal - 0.5) ** 2 + (vertical - 0.5) ** 2
        gain = GAIN_LOW + (GAIN_HIGH - GAIN_LOW) * torch.exp(-radius / (2 * SPOTLIGHT_SIGMA**2))
    return gain.expand(height, width).contiguous()


def apply_spatial_gain(image: np.ndarray, kind: str) -> np.ndarray:
    """Multiply an HWC RGB image in ``[0, 1]`` by one frozen gain and clip to ``[0, 1]``."""
    if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
        raise ValueError("Expected a nonempty HWC RGB image")
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Expected floating point pixels in [0, 1]")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Expected finite pixels in [0, 1]")
    height, width = image.shape[:2]
    gain = spatial_gain(height, width, kind).numpy()
    return np.clip(image * gain[..., None], 0, 1).astype(image.dtype, copy=False)


def apply_corner(image: np.ndarray, brightness: float, gamma: float) -> np.ndarray:
    """One of the four frozen ``clip(brightness * image**gamma, 0, 1)`` corners."""
    if (brightness, gamma) not in BRIGHTNESS_GAMMA_CORNERS:
        raise ValueError("Corner is outside the frozen brightness/gamma set")
    return relight(image, brightness=brightness, gamma=gamma)


def relight_validation_image(image: np.ndarray) -> dict[str, np.ndarray]:
    """Seven frozen relights of one normal validation image. No test image is read."""
    views = {}
    for brightness, gamma in BRIGHTNESS_GAMMA_CORNERS:
        views[f"corner_{brightness}_{gamma}"] = apply_corner(image, brightness, gamma)
    for kind in SPATIAL_MAPS:
        views[kind] = apply_spatial_gain(image, kind)
    if tuple(views) != RELIGHT_NAMES:
        raise RuntimeError("Relight names drifted from the frozen list")
    return views


def refuse_test_path(path: str) -> None:
    """Selection may read train or validation only."""
    parts = set(path.replace("\\", "/").split("/"))
    if parts & FORBIDDEN_SPLIT_NAMES:
        raise ValueError(f"Refusing a test path during score-field selection: {path}")


def expected_run_identities() -> set[tuple[str, int, str]]:
    """The formal 256 study matrix: eight categories, seeds 0–2, arms B0 and B1."""
    return {
        (category, seed, variant)
        for category in CATEGORIES
        for seed in STUDY_SEEDS
        for variant in STUDY_VARIANTS
    }


def validation_image_id(category: str, image_id: str) -> None:
    """Accept only ``<category>/validation/good/<file>``."""
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("Validation image id must be a nonempty string")
    refuse_test_path(image_id)
    parts = image_id.replace("\\", "/").split("/")
    if (
        len(parts) != 4
        or parts[0] != category
        or parts[1] != "validation"
        or parts[2] != "good"
        or parts[3] in {"", ".", ".."}
        or any(part in {".", ".."} or ":" in part for part in parts)
    ):
        raise ValueError(f"Refusing a non-validation image id: {image_id}")


def validate_selection_run(run: dict) -> None:
    """One released B0 or B1 run, with matching normal-validation maps."""
    category, variant = run.get("category"), run.get("variant")
    seed = run.get("seed")
    if (
        category not in CATEGORIES
        or type(seed) is not int
        or seed not in STUDY_SEEDS
        or variant not in STUDY_VARIANTS
    ):
        raise ValueError("Unexpected category, seed or arm")
    if "image_ids" not in run:
        raise ValueError("Missing validation image ids")
    regular = np.asarray(run["regular"])
    if regular.ndim != 3 or regular.shape[0] < 1 or not np.isfinite(regular).all():
        raise ValueError("Regular maps must be a nonempty finite (N, H, W) array")
    relit = run["relit"]
    if tuple(relit) != RELIGHT_NAMES:
        raise ValueError("Relight conditions must be the frozen seven, in order")
    for name in RELIGHT_NAMES:
        view = np.asarray(relit[name])
        if view.shape != regular.shape or not np.isfinite(view).all():
            raise ValueError("Relit maps must match the regular-light maps")
    image_ids = list(run["image_ids"])
    if len(image_ids) != len(regular) or len(set(image_ids)) != len(image_ids):
        raise ValueError("Image ids do not match the validation maps")
    for image_id in image_ids:
        validation_image_id(category, str(image_id))


def pooled_std(maps: np.ndarray) -> float:
    """Population standard deviation of every patch in ``(N, H, W)``."""
    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 3 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite batch of maps")
    return float(values.std(ddof=0))


def _residual_batch(maps: np.ndarray, downsample_size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(maps, dtype=np.float32))
    return residual_map(tensor, downsample_size).numpy()


def evaluate_run(regular: np.ndarray, relit: dict[str, np.ndarray], downsample_size: int) -> dict:
    """Threshold the residual on regular-light maps and score each relight.

    ``regular`` and each relight array have shape ``(N, H, W)`` and contain
    only normal validation images. Labels are not an input.
    """
    if tuple(relit) != RELIGHT_NAMES:
        raise ValueError("Relight conditions must be the frozen seven, in order")
    corrected = _residual_batch(regular, downsample_size)
    scores = corrected.reshape(len(corrected), -1).max(axis=1)
    threshold = normal_validation_threshold(scores, TARGET_FPR)
    uncorrected_std = pooled_std(regular)
    corrected_std = pooled_std(corrected)
    ratio = corrected_std / uncorrected_std if uncorrected_std > 1e-8 else 0.0
    conditions = {}
    for name in RELIGHT_NAMES:
        view = relit[name]
        if view.shape != regular.shape:
            raise ValueError("Relit maps must match the regular-light maps")
        view_scores = _residual_batch(view, downsample_size).reshape(len(view), -1).max(axis=1)
        false_positives = int(np.sum(view_scores > threshold))
        conditions[name] = {
            "false_positives": false_positives,
            "normal_images": int(len(view_scores)),
            "fpr": false_positives / len(view_scores),
        }
    regular_hits = int(np.sum(scores > threshold))
    return {
        "threshold": threshold,
        "regular_fpr": regular_hits / len(scores),
        "uncorrected_std": uncorrected_std,
        "corrected_std": corrected_std,
        "std_ratio": ratio,
        "std_ok": bool(ratio + 1e-12 >= STD_RATIO_FLOOR),
        "conditions": conditions,
    }


def select_downsample_size(runs: Iterable[dict]) -> dict:
    """One ``k`` for the full 48-run matrix, or none if any size fails a std check.

    A size is eligible only when every run keeps at least half of its own
    uncorrected patch-score standard deviation. Among eligible sizes, choose
    the one whose worst absolute relight-FPR gap from 0.05 is smallest. Equal
    gaps keep the smaller grid.
    """
    materialized = list(runs)
    if not materialized:
        raise ValueError("No validation runs were provided")
    identities = []
    cohorts = {}
    for run in materialized:
        validate_selection_run(run)
        category = run["category"]
        cohort = frozenset(run["image_ids"])
        if category in cohorts and cohorts[category] != cohort:
            raise ValueError(f"Validation cohort mismatch within category {category}")
        cohorts[category] = cohort
        identities.append((run["category"], run["seed"], run["variant"]))
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate category, seed and variant")
    found = set(identities)
    expected = expected_run_identities()
    if found != expected:
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        raise ValueError(
            f"Selection requires all 48 runs; missing {missing}; unexpected {unexpected}"
        )
    reports = []
    for size in DOWNSAMPLE_SIZES:
        per_run = []
        for run in materialized:
            report = evaluate_run(run["regular"], run["relit"], size)
            report.update(
                {
                    "category": run["category"],
                    "seed": run["seed"],
                    "variant": run["variant"],
                    "downsample_size": size,
                }
            )
            per_run.append(report)
        eligible = all(item["std_ok"] for item in per_run)
        worst = max(
            abs(condition["fpr"] - TARGET_FPR)
            for item in per_run
            for condition in item["conditions"].values()
        )
        reports.append(
            {
                "downsample_size": size,
                "eligible": eligible,
                "worst_abs_fpr_gap": worst,
                "runs": per_run,
            }
        )
    eligible = [item for item in reports if item["eligible"]]
    if not eligible:
        chosen = None
    else:
        chosen = min(
            eligible,
            key=lambda item: (item["worst_abs_fpr_gap"], item["downsample_size"]),
        )["downsample_size"]
    return {
        "selected_downsample_size": chosen,
        "public_test_used": False,
        "target_fpr": TARGET_FPR,
        "std_ratio_floor": STD_RATIO_FLOOR,
        "sizes": reports,
    }
