"""Deterministic checks for the frozen anomaly-map residual. No dataset is read."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from lc_simplenet.metrics import normal_validation_threshold
from lc_simplenet.score_field import (
    DOWNSAMPLE_SIZES,
    RELIGHT_NAMES,
    TARGET_FPR,
    apply_spatial_gain,
    evaluate_run,
    expected_run_identities,
    image_scores,
    low_frequency_field,
    refuse_test_path,
    relight_validation_image,
    residual_map,
    select_downsample_size,
    spatial_gain,
    validation_image_id,
)


def test_constant_map_has_zero_residual():
    image = torch.full((32, 40), 3.5)
    for size in DOWNSAMPLE_SIZES:
        field = low_frequency_field(image, size)
        residual = residual_map(image, size)
        assert torch.allclose(field, image)
        assert torch.allclose(residual, torch.zeros_like(image), atol=1e-6)


def test_repeated_calls_match_on_a_varying_map():
    torch.manual_seed(0)
    image = torch.randn(32, 32)
    first = residual_map(image, 8)
    second = residual_map(image, 8)
    assert torch.equal(first, second)


def test_horizontal_gain_follows_pixel_centers_and_clips():
    gain = spatial_gain(1, 2, "horizontal")
    assert torch.allclose(gain, torch.tensor([[0.9, 1.1]]))
    clipped = apply_spatial_gain(np.ones((1, 2, 3), dtype=np.float32), "horizontal")
    assert clipped.shape == (1, 2, 3)
    assert np.allclose(clipped[0, 0], 0.9)
    assert np.allclose(clipped[0, 1], 1.0)


def test_spotlight_peaks_at_the_center_and_stays_inside_the_gain_range():
    gain = spatial_gain(5, 5, "spotlight")
    assert gain[2, 2].item() == pytest.approx(1.2, abs=1e-5)
    assert float(gain.min()) >= 0.8 - 1e-6
    assert float(gain.max()) <= 1.2 + 1e-5


def test_edge_spike_is_not_removed():
    image = torch.zeros(32, 32)
    image[:, -1] = 5
    residual = residual_map(image, 4)
    assert float(residual[:, -1].max()) > 1


def test_small_defect_survives_the_coarsest_field():
    image = torch.zeros(32, 32)
    image[14:17, 14:17] = 8
    residual = residual_map(image, 4)
    assert float(residual[15, 15]) > 4


def test_relights_are_the_frozen_seven_and_do_not_need_a_test_image():
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    views = relight_validation_image(image)
    assert tuple(views) == RELIGHT_NAMES
    assert all(view.shape == image.shape for view in views.values())
    assert all(0 <= float(view.min()) and float(view.max()) <= 1 for view in views.values())


def test_test_paths_are_refused():
    with pytest.raises(ValueError, match="test"):
        refuse_test_path("can/test_public/bad/000.png")
    with pytest.raises(ValueError, match="test"):
        refuse_test_path("can/test_private/000_regular.png")


def _peaked(width: int) -> np.ndarray:
    maps = np.zeros((20, 32, 32), dtype=np.float32)
    start = 16 - width // 2
    maps[:, start : start + width, start : start + width] = 4
    return maps


def _ids(category: str, count: int) -> list[str]:
    return [f"{category}/validation/good/{index:03d}.png" for index in range(count)]


def _run(category: str, seed: int, variant: str, maps: np.ndarray) -> dict:
    return {
        "category": category,
        "seed": seed,
        "variant": variant,
        "image_ids": _ids(category, len(maps)),
        "regular": maps,
        "relit": {name: maps.copy() for name in RELIGHT_NAMES},
    }


def _matrix(maps: np.ndarray) -> list[dict]:
    categories = [
        "can",
        "fabric",
        "fruit_jelly",
        "rice",
        "sheet_metal",
        "vial",
        "wallplugs",
        "walnuts",
    ]
    return [
        _run(category, seed, variant, maps)
        for category in categories
        for seed in (0, 1, 2)
        for variant in ("B0", "B1")
    ]


def test_one_global_size_is_rejected_when_any_run_loses_half_its_spread():
    sharp = _peaked(1)
    broad = _peaked(4)
    runs = _matrix(sharp)
    runs[0] = _run("can", 0, "B0", broad)
    result = select_downsample_size(runs)
    by_size = {item["downsample_size"]: item for item in result["sizes"]}
    assert result["public_test_used"] is False
    assert set(by_size) == set(DOWNSAMPLE_SIZES)
    assert by_size[16]["eligible"] is False
    assert any(not item["std_ok"] for item in by_size[16]["runs"])
    assert all(item["std_ok"] for item in by_size[4]["runs"])
    assert all(item["std_ok"] for item in by_size[8]["runs"])
    eligible = [item for item in result["sizes"] if item["eligible"]]
    expected = sorted(
        eligible, key=lambda item: (item["worst_abs_fpr_gap"], item["downsample_size"])
    )
    assert result["selected_downsample_size"] == expected[0]["downsample_size"]


def test_selection_requires_the_full_matrix_and_validation_ids():
    assert len(expected_run_identities()) == 48
    runs = _matrix(_peaked(1))
    with pytest.raises(ValueError, match="48"):
        select_downsample_size(runs[:-1])
    duplicated = list(runs)
    duplicated[-1] = _run("can", 0, "B0", _peaked(1))
    with pytest.raises(ValueError, match="Duplicate"):
        select_downsample_size(duplicated)
    unexpected = list(runs)
    unexpected[-1] = _run("grid", 0, "B0", _peaked(1))
    with pytest.raises(ValueError, match="Unexpected"):
        select_downsample_size(unexpected)
    leaked = list(runs)
    leaked[0]["image_ids"] = ["can/test_public/good/000.png", *_ids("can", 19)]
    with pytest.raises(ValueError, match="test"):
        select_downsample_size(leaked)
    shaped = list(runs)
    shaped[0]["relit"] = dict(shaped[0]["relit"])
    shaped[0]["relit"]["horizontal"] = np.zeros((20, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="match"):
        select_downsample_size(shaped)
    missing = list(runs)
    missing[0]["relit"] = {
        name: missing[0]["relit"][name] for name in RELIGHT_NAMES if name != "spotlight"
    }
    with pytest.raises(ValueError, match="frozen seven"):
        select_downsample_size(missing)


def test_threshold_matches_the_higher_regular_quantile_and_image_denominator():
    regular = np.random.default_rng(0).normal(size=(20, 32, 32)).astype(np.float32)
    relit = {name: regular.copy() for name in RELIGHT_NAMES}
    report = evaluate_run(regular, relit, 8)
    scores = image_scores(torch.from_numpy(regular), 8).numpy()
    assert report["threshold"] == normal_validation_threshold(scores, TARGET_FPR)
    for condition in report["conditions"].values():
        assert condition["normal_images"] == 20
        assert condition["fpr"] == condition["false_positives"] / 20


def test_validation_ids_reject_test_and_ground_truth():
    validation_image_id("can", "can/validation/good/000.png")
    with pytest.raises(ValueError, match="test"):
        validation_image_id("can", "can/test_private/000.png")
    with pytest.raises(ValueError, match="ground_truth"):
        validation_image_id("can", "can/validation/ground_truth/000.png")


def _select_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "select_score_field.py"
    spec = importlib.util.spec_from_file_location("select_score_field_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loader_rejects_a_missing_map(tmp_path):
    path = tmp_path / "can-B0-seed0.npz"
    regular = np.zeros((2, 32, 32), dtype=np.float32)
    np.savez(
        path,
        category="can",
        seed=np.int64(0),
        variant="B0",
        image_ids=np.array(["can/validation/good/000.png", "can/validation/good/001.png"]),
        regular=regular,
    )
    with pytest.raises(ValueError, match="missing"):
        _select_script().load_run(path)


@pytest.mark.parametrize(
    "seed", [np.float64(0.5), np.bool_(False), np.array("0"), np.array([0, 1])]
)
def test_loader_rejects_seed_metadata_that_would_be_coerced(tmp_path, seed):
    path = tmp_path / "can-B0.npz"
    regular = np.zeros((2, 32, 32), dtype=np.float32)
    np.savez(
        path,
        category="can",
        seed=seed,
        variant="B0",
        image_ids=_ids("can", 2),
        regular=regular,
        **{name: regular for name in RELIGHT_NAMES},
    )
    with pytest.raises(ValueError, match="seed.*scalar integer"):
        _select_script().load_run(path)


def test_loader_accepts_the_generated_scalar_schema(tmp_path):
    path = tmp_path / "can-B0.npz"
    regular = np.zeros((2, 32, 32), dtype=np.float32)
    np.savez(
        path,
        category="can",
        seed=np.int64(0),
        variant="B0",
        image_ids=_ids("can", 2),
        regular=regular,
        **{name: regular for name in RELIGHT_NAMES},
    )
    run = _select_script().load_run(path)
    assert (run["category"], run["seed"], run["variant"]) == ("can", 0, "B0")
    assert run["image_ids"] == _ids("can", 2)


@pytest.mark.parametrize(
    "field,value", [("category", ["can", "fabric"]), ("variant", ["B0", "B1"])]
)
def test_loader_rejects_ambiguous_identity_arrays(tmp_path, field, value):
    path = tmp_path / "ambiguous.npz"
    regular = np.zeros((2, 32, 32), dtype=np.float32)
    payload = {
        "category": "can",
        "seed": np.int64(0),
        "variant": "B0",
        "image_ids": _ids("can", 2),
        "regular": regular,
        **{name: regular for name in RELIGHT_NAMES},
    }
    payload[field] = np.asarray(value)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=field + ".*scalar string"):
        _select_script().load_run(path)


def test_image_score_is_the_max_residual():
    maps = torch.zeros(2, 16, 16)
    maps[0, 0, 0] = 3
    maps[1, :, :] = 1
    scores = image_scores(maps, 4)
    assert scores.shape == (2,)
    assert float(scores[1]) == pytest.approx(0, abs=1e-5)
    assert float(scores[0]) > 0


@pytest.mark.parametrize("change", ["replace", "remove"])
def test_selection_rejects_mismatched_category_cohorts(change):
    runs = _matrix(_peaked(1))
    run = runs[0]
    if change == "replace":
        run["image_ids"][0] = f"{run['category']}/validation/good/different.png"
    else:
        run["image_ids"] = run["image_ids"][1:]
        run["regular"] = run["regular"][1:]
        run["relit"] = {name: maps[1:] for name, maps in run["relit"].items()}
    with pytest.raises(ValueError, match="cohort mismatch"):
        select_downsample_size(runs)


def test_selection_accepts_reordered_identical_cohorts():
    runs = _matrix(_peaked(1))
    expected = select_downsample_size(runs)
    run = runs[0]
    run["image_ids"] = run["image_ids"][::-1]
    run["regular"] = run["regular"][::-1]
    run["relit"] = {name: maps[::-1] for name, maps in run["relit"].items()}
    assert select_downsample_size(runs) == expected
