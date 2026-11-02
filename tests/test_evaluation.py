import json

import pytest

from lc_simplenet.evaluation import (
    calibrate,
    evaluate_run,
    image_auroc,
    load_scores,
    macro_average,
    mean_sd,
    summarize_seeds,
)


def rows(pairs, prefix="img"):
    return [
        {"image_id": f"{prefix}{i}", "label": label, "score": score}
        for i, (score, label) in enumerate(pairs)
    ]


def test_auroc_known_values():
    assert image_auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert image_auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    # 3 of 4 positive/negative pairs ranked correctly
    assert image_auroc([0.1, 0.6, 0.4, 0.9], [0, 0, 1, 1]) == 0.75
    # ties count one half
    assert image_auroc([0.5, 0.5], [0, 1]) == 0.5
    assert image_auroc([0.1, 0.2], [0, 0]) is None


def test_calibration_records_rate_and_rejects_defects():
    val = rows([(s, 0) for s in [0, 1, 2, 3, 4]])
    cal = calibrate(val, 0.25)
    assert cal["threshold"] == 3
    assert cal["validation_images"] == 5 and cal["validation_false_positives"] == 1
    assert cal["validation_fpr"] == 0.2
    with pytest.raises(ValueError):
        calibrate(rows([(0.1, 0), (0.9, 1)]), 0.25)


def test_conditions_share_one_frozen_threshold():
    val = rows([(s, 0) for s in [0, 1, 2, 3, 4]], "v")  # threshold 3
    test = rows([(1, 0), (4, 0), (5, 1), (2, 1)], "t")
    cond = {"t0": "regular", "t1": "regular", "t2": "shifted", "t3": "shifted"}
    result = evaluate_run(val, test, 0.25, cond)
    assert result["calibration"]["threshold"] == 3
    regular, shifted = result["conditions"]["regular"], result["conditions"]["shifted"]
    assert (regular["false_positives"], regular["normal_images"]) == (1, 2)
    assert regular["recall"] is None and shifted["fpr"] is None
    assert (shifted["true_positives"], shifted["defective_images"]) == (1, 2)
    assert result["conditions"]["all"]["recall"] == 0.5


def test_conditions_must_cover_every_test_image():
    val = rows([(s, 0) for s in range(5)], "v")
    with pytest.raises(ValueError):
        evaluate_run(val, rows([(1, 0)], "t"), 0.25, {})
    assert list(evaluate_run(val, rows([(1, 0)], "t"), 0.25)["conditions"]) == ["all"]


def test_seed_summary_uses_sample_sd_and_flags_missing():
    assert mean_sd([0.0, 0.5, 1.0]) == {"mean": 0.5, "sd": 0.5, "n": 3}
    assert mean_sd([0.2, None])["mean"] is None
    val = rows([(s, 0) for s in range(5)], "v")
    runs = {
        seed: evaluate_run(val, rows([(1, 0), (4, 0), (5, 1)], "t"), 0.25) for seed in (0, 1, 2)
    }
    summary = summarize_seeds(runs)["all"]
    assert summary["fpr"] == {"mean": 0.5, "sd": 0.0, "n": 3}
    assert summary["seeds"] == [0, 1, 2]


def test_macro_average_is_unweighted_over_categories():
    def cat(value):
        return {"all": {"fpr": {"mean": value}}}

    cats = {"a": cat(0.1), "b": cat(0.3), "c": cat(0.5)}
    assert macro_average(cats, "all", "fpr")["mean"] == pytest.approx(0.3)
    assert macro_average(cats, "all", "fpr", exclude=["c"]) == {"mean": 0.2, "categories": 2}
    assert macro_average({"a": cat(None)}, "all", "fpr")["mean"] is None


def test_load_scores_rejects_duplicates(tmp_path):
    path = tmp_path / "scores.json"
    good = [{"image_id": "a", "label": 0, "score": 0.1}]
    path.write_text(json.dumps(good))
    assert load_scores(path) == good
    path.write_text(json.dumps(good * 2))
    with pytest.raises(ValueError):
        load_scores(path)


def test_optional_lighting_pools_reuse_calibration_and_keep_suffix_groups():
    val = rows([(s, 0) for s in range(5)], "v")
    test = rows([(1, 0), (4, 0), (5, 1), (2, 1)], "t")
    conditions = dict(
        zip(
            [r["image_id"] for r in test],
            ["regular", "overexposed", "underexposed", "shift_4"],
            strict=True,
        )
    )
    original = evaluate_run(val, test, 0.25, conditions)
    result = evaluate_run(val, test, 0.25, conditions, lighting_aggregates=True)
    assert result["calibration"] == original["calibration"]
    for name, row in original["conditions"].items():
        assert row == result["conditions"][name]
    assert result["conditions"]["nonregular"]["normal_images"] == 1
    assert result["conditions"]["nonregular"]["defective_images"] == 2
    assert result["conditions"]["exposure_changed"]["recall"] == 1.0
    assert result["conditions"]["additional_light"]["recall"] == 0.0
    assert result["conditions"]["additional_light"]["fpr"] is None


@pytest.mark.parametrize("name", ["all", "nonregular", "exposure_changed", "additional_light"])
def test_condition_names_cannot_overwrite_or_duplicate_aggregate_groups(name):
    val = rows([(s, 0) for s in range(5)], "v")
    with pytest.raises(ValueError, match="reserved"):
        evaluate_run(val, rows([(1, 0)], "t"), 0.25, {"t0": name})
