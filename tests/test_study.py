import pytest

from lc_simplenet.study import choose_ad_categories, choose_ad_scope, select_consistency_weight


def profile(category, seconds, validation=42):
    return {
        "category": category,
        "train_images": 160,
        "validation_images": validation,
        "batch_size": 8,
        "paired_p90_seconds": seconds,
        "paired_nontraining_seconds": 5,
    }


def test_scope_selection_uses_cost_type_and_counts():
    rows = [
        profile("bottle", 0.1),
        profile("wood", 0.2),
        profile("grid", 0.15),
        profile("leather", 0.01, validation=12),
        profile("capsule", 0.001),
    ]
    assert choose_ad_categories(rows, 20000) == ["bottle", "grid"]
    assert choose_ad_categories(rows, 2500) == ["bottle"]
    with pytest.raises(ValueError, match="exceeds"):
        choose_ad_categories(rows, 100)


def test_scope_prefers_two_categories_at_the_predeclared_shorter_horizon():
    rows = [profile("bottle", 0.1), profile("grid", 0.15)]
    scope = choose_ad_scope(rows, 3200)
    assert scope == {"categories": ["bottle", "grid"], "source_passes": 40}


def trial(variant, weight, mse, std=1.0, hinge=0.2):
    return {
        "category": "bottle",
        "variant": variant,
        "candidate": weight,
        "metrics": {"pair_mse": mse, "patch_score_std": std, "synthetic_hinge": hinge},
    }


def test_weight_selection_rejects_collapse_and_uses_declared_tie_rule():
    trials = [trial("B1", weight, 1.0) for weight in (0.01, 0.1, 1.0)]
    trials += [trial("B2", 0.01, 0.505), trial("B2", 0.1, 0.5), trial("B2", 1.0, 0.01, std=0.1)]
    result = select_consistency_weight(trials, [0.01, 0.1, 1.0])
    assert result["lambda_score"] == 0.01
    assert result["candidates"][2]["eligible"] is False


def test_weight_selection_stops_when_discrimination_safeguard_fails():
    trials = [trial("B1", 0.1, 1.0), trial("B2", 0.1, 0.1, hinge=0.4)]
    with pytest.raises(ValueError, match="No positive"):
        select_consistency_weight(trials, [0.1])
