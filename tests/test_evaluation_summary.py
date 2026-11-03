from types import SimpleNamespace

import pytest

from lc_simplenet.evaluation_summary import aggregate_results, filter_runs


def fixture():
    results, matrix = {}, []
    for category in ("can", "sheet_metal"):
        for method in ("B0", "B1", "B2@0.1"):
            for seed in (0, 1, 2):
                recall = 1.0 if method == "B0" else 0.9 if category == "can" else 1.0
                row = {"fpr": 0.1 * seed, "recall": recall, "image_auroc": 0.8}
                results[category, method, seed] = {
                    "evaluation": {"conditions": {"regular": row}},
                    "test": [{"image_id": f"{category}/test/0", "label": 0}],
                }
                matrix.append(
                    {"category": category, "method": method, "seed": seed, "status": "ready"}
                )
    return results, matrix


def test_macros_use_categories_equally_and_seed_sd_with_explicit_recall_failures():
    result = aggregate_results(*fixture())
    b1 = next(r for r in result["macros"] if r["method"] == "B1" and r["scope"] == "all_categories")
    assert b1["recall"]["mean"] == pytest.approx(0.95)
    assert b1["fpr"]["sd"] == pytest.approx(0.1)
    checks = [r for r in result["comparisons"] if r["candidate"] == "B1"]
    assert {r["scope"]: r["recall_preserved"] for r in checks} == {
        "all_categories": True,
        "without_sheet_metal": False,
    }
    category = next(
        r
        for r in result["category_recall_checks"]
        if r["category"] == "can" and r["candidate"] == "B1"
    )
    assert category["exceeds_tolerance"]


def test_missing_expected_seed_makes_macro_unavailable():
    results, matrix = fixture()
    del results["can", "B1", 2]
    result = aggregate_results(results, matrix)
    row = next(
        r for r in result["macros"] if r["method"] == "B1" and r["scope"] == "all_categories"
    )
    assert row["status"] == "incomplete" and row["recall"]["mean"] is None
    assert result["status"] == "incomplete"
    assert all(
        r["recall_preserved"] is None for r in result["comparisons"] if r["candidate"] == "B1"
    )


def test_population_mismatch_is_rejected_before_comparison():
    results, matrix = fixture()
    results["can", "B1", 0]["test"][0]["label"] = 1
    with pytest.raises(ValueError, match="IDs or labels"):
        aggregate_results(results, matrix)


def test_explicit_scope_filters_all_data_not_only_the_matrix():
    runs = [
        SimpleNamespace(category=c, method=m, seed=s)
        for c in ("can", "fabric")
        for m in ("B0", "B1")
        for s in (0, 1)
    ]
    chosen = filter_runs(runs, ["fabric"], [1], ["B1"])
    assert len(chosen) == 1
    assert (chosen[0].category, chosen[0].method, chosen[0].seed) == ("fabric", "B1", 1)


def test_validation_only_is_not_a_finished_evaluation():
    results, matrix = fixture()
    for row in results.values():
        row["evaluation"] = None
        row["test"] = []
    result = aggregate_results(results, matrix)
    assert result["status"] == "not_evaluated" and not result["comparisons"]


def test_structurally_absent_lighting_is_never_called_a_complete_macro():
    results, matrix = fixture()
    for (category, _, _), row in results.items():
        if category == "can":
            row["evaluation"]["conditions"]["shift_4"] = {
                "fpr": 0.0,
                "recall": 1.0,
                "image_auroc": 1.0,
            }
    result = aggregate_results(results, matrix)
    rows = [r for r in result["macros"] if r["condition"] == "shift_4"]
    assert all(
        r["status"] == "not_applicable"
        and r["recall"]["mean"] is None
        and r["categories_without_condition"] == ["sheet_metal"]
        for r in rows
        if r["scope"] == "all_categories"
    )
    assert all(
        r["status"] == "complete" and r["recall"]["mean"] == 1.0
        for r in rows
        if r["scope"] == "without_sheet_metal"
    )
    assert result["status"] == "complete"


def test_lighting_groups_must_match_between_methods():
    results, matrix = fixture()
    results["can", "B1", 0]["evaluation"]["conditions"]["different"] = {}
    with pytest.raises(ValueError, match="condition groups differ"):
        aggregate_results(results, matrix)
