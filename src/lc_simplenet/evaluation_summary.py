"""Complete, paired category macros and the declared recall-preservation rule."""

import math

import numpy as np

from lc_simplenet.evaluation import RATE_KEYS, mean_sd


def filter_runs(runs, categories, seeds, methods=None):
    """Apply the requested study scope to data as well as to the run matrix."""
    return [
        r
        for r in runs
        if r.category in categories and r.seed in seeds and (methods is None or r.method in methods)
    ]


def validate_test_population(results):
    """Methods/seeds in a category must score identical image IDs and labels."""
    populations = {}
    group_names = {}
    for (category, _, _), result in results.items():
        if not result["evaluation"]:
            continue
        population = {row["image_id"]: row["label"] for row in result["test"]}
        if category in populations and populations[category] != population:
            raise ValueError(f"test image IDs or labels differ across runs for {category}")
        populations[category] = population
        names = set(result["evaluation"]["conditions"])
        if category in group_names and group_names[category] != names:
            raise ValueError(f"lighting condition groups differ across runs for {category}")
        group_names[category] = names


def aggregate_results(results, matrix, recall_tolerance=0.05):
    """Macro per seed, then mean/sample SD; missing runs never become a full result.

    The tolerance is in rate units (0.05 = five percentage points), per condition.
    This reports the declared rule; it does not choose a method or hyperparameter.
    """
    if not math.isfinite(recall_tolerance) or not 0 <= recall_tolerance <= 1:
        raise ValueError("recall_tolerance must be finite and in [0,1]")
    validate_test_population(results)
    categories = sorted({row["category"] for row in matrix})
    seeds = sorted({row["seed"] for row in matrix})
    methods = sorted({row["method"] for row in matrix})
    conditions = sorted(
        {
            name
            for result in results.values()
            if result["evaluation"]
            for name in result["evaluation"]["conditions"]
        }
    )
    scopes = {"all_categories": categories}
    if "sheet_metal" in categories and len(categories) > 1:
        scopes["without_sheet_metal"] = [c for c in categories if c != "sheet_metal"]
    macros, comparisons, category_checks = [], [], []
    for scope, chosen in scopes.items():
        for method in methods:
            for condition in conditions:
                per_seed = {}
                missing = []
                unavailable = set()
                for seed in seeds:
                    values = []
                    for category in chosen:
                        evaluated = results.get((category, method, seed), {}).get("evaluation")
                        row = evaluated["conditions"].get(condition) if evaluated else None
                        if row is None:
                            if evaluated:
                                unavailable.add(category)
                            else:
                                missing.append({"category": category, "seed": seed})
                        values.append(row)
                    per_seed[seed] = {
                        key: float(np.mean([v[key] for v in values]))
                        if all(v is not None and v[key] is not None for v in values)
                        else None
                        for key in RATE_KEYS
                    }
                macros.append(
                    {
                        "scope": scope,
                        "method": method,
                        "condition": condition,
                        "categories": chosen,
                        "expected_seeds": seeds,
                        "status": "incomplete"
                        if missing
                        else "not_applicable"
                        if unavailable
                        else "complete",
                        "missing": missing,
                        "categories_without_condition": sorted(unavailable),
                        "per_seed": per_seed,
                        **{key: mean_sd([per_seed[s][key] for s in seeds]) for key in RATE_KEYS},
                    }
                )
    pairs = [("B1", "B0")] + [
        (m, reference) for m in methods if m.startswith("B2@") for reference in ("B0", "B1")
    ]
    indexed = {(row["scope"], row["method"], row["condition"]): row for row in macros}
    for candidate, reference in pairs:
        if candidate not in methods or reference not in methods:
            continue
        for scope in scopes:
            for condition in conditions:
                first, second = [indexed[scope, m, condition] for m in (candidate, reference)]
                ready = first["status"] == second["status"] == "complete"
                recalls = [row["recall"]["mean"] for row in (first, second)]
                delta = recalls[0] - recalls[1] if ready and None not in recalls else None
                comparisons.append(
                    {
                        "scope": scope,
                        "candidate": candidate,
                        "reference": reference,
                        "condition": condition,
                        "recall_delta": delta,
                        "recall_tolerance": recall_tolerance,
                        "recall_preserved": delta >= -recall_tolerance - 1e-12
                        if delta is not None
                        else None,
                        "fpr_delta": first["fpr"]["mean"] - second["fpr"]["mean"]
                        if ready
                        and first["fpr"]["mean"] is not None
                        and second["fpr"]["mean"] is not None
                        else None,
                    }
                )
        for category in categories:
            for condition in conditions:
                rates = []
                for method in (candidate, reference):
                    values = []
                    for seed in seeds:
                        evaluated = results.get((category, method, seed), {}).get("evaluation")
                        row = evaluated["conditions"].get(condition) if evaluated else None
                        values.append(row["recall"] if row else None)
                    rates.append(mean_sd(values)["mean"])
                delta = rates[0] - rates[1] if None not in rates else None
                category_checks.append(
                    {
                        "category": category,
                        "condition": condition,
                        "candidate": candidate,
                        "reference": reference,
                        "recall_delta": delta,
                        "exceeds_tolerance": delta < -recall_tolerance - 1e-12
                        if delta is not None
                        else None,
                    }
                )
    return {
        "schema": 1,
        "aggregation": "unweighted category macro per seed, then mean/sample SD",
        "recall_tolerance": recall_tolerance,
        "macros": macros,
        "comparisons": comparisons,
        "category_recall_checks": category_checks,
        "status": "not_evaluated"
        if not conditions
        else "incomplete"
        if any(row["status"] == "incomplete" for row in macros)
        else "complete",
    }
