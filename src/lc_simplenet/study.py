"""Predeclared cost and normal-validation decisions for the AD training matrix."""

import math

import numpy as np

TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def matrix_cost_seconds(profile: dict, *, passes: int = 80) -> float:
    updates = passes * math.ceil(profile["train_images"] / profile["batch_size"])
    return 9 * (
        updates * profile["paired_p90_seconds"] * 1.25
        + 2 * profile["paired_nontraining_seconds"]
        + 20
    )


def choose_ad_categories(
    profiles: list[dict], available_seconds: float, *, passes: int = 80
) -> list[str]:
    by_category = {row["category"]: row for row in profiles}
    bottle = by_category["bottle"]
    if bottle["validation_images"] < 20:
        raise ValueError("Bottle validation membership is too small for the declared study")
    remaining = available_seconds - matrix_cost_seconds(bottle, passes=passes)
    if remaining < 0:
        raise ValueError("The complete three-seed bottle matrix exceeds the available budget")
    selected = ["bottle"]
    candidates = sorted(
        (row for row in profiles if row["category"] in TEXTURES and row["validation_images"] >= 20),
        key=lambda row: (matrix_cost_seconds(row, passes=passes), row["category"]),
    )
    if candidates and matrix_cost_seconds(candidates[0], passes=passes) <= remaining:
        selected.append(candidates[0]["category"])
    return selected


def choose_ad_scope(profiles: list[dict], available_seconds: float) -> dict:
    feasible = []
    for passes in (80, 40):
        try:
            categories = choose_ad_categories(profiles, available_seconds, passes=passes)
        except ValueError:
            continue
        choice = {"categories": categories, "source_passes": passes}
        if len(categories) == 2:
            return choice
        feasible.append(choice)
    if feasible:
        return feasible[0]
    raise ValueError("No complete three-seed matrix fits the available budget")


def select_consistency_weight(trials: list[dict], candidates: list[float]) -> dict:
    """Minimize category-macro relative pair MSE under fixed normal-only safeguards."""
    categories = sorted({row["category"] for row in trials})
    baseline, paired = {}, {}
    for row in trials:
        key = (row["category"], row["candidate"])
        target = baseline if row["variant"] == "B1" else paired if row["variant"] == "B2" else None
        if target is not None:
            if key in target:
                raise ValueError("Duplicate selection trial")
            target[key] = row["metrics"]
    evaluations = []
    for weight in candidates:
        ratios, eligible = [], True
        for category in categories:
            base, current = baseline[(category, weight)], paired[(category, weight)]
            values = [
                entry[key]
                for entry in (base, current)
                for key in ("pair_mse", "patch_score_std", "synthetic_hinge")
            ]
            if not np.isfinite(values).all() or base["patch_score_std"] <= 1e-6:
                eligible = False
                continue
            eligible &= current["patch_score_std"] >= 0.5 * base["patch_score_std"]
            eligible &= current["synthetic_hinge"] <= 1.25 * base["synthetic_hinge"] + 1e-6
            ratios.append(current["pair_mse"] / max(base["pair_mse"], 1e-12))
        evaluations.append(
            {
                "lambda_score": weight,
                "eligible": bool(eligible),
                "relative_pair_mse": float(np.mean(ratios)) if ratios else None,
            }
        )
    feasible = [row for row in evaluations if row["eligible"]]
    if not feasible:
        raise ValueError("No positive consistency weight satisfies the predeclared safeguards")
    best = min(row["relative_pair_mse"] for row in feasible)
    chosen = min(
        row["lambda_score"] for row in feasible if row["relative_pair_mse"] <= best * 1.01 + 1e-12
    )
    return {
        "lambda_score": chosen,
        "candidates": evaluations,
        "scope": "normal-validation choice; does not establish defect recall",
    }
