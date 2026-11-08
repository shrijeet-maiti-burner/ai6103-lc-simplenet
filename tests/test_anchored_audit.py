import copy
import importlib.util
import itertools
from pathlib import Path

import pytest

pytest.importorskip("torch")

from lc_simplenet import anchored_pilot as pilot  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "independent_anchored_audit", Path(__file__).parents[1] / "scripts/audit_anchored_pilot.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture
def completed_run(tmp_path):
    policy = copy.deepcopy(pilot.read(pilot.POLICY))
    policy.update(categories=["can"], development_parents=1, lights=[[1, 1], [0.8, 0.8]])
    policy["qualification"]["categories_corner_fpr_improved_vs_augmentation_min"] = 1
    parent = "can/train/good/001.png"
    pilot.write(
        tmp_path / "protocol.json",
        {"policy": policy, "roles": {"can": {"development": [{"image_id": parent}]}}},
    )
    all_metrics = []
    for seed in policy["seeds"]:
        target = tmp_path / "training/can" / f"seed-{seed}"
        calibration = {
            arm: {
                "scores": [0.0] * 32,
                "patch_std": 1.0,
                "gaussian_hinge": 1.0,
                "replay_bit_exact": True,
            }
            for arm in policy["arms"]
        }
        pilot.write(target / "calibration.json", calibration)
        predictions = []
        for arm, light, kind in itertools.product(
            policy["arms"], range(2), ("normal", "spot", "scratch", "blend")
        ):
            flag = kind != "normal" or arm != "candidate"
            predictions.append(
                {
                    "category": "can",
                    "seed": seed,
                    "role": "development",
                    "arm": arm,
                    "light_index": light,
                    "parent_id": parent,
                    "kind": kind,
                    "label": int(kind != "normal"),
                    "changed_pixels": 0 if kind in {"normal", "blend"} else 12,
                    "score": 1.0 if flag else -1.0,
                    "threshold": 0.0,
                    "flagged": flag,
                }
            )
        pilot.write(target / "development-predictions.json", predictions)
        for arm, condition in itertools.product(policy["arms"], ("identity", "corners")):
            (target / f"{arm}.pt").write_bytes(b"hash-only fixture, not a real checkpoint")
            all_metrics.append(
                {
                    "category": "can",
                    "seed": seed,
                    "arm": arm,
                    "condition": condition,
                    "normal": 1,
                    "synthetic_changes": 3,
                    "false_positives": int(arm != "candidate"),
                    "true_positives": 3,
                    "erased_changes": 1,
                    "image_auroc": 1.0 if arm == "candidate" else 0.5,
                    "patch_std": 1.0,
                    "gaussian_hinge": 1.0,
                }
            )
    pilot.write(tmp_path / "screening-metrics.json", all_metrics)
    pilot.write(tmp_path / "screening-decision.json", pilot.qualify(all_metrics, policy, [0, 1]))
    return tmp_path


def test_independent_recount_matches_complete_passing_gate(completed_run):
    result = AUDIT.audit(completed_run, "development")
    assert result["decision_reproduced"] and result["passed"]
    assert result["metrics"] == 20
    assert len(result["artifact_sha256"]) == 14


@pytest.mark.parametrize(
    "damage", ["missing_prediction", "changed_threshold", "missing_checkpoint", "changed_gate"]
)
def test_independent_recount_rejects_corruption(completed_run, damage):
    target = completed_run / "training/can/seed-0"
    if damage == "missing_checkpoint":
        (target / "teacher.pt").unlink()
    elif damage == "changed_gate":
        path = completed_run / "screening-decision.json"
        decision = pilot.read(path)
        decision["passed"] = False
        pilot.write(path, decision)
    else:
        path = target / "development-predictions.json"
        predictions = pilot.read(path)
        if damage == "missing_prediction":
            predictions.pop()
        else:
            predictions[0]["threshold"] = 1.1
        pilot.write(path, predictions)
    with pytest.raises((AssertionError, FileNotFoundError)):
        AUDIT.audit(completed_run, "development")
