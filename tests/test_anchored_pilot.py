import copy
import itertools

import numpy as np
import pytest

pytest.importorskip("torch")

from lc_simplenet import anchored_pilot as pilot  # noqa: E402


def passing_metrics():
    policy = pilot.read(pilot.POLICY)
    rows = []
    for category, seed, arm, condition in itertools.product(
        policy["categories"], policy["seeds"], policy["arms"], ("identity", "corners")
    ):
        normal = 8 if condition == "identity" else 32
        rows.append(
            {
                "category": category,
                "seed": seed,
                "arm": arm,
                "condition": condition,
                "normal": normal,
                "synthetic_changes": normal * 3,
                "false_positives": 0 if arm == "candidate" else 4,
                "true_positives": normal * 3,
                "patch_std": 1.0,
                "gaussian_hinge": 1.0,
            }
        )
    return policy, rows


def test_complete_gate_and_exact_recall_regression():
    policy, rows = passing_metrics()
    assert pilot.qualify(rows, policy, [0, 1])["passed"]
    row = next(r for r in rows if r["arm"] == "candidate" and r["condition"] == "corners")
    row["true_positives"] -= 1
    decision = pilot.qualify(rows, policy, [0, 1])
    assert not decision["passed"]
    assert not decision["per_seed"][0]["comparisons"]["teacher"]["passed"]


def test_teacher_is_a_required_control():
    policy, rows = passing_metrics()
    for row in rows:
        if row["arm"] == "teacher":
            row["false_positives"] = 0
    decision = pilot.qualify(rows, policy, [0, 1])
    assert not decision["passed"]
    assert all(not r["comparisons"]["teacher"]["passed"] for r in decision["per_seed"])


@pytest.mark.parametrize("damage", ["missing", "duplicate", "dropped_changes", "fractional", "nan"])
def test_missing_or_malformed_evidence_cannot_qualify(damage):
    policy, rows = passing_metrics()
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows[-1] = rows[0]
    elif damage == "dropped_changes":
        rows[0]["synthetic_changes"] -= 1
    elif damage == "fractional":
        rows[0]["false_positives"] = 0.5
    else:
        rows[0]["patch_std"] = float("nan")
    with pytest.raises(ValueError):
        pilot.qualify(rows, policy, [0, 1])


@pytest.mark.parametrize("stage", [pilot.prepare, pilot.train])
def test_failed_screen_refuses_confirmation_before_data_access(tmp_path, monkeypatch, stage):
    policy, rows = passing_metrics()
    for row in rows:
        if row["arm"] == "candidate":
            row["false_positives"] = row["normal"]
    decision = pilot.qualify(rows, policy, [0, 1])
    pilot.write(tmp_path / "screening-metrics.json", rows)
    pilot.write(tmp_path / "screening-decision.json", decision)
    monkeypatch.setattr(pilot, "verify_protocol", lambda _: {"policy": policy})

    def forbidden(*args, **kwargs):
        pytest.fail("failed screening must not open features, an encoder, or confirmation images")

    monkeypatch.setattr(pilot, "backend_for", forbidden)
    monkeypatch.setattr(pilot, "cached", forbidden)
    if stage == pilot.prepare:
        stage(tmp_path, tmp_path / "nonexistent-data", "confirmation")
    else:
        stage(tmp_path, "confirmation")
    assert not pilot.read(tmp_path / "confirmation-status.json")["confirmation_decoded"]
    assert not (tmp_path / "confirmation-cache").exists()


def test_erased_synthetic_blend_is_retained():
    normal = np.ones((256, 256, 3), dtype=np.float32)
    first = pilot.synthetic_changes(normal, normal, "can/train/good/001.png", "development")
    second = pilot.synthetic_changes(normal, normal, "can/train/good/001.png", "development")
    assert len(first) == 3
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first[2], normal)
    assert np.any(first[0] != normal) and np.any(first[1] != normal)


def test_cached_case_metadata_is_complete_and_hashed(tmp_path):
    cases = [
        {
            "parent_index": 0,
            "parent_id": "can/train/good/001.png",
            "light_index": i,
            "changed_pixels": [0, 20, 10, 0],
        }
        for i in range(5)
    ]
    path = tmp_path / "cache/can/cases.json"
    pilot.write(path, cases)
    protocol = {
        "policy": {"lights": [None] * 5},
        "roles": {"can": {"development": [{"image_id": "can/train/good/001.png"}]}},
    }
    manifest = {"cache/can/cases.json": pilot.digest(path)}
    assert (
        pilot.verified_cases(tmp_path, "cache", "can", "development", protocol, manifest) == cases
    )
    pilot.write(path, cases[:-1])
    with pytest.raises(ValueError, match="metadata changed"):
        pilot.verified_cases(tmp_path, "cache", "can", "development", protocol, manifest)
    manifest["cache/can/cases.json"] = pilot.digest(path)
    with pytest.raises(ValueError, match="complete unique"):
        pilot.verified_cases(tmp_path, "cache", "can", "development", protocol, manifest)


def test_plan_excludes_recorded_exposure_and_rejects_rgb_overlap(tmp_path, monkeypatch):
    policy = copy.deepcopy(pilot.read(pilot.POLICY))
    policy.update(
        categories=["can"],
        fit_parents=2,
        calibration_parents=2,
        development_parents=2,
        confirmation_parents=2,
    )
    policy_path = tmp_path / "policy.json"
    pilot.write(policy_path, policy)
    monkeypatch.setattr(pilot, "POLICY", policy_path)
    monkeypatch.setattr(pilot, "source_hashes", lambda: {})
    monkeypatch.setattr(
        pilot,
        "source_record",
        lambda _: {
            "git_dirty": False,
            "upstream_dirty": False,
            "upstream_commit": pilot.UPSTREAM_SHA,
        },
    )
    rows = []
    for index in range(10):
        image_id = f"can/train/good/{index:03d}.png"
        path = tmp_path / "data" / image_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]))
        rows.append(
            {
                "image_id": image_id,
                "split": "train",
                "sha256": pilot.digest(path),
                "decoded_rgb_sha256": f"rgb-{index}",
            }
        )
    reused_path, excluded_path, audit_path = [
        tmp_path / f"{x}.json" for x in ("roles", "exclusion", "rgb")
    ]
    pilot.write(reused_path, {"images": {"can": {"fit": rows[:2], "calibration": rows[2:4]}}})
    pilot.write(excluded_path, {"excluded_image_ids": [r["image_id"] for r in rows[:6]]})
    pilot.write(audit_path, {"rows": rows})
    pilot.plan(tmp_path / "run", tmp_path / "data", reused_path, excluded_path, audit_path)
    roles = pilot.read(tmp_path / "run/protocol.json")["roles"]["can"]
    new_ids = {r["image_id"] for role in ("development", "confirmation") for r in roles[role]}
    assert new_ids == {r["image_id"] for r in rows[6:]}
    rows[6]["decoded_rgb_sha256"] = rows[0]["decoded_rgb_sha256"]
    pilot.write(audit_path, {"rows": rows})
    with pytest.raises(ValueError, match="RGB overlap"):
        pilot.plan(tmp_path / "rejected", tmp_path / "data", reused_path, excluded_path, audit_path)
    assert not (tmp_path / "rejected").exists()
