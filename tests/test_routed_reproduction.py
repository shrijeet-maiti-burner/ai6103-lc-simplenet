"""Prevent incomplete or altered reproduction inputs from becoming test results."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reproduce_routed_ad2.py"
spec = importlib.util.spec_from_file_location("routed_reproduction_cli", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unverified_execution_archive_is_rejected(tmp_path):
    path = tmp_path / "source.zip"
    path.write_bytes(b"a different execution source")
    with pytest.raises(ValueError, match="archive hash differs"), module.execution_source(path):
        pytest.fail("unverified code must not be extracted")


@pytest.mark.parametrize("mechanics,completed", [(True, True), (False, False)])
def test_incomplete_training_cannot_enter_target_scoring(tmp_path, mechanics, completed):
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "mechanics_only": mechanics,
                "source_archive_sha256": module.SOURCE_SHA256,
            }
        )
    )
    (tmp_path / "complete.json").write_text(json.dumps({"completed": completed}))
    args = argparse.Namespace(run=tmp_path)
    with pytest.raises(ValueError, match="completed full-budget run"):
        module.score(args, tmp_path / "no-target-files-exist")


def test_bad_source_role_leaves_no_published_staging(tmp_path):
    source = tmp_path / "source"
    (source / "configs").mkdir(parents=True)
    (source / "references/ad2-normal-manifests").mkdir(parents=True)
    (source / "configs/ad2-study-policy.json").write_text(
        json.dumps(
            {
                "category_counts_train_validation": {"can": [1, 0]},
            }
        )
    )
    (source / "references/ad2-normal-manifests/can.json").write_text(
        json.dumps(
            {
                "image_id": "can/train/bad/not-normal.png",
                "split": "train",
                "label": 1,
                "mask": None,
            }
        )
    )
    output = tmp_path / "prepared"
    args = argparse.Namespace(
        out=output, category="can", data_root=tmp_path / "unread", textures_root=tmp_path / "unread"
    )
    with pytest.raises(ValueError, match="only original normal sources"):
        module.prepare(args, source)
    assert list(output.iterdir()) == []
