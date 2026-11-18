"""Reject incomplete or altered evidence before decoding presentation images."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_blade_gallery.py"
spec = importlib.util.spec_from_file_location("blade_gallery_handoff", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, *, decisions=59004, correct_protocol=True):
    (tmp_path / "real-test").mkdir()
    protocol = tmp_path / "real-test-protocol.json"
    protocol.write_text(json.dumps({"domain_order": ["illumination"]}))
    (tmp_path / "real-test-decision.json").write_text(
        json.dumps(
            {
                "all_image_decisions_independently_recounted": decisions,
                "exploratory": True,
                "source_confirmation_all_gates_passed": False,
                "real_test_protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()
                if correct_protocol
                else "0" * 64,
            }
        )
    )
    (tmp_path / "real-test/complete.json").write_text(
        json.dumps({"domains": {"illumination": {"path": "scores"}}})
    )


@pytest.mark.parametrize("decisions,correct_protocol", [(59003, True), (59004, False)])
def test_incomplete_or_mixed_protocol_cannot_render(tmp_path, decisions, correct_protocol):
    fixture(tmp_path, decisions=decisions, correct_protocol=correct_protocol)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="complete unaltered"):
        module.run(tmp_path, output)
    assert not output.exists()


def test_changed_scores_are_rejected_before_native_inputs(tmp_path):
    fixture(tmp_path)
    scores = tmp_path / "scores"
    scores.mkdir()
    (scores / "image-scores.npz").write_bytes(b"changed scores, not a model output")
    (scores / "complete.json").write_text(
        json.dumps({"artifacts": {"image-scores.npz": {"sha256": "0" * 64}}})
    )
    with pytest.raises(ValueError, match="recorded artifact changed"):
        module.run(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
