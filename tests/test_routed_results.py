"""Check the delivered matrix and reject a rehashed incomplete copy."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "routed_results_verifier", ROOT / "scripts/verify_routed_results.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_complete_delivered_matrix_retains_both_outcomes():
    result = verifier.verify(ROOT / "evidence")
    assert result["ad2_image_decisions"] == 22764
    assert result["blade_image_decisions"] == 59004
    assert result["ad2_targets_met"] is False
    assert result["blade_mean_targets_met"] is True


def test_missing_seed_row_fails_even_with_updated_file_hash(tmp_path):
    folder = tmp_path / "evidence"
    shutil.copytree(ROOT / "evidence", folder)
    path = folder / "blade-domain-seed-metrics.csv"
    raw = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(raw[:-1]))
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete blade matrix"):
        verifier.verify(folder)
