import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_release", ROOT / "scripts/prepare_study_artifacts.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(path, name="paired/can-B0-seed0/run.json"):
    with zipfile.ZipFile(path / "tiny.zip", "w") as archive:
        archive.writestr(name, "{}")
        archive.writestr("single/can/unused.pt", b"unused")
    checksum = hashlib.sha256((path / "tiny.zip").read_bytes()).hexdigest()
    (path / "tiny.sha256").write_text(checksum + "  tiny.zip\n")


def test_checksums_and_scope_before_staging(tmp_path):
    fixture(tmp_path)
    result = module.stage(tmp_path, tmp_path / "out")
    assert result["paired_runs"] == 1 and result["files"] == 1
    assert not (tmp_path / "out/single").exists()
    with pytest.raises(FileExistsError):
        module.stage(tmp_path, tmp_path / "out")


def test_changed_archive_fails_before_output_creation(tmp_path):
    fixture(tmp_path)
    (tmp_path / "tiny.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.stage(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_release_traversal_rejected(tmp_path):
    fixture(tmp_path, "paired/../../escape.txt")
    with pytest.raises(ValueError, match="unsafe release path"):
        module.stage(tmp_path, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()
