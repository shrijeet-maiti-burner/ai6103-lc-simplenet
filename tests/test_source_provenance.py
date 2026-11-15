import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "source_provenance_writer",
    Path(__file__).resolve().parents[1] / "scripts/write_source_provenance.py",
)
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)


@pytest.fixture
def source(tmp_path, monkeypatch):
    (tmp_path / "source.py").write_text("print('source')\n")
    identity = {
        "source_kind": "git",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "upstream_dirty": False,
        "upstream_commit": writer.UPSTREAM_SHA,
    }
    monkeypatch.setattr(writer, "source_record", lambda _: identity)
    monkeypatch.setattr(writer, "git", lambda *_: "source.py\0")
    return tmp_path, identity


def test_manifest_binds_source_bytes_and_names_its_checksum(source):
    root, _ = source
    output = root / "results/run-source.json"
    receipt = writer.write_provenance(root, output)
    manifest = json.loads(output.read_text())
    assert manifest["files"] == {
        "source.py": hashlib.sha256((root / "source.py").read_bytes()).hexdigest()
    }
    assert manifest["git_head"] == "a" * 40
    assert output.with_suffix(".sha256").read_text().split() == [receipt["sha256"], output.name]
    assert receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        writer.write_provenance(root, output)


def test_dirty_source_is_refused_before_writing(source):
    root, identity = source
    identity["git_dirty"] = True
    with pytest.raises(ValueError, match="clean"):
        writer.write_provenance(root, root / "manifest.json")
    assert not (root / "manifest.json").exists()


def test_existing_sidecar_is_preserved(source):
    root, _ = source
    (root / "manifest.sha256").write_text("existing")
    with pytest.raises(FileExistsError):
        writer.write_provenance(root, root / "manifest.json")
    assert not (root / "manifest.json").exists()
    assert (root / "manifest.sha256").read_text() == "existing"


def test_verified_source_archive_needs_no_git(source, monkeypatch):
    root, identity = source
    identity["source_kind"] = "source_archive"
    (root / writer.BUNDLE_NAME).write_text(json.dumps({"files": {"source.py": "unused"}}))
    monkeypatch.setattr(writer, "git", lambda *_: pytest.fail("archive execution invoked git"))
    writer.write_provenance(root, root / "manifest.json")
    assert json.loads((root / "manifest.json").read_text())["source_kind"] == "source_archive"
