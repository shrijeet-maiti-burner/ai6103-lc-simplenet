import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from lc_simplenet.source_identity import BUNDLE_NAME, UPSTREAM_PATH, git, source_record

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bundle_builder", ROOT / "scripts/build_source_bundle.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def commit(root):
    git(
        root,
        "-c",
        "user.name=bundle test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "source fixture",
    )


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("bundle construction needs Git; archive execution does not")
    root = tmp_path / "project"
    root.mkdir()
    git(root, "init", "-q")
    (root / "main.py").write_bytes(b"print('source')\n")
    upstream = root / UPSTREAM_PATH
    upstream.mkdir(parents=True)
    git(upstream, "init", "-q")
    (upstream / "simplenet.py").write_bytes(b"# upstream fixture\n")
    (upstream / "LICENSE").write_bytes(b"fixture licence\n")
    cache = upstream / "__pycache__/simplenet.cpython-38.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"obsolete author-tracked bytecode")
    git(upstream, "add", ".")
    commit(upstream)
    pin = git(upstream, "rev-parse", "HEAD")
    git(root, "add", "main.py")
    (root / "main.py").chmod(0o755)
    git(root, "update-index", "--chmod=+x", "main.py")
    git(root, "update-index", "--add", "--cacheinfo", f"160000,{pin},{UPSTREAM_PATH}")
    commit(root)
    monkeypatch.setattr(builder, "UPSTREAM_SHA", pin)
    return root


def extract_bundle(root, tmp_path):
    output = tmp_path / "source.zip"
    builder.build_bundle(root, output)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        archive.extractall(extracted)
    return output, extracted


def test_deterministic_zip_includes_upstream_and_excludes_untracked(checkout, tmp_path):
    (checkout / "private-secret.txt").write_text("must stay outside the zip")
    first, extracted = extract_bundle(checkout, tmp_path)
    second = tmp_path / "second.zip"
    builder.build_bundle(checkout, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.getinfo("main.py").external_attr >> 16 == 0o100755
        assert archive.getinfo(UPSTREAM_PATH + "/LICENSE").external_attr >> 16 == 0o100644
    assert (extracted / UPSTREAM_PATH / "simplenet.py").is_file()
    assert (extracted / UPSTREAM_PATH / "LICENSE").is_file()
    assert not (extracted / "private-secret.txt").exists()
    assert not list(extracted.rglob(".git"))
    assert not list(extracted.rglob("*.pyc"))
    manifest = json.loads((extracted / BUNDLE_NAME).read_text())
    assert manifest["omitted_bytecode"] == [UPSTREAM_PATH + "/__pycache__/simplenet.cpython-38.pyc"]
    assert manifest["files"]["main.py"] == hashlib.sha256(b"print('source')\n").hexdigest()
    assert source_record(extracted)["git_dirty"] is False


def test_archive_runs_without_git_and_does_not_inherit_parent(checkout, tmp_path, monkeypatch):
    _, extracted = extract_bundle(checkout, tmp_path)
    git(tmp_path, "init", "-q")
    expected = git(checkout, "rev-parse", "HEAD")

    def no_git(*args, **kwargs):
        raise AssertionError("archive execution must not invoke Git")

    monkeypatch.setattr("lc_simplenet.source_identity.git", no_git)
    record = source_record(extracted)
    assert record["source_kind"] == "source_archive"
    assert record["git_commit"] == expected
    assert record["git_diff_sha256"] is None
    assert record["source_files_checked"] == 3


@pytest.mark.parametrize("target", ["main.py", UPSTREAM_PATH + "/simplenet.py"])
def test_missing_archive_source_is_recorded(checkout, tmp_path, target):
    _, extracted = extract_bundle(checkout, tmp_path)
    (extracted / target).unlink()
    record = source_record(extracted)
    assert record["git_dirty"] is True
    assert record["source_files_changed"] == [target]
    assert record["upstream_dirty"] == target.startswith(UPSTREAM_PATH)


def test_missing_manifest_cannot_use_an_enclosing_repository(checkout):
    child = checkout / "unrelated-folder"
    child.mkdir()
    with pytest.raises(ValueError, match="this project's Git checkout"):
        source_record(child)


def test_non_ascii_git_diff_has_a_portable_source_identity(checkout):
    (checkout / "main.py").write_text("# gamma: γ → γ²\n", encoding="utf-8")
    record = source_record(checkout)
    assert record["git_dirty"] is True
    assert len(record["git_diff_sha256"]) == 64


def test_dirty_source_or_wrong_pin_cannot_be_packaged(checkout, tmp_path, monkeypatch):
    (checkout / "main.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="Commit tracked changes"):
        builder.build_bundle(checkout, tmp_path / "dirty.zip")
    git(checkout, "restore", "main.py")
    monkeypatch.setattr(builder, "UPSTREAM_SHA", "0" * 40)
    with pytest.raises(ValueError, match="required pinned commit"):
        builder.build_bundle(checkout, tmp_path / "wrong.zip")


def test_manifest_path_cannot_escape_extraction_root(checkout, tmp_path):
    _, extracted = extract_bundle(checkout, tmp_path)
    path = extracted / BUNDLE_NAME
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside.txt"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe source bundle path"):
        source_record(extracted)
