import io
import json
import tarfile
from pathlib import Path

import pytest

from lc_simplenet.ad2_archive import ArchiveError, extract_archive, inspect_archive


def _regular(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.type = tarfile.REGTYPE
    return info


def _write_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> Path:
    with tarfile.open(path, "w:gz") as bundle:
        for info, payload in members:
            bundle.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return path


def _fabric_archive(path: Path, payload: bytes = b"normal-png-bytes") -> Path:
    return _write_tar(
        path,
        [
            (_regular("license.txt", b"cc-by-nc-sa"), b"cc-by-nc-sa"),
            (_regular("fabric/train/good/000.png", payload), payload),
        ],
    )


def test_safe_archive_extracts_and_records_hash(tmp_path):
    payload = b"normal-png-bytes"
    archive = _fabric_archive(tmp_path / "fabric.tar.gz", payload)
    root = tmp_path / "mvtec_ad2"
    report_path = tmp_path / "audits/ad2-acquisition.json"
    report = extract_archive(archive, root, acquisition_out=report_path)
    assert report["extract_skipped"] is False
    assert (root / "fabric/train/good/000.png").read_bytes() == payload
    assert (root / "license.txt").read_text() == "cc-by-nc-sa"
    saved = json.loads(report_path.read_text())
    assert saved["archive_sha256"] == report["archive_sha256"]
    assert saved["top_level"]["fabric"] == 1
    assert "note" in saved


def test_repeated_extract_skips_when_sha_matches(tmp_path):
    payload = b"normal-png-bytes"
    archive = _fabric_archive(tmp_path / "fabric.tar.gz", payload)
    root = tmp_path / "mvtec_ad2"
    report_path = tmp_path / "audits/ad2-acquisition.json"
    first = extract_archive(archive, root, acquisition_out=report_path)
    marker = root / "fabric/train/good/000.png"
    acquired = json.loads(report_path.read_text())["acquired_at_utc"]
    second = extract_archive(archive, root, acquisition_out=report_path)
    assert second["extract_skipped"] is True
    assert second["archive_sha256"] == first["archive_sha256"]
    assert marker.read_bytes() == payload
    assert json.loads(report_path.read_text())["acquired_at_utc"] == acquired


def test_changed_archive_refuses_overwrite(tmp_path):
    first_payload = b"first-bytes"
    archive = _fabric_archive(tmp_path / "fabric.tar.gz", first_payload)
    root = tmp_path / "mvtec_ad2"
    report_path = tmp_path / "audits/ad2-acquisition.json"
    extract_archive(archive, root, acquisition_out=report_path)
    changed = _fabric_archive(tmp_path / "other.tar.gz", b"second-bytes")
    with pytest.raises(ArchiveError, match="SHA-256 differs"):
        extract_archive(changed, root, acquisition_out=report_path)
    assert (root / "fabric/train/good/000.png").read_bytes() == first_payload


def test_existing_conflicting_extraction_refuses(tmp_path):
    root = tmp_path / "mvtec_ad2"
    existing = root / "fabric/train/good/000.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already-present")
    archive = _fabric_archive(tmp_path / "fabric.tar.gz")
    with pytest.raises(FileExistsError, match="conflicts"):
        extract_archive(archive, root, acquisition_out=tmp_path / "audits/ad2-acquisition.json")
    assert existing.read_bytes() == b"already-present"


@pytest.mark.parametrize(
    ("name", "link", "kind"),
    [
        ("../escape.png", None, "file"),
        ("/tmp/absolute.png", None, "file"),
        ("fabric/../secret.png", None, "file"),
        ("fabric/link.png", "train/good/000.png", "symlink"),
    ],
)
def test_unsafe_members_are_rejected(tmp_path, name, link, kind):
    info = tarfile.TarInfo(name)
    payload: bytes | None = b"x"
    if kind == "symlink":
        info.type = tarfile.SYMTYPE
        info.linkname = link
        payload = None
        info.size = 0
    else:
        info.size = 1
        info.type = tarfile.REGTYPE
    archive = _write_tar(tmp_path / "bad.tar.gz", [(info, payload)])
    with pytest.raises(ArchiveError, match="Refusing"):
        inspect_archive(archive)
    with pytest.raises(ArchiveError, match="Refusing"):
        extract_archive(archive, tmp_path / "out")


def test_inspect_does_not_require_an_extract_root(tmp_path):
    archive = _write_tar(
        tmp_path / "can.tar.gz",
        [(_regular("can/validation/good/000.png", b"abc"), b"abc")],
    )
    inventory = inspect_archive(archive)
    assert inventory["top_level"] == {"can": 1}
    assert inventory["kinds"] == {"file": 1}
    assert "entries" in inventory
