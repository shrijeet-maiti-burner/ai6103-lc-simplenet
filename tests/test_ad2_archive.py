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


def test_safe_archive_extracts_and_records_hash(tmp_path):
    payload = b"normal-png-bytes"
    archive = _write_tar(
        tmp_path / "fabric.tar.gz",
        [
            (_regular("license.txt", b"cc-by-nc-sa"), b"cc-by-nc-sa"),
            (_regular("fabric/train/good/000.png", payload), payload),
        ],
    )
    root = tmp_path / "mvtec_ad2"
    report_path = tmp_path / "audits/ad2-acquisition.json"
    report = extract_archive(archive, root, acquisition_out=report_path)
    assert (root / "fabric/train/good/000.png").read_bytes() == payload
    assert (root / "license.txt").read_text() == "cc-by-nc-sa"
    saved = json.loads(report_path.read_text())
    assert saved["archive_sha256"] == report["archive_sha256"]
    assert saved["top_level"]["fabric"] == 1
    assert "note" in saved
    with pytest.raises(FileExistsError, match="overwrite"):
        extract_archive(archive, root, acquisition_out=tmp_path / "audits/other.json")


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
