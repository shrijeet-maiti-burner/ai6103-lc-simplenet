import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from lc_simplenet.data_manifest import build_normal_manifest, write_normal_manifest


def _image(root: Path, relative: str, value: int) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (5, 7), (value, value // 2, 255 - value)).save(path)
    return path


def test_ad_holdout_keeps_duplicates_together_and_ignores_test(tmp_path):
    for i in range(5):
        _image(tmp_path, f"bottle/train/good/{i:03}.png", i * 30)
    _image(tmp_path, "bottle/train/good/duplicate.png", 0)
    forbidden = tmp_path / "bottle/test/bad/corrupt.png"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("must not be read")
    records, audit = build_normal_manifest(tmp_path, "mvtec_ad", ["bottle"], seed=3)
    again, _ = build_normal_manifest(tmp_path, "mvtec_ad", ["bottle"], seed=3)
    assert records == again
    duplicate = [r for r in records if r["relative_path"].endswith(("000.png", "duplicate.png"))]
    assert len({r["split"] for r in duplicate}) == 1
    assert {r["split"] for r in records} == {"train", "validation"}
    assert all(r["label"] == 0 and r["mask"] is None for r in records)
    assert audit["duplicate_rgb_groups"] == 1
    assert audit["test_directories_read"] is False
    assert all(not Path(r["relative_path"]).is_absolute() for r in records)


def test_ad2_preserves_official_split_and_rejects_overlap(tmp_path):
    _image(tmp_path, "fabric/train/good/000.png", 10)
    _image(tmp_path, "fabric/validation/good/000.png", 20)
    records, audit = build_normal_manifest(tmp_path, "mvtec_ad2", ["fabric"])
    assert audit["counts"] == {"fabric/train": 1, "fabric/validation": 1}
    assert audit["split_policy"] == "official"
    _image(tmp_path, "fabric/validation/good/001.png", 10)
    with pytest.raises(ValueError, match="crossing train/validation"):
        build_normal_manifest(tmp_path, "mvtec_ad2", ["fabric"])
    assert records[0]["width"] == 5 and records[0]["height"] == 7


def test_manifest_is_portable_auditable_and_not_overwritten(tmp_path):
    _image(tmp_path, "fabric/train/good/000.png", 10)
    _image(tmp_path, "fabric/validation/good/000.png", 20)
    records, audit = build_normal_manifest(tmp_path, "mvtec_ad2", ["fabric"])
    output = tmp_path / "manifest.jsonl"
    write_normal_manifest(records, audit, output)
    loaded = [json.loads(line) for line in output.read_text().splitlines()]
    assert loaded == records
    saved_audit = json.loads(output.with_suffix(".audit.json").read_text())
    assert saved_audit["manifest_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_normal_manifest(records, audit, output)


def test_invalid_category_and_insufficient_content_fail(tmp_path):
    with pytest.raises(ValueError, match="Unknown categories"):
        build_normal_manifest(tmp_path, "mvtec_ad2", ["../bottle"])
    _image(tmp_path, "bottle/train/good/000.png", 10)
    _image(tmp_path, "bottle/train/good/001.png", 10)
    with pytest.raises(ValueError, match="two distinct"):
        build_normal_manifest(tmp_path, "mvtec_ad", ["bottle"])


@pytest.mark.parametrize("mode", ["RGBA", "P", "LA", "I;16"])
def test_unsupported_modes_fail_during_inventory(tmp_path, mode):
    source = tmp_path / "bottle/train/good/unsupported.png"
    source.parent.mkdir(parents=True)
    Image.new(mode, (5, 7)).save(source)
    with pytest.raises(ValueError, match=r"unsupported image mode .*: unsupported\.png"):
        build_normal_manifest(tmp_path, "mvtec_ad", ["bottle"])
