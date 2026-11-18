import hashlib
from pathlib import Path

import pytest
from PIL import Image

from lc_simplenet.ad2_audit import AuditError, audit_mvtec_ad2, write_audit
from lc_simplenet.data import NormalManifestDataset
from lc_simplenet.data_manifest import build_normal_manifest, write_normal_manifest


def _png(path: Path, value: int = 10, size=(4, 6), mode="RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "RGB":
        Image.new("RGB", size, (value, value // 2, 255 - value)).save(path)
    else:
        Image.new(mode, size, value).save(path)


def _fabric_tree(root: Path) -> None:
    for i in range(2):
        _png(root / f"fabric/train/good/{i:03}.png", 20 + i)
    _png(root / "fabric/validation/good/000.png", 40)
    _png(root / "fabric/test_public/good/000.png", 50)
    _png(root / "fabric/test_public/bad/000.png", 60)
    _png(root / "fabric/test_public/ground_truth/bad/000.png", 255, mode="L")
    _png(root / "fabric/test_private/000_regular.png", 70)
    _png(root / "fabric/test_private_mixed/000_mixed.png", 80)
    (root / "license.txt").write_text("cc")


def test_audit_tags_paper_counts_separately_and_does_not_label_private(tmp_path):
    _fabric_tree(tmp_path)
    report = audit_mvtec_ad2(tmp_path)
    fabric = report["categories"][0]
    assert report["present_categories"] == ["fabric"]
    assert "can" in report["missing_categories"]
    assert report["private_labels_inferred"] is False
    assert all(image["private_label"] is None for image in fabric["images"])
    train = next(row for row in fabric["comparisons"] if row["field"] == "train")
    assert train["expected"] == 387
    assert train["expected_source"].startswith("paper:")
    assert train["observed"] == 2
    assert train["observed_source"] == "files"
    assert train["match"] is False
    assert fabric["counts"]["test_public_masks"]["observed"] == 1
    ids = {image["relative_path"] for image in fabric["images"]}
    assert "fabric/test_private/000_regular.png" in ids


def test_unexpected_entries_and_duplicate_bytes_are_reported(tmp_path):
    _fabric_tree(tmp_path)
    extra = tmp_path / "not_a_category"
    extra.mkdir()
    _png(tmp_path / "fabric/train/good/dup.png", 20)
    report = audit_mvtec_ad2(tmp_path)
    assert "not_a_category" in report["unexpected_entries"]
    assert report["duplicate_sha256_groups"]


def test_symlink_dataset_root_is_rejected_before_resolution(tmp_path):
    root = tmp_path / "dataset"
    _fabric_tree(root)
    link = tmp_path / "linked-dataset"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(AuditError, match="symlink dataset root"):
        audit_mvtec_ad2(link)


@pytest.mark.parametrize(
    "kind",
    ["file_outside", "file_inside", "directory_outside"],
)
def test_symlink_escape_is_rejected(tmp_path, kind):
    root = tmp_path / "dataset"
    _fabric_tree(root)
    if kind == "file_outside":
        outside = tmp_path / "outside.png"
        _png(outside, 9)
        (root / "fabric/train/good/link.png").symlink_to(outside)
    elif kind == "file_inside":
        target = root / "fabric/train/good/000.png"
        (root / "fabric/train/good/link.png").symlink_to(target)
    else:
        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()
        (root / "fabric/sneaky").symlink_to(outside_dir)
    with pytest.raises(AuditError, match="symlink"):
        audit_mvtec_ad2(root)


def test_normal_manifest_builder_still_ignores_test_and_rejects_leakage(tmp_path):
    _fabric_tree(tmp_path)
    records, audit = build_normal_manifest(tmp_path, "mvtec_ad2", ["fabric"])
    assert audit["test_directories_read"] is False
    assert {r["split"] for r in records} == {"train", "validation"}
    assert all("/test_" not in r["relative_path"] for r in records)
    leaked = dict(records[0])
    leaked["relative_path"] = "fabric/test_public/bad/000.png"
    leaked["image_id"] = leaked["relative_path"]
    leaked["label"] = 0
    leaked["mask"] = None
    output = tmp_path / "leaked.jsonl"
    write_normal_manifest([leaked], {**audit, "records": 1}, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="normal-image/split/path"):
        NormalManifestDataset(
            output,
            tmp_path,
            dataset="mvtec_ad2",
            category="fabric",
            split="train",
            expected_sha256=digest,
        )


def test_write_audit_refuses_overwrite(tmp_path):
    _fabric_tree(tmp_path)
    report = audit_mvtec_ad2(tmp_path)
    output = tmp_path / "audit.json"
    write_audit(report, output)
    assert output.with_suffix(".md").exists()
    with pytest.raises(FileExistsError):
        write_audit(report, output)
