import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from lc_simplenet.data import NormalManifestDataset
from lc_simplenet.data_manifest import build_normal_manifest, write_normal_manifest


def _fixture(root, dataset="mvtec_ad"):
    category = "bottle" if dataset == "mvtec_ad" else "fabric"
    for i in range(5):
        path = root / category / "train/good" / f"{i:03}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (5, 7), i * 30).save(path)
    if dataset == "mvtec_ad2":
        path = root / category / "validation/good/000.png"
        path.parent.mkdir(parents=True)
        Image.new("L", (5, 7), 200).save(path)
    records, audit = build_normal_manifest(root, dataset, [category])
    manifest = root / "normal.jsonl"
    write_normal_manifest(records, audit, manifest)
    return manifest, category, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _load(root, manifest, category, digest, *, split="train", dataset="mvtec_ad", **kwargs):
    return NormalManifestDataset(
        manifest,
        root,
        dataset=dataset,
        category=category,
        split=split,
        expected_sha256=digest,
        **kwargs,
    )


def test_ad_holdout_is_enforced_despite_shared_source_folder(tmp_path):
    manifest, category, digest = _fixture(tmp_path)
    train = _load(tmp_path, manifest, category, digest)
    validation = _load(tmp_path, manifest, category, digest, split="validation")
    train_ids = {r.image_id for r in train.records}
    validation_ids = {r.image_id for r in validation.records}
    assert len(train) == 4 and len(validation) == 1
    assert train_ids.isdisjoint(validation_ids)
    assert len(train_ids | validation_ids) == 5
    assert all("/train/good/" in name for name in validation_ids)
    sample = validation[0]
    assert sample["image"].shape == (7, 5, 3)
    assert sample["image"].dtype == np.float32
    np.testing.assert_array_equal(sample["image"][:, :, 0], sample["image"][:, :, 2])
    assert sample["label"] == 0 and sample["split"] == "validation"


def test_ad2_split_and_explicit_resize_keep_native_metadata(tmp_path):
    manifest, category, digest = _fixture(tmp_path, "mvtec_ad2")
    validation = _load(
        tmp_path,
        manifest,
        category,
        digest,
        split="validation",
        dataset="mvtec_ad2",
        output_size=(4, 6),
    )
    sample = validation[0]
    assert len(validation) == 1 and sample["image"].shape == (4, 6, 3)
    np.testing.assert_array_equal(sample["original_size"], [7, 5])
    np.testing.assert_array_equal(sample["image_size"], [4, 6])
    np.testing.assert_allclose(sample["image"], 200 / 255)


def test_changed_manifest_and_changed_pixels_fail(tmp_path):
    manifest, category, digest = _fixture(tmp_path)
    source = _load(tmp_path, manifest, category, digest)
    (tmp_path / source.records[0].image_id).write_bytes(b"changed image")
    with pytest.raises(ValueError, match="Image bytes changed"):
        source[0]
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Manifest SHA-256"):
        _load(tmp_path, manifest, category, digest)


@pytest.mark.parametrize("bad_path", ["bottle/test/good/000.png", "bottle/train/good/../000.png"])
def test_test_paths_and_traversal_fail_before_any_image_is_opened(tmp_path, bad_path):
    manifest, category, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["image_id"] = rows[0]["relative_path"] = bad_path
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="normal-image/split/path"):
        _load(tmp_path, manifest, category, digest)


def test_cross_split_content_leakage_is_rejected_before_filtering(tmp_path):
    manifest, category, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    train = next(row for row in rows if row["split"] == "train")
    validation = next(row for row in rows if row["split"] == "validation")
    validation["rgb_sha256"] = train["rgb_sha256"]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="crosses train/validation"):
        _load(tmp_path, manifest, category, digest)


@pytest.mark.parametrize("field,value", [("split", "validation"), ("label", 1)])
def test_ad2_rejects_changed_official_membership_and_non_normal_labels(tmp_path, field, value):
    manifest, category, _ = _fixture(tmp_path, "mvtec_ad2")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    train = next(row for row in rows if row["split"] == "train")
    train[field] = value
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="normal-image/split/path"):
        _load(tmp_path, manifest, category, digest, dataset="mvtec_ad2")


def test_default_torch_collation_preserves_raw_bhwc_contract(tmp_path):
    torch = pytest.importorskip("torch")
    manifest, category, digest = _fixture(tmp_path)
    source = _load(tmp_path, manifest, category, digest)
    batch = next(iter(torch.utils.data.DataLoader(source, batch_size=2, shuffle=False)))
    assert batch["image"].shape == (2, 7, 5, 3)
    assert batch["image"].dtype == torch.float32
    assert batch["original_size"].shape == (2, 2)
    assert batch["image"].min() >= 0 and batch["image"].max() <= 1
