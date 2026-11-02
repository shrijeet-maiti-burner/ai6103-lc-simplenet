import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lc_simplenet.data import NormalManifestDataset
from lc_simplenet.data_manifest import build_normal_manifest, write_normal_manifest


def _fixture(root, size=(40, 40)):
    rng = np.random.default_rng(0)
    for i in range(5):
        path = root / "bottle/train/good" / f"{i:03}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    records, audit = build_normal_manifest(root, "mvtec_ad", ["bottle"])
    manifest = root / "normal.jsonl"
    write_normal_manifest(records, audit, manifest)
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _source(root, manifest, digest, split, resize):
    return NormalManifestDataset(
        manifest,
        root,
        dataset="mvtec_ad",
        category="bottle",
        split=split,
        expected_sha256=digest,
        output_size=(resize, resize),
    )


def test_input_matches_upstream_mvtec_train_transform(tmp_path):
    torch = pytest.importorskip("torch")
    from lc_simplenet.baseline import UpstreamInput, upstream_module

    manifest, digest = _fixture(tmp_path)
    upstream = upstream_module("datasets.mvtec").MVTecDataset(
        str(tmp_path), "bottle", resize=33, imagesize=28
    )
    by_path = {item[2]: i for i, item in enumerate(upstream.data_to_iterate)}
    for split in ("train", "validation"):
        ours = UpstreamInput(_source(tmp_path, manifest, digest, split, 33), 28)
        for sample in ours:
            expected = upstream[by_path[str(tmp_path / sample["image_id"])]]["image"]
            assert torch.equal(sample["image"], expected)


def test_upstream_imports_write_no_bytecode_into_the_submodule(monkeypatch):
    pytest.importorskip("torch")
    from lc_simplenet.baseline import ROOT, upstream_module

    source = ROOT / "third_party/SimpleNet/datasets/mvtec.py"
    cache = Path(importlib.util.cache_from_source(str(source)))
    if cache.exists():
        pytest.skip("a cache from an earlier import already exists")
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    for name in ("datasets", "datasets.mvtec"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    upstream_module("datasets.mvtec")
    assert not cache.exists()
    assert sys.dont_write_bytecode is False


def test_center_crop_box_matches_torchvision():
    torch = pytest.importorskip("torch")
    import torchvision.transforms.functional as TF

    from lc_simplenet.baseline import center_crop_box

    assert center_crop_box(329, 288) == [20, 20, 308, 308]
    for resize, size in ((329, 288), (33, 28), (30, 28)):
        canvas = torch.arange(resize * resize).reshape(1, resize, resize)
        top, left, bottom, right = center_crop_box(resize, size)
        assert torch.equal(TF.center_crop(canvas, [size, size]), canvas[:, top:bottom, left:right])


def test_input_rejects_non_square_sources(tmp_path):
    pytest.importorskip("torch")
    from lc_simplenet.baseline import UpstreamInput

    manifest, digest = _fixture(tmp_path, size=(40, 30))
    with pytest.raises(ValueError, match="square"):
        UpstreamInput(_source(tmp_path, manifest, digest, "train", 33), 28)


def test_stretch_resizes_non_square_ad2_full_frame(tmp_path):
    torch = pytest.importorskip("torch")
    import torchvision.transforms.functional as TF

    from lc_simplenet.baseline import IMAGENET_MEAN, IMAGENET_STD, UpstreamInput

    rng = np.random.default_rng(0)
    for split, count in (("train", 3), ("validation", 2)):
        for i in range(count):
            path = tmp_path / "fabric" / split / "good" / f"{i:03}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)).save(path)
    records, audit = build_normal_manifest(tmp_path, "mvtec_ad2", ["fabric"])
    manifest = tmp_path / "ad2.jsonl"
    write_normal_manifest(records, audit, manifest)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    source = NormalManifestDataset(
        manifest,
        tmp_path,
        dataset="mvtec_ad2",
        category="fabric",
        split="train",
        expected_sha256=digest,
        output_size=(32, 32),
    )
    with pytest.raises(ValueError, match="square"):
        UpstreamInput(source, 32)
    for sample in UpstreamInput(source, 32, stretch=True):
        with Image.open(tmp_path / sample["image_id"]) as image:
            resized = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        expected = torch.from_numpy(np.asarray(resized, dtype=np.float32) / np.float32(255))
        expected = TF.normalize(expected.permute(2, 0, 1), IMAGENET_MEAN, IMAGENET_STD)
        assert torch.equal(sample["image"], expected)


def test_final_state_trains_and_reloads_to_identical_scores(tmp_path):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from lc_simplenet import baseline

    manifest, digest = _fixture(tmp_path, size=(80, 80))
    settings = {**baseline.UPSTREAM_AD, "meta_epochs": 1, "gan_epochs": 1}

    def loader(split):
        source = baseline.UpstreamInput(_source(tmp_path, manifest, digest, split, 72), 64)
        return torch.utils.data.DataLoader(source, batch_size=2)

    def model():
        torch.manual_seed(0)
        backbone = torchvision.models.wide_resnet50_2(weights=None)
        return baseline.build_model(backbone, torch.device("cpu"), 64, settings)

    trained, initial = model(), model()
    baseline.train_fixed(trained, loader("train"), tmp_path / "logs")
    state = baseline.trained_state(trained)
    assert set(state) == {"discriminator", "pre_projection"}
    assert any(
        not torch.equal(value, initial.discriminator.state_dict()[key].cpu())
        for key, value in state["discriminator"].items()
    )
    checkpoint = baseline.training_checkpoint(trained, settings, 0)
    assert checkpoint["schema"] == 2 and checkpoint["seed"] == 0
    assert checkpoint["optimizers"]["discriminator"]["state"]
    assert checkpoint["optimizers"]["adapter"]["state"]
    assert set(checkpoint["rng_state"]) == {"python_rng", "numpy_rng", "torch_rng", "cuda_rng"}
    torch.save(checkpoint, tmp_path / "state.pt")
    restored = torch.load(tmp_path / "state.pt", weights_only=True)
    baseline.load_trained_state(initial, restored["state"])
    first = baseline.predict(trained, loader("validation"))
    second = baseline.predict(initial, loader("validation"))
    np.testing.assert_array_equal(first["image_id"], second["image_id"])
    np.testing.assert_array_equal(first["image_score"], second["image_score"])
    np.testing.assert_array_equal(first["map"], second["map"])
    assert first["map"].shape == (1, 64, 64) and first["image_score"].dtype == np.float32
