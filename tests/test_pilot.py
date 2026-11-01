import copy
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lc_simplenet.pilot import (  # noqa: E402
    ROOT,
    PilotConfig,
    PilotTrainer,
    assert_same_state,
    prediction_geometry,
    state_hash,
    validate_input_geometry,
)


class TinyData:
    dataset, category, split = "mvtec_ad", "bottle", "train"
    manifest_sha256, output_size = "a" * 64, (32, 32)
    records = (SimpleNamespace(width=32, height=32),) * 3

    def __len__(self):
        return 3

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        return {
            "image": rng.uniform(0.05, 0.9, (*self.output_size, 3)).astype(np.float32),
            "image_id": f"bottle/train/good/{index:03}.png",
        }


class TinyBackend:
    def __init__(self, config, device):
        self.encoder = torch.nn.Conv2d(3, 8, 1).to(device).eval().requires_grad_(False)
        self.adapter = torch.nn.Linear(8, 8).to(device)
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.BatchNorm1d(8),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(8, 1, bias=False),
        ).to(device)
        self.encoder_hash = state_hash(self.encoder.state_dict())

    def embed(self, images):
        with torch.no_grad():
            output = self.encoder(torch.nn.functional.adaptive_avg_pool2d(images, (2, 2)))
            return output.permute(0, 2, 3, 1).reshape(-1, 8), [(2, 2)]


def config(**kwargs):
    return PilotConfig("mvtec_ad", "bottle", "a" * 64, image_size=(32, 32), **kwargs)


def trainer(settings=None, data=None):
    return PilotTrainer(settings or config(), data or TinyData(), backend_factory=TinyBackend)


@pytest.mark.parametrize("ad_resize", [None, 37])
def test_zero_consistency_replays_b1_exactly_including_bn_rng_and_optimizers(ad_resize):
    torch.set_num_threads(2)
    data = TinyData()
    data.output_size = (ad_resize, ad_resize) if ad_resize else (32, 32)
    first = trainer(config(variant="B1", ad_resize=ad_resize), data)
    records = [first.train_step() for _ in range(5)]
    expected = copy.deepcopy(first.checkpoint())
    second = trainer(config(variant="B2", lambda_score=0, ad_resize=ad_resize), data)
    assert_same_state(records, [second.train_step() for _ in range(5)])
    actual = second.checkpoint()
    expected.pop("config")
    actual.pop("config")
    assert_same_state(expected, actual)


@pytest.mark.parametrize("ad_resize", [None, 37])
def test_cold_resume_replays_next_epoch_and_predictions_exactly(tmp_path, ad_resize):
    data = TinyData()
    data.output_size = (ad_resize, ad_resize) if ad_resize else (32, 32)
    settings = config(variant="B2", lambda_score=0.1, ad_resize=ad_resize)
    first = trainer(settings, data)
    first.train_step()
    first.train_step()  # Partial last batch; next update starts a new permutation.
    checkpoint = tmp_path / "step2.pt"
    first.save(checkpoint)
    expected_record = first.train_step()
    expected_state = copy.deepcopy(first.checkpoint())
    expected_maps = first.predict([data[0]["image"]])
    resumed = trainer(settings, data)
    resumed.load(checkpoint)
    assert_same_state(expected_record, resumed.train_step())
    assert_same_state(expected_state, resumed.checkpoint())
    np.testing.assert_array_equal(expected_maps, resumed.predict([data[0]["image"]]))
    assert state_hash(resumed.backend.encoder.state_dict()) == resumed.backend.encoder_hash
    assert all(p.grad is None for p in resumed.backend.encoder.parameters())


def test_consistency_changes_update_without_changing_batch_or_noise_stream():
    first = trainer(config(variant="B1"))
    initial = first.trainable_hash()
    base = first.train_step()
    base_state = copy.deepcopy(first.checkpoint())
    second = trainer(config(variant="B2", lambda_score=1))
    constrained = second.train_step()
    assert base["image_ids"] == constrained["image_ids"]
    assert constrained["consistency"] > 0
    assert constrained["loss"] == pytest.approx(base["loss"] + base["consistency"])
    assert initial != first.trainable_hash() != second.trainable_hash()
    torch.testing.assert_close(
        base_state["noise_rng"], second.checkpoint()["noise_rng"], rtol=0, atol=0
    )
    assert all(p.grad is not None for p in second.backend.adapter.parameters())
    assert all(p.grad is not None for p in second.backend.discriminator.parameters())


def test_views_are_aligned_before_normalization_and_b0_is_identity():
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, 10, 17] = 0.5
    paired = trainer(config(variant="B1", brightness=(0.8, 0.8), gamma=(2, 2)))
    views, clipping = paired.paired_views(image)
    torch.testing.assert_close(views[0], image[0])
    torch.testing.assert_close(views[1, :, 10, 17], torch.full((3,), 0.2))
    torch.testing.assert_close(views[0] > 0, views[1] > 0)
    assert clipping == 0
    baseline, _ = trainer().paired_views(image)
    torch.testing.assert_close(baseline[0], baseline[1], rtol=0, atol=0)


def test_rejects_validation_training_and_incompatible_checkpoints(tmp_path):
    validation = TinyData()
    validation.split = "validation"
    with pytest.raises(ValueError, match="Training source"):
        trainer(data=validation)
    first = trainer()
    first.train_step()
    path = tmp_path / "checkpoint.pt"
    first.save(path)
    with pytest.raises(FileExistsError):
        first.save(path)
    with pytest.raises(ValueError, match="configuration mismatch"):
        trainer(replace(config(), noise_std=0.02)).load(path)
    corrupted = torch.load(path, weights_only=True)
    corrupted["continuation_contract"]["torch"] = "different runtime"
    torch.save(corrupted, tmp_path / "incompatible.pt")
    with pytest.raises(ValueError, match="matching sources and runtime"):
        trainer().load(tmp_path / "incompatible.pt")


@pytest.mark.parametrize("ad_resize", [None, 72])
def test_upstream_pilot_update_predict_and_cold_resume(tmp_path, ad_resize):
    pytest.importorskip("torchvision")
    torch.set_num_threads(2)
    data = TinyData()
    data.output_size = (ad_resize, ad_resize) if ad_resize else (64, 64)
    settings = config(variant="B2", lambda_score=0.1, weights="random")
    settings = replace(settings, image_size=(64, 64), ad_resize=ad_resize)
    first = PilotTrainer(settings, data)
    before = first.trainable_hash()
    first.train_step()
    assert first.trainable_hash() != before
    path = tmp_path / "upstream.pt"
    first.save(path)
    expected = first.train_step()
    expected_hash = first.trainable_hash()
    expected_maps = first.predict([data[0]["image"]])
    assert expected_maps.shape == (1, 8, 8) and np.isfinite(expected_maps).all()
    assert state_hash(first.backend.encoder.state_dict()) == first.backend.encoder_hash
    assert all(
        module.num_batches_tracked.item() == 0
        for module in first.backend.encoder.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )
    del first
    resumed = PilotTrainer(settings, data)
    resumed.load(path)
    assert_same_state(expected, resumed.train_step())
    assert resumed.trainable_hash() == expected_hash
    np.testing.assert_array_equal(expected_maps, resumed.predict([data[0]["image"]]))


@pytest.mark.parametrize("resize", [73, 74])
def test_ad_crop_matches_upstream_input_and_retains_geometry(tmp_path, monkeypatch, resize):
    import hashlib

    from PIL import Image

    from lc_simplenet.data import NormalManifestDataset
    from lc_simplenet.data_manifest import build_normal_manifest, write_normal_manifest

    monkeypatch.syspath_prepend(str(ROOT / "third_party/SimpleNet"))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    from datasets.mvtec import MVTecDataset

    for index in range(5):
        path = tmp_path / "bottle/train/good" / f"{index:03}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.random.default_rng(index).integers(0, 256, (80, 80, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
    records, audit = build_normal_manifest(tmp_path, "mvtec_ad", ["bottle"])
    manifest = tmp_path / "normal.jsonl"
    write_normal_manifest(records, audit, manifest)
    settings = replace(
        config(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        image_size=(64, 64),
        ad_resize=resize,
    )
    sources = {
        split: NormalManifestDataset(
            manifest,
            tmp_path,
            dataset="mvtec_ad",
            category="bottle",
            split=split,
            expected_sha256=settings.manifest_sha256,
            output_size=settings.load_size,
        )
        for split in ("train", "validation")
    }
    engine = trainer(settings, sources["train"])
    upstream = MVTecDataset(str(tmp_path), "bottle", resize=resize, imagesize=64)
    by_path = {item[2]: i for i, item in enumerate(upstream.data_to_iterate)}
    for source in sources.values():
        validate_input_geometry(settings, source)
        for sample in source:
            actual = engine.normalize(engine.prepare_pixels([sample["image"]]))[0]
            expected = upstream[by_path[str(tmp_path / sample["image_id"])]]["image"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            geometry = prediction_geometry(settings, sample)
            assert geometry["original_size"] == [80, 80]
            assert geometry["resized_size"] == [resize, resize]
            assert geometry["image_size"] == [64, 64]
            assert geometry["map_coordinates"] == "model_input_crop"
            left, top, right, bottom = geometry["crop_box_xyxy"]
            assert right - left == bottom - top == 64
            assert (left, top) == (int(round((resize - 64) / 2)),) * 2


@pytest.mark.parametrize(
    "change",
    [
        {"dataset": "mvtec_ad2", "category": "fabric", "ad_resize": 40},
        {"ad_resize": 16},
        {"ad_resize": 40.0},
        {"ad_resize": 48, "image_size": (32, 40)},
    ],
)
def test_rejects_incompatible_crop_configuration(change):
    with pytest.raises(ValueError, match="ad_resize"):
        replace(config(), **change)


def test_ad_crop_rejects_rectangular_sources_and_wrong_prediction_size():
    settings = config(ad_resize=37)
    data = TinyData()
    data.output_size = settings.load_size
    data.records = (SimpleNamespace(width=80, height=60),)
    with pytest.raises(ValueError, match="square source"):
        validate_input_geometry(settings, data)
    data.records = TinyData.records
    engine = trainer(settings, data)
    with pytest.raises(ValueError, match="matching load_size"):
        engine.predict([np.zeros((32, 32, 3), dtype=np.float32)])
