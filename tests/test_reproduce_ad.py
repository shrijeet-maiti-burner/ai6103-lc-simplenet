import json
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from lc_simplenet.data_manifest import build_normal_manifest, write_normal_manifest


def _mvtec_fixture(root):
    rng = np.random.default_rng(0)

    def image(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(path)

    for i in range(5):
        image(root / "bottle/train/good" / f"{i:03}.png")
    image(root / "bottle/test/good/000.png")
    image(root / "bottle/test/broken/000.png")
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    (root / "bottle/ground_truth/broken").mkdir(parents=True)
    Image.fromarray(mask).save(root / "bottle/ground_truth/broken/000_mask.png")
    records, audit = build_normal_manifest(root, "mvtec_ad", ["bottle"])
    manifest = root.parent / "manifests/ad-bottle.jsonl"
    manifest.parent.mkdir()
    write_normal_manifest(records, audit, manifest)
    return manifest, json.loads(manifest.with_suffix(".audit.json").read_text())["manifest_sha256"]


def test_per_epoch_scoring_leaves_training_identical_to_the_baseline(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from lc_simplenet import baseline

    data = tmp_path / "data"
    manifest, digest = _mvtec_fixture(data)
    real = baseline.upstream_module
    torch.manual_seed(0)
    weights = torchvision.models.wide_resnet50_2(weights=None).state_dict()

    def load(_):
        # Fixed random weights instead of the ImageNet download. Like a pretrained load, it
        # builds the network first, and every call (including the reload) gets the same weights.
        backbone = torchvision.models.wide_resnet50_2(weights=None)
        backbone.load_state_dict(weights)
        return backbone

    def upstream(name):
        return SimpleNamespace(load=load) if name == "backbones" else real(name)

    monkeypatch.setattr(baseline, "upstream_module", upstream)
    # train_baseline hashes the downloaded ImageNet file for run.json; these weights have none.
    hub = tmp_path / "hub"
    (hub / "checkpoints").mkdir(parents=True)
    (hub / "checkpoints/wide_resnet50_2-95faca4d.pth").write_bytes(b"stand-in")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub))
    # In-process loading for both scripts: workers do not change the RNG stream (the loader
    # draws its base seed either way), and spawned workers can hang on macOS.
    loader = torch.utils.data.DataLoader
    monkeypatch.setattr(
        torch.utils.data, "DataLoader", lambda *a, **k: loader(*a, **{**k, "num_workers": 0})
    )
    monkeypatch.setenv("LC_DATA", str(data))
    monkeypatch.setenv("LC_MANIFESTS", str(manifest.parent))
    monkeypatch.setenv("LC_DEVICE", "cpu")
    monkeypatch.setenv("LC_META_EPOCHS", "2")
    reproduced, reference = tmp_path / "reproduced", tmp_path / "reference"
    scripts = baseline.ROOT / "scripts"
    monkeypatch.setattr(sys, "argv", ["reproduce_ad.py", "bottle", "0", str(reproduced)])
    runpy.run_path(str(scripts / "reproduce_ad.py"), run_name="__main__")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_baseline.py",
            "--manifest", str(manifest),
            "--manifest-sha256", digest,
            "--root", str(data),
            "--category", "bottle",
            "--out", str(reference),
            "--meta-epochs", "2",
            "--device", "cpu",
        ],
    )  # fmt: skip
    runpy.run_path(str(scripts / "train_baseline.py"), run_name="__main__")

    ours = torch.load(reproduced / "checkpoint.pt")["state"]
    theirs = torch.load(reference / "checkpoint.pt")["state"]
    assert ours.keys() == theirs.keys()
    for part in ours:
        for key, value in ours[part].items():
            assert torch.equal(value, theirs[part][key]), (part, key)

    summary = json.loads((reproduced / "summary.json").read_text())
    rows = [json.loads(line) for line in (reproduced / "trajectory.jsonl").read_text().splitlines()]
    assert [row["meta_epoch"] for row in rows] == [1, 2]
    assert rows[-1] == summary["final"]
    assert summary["reload_max_abs_score_diff"] == 0 and summary["reload_max_abs_map_diff"] == 0
    assert summary["scorer_vs_upstream_predict_max_abs_diff"] == {"score": 0.0, "map": 0.0}
    assert (summary["test_images"], summary["test_defective"]) == (2, 1)
