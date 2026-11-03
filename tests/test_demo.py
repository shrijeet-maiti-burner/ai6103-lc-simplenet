import base64
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from lc_simplenet import demo  # noqa: E402
from lc_simplenet.pilot import state_hash  # noqa: E402


class TinyBackend:
    def __init__(self, config, device):
        torch.manual_seed(12)
        self.encoder = torch.nn.Conv2d(3, 4, 1).to(device).eval().requires_grad_(False)
        self.adapter = torch.nn.Linear(4, 4).to(device)
        self.discriminator = torch.nn.Linear(4, 1).to(device)
        self.encoder_hash = state_hash(self.encoder.state_dict())

    def embed(self, pixels):
        with torch.no_grad():
            features = self.encoder(torch.nn.functional.adaptive_avg_pool2d(pixels, (2, 2)))
        return features.permute(0, 2, 3, 1).reshape(-1, 4), [(2, 2)]


def make_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "UpstreamBackend", TinyBackend)
    category, image_id = "fabric", "fabric/validation/good/000_regular.png"
    path = tmp_path / "data" / image_id
    path.parent.mkdir(parents=True)
    Image.new("RGB", (40, 48), (128, 80, 160)).save(path)
    pixels = np.asarray(Image.open(path).resize((32, 32)), dtype=np.float32) / np.float32(255)
    image = torch.from_numpy(pixels).permute(2, 0, 1)[None]
    normalized = (image - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / (
        torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    )
    for variant in ("B0", "B1", "B2"):
        config = {
            "dataset": "mvtec_ad2",
            "category": category,
            "manifest_sha256": "a" * 64,
            "seed": 0,
            "variant": variant,
            "lambda_score": 0.1 if variant == "B2" else 0,
            "image_size": [32, 32],
            "weights": "random",
        }
        backend = TinyBackend(SimpleNamespace(), torch.device("cpu"))
        with torch.inference_mode():
            scores = -backend.discriminator(backend.adapter(backend.embed(normalized)[0]))
        directory = tmp_path / "runs" / variant
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps({"config": config, "target_fpr": 0.05}))
        (directory / "validation-scores.json").write_text(
            json.dumps([{"image_id": image_id, "label": 0, "score": float(scores.max())}])
        )
        np.savez(directory / "validation-patch-maps.npz", scores=scores.numpy().reshape(1, 2, 2))
        torch.save(
            {
                "config": config,
                "encoder_hash": backend.encoder_hash,
                "adapter": backend.adapter.state_dict(),
                "discriminator": backend.discriminator.state_dict(),
            },
            directory / "heads.pt",
        )
    return demo.InspectionRuntime(tmp_path / "runs", tmp_path / "data")


def query():
    return {
        "category": "fabric",
        "seed": 0,
        "image_id": "fabric/validation/good/000_regular.png",
        "brightness": 1.0,
        "gamma": 1.0,
        "weight": 0.1,
    }


def test_real_model_path_has_frozen_calibration_and_explicit_measured_cache(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, monkeypatch)
    reference = runtime.inspect(query())
    assert len(reference["models"]) == 3 and reference["mode"] == "live inference"
    assert all(
        m["pair_mse"] == 0 and m["validation_replay_max_score_error"] < 1e-6
        for m in reference["models"]
    )
    altered = runtime.inspect({**query(), "brightness": 0.8, "gamma": 1.2})
    assert [m["threshold"] for m in reference["models"]] == [
        m["threshold"] for m in altered["models"]
    ]
    assert [m["heatmap_limits"] for m in reference["models"]] == [
        m["heatmap_limits"] for m in altered["models"]
    ]
    assert any(m["pair_mse"] > 0 for m in altered["models"])
    assert runtime.inspect(query())["mode"] == "cached inference"
    assert runtime.inspect(query())["image_sha256"] == reference["image_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        {"brightness": float("nan")},
        {"gamma": 2},
        {"category": []},
        {"seed": True},
        {"weight": True},
        {"image_id": "../../secret.png"},
        {"unexpected": 1},
    ],
)
def test_bad_requests_rejected_before_model_loading(tmp_path, monkeypatch, change):
    runtime = make_runtime(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        runtime.inspect({**query(), **change})
    assert not runtime.loaded


def test_fixed_heatmap_scale_does_not_rescale_each_image():
    first = demo.heatmap_data(np.array([[0.25, 0.5]], dtype=np.float32), [0, 1])
    second = demo.heatmap_data(np.array([[0.25, 0.9]], dtype=np.float32), [0, 1])
    import base64

    arrays = [
        np.asarray(Image.open(io.BytesIO(base64.b64decode(s.split(",")[1]))))
        for s in (first, second)
    ]
    np.testing.assert_array_equal(arrays[0][0, 0], arrays[1][0, 0])


def test_unmatched_controls_are_rejected_before_loading_heads(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, monkeypatch)
    entry = runtime.entries["fabric", 0, "B1"]
    entry["config"] = replace(entry["config"], noise_std=0.02)
    with pytest.raises(ValueError, match="share controls"):
        runtime.inspect(query())
    assert not runtime.loaded


def test_recorded_predictions_are_labelled_and_unrecorded_settings_rejected(tmp_path, monkeypatch):
    live = make_runtime(tmp_path, monkeypatch)
    frame = live.inspect(query())
    path = tmp_path / "recording.json"
    path.write_text(json.dumps({"catalog": live.describe(), "frames": [frame]}))
    runtime = demo.RecordedRuntime(path)
    assert runtime.describe()["mode"] == "recorded predictions"
    assert runtime.inspect(query())["mode"] == "recorded predictions"
    with pytest.raises(ValueError, match="not recorded"):
        runtime.inspect({**query(), "brightness": 0.8})


def test_uploaded_normal_bytes_keep_scores_but_never_inherit_the_known_label(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, monkeypatch)
    reference = runtime.inspect(query())
    raw = (tmp_path / "data" / query()["image_id"]).read_bytes()
    request = {
        "query": query(),
        "image_data": "data:image/png;base64," + base64.b64encode(raw).decode(),
    }
    uploaded = runtime.inspect_upload(request)
    assert uploaded["label"] is None and uploaded["query"]["image_id"].startswith("upload/")
    assert uploaded["image_sha256"] == reference["image_sha256"]
    assert [m["score"] for m in uploaded["models"]] == [m["score"] for m in reference["models"]]
    assert all(m["validation_replay_max_score_error"] is None for m in uploaded["models"])
    assert all(error < 1e-6 for error in uploaded["validation_guard"]["score_errors"].values())
    result = runtime.sweep_upload(request)
    assert len(result["rows"]) == 9 and result["query"]["image_id"].startswith("upload/")
    assert {m["threshold"] for row in result["rows"] for m in row["models"]} == {
        reference["models"][0]["threshold"]
    }


@pytest.mark.parametrize(
    "bad", [None, "not an image", "data:image/png;base64,%%", "data:image/svg+xml;base64,abc"]
)
def test_invalid_uploads_fail_before_loading_heads(tmp_path, monkeypatch, bad):
    runtime = make_runtime(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        runtime.inspect_upload({"query": query(), "image_data": bad})
    assert not runtime.loaded


def test_transparent_uploads_require_an_explicit_rgb_conversion(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, monkeypatch)
    stream = io.BytesIO()
    Image.new("RGBA", (4, 4), (255, 0, 0, 0)).save(stream, format="PNG")
    with pytest.raises(ValueError, match="RGB"):
        runtime.inspect_upload(
            {
                "query": query(),
                "image_data": "data:image/png;base64,"
                + base64.b64encode(stream.getvalue()).decode(),
            }
        )
    assert not runtime.loaded
