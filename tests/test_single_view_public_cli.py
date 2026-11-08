import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

spec = importlib.util.spec_from_file_location(
    "single_view_public_cli",
    Path(__file__).resolve().parents[1] / "scripts/score_single_view_public.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoint.pt").write_bytes(b"test checkpoint")
    (run / "summary.json").write_text("{}")
    np.savez(run / "validation_normal.npz", image_id=["normal"], image_score=[0.25])
    snapshot = tmp_path / "snapshot.pt"
    torch.save({"test": True}, snapshot)
    guard = {
        "passed": True,
        "score_max_error": 0,
        "map_max_error": 0,
        "normal_flag_differences": 0,
        **{
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in {
                "snapshot_sha256": snapshot,
                "checkpoint_sha256": run / "checkpoint.pt",
                "saved_normal_sha256": run / "validation_normal.npz",
            }.items()
        },
    }
    monkeypatch.setattr(module.normal_cli, "replay", lambda _: guard)
    args = SimpleNamespace(
        run=run,
        snapshot=snapshot,
        out=tmp_path / "out",
        public_root=tmp_path / "public",
        device="cpu",
        manifest=tmp_path / "normal.jsonl",
        manifest_sha256="0" * 64,
        data=tmp_path / "normal-data",
    )
    return args, guard


@pytest.mark.parametrize(
    "failure", ["failed", "numeric_only", "changed_snapshot", "changed_head", "changed_normal"]
)
def test_failed_or_changed_reference_never_opens_public_input(tmp_path, monkeypatch, failure):
    args, guard = inputs(tmp_path, monkeypatch)
    if failure == "failed":
        guard["passed"] = False
    elif failure == "numeric_only":
        guard["map_max_error"] = 1e-6
    else:
        target = {
            "changed_snapshot": args.snapshot,
            "changed_head": args.run / "checkpoint.pt",
            "changed_normal": args.run / "validation_normal.npz",
        }[failure]
        target.write_bytes(b"changed after replay")
    monkeypatch.setattr(module, "list_public_test", lambda *_: pytest.fail("public input opened"))
    with pytest.raises(ValueError, match="public input remains closed"):
        module.score(args)
    assert not (args.out / "test-scores.json").exists()


def test_original_cpu_preprocessing_batching_and_raw_score_are_retained(tmp_path, monkeypatch):
    args, _ = inputs(tmp_path, monkeypatch)
    for index in range(9):
        directory = args.public_root / "can/test_public" / ("good" if index < 8 else "bad")
        directory.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 24), (0, 0, 0)).save(directory / f"{index:03d}_regular.png")
    summary = {
        "category": "can",
        "seed": 0,
        "dataset": "mvtec_ad2",
        "input": {"resize": 32, "imagesize": 32, "batch_size": 8},
    }
    monkeypatch.setattr(
        module.normal_cli,
        "original_model",
        lambda *_: (SimpleNamespace(backbone=None), summary, "0" * 64),
    )
    monkeypatch.setattr(module, "restore_backbone_state", lambda *_, **__: {"buffers_restored": 1})
    batches = []

    def predict(_, batch):
        batches.append(len(batch))
        expected = -torch.tensor([0.485 / 0.229, 0.456 / 0.224, 0.406 / 0.225])
        torch.testing.assert_close(batch, expected[None, :, None, None].expand_as(batch))
        return np.full(len(batch), 7.0, dtype=np.float32), np.zeros(
            (len(batch), 32, 32), np.float32
        )

    monkeypatch.setattr(module, "predict_reference_batch", predict)
    meta = module.score(args)
    assert batches == [8, 1] and meta["images"] == 9
    assert meta["normal"] == 8 and meta["defective"] == 1
    assert meta["scoring_git_commit"] == meta["source"]["git_commit"]
    rows = json.loads((args.out / "test-scores.json").read_text())
    assert all(row["score"] == 7 and row["original_size"] == [24, 48] for row in rows)
    assert sum(row["label"] for row in rows) == 1
    assert (
        meta["image_ids_sha256"]
        == hashlib.sha256("\n".join(row["image_id"] for row in rows).encode()).hexdigest()
    )
    with np.load(args.out / "test-maps.npz") as maps:
        assert maps["maps"].shape == (9, 32, 32) and not maps["maps"].any()
        assert maps["image_ids"].tolist() == [row["image_id"] for row in rows]
