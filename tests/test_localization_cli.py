import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("scipy")
spec = importlib.util.spec_from_file_location(
    "localization_cli",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_public_localization.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, checkpoint_file="heads.pt"):
    run = tmp_path / "run"
    run.mkdir()
    root = tmp_path / "data"
    mask = root / "can/test_public/ground_truth/bad/001_regular_mask.png"
    mask.parent.mkdir(parents=True)
    truth = np.zeros((4, 6), dtype=np.uint8)
    truth[:, 3:] = 255
    Image.fromarray(truth).save(mask)
    ids = ["can/test_public/good/000_regular.png", "can/test_public/bad/001_regular.png"]
    rows = [
        dict(
            image_id=name,
            label=i,
            original_size=[4, 6],
            image_size=[2, 3],
            map_coordinates="full_frame",
            crop_box_xyxy=None,
        )
        for i, name in enumerate(ids)
    ]
    (run / "test-scores.json").write_text(json.dumps(rows))
    (run / checkpoint_file).write_bytes(b"checkpoint fixture")
    (run / "test-scores.meta.json").write_text(
        json.dumps(
            {
                "images": 2,
                "checkpoint_file": checkpoint_file,
                "checkpoint_sha256": hashlib.sha256(b"checkpoint fixture").hexdigest(),
                "scoring_git_commit": "source-fixture",
            }
        )
    )
    np.savez(
        run / "test-maps.npz",
        maps=np.array([np.zeros((2, 3)), [[0, 0, 1], [0, 0, 1]]], dtype=np.float32),
        image_ids=ids,
    )
    return run, root, tmp_path / "out.json"


@pytest.mark.parametrize("checkpoint_file", ["heads.pt", "final.pt", "checkpoint.pt"])
def test_native_metric_path_retains_geometry_and_mask_evidence(tmp_path, checkpoint_file):
    run, root, output = fixture(tmp_path, checkpoint_file)
    result = module.evaluate(run, root, output)
    assert result["images"] == 2 and result["regions"] == 1
    assert result["normal_pixels"] == 36 and result["defect_pixels"] == 12
    assert 0 <= result["au_pro"] <= 1 and len(result["mask_sha256"]) == 1
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError):
        module.evaluate(run, root, output)


def test_external_cli_preserves_exact_metrics_and_cleans_only_its_workspace(tmp_path):
    run, root, output = fixture(tmp_path)
    reference = module.evaluate(run, root, output)
    external_out = tmp_path / "external.json"
    work = tmp_path / "owned-buckets"
    actual = module.evaluate(
        run, root, external_out, histogram_mib=64, engine="external", work_dir=work
    )
    assert actual["pixel_auroc"] == pytest.approx(reference["pixel_auroc"], abs=1e-14)
    assert actual["au_pro"] == pytest.approx(reference["au_pro"], abs=1e-14)
    assert len(actual["engine_source_sha256"]) == 64
    assert json.loads(external_out.read_text()) == actual
    assert not work.exists()


def test_existing_external_workspace_is_never_reused_or_removed(tmp_path):
    run, root, output = fixture(tmp_path)
    work = tmp_path / "existing"
    work.mkdir()
    sentinel = work / "unrelated.txt"
    sentinel.write_text("retain")
    with pytest.raises(FileExistsError):
        module.evaluate(run, root, output, engine="external", work_dir=work)
    assert sentinel.read_text() == "retain" and not output.exists()


@pytest.mark.parametrize("fault", ["order", "crop", "mask", "checkpoint"])
def test_mismatched_provenance_or_geometry_is_rejected(tmp_path, fault):
    run, root, output = fixture(tmp_path)
    if fault == "order":
        with np.load(run / "test-maps.npz") as archive:
            maps, ids = archive["maps"], archive["image_ids"][::-1]
        np.savez(run / "test-maps.npz", maps=maps, image_ids=ids)
    elif fault == "crop":
        rows = json.loads((run / "test-scores.json").read_text())
        rows[0]["crop_box_xyxy"] = [0, 0, 2, 2]
        (run / "test-scores.json").write_text(json.dumps(rows))
    elif fault == "mask":
        Image.new("L", (3, 3), 255).save(
            root / "can/test_public/ground_truth/bad/001_regular_mask.png"
        )
    else:
        (run / "heads.pt").write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError):
        module.evaluate(run, root, output)
    assert not output.exists()
