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


def fixture(tmp_path):
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
    (run / "heads.pt").write_bytes(b"checkpoint fixture")
    (run / "test-scores.meta.json").write_text(
        json.dumps(
            {
                "images": 2,
                "checkpoint_file": "heads.pt",
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


def test_native_metric_path_retains_geometry_and_mask_evidence(tmp_path):
    run, root, output = fixture(tmp_path)
    result = module.evaluate(run, root, output)
    assert result["images"] == 2 and result["regions"] == 1
    assert result["normal_pixels"] == 36 and result["defect_pixels"] == 12
    assert 0 <= result["au_pro"] <= 1 and len(result["mask_sha256"]) == 1
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError):
        module.evaluate(run, root, output)


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
