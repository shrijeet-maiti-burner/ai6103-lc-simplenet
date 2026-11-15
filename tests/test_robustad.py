import copy

import pytest

from lc_simplenet.robustad import REVISION, TRAIN_DIRECTORIES, normal_training_manifest


def fixture():
    rows = {
        category: [
            {"file_name": "normal/001.jpg", "label": 0, "mask": None},
            {"file_name": "anomaly/002.jpg", "label": 1, "mask": "masks/002.png"},
        ]
        for category in TRAIN_DIRECTORIES
    }
    rows["PiledBags"][0].pop("mask")  # the classification-only source omits this field
    repository = {
        "sha": REVISION,
        "siblings": [
            {
                "rfilename": f"{category}/{directory}/normal/001.jpg",
                "size": 42,
                "lfs": {"size": 42, "sha256": "a" * 64},
            }
            for category, directory in TRAIN_DIRECTORIES.items()
        ],
    }
    return repository, rows


def test_train_anomalies_are_excluded_without_requiring_their_image_bytes():
    repository, rows = fixture()
    result = normal_training_manifest(repository, rows)
    assert result["normal_images"] == result["train_anomalies_excluded"] == 3
    assert result["bytes"] == 126
    assert all("/normal/" in row["image_id"] and row["label"] == 0 for row in result["selected"])


@pytest.mark.parametrize(
    "fault",
    ["revision", "category", "label", "mask", "missing_mask", "path", "duplicate", "hash", "size"],
)
def test_invalid_normal_metadata_or_source_identity_is_rejected(fault):
    repository, rows = fixture()
    if fault == "revision":
        repository["sha"] = "another-revision"
    elif fault == "category":
        rows.pop("PCB")
    elif fault == "label":
        rows["PCB"][0]["label"] = False
    elif fault == "mask":
        rows["PCB"][0]["mask"] = "a-mask.png"
    elif fault == "missing_mask":
        rows["PCB"][0].pop("mask")
    elif fault == "path":
        rows["PCB"][0]["file_name"] = "../test0/001.jpg"
    elif fault == "duplicate":
        rows["PCB"].append(copy.deepcopy(rows["PCB"][0]))
    elif fault == "hash":
        repository["siblings"][0]["lfs"]["sha256"] = "not-a-hash"
    else:
        repository["siblings"][0]["size"] = 43
    with pytest.raises(ValueError):
        normal_training_manifest(repository, rows)
