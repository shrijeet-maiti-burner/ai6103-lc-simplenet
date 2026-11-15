"""Select metadata-confirmed normal training files from the pinned author release."""

import re
from pathlib import PurePosixPath

REPOSITORY = "AmazonScience/RobustAD"
REVISION = "82c642b2d18be60f2dab104f8dd4dea1f4ccfacb"
TRAIN_DIRECTORIES = {
    "MetalParts": "metal_parts_data_dir_train",
    "PCB": "pcb_data_dir_train",
    "PiledBags": "piled_bags_data_dir_train",
}
METADATA_SHA256 = {
    "MetalParts": "7d587bc34e31e29d2fae95f7486d024d5e20b8477ab24048a33f2e5d387e65f1",
    "PCB": "244491445aa55810c6b6b5f646034a557f647701a705bddcdb0b2cf17e87926a",
    "PiledBags": "65ac3fdcde4dcbea37672c32c126450020e1c662a417f226c88e3233ab397c8d",
}


def normal_training_manifest(repository, metadata):
    """Never infer normality from the train split alone; it contains anomalies.

    This validates source metadata and file identities, not downloaded pixels,
    capture independence, a train/evaluation split or model compatibility.
    """
    if repository.get("sha") != REVISION or set(metadata) != set(TRAIN_DIRECTORIES):
        raise ValueError("unexpected RobustAD revision or category set")
    entries = {}
    for entry in repository["siblings"]:
        name = entry["rfilename"]
        if name in entries:
            raise ValueError("duplicate repository file identity")
        entries[name] = entry
    selected, excluded, seen = [], {}, set()
    for category, directory in TRAIN_DIRECTORIES.items():
        excluded[category] = 0
        for row in metadata[category]:
            if type(row.get("label")) is not int or row["label"] not in {0, 1}:
                raise ValueError("training metadata needs an explicit binary integer label")
            name = row.get("file_name")
            if not isinstance(name, str):
                raise ValueError("missing metadata file name")
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or len(path.parts) != 2
                or any(part in {".", ".."} or ":" in part or "\\" in part for part in path.parts)
                or path.as_posix() != name
            ):
                raise ValueError("expected one category-local training path")
            identifier = f"{category}/{directory}/{name}"
            if identifier in seen:
                raise ValueError("duplicate training metadata identity")
            seen.add(identifier)
            if row["label"] == 1:
                excluded[category] += 1
                continue
            if (
                path.parts[0] != "normal"
                or row.get("mask") is not None
                or (category != "PiledBags" and "mask" not in row)
            ):
                raise ValueError("normal training row must use normal/ and have no mask")
            entry = entries.get(identifier, {})
            lfs = entry.get("lfs", {})
            if (
                type(entry.get("size")) is not int
                or entry["size"] <= 0
                or entry["size"] != lfs.get("size")
                or not isinstance(lfs.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", lfs["sha256"]) is None
            ):
                raise ValueError("normal file lacks a valid author LFS size/hash")
            selected.append(
                {
                    "image_id": identifier,
                    "category": category,
                    "label": 0,
                    "split": "train",
                    "bytes": entry["size"],
                    "sha256": lfs["sha256"],
                }
            )
    selected.sort(key=lambda row: row["image_id"])
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "scope": "metadata-only normal training manifest; no pixels downloaded or scored",
        "selected": selected,
        "normal_images": len(selected),
        "bytes": sum(row["bytes"] for row in selected),
        "train_anomalies_excluded": sum(excluded.values()),
        "excluded_by_category": excluded,
    }
