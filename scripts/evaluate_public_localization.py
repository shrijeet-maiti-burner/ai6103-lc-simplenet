"""Evaluate saved public-test maps at native resolution without choosing model settings."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from lc_simplenet.localization import restore_native
from lc_simplenet.streaming_localization import NativeMetrics


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def evaluate(run_dir, root, output, fpr_limit=0.05, histogram_mib=2048, progress=None):
    if output.exists():
        raise FileExistsError("refusing to overwrite localization evidence")
    rows = json.loads((run_dir / "test-scores.json").read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("no saved public test scores")
    identifiers = [row["image_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate test image ID")
    metadata_path = run_dir / "test-scores.meta.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["images"] != len(rows) or metadata["checkpoint_file"] not in {
        "heads.pt",
        "final.pt",
    }:
        raise ValueError("scoring metadata does not identify the expected population/checkpoint")
    if digest(run_dir / metadata["checkpoint_file"]) != metadata["checkpoint_sha256"]:
        raise ValueError("scored checkpoint hash does not match the retained checkpoint")
    root = root.resolve(strict=True)
    maps_path = run_dir / "test-maps.npz"
    metrics = NativeMetrics(fpr_limit, histogram_mib * 2**20)
    mask_hashes = {}
    with np.load(maps_path, allow_pickle=False) as archive:
        if archive["image_ids"].tolist() != identifiers or len(archive["maps"]) != len(rows):
            raise ValueError("saved map image IDs/order differ from test-scores.json")
        for index, (row, scores) in enumerate(zip(rows, archive["maps"], strict=True), 1):
            parts = PurePosixPath(row["image_id"]).parts
            if (
                len(parts) != 4
                or parts[1] != "test_public"
                or parts[2] not in {"good", "bad"}
                or any(p in {".", ".."} or ":" in p or "\\" in p for p in parts)
            ):
                raise ValueError("expected an original public test image ID")
            if row["label"] != (parts[2] == "bad"):
                raise ValueError("saved label differs from public folder")
            if list(scores.shape) != row["image_size"]:
                raise ValueError("saved smoothed map does not match recorded model-input size")
            native = restore_native(
                scores,
                row["original_size"],
                map_coordinates=row["map_coordinates"],
                crop_box_xyxy=row["crop_box_xyxy"],
            )
            if row["label"] == 0:
                truth = np.zeros(native.shape, dtype=np.uint8)
            else:
                relative = (
                    Path(parts[0])
                    / "test_public/ground_truth/bad"
                    / (Path(parts[3]).stem + "_mask.png")
                )
                mask = (root / relative).resolve(strict=True)
                if not mask.is_relative_to(root):
                    raise ValueError("mask escapes dataset root")
                with Image.open(mask) as image:
                    truth = np.asarray(image)
                if truth.shape != native.shape or not np.isin(truth, [0, 255]).all():
                    raise ValueError("mask must have native geometry and binary values 0/255")
                truth = (truth > 0).astype(np.uint8)
                mask_hashes[relative.as_posix()] = digest(mask)
            metrics.add(native, truth)
            if progress and (index % 10 == 0 or index == len(rows)):
                progress(
                    {
                        "processed": index,
                        "images": len(rows),
                        "histogram_bytes": metrics.histogram_bytes,
                    }
                )
    result = {
        **metrics.result(),
        "scope": "local annotated AD 2 public subset; not private-server results",
        "geometry": "saved input-size smoothed continuous maps, PIL bilinear to native masks",
        "postprocessing": "no per-image/test-condition scaling, threshold fitting or morphology",
        "test_scores_sha256": digest(run_dir / "test-scores.json"),
        "test_maps_sha256": digest(maps_path),
        "scoring_metadata_sha256": digest(metadata_path),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "scoring_git_commit": metadata.get("scoring_git_commit"),
        "run_config_sha256": metadata.get("run_config_sha256"),
        "image_ids_sha256": hashlib.sha256("\n".join(identifiers).encode()).hexdigest(),
        "mask_sha256": mask_hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fpr-limit", type=float, default=0.05)
    parser.add_argument("--histogram-mib", type=int, default=2048)
    args = parser.parse_args()
    result = evaluate(
        args.run_dir,
        args.root,
        args.out,
        args.fpr_limit,
        args.histogram_mib,
        progress=lambda row: print(json.dumps(row), flush=True),
    )
    print(json.dumps({k: result[k] for k in ("status", "images", "pixel_auroc", "au_pro")}))


if __name__ == "__main__":
    main()
