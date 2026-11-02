"""Stage original public images/masks and link existing normal data; never score test data."""

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from lc_simplenet.data_manifest import AD2_CATEGORIES


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def stage(archive, expected_sha256, normals, output):
    if digest(archive) != expected_sha256:
        raise ValueError("archive checksum differs from the declared acquisition pin")
    if output.resolve().is_relative_to(normals.resolve()):
        raise ValueError("choose a separate staging root")
    with zipfile.ZipFile(archive) as source:
        selected = []
        for item in source.infolist():
            parts = PurePosixPath(item.filename).parts
            if len(parts) < 2 or parts[0] != "mvtec_ad_2" or parts[1] not in AD2_CATEGORIES:
                continue
            relative = parts[1:]
            image = (
                len(relative) == 4
                and relative[1] == "test_public"
                and relative[2] in {"good", "bad"}
            )
            mask = len(relative) == 5 and relative[1:4] == ("test_public", "ground_truth", "bad")
            if not (image or mask) or not relative[-1].endswith(".png"):
                continue
            if any(p in {".", ".."} or ":" in p or "\\" in p for p in relative):
                raise ValueError("unsafe public archive path")
            selected.append((item, relative))
        counts = Counter((parts[0], "/".join(parts[1:-1])) for _, parts in selected)
        if any(
            (category, "test_public/good") not in counts
            or (category, "test_public/bad") not in counts
            for category in AD2_CATEGORIES
        ):
            raise ValueError("archive does not contain all eight original public categories")
        output.mkdir(parents=True, exist_ok=False)
        linked = 0
        for category in sorted(AD2_CATEGORIES):
            for split in ("train", "validation"):
                for path in sorted((normals / category / split / "good").glob("*.png")):
                    target = output / path.relative_to(normals)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.hardlink_to(path.resolve(strict=True))
                    linked += 1
        for item, relative in selected:
            target = output.joinpath(*relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as stream, target.open("xb") as destination:
                shutil.copyfileobj(stream, destination)
        rows = [
            {"category": category, "folder": folder, "files": count}
            for (category, folder), count in sorted(counts.items())
        ]
        manifest = {
            "archive_sha256": expected_sha256,
            "public_files": len(selected),
            "normal_hardlinks": linked,
            "counts": rows,
            "scope": "file staging only; test images/masks not decoded or scored",
            "provenance_limit": "pinned mirror archive; official tar identity unverified",
        }
        (output / "staging.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--normals", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.archive, args.archive_sha256, args.normals, args.out), indent=2))
