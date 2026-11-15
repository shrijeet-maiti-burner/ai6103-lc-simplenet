"""Audit locally supplied author metadata without downloading train/test pixels."""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc_simplenet.robustad import (  # noqa: E402
    METADATA_SHA256,
    REPOSITORY,
    REVISION,
    TRAIN_DIRECTORIES,
    normal_training_manifest,
)


def fetch_source(source):
    """Retrieve only repository metadata, training JSONL and the author card."""
    source.mkdir(parents=True, exist_ok=False)
    files = {
        "repository.json": f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{REVISION}?blobs=true",
        "AUTHOR_README.md": f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/README.md",
        **{
            f"{category}-train-metadata.jsonl": (
                f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
                f"{category}/{directory}/metadata.jsonl"
            )
            for category, directory in TRAIN_DIRECTORIES.items()
        },
    }
    for name, url in files.items():
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read(8 * 2**20 + 1)
        if len(raw) > 8 * 2**20:
            raise ValueError("metadata response exceeds the fixed 8 MiB limit")
        with (source / name).open("xb") as stream:
            stream.write(raw)
    return source


def audit(source, out):
    repository_path = source / "repository.json"
    repository = json.loads(repository_path.read_text(encoding="utf-8"))
    metadata, hashes = {}, {}
    for category, expected in METADATA_SHA256.items():
        path = source / f"{category}-train-metadata.jsonl"
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError("metadata bytes differ from the pinned author release")
        hashes[path.name] = actual
        metadata[category] = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    result = normal_training_manifest(repository, metadata)
    counts = {
        category: sum(r["category"] == category for r in result["selected"])
        for category in metadata
    }
    if set(counts.values()) != {180} or result["train_anomalies_excluded"] != 201:
        raise ValueError("pinned release population differs; investigate before acquiring pixels")
    result.update(
        category_counts=counts,
        metadata_sha256=hashes,
        repository_record_sha256=hashlib.sha256(repository_path.read_bytes()).hexdigest(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path)
    inputs.add_argument("--fetch-source", type=Path, help="new directory for author metadata only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = fetch_source(args.fetch_source) if args.fetch_source is not None else args.source
    result = audit(source, args.out)
    print(
        json.dumps({k: result[k] for k in ("normal_images", "train_anomalies_excluded", "bytes")})
    )
