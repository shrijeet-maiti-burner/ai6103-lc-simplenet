"""Verify release archive checksums and stage paired run artifacts locally."""

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def stage(archives: Path, output: Path):
    expected = {}
    for manifest in sorted(archives.glob("*.sha256")):
        for line in manifest.read_text().splitlines():
            checksum, name = line.split(maxsplit=1)
            name = name.lstrip(" *")
            if Path(name).name != name or len(checksum) != 64:
                raise ValueError("invalid release checksum manifest")
            expected[name] = checksum
    zipped = sorted(archives.glob("*.zip"))
    if not zipped or any(path.name not in expected for path in zipped):
        raise ValueError("every archive needs a published checksum")
    for path in zipped:
        if digest(path) != expected[path.name]:
            raise ValueError(f"archive checksum mismatch: {path.name}")
    output.mkdir(parents=True, exist_ok=False)
    extracted = 0
    total = 0
    for path in zipped:
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                parts = PurePosixPath(item.filename).parts
                if not parts or parts[0] != "paired" or item.is_dir():
                    continue
                if any(p in {".", ".."} or ":" in p or "\\" in p for p in parts):
                    raise ValueError("unsafe release path")
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("release symlinks are unsupported")
                total += item.file_size
                if total > 8 * 2**30 or item.file_size > 200 * 2**20 or extracted > 10000:
                    raise ValueError("release exceeds staging limits")
                target = output.joinpath(*parts)
                if not target.resolve().is_relative_to(output.resolve()):
                    raise ValueError("release path escapes output")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
                extracted += 1
    runs = list((output / "paired").glob("*/run.json"))
    result = {
        "archive_sha256": expected,
        "files": extracted,
        "bytes": total,
        "paired_runs": len(runs),
        "scope": "verified released artifacts; no test scoring",
    }
    (output / "artifact-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.archives, args.out), indent=2))
