"""Build a deterministic source zip from committed files, including pinned SimpleNet."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

from lc_simplenet.source_identity import (
    BUNDLE_NAME,
    UPSTREAM_PATH,
    UPSTREAM_SHA,
    git,
    source_record,
)

ROOT = Path(__file__).resolve().parents[1]


def committed_files(root: Path, prefix: str = "", *, executables: set[str]) -> dict[str, bytes]:
    archive = subprocess.check_output(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            "-C",
            str(root),
            "archive",
            "--format=tar",
            "HEAD",
        ]
    )
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Source bundles do not accept symlinks: {member.name}")
            files[prefix + member.name] = source.extractfile(member).read()
            if member.mode & 0o111:
                executables.add(prefix + member.name)
    return files


def build_bundle(root: Path, output: Path) -> dict:
    identity = source_record(root)
    if identity["source_kind"] != "git":
        raise ValueError("Build source bundles from an initialized Git checkout")
    tracked_changes = git(root, "status", "--porcelain", "--untracked-files=no")
    if tracked_changes or identity["upstream_dirty"]:
        raise ValueError("Commit tracked changes before building a source bundle")
    if identity["upstream_commit"] != UPSTREAM_SHA:
        raise ValueError("SimpleNet does not match the required pinned commit")
    linked = git(root, "ls-tree", "HEAD", UPSTREAM_PATH).split()
    if linked[:3] != ["160000", "commit", UPSTREAM_SHA]:
        raise ValueError("The committed SimpleNet gitlink does not match the required pin")
    executables = set()
    files = committed_files(root, executables=executables)
    files.update(
        committed_files(root / UPSTREAM_PATH, UPSTREAM_PATH + "/", executables=executables)
    )
    if BUNDLE_NAME in files:
        raise ValueError("SOURCE_BUNDLE.json is generated and must not be tracked")
    manifest = {
        "schema": 1,
        "git_commit": identity["git_commit"],
        "upstream_path": UPSTREAM_PATH,
        "upstream_commit": UPSTREAM_SHA,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    }
    files[BUNDLE_NAME] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrite; fixed timestamps and committed executable flags preserve reproducibility.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (0o100755 if name in executables else 0o100644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data, compresslevel=9)
    return {
        "file": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_files": len(manifest["files"]),
        "git_commit": identity["git_commit"],
        "upstream_commit": UPSTREAM_SHA,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(ROOT, args.out), indent=2))


if __name__ == "__main__":
    main()
