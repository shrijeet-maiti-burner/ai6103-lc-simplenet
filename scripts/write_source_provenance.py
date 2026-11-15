"""Write a fresh source manifest and checksum for a clean checkout or source ZIP."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from lc_simplenet.source_identity import (
    BUNDLE_NAME,
    UPSTREAM_PATH,
    UPSTREAM_SHA,
    git,
    source_record,
)

ROOT = Path(__file__).resolve().parents[1]


def write_provenance(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    sidecar = output.with_suffix(".sha256")
    if output == sidecar or output.exists() or sidecar.exists():
        raise FileExistsError("Refusing to replace a source manifest or checksum")
    identity = source_record(root)
    if identity["git_dirty"] or identity["upstream_dirty"]:
        raise ValueError("Source must be clean before recording provenance")
    if identity["upstream_commit"] != UPSTREAM_SHA:
        raise ValueError("SimpleNet does not match the pinned source")
    if identity["source_kind"] == "source_archive":
        names = json.loads((root / BUNDLE_NAME).read_text())["files"]
    else:
        names = filter(None, git(root, "ls-files", "-z", "--recurse-submodules").split("\0"))
    files = {}
    for name in sorted(names):
        relative = PurePosixPath(name)
        path = (root / name).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
            raise ValueError("Unsafe source path")
        if output == path or sidecar == path:
            raise ValueError("Write generated provenance outside tracked source files")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "git_head": identity["git_commit"],
        "source_kind": identity["source_kind"],
        "purpose": "current source identity for a new execution; not a historical run receipt",
        "submodule": {"path": UPSTREAM_PATH, "commit": UPSTREAM_SHA},
        "files": files,
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    with sidecar.open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {output.name}\n")
    return {"manifest": output.name, "sha256": digest, "files": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_provenance(ROOT, args.out)))
