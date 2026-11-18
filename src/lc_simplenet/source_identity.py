"""Identify a checkout or verified source archive without inheriting a parent repository."""

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

UPSTREAM_PATH = "third_party/SimpleNet"
UPSTREAM_SHA = "351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b"
BUNDLE_NAME = "SOURCE_BUNDLE.json"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, encoding="utf-8"
    ).strip()


def source_record(root: Path) -> dict:
    root = root.resolve()
    bundle = root / BUNDLE_NAME
    if bundle.is_file():
        raw = bundle.read_bytes()
        record = json.loads(raw)
        if record["schema"] != 1 or record["upstream_path"] != UPSTREAM_PATH:
            raise ValueError("Unsupported source bundle manifest")
        changed = []
        files = record["files"]
        if not files or not any(name.startswith(UPSTREAM_PATH + "/") for name in files):
            raise ValueError("Source bundle must include upstream files")
        for name, expected in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError(f"Unsafe source bundle path: {name}")
            path = root / name
            safe = path.resolve().is_relative_to(root) and not path.is_symlink()
            if not safe or not path.is_file():
                changed.append(name)
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                changed.append(name)
        upstream_dirty = any(name.startswith(UPSTREAM_PATH + "/") for name in changed)
        return {
            "source_kind": "source_archive",
            "git_commit": record["git_commit"],
            "git_dirty": bool(changed),
            "git_status": "\n".join(f" M {name}" for name in changed),
            "git_diff_sha256": None,
            "upstream_commit": record["upstream_commit"],
            "upstream_dirty": upstream_dirty,
            "source_bundle_sha256": hashlib.sha256(raw).hexdigest(),
            "source_files_checked": len(files),
            "source_files_changed": changed,
        }
    # Git searches parent directories by default. Only an explicit checkout is valid.
    if (
        not (root / ".git").exists()
        or Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root
    ):
        raise ValueError("Expected this project's Git checkout or SOURCE_BUNDLE.json")
    upstream = root / UPSTREAM_PATH
    if not (upstream / ".git").exists():
        raise ValueError("SimpleNet is missing: initialize submodules or use a complete source zip")
    status = git(root, "status", "--porcelain")
    return {
        "source_kind": "git",
        "git_commit": git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status,
        "git_diff_sha256": hashlib.sha256(git(root, "diff", "HEAD").encode()).hexdigest(),
        "upstream_commit": git(upstream, "rev-parse", "HEAD"),
        "upstream_dirty": bool(git(upstream, "status", "--porcelain")),
    }
