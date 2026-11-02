"""Inspect and extract a user-supplied MVTec AD 2 archive. No download."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

CHUNK = 2**20
SPACE_MARGIN = 512 * 1024 * 1024
REGTYPE = {tarfile.REGTYPE, tarfile.AREGTYPE}


class ArchiveError(ValueError):
    """Unsafe or incomplete AD 2 archive contents."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_name(member: tarfile.TarInfo) -> str:
    name = member.name.replace("\\", "/")
    if name.endswith("/") and member.isdir():
        name = name.rstrip("/")
    return name


def _relative_parts(name: str) -> tuple[str, ...]:
    relative = PurePosixPath(name)
    if name.startswith("/") or relative.is_absolute() or not relative.parts:
        raise ArchiveError(f"Refusing absolute or empty archive path: {name}")
    if any(part in {".", ".."} or ":" in part or "\0" in part for part in relative.parts):
        raise ArchiveError(f"Refusing unsafe archive path: {name}")
    return relative.parts


def destination_for(root: Path, name: str) -> Path:
    root = root.resolve()
    dest = root.joinpath(*_relative_parts(name)).resolve()
    if dest != root and not dest.is_relative_to(root):
        raise ArchiveError(f"Archive path escapes the extract root: {name}")
    return dest


def inspect_archive(archive: Path) -> dict:
    """Inventory members and reject traversal, links and special files."""
    archive = archive.resolve(strict=True)
    if not tarfile.is_tarfile(archive):
        raise ArchiveError(f"Not a tar archive: {archive.name}")
    members = []
    kinds: dict[str, int] = {}
    prefixes: dict[str, int] = {}
    unpacked = 0
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle:
            name = _member_name(member)
            _relative_parts(name)
            if member.issym() or member.islnk():
                raise ArchiveError(f"Refusing link entry {name} -> {member.linkname}")
            if member.isdev() or member.isfifo():
                raise ArchiveError(f"Refusing special file: {name}")
            if member.isdir():
                kind = "dir"
            elif member.type in REGTYPE or member.isreg():
                kind = "file"
                unpacked += max(member.size, 0)
            else:
                raise ArchiveError(f"Refusing unsupported tar type {member.type!r}: {name}")
            members.append({"name": name, "kind": kind, "size": member.size})
            kinds[kind] = kinds.get(kind, 0) + 1
            top = name.split("/", 1)[0]
            prefixes[top] = prefixes.get(top, 0) + 1
    return {
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "members": len(members),
        "kinds": kinds,
        "unpacked_bytes": unpacked,
        "top_level": dict(sorted(prefixes.items())),
        "entries": members,
    }


def _require_space(root: Path, unpacked_bytes: int) -> None:
    probe = root if root.exists() else root.parent
    if not probe.exists():
        probe = Path(probe.anchor) if probe.anchor else Path("/")
    free = shutil.disk_usage(probe).free
    needed = unpacked_bytes + SPACE_MARGIN
    if free < needed:
        raise ArchiveError(f"Need {needed} bytes including margin; {probe} has {free} bytes free")


def _read_acquisition(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArchiveError("Acquisition record is not valid JSON") from exc
    if type(data) is not dict or type(data.get("archive_sha256")) is not str:
        raise ArchiveError("Acquisition record is missing archive_sha256")
    return data


def _member_matches(root: Path, item: dict) -> bool:
    dest = destination_for(root, item["name"])
    if dest.is_symlink():
        return False
    if item["kind"] == "dir":
        return dest.is_dir()
    return dest.is_file() and dest.stat().st_size == item["size"]


def _members_present(root: Path, entries: list[dict]) -> bool:
    if not any(item["kind"] == "file" for item in entries):
        return False
    return all(_member_matches(root, item) for item in entries)


def _existing_member_names(root: Path, names: list[str]) -> list[str]:
    return [name for name in names if destination_for(root, name).exists()]


def _public_report(
    archive: Path,
    digest: str,
    inventory: dict,
    source_url: str,
    acquired_at: str,
    skipped: bool,
) -> dict:
    return {
        "source_url": source_url,
        "acquired_at_utc": acquired_at,
        "archive_filename": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest,
        "unpacked_bytes": inventory["unpacked_bytes"],
        "members": inventory["members"],
        "kinds": inventory["kinds"],
        "top_level": inventory["top_level"],
        "extract_completed": True,
        "extract_skipped": skipped,
    }


def _skip_if_recorded(
    *,
    root: Path,
    digest: str,
    inventory: dict,
    names: list[str],
    acquisition_out: Path | None,
    force: bool,
) -> dict | None:
    """Return the recorded acquisition when extract can be skipped."""
    if force:
        return None
    if acquisition_out is not None and acquisition_out.exists():
        recorded = _read_acquisition(acquisition_out)
        if recorded["archive_sha256"] != digest:
            raise ArchiveError(
                "Archive SHA-256 differs from the recorded acquisition; "
                "refusing to overwrite the existing extraction"
            )
        recorded_root = recorded.get("extract_root")
        if recorded_root is not None and Path(recorded_root).resolve() != root:
            raise ArchiveError(
                "Extract root differs from the recorded acquisition; refusing to overwrite"
            )
        if not recorded.get("extract_completed"):
            raise ArchiveError("Recorded acquisition is incomplete; refusing to overwrite")
        if not _members_present(root, inventory["entries"]):
            raise ArchiveError("Recorded extraction is missing files; refusing to overwrite")
        return recorded
    if _existing_member_names(root, names):
        raise FileExistsError(
            "Existing extraction conflicts with the archive or recorded acquisition; "
            "refusing to overwrite"
        )
    return None


def extract_archive(
    archive: Path,
    root: Path,
    *,
    force: bool = False,
    expected_sha256: str | None = None,
    acquisition_out: Path | None = None,
    source_url: str = "manual download; no URL used by this repository",
) -> dict:
    """Hash, inspect, then extract unless a matching recorded extraction already exists."""
    archive = archive.resolve(strict=True)
    root = root.resolve()
    digest = file_sha256(archive)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ArchiveError("Archive SHA-256 differs from the declared pin")
    inventory = inspect_archive(archive)
    names = [item["name"] for item in inventory["entries"]]
    recorded = _skip_if_recorded(
        root=root,
        digest=digest,
        inventory=inventory,
        names=names,
        acquisition_out=acquisition_out,
        force=force,
    )
    if recorded is not None:
        return _public_report(
            archive,
            digest,
            inventory,
            recorded.get("source_url", source_url),
            recorded.get("acquired_at_utc", ""),
            True,
        )
    _require_space(root, inventory["unpacked_bytes"])
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle:
            name = _member_name(member)
            dest = destination_for(root, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if dest.exists() and not force:
                raise FileExistsError(f"Refusing to overwrite {name}")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ArchiveError(f"Could not read archive member: {name}")
            dest.write_bytes(extracted.read())
    report = _public_report(
        archive,
        digest,
        inventory,
        source_url,
        datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        False,
    )
    if acquisition_out is not None:
        acquisition_out.parent.mkdir(parents=True, exist_ok=True)
        if acquisition_out.exists() and not force:
            raise FileExistsError("Acquisition record already exists")
        local = {
            **report,
            "archive_path": str(archive),
            "extract_root": str(root),
            "note": "Local paths are for the operator only; do not commit this file.",
        }
        acquisition_out.write_text(json.dumps(local, indent=2) + "\n", encoding="utf-8")
    return report
