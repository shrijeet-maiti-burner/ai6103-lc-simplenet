"""Collect recorded normal-parent exposure and combine existing decoded-RGB audits.

This inventories supplied local protocols, not unrecorded teammate activity.
It does not decode images or infer absence of prior full-data training.
"""

import argparse
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc_simplenet.anchored_pilot import digest, read, write  # noqa: E402


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-runs", type=Path, required=True)
    parser.add_argument("--rgb-audit-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve exposure manifests")
    excluded, sources = set(), {}
    for path in sorted(args.recorded_runs.rglob("protocol.json")):
        if any(p in {".venv", "third_party", "extracted", "source-extracted"} for p in path.parts):
            continue
        identities = set()
        for value in strings(read(path)):
            parts = PurePosixPath(value).parts
            if len(parts) == 4 and parts[1:3] == ("train", "good") and value.endswith(".png"):
                identities.add(value)
        if identities:
            excluded.update(identities)
            sources[path.relative_to(args.recorded_runs).as_posix()] = digest(path)
    if not excluded:
        raise ValueError("no recorded exposure found; do not assume fresh parents")
    policy = read(ROOT / "configs/anchored-pilot-policy.json")
    rows, audit_sources = [], {}
    for category in policy["categories"]:
        path = args.rgb_audit_dir / f"{category}.json"
        rows.extend(read(path))
        audit_sources[path.name] = digest(path)
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate RGB-audit identities")
    args.out.mkdir(parents=True)
    write(
        args.out / "exclusions.json",
        {
            "scope": "recorded local normal-parent identities; not a global exposure guarantee",
            "excluded_image_ids": sorted(excluded),
            "source_sha256": sources,
        },
    )
    write(args.out / "rgb-audit.json", {"rows": rows, "source_sha256": audit_sources})
    print(
        {
            "recorded_protocols": len(sources),
            "excluded_parents": len(excluded),
            "audited_images": len(rows),
        }
    )


if __name__ == "__main__":
    main()
