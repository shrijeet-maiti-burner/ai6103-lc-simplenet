"""Hash and safely extract a user-provided MVTec AD 2 archive. No download."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lc_simplenet.ad2_archive import extract_archive, inspect_archive

ROOT = Path(__file__).resolve().parents[1]


def _path(value: str | None, env: str) -> Path | None:
    text = value or os.environ.get(env)
    return Path(text).expanduser() if text else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Overrides MVTEC_AD2_ARCHIVE")
    parser.add_argument("--root", type=Path, help="Overrides MVTEC_AD2_ROOT")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--acquisition-out",
        type=Path,
        default=ROOT / "data/audits/ad2-acquisition.json",
    )
    args = parser.parse_args()
    archive = _path(None if args.archive is None else str(args.archive), "MVTEC_AD2_ARCHIVE")
    root = _path(None if args.root is None else str(args.root), "MVTEC_AD2_ROOT")
    if archive is None:
        raise SystemExit("Set --archive or MVTEC_AD2_ARCHIVE")
    inventory = inspect_archive(archive)
    if args.inspect_only:
        summary = {k: inventory[k] for k in inventory if k != "entries"}
        print(json.dumps(summary, indent=2))
        return
    if root is None:
        raise SystemExit("Set --root or MVTEC_AD2_ROOT")
    report = extract_archive(
        archive,
        root,
        force=args.force,
        expected_sha256=args.expected_sha256,
        acquisition_out=args.acquisition_out,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
