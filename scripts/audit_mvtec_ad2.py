"""CPU-only on-disk AD 2 audit. Does not invent private labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lc_simplenet.ad2_audit import audit_mvtec_ad2, write_audit

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Overrides MVTEC_AD2_ROOT")
    parser.add_argument("--out", type=Path, default=ROOT / "data/audits/ad2-tree.json")
    args = parser.parse_args()
    root = args.root or (
        Path(os.environ["MVTEC_AD2_ROOT"]) if os.environ.get("MVTEC_AD2_ROOT") else None
    )
    if root is None:
        raise SystemExit("Set --root or MVTEC_AD2_ROOT")
    report = audit_mvtec_ad2(root)
    write_audit(report, args.out)
    printable = {
        "present_categories": report["present_categories"],
        "missing_categories": report["missing_categories"],
        "unexpected_entries": report["unexpected_entries"],
        "summary": report["summary"],
        "comparisons": {item["category"]: item["comparisons"] for item in report["categories"]},
        "duplicate_sha256_groups": len(report["duplicate_sha256_groups"]),
        "private_labels_inferred": report["private_labels_inferred"],
        "audit_json": str(args.out),
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
