"""Create the normal-data handoff without reading test images or labels."""

import argparse
import json
import os
from pathlib import Path

from lc_simplenet.data_manifest import build_normal_manifest, write_normal_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec_ad", "mvtec_ad2"], required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--category", action="append", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    env_key = "MVTEC_AD2_ROOT" if args.dataset == "mvtec_ad2" else "MVTEC_AD_ROOT"
    root = args.root or (Path(os.environ[env_key]) if os.environ.get(env_key) else None)
    if root is None:
        raise SystemExit(f"Set --root or {env_key}")
    records, audit = build_normal_manifest(
        root, args.dataset, args.category, seed=args.seed, val_fraction=args.val_fraction
    )
    write_normal_manifest(records, audit, args.out)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
