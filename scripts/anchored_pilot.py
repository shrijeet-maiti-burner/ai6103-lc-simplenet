"""Plan, prepare or run the isolated fixed-teacher synthetic pilot."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc_simplenet.anchored_pilot import plan, prepare, train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("plan", "prepare", "train", "prepare-confirmation", "confirm")
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--normal-root", type=Path)
    parser.add_argument("--reused-roles", type=Path)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--rgb-audit", type=Path)
    args = parser.parse_args()
    if args.stage == "plan":
        if any(
            x is None
            for x in (args.normal_root, args.reused_roles, args.exclusions, args.rgb_audit)
        ):
            parser.error("plan needs normal-root, reused-roles, exclusions and rgb-audit")
        plan(args.out, args.normal_root, args.reused_roles, args.exclusions, args.rgb_audit)
    elif args.stage in {"prepare", "prepare-confirmation"}:
        if args.normal_root is None:
            parser.error("preparation needs normal-root")
        prepare(
            args.out,
            args.normal_root,
            "confirmation" if args.stage.endswith("confirmation") else "development",
        )
    else:
        train(args.out, "confirmation" if args.stage == "confirm" else "development")


if __name__ == "__main__":
    main()
