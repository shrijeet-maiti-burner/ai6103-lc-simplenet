#!/usr/bin/env python3
"""Choose one residual grid from normal validation maps only.

The arrays must already be anomaly maps of regular-light and frozen-relight
normal validation images. This script does not open a test directory, does not
load a public-test label, and does not write into a released run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lc_simplenet.score_field import (
    RELIGHT_NAMES,
    refuse_test_path,
    select_downsample_size,
    validate_selection_run,
)

REQUIRED = ("category", "seed", "variant", "regular", "image_ids")


def load_run(path: Path) -> dict:
    refuse_test_path(path.as_posix())
    payload = np.load(path)
    missing = [name for name in (*REQUIRED, *RELIGHT_NAMES) if name not in payload]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    regular = np.asarray(payload["regular"], dtype=np.float32)
    relit = {name: np.asarray(payload[name], dtype=np.float32) for name in RELIGHT_NAMES}
    category = str(np.array(payload["category"]).reshape(-1)[0])
    variant = str(np.array(payload["variant"]).reshape(-1)[0])
    image_ids = [str(item) for item in np.asarray(payload["image_ids"]).reshape(-1)]
    refuse_test_path(category)
    run = {
        "category": category,
        "seed": int(np.asarray(payload["seed"]).reshape(-1)[0]),
        "variant": variant,
        "image_ids": image_ids,
        "regular": regular,
        "relit": relit,
    }
    validate_selection_run(run)
    return run


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    refuse_test_path(args.out.as_posix())
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite a selection result")
    runs = [load_run(path) for path in args.maps]
    result = select_downsample_size(runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
