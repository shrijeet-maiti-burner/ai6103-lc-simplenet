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
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in (*REQUIRED, *RELIGHT_NAMES) if name not in payload]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        seed_array = np.asarray(payload["seed"])
        if seed_array.ndim != 0 or seed_array.dtype.kind not in "iu":
            raise ValueError("seed metadata must be a scalar integer")
        seed = seed_array.item()
        strings = {}
        for name in ("category", "variant"):
            value = np.asarray(payload[name])
            if value.ndim != 0 or value.dtype.kind != "U":
                raise ValueError(f"{name} metadata must be a scalar string")
            strings[name] = value.item()
        ids = np.asarray(payload["image_ids"])
        if ids.ndim != 1 or ids.dtype.kind != "U":
            raise ValueError("image_ids metadata must be a one-dimensional string array")
        image_ids = ids.tolist()
        regular = np.asarray(payload["regular"], dtype=np.float32)
        relit = {name: np.asarray(payload[name], dtype=np.float32) for name in RELIGHT_NAMES}
    category, variant = strings["category"], strings["variant"]
    refuse_test_path(category)
    run = {
        "category": category,
        "seed": seed,
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
