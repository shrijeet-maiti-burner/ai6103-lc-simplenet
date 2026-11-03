"""Inventory documented public lighting suffixes without opening image pixels."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from lc_simplenet.data_manifest import AD2_CATEGORIES
from lc_simplenet.public_scoring import condition_from_name, list_public_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    conditions, counts = {}, Counter()
    for category in sorted(AD2_CATEGORIES):
        allowed = {"regular", "underexposed", "overexposed", "shift_1"}
        if category != "fruit_jelly":
            allowed |= {"shift_2", "shift_3"}
        if category == "vial":
            allowed.add("shift_4")
        for image in list_public_test(args.root, category):
            condition = condition_from_name(image.image_id)
            if condition not in allowed:
                raise ValueError(f"undocumented condition for {category}: {condition}")
            conditions[image.image_id] = condition
            counts[category, condition, image.label] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(conditions, stream, indent=2)
    metadata = {
        "scope": "public filename/label-folder inventory; pixels not opened",
        "source": "https://arxiv.org/html/2503.21622v1#S3.T6",
        "physical_mapping_unverified": ["vial/shift_4"],
        "mapping_limit": "vial shift_4 exists in the archive but is absent from arxiv v1 Table VI; "
        "retain the observed group name without inventing its lamp configuration",
        "images": len(conditions),
        "ordered_image_ids_sha256": hashlib.sha256("\n".join(conditions).encode()).hexdigest(),
        "conditions_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "counts": [
            {"category": c, "condition": light, "label": label, "images": n}
            for (c, light, label), n in sorted(counts.items())
        ],
    }
    with args.out.with_suffix(".meta.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(json.dumps({"images": len(conditions), "output": str(args.out)}))


if __name__ == "__main__":
    main()
