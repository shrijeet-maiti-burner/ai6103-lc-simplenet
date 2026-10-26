"""Normal-only source inventory; test images and labels are deliberately excluded.

AD 2 folder semantics were checked against MVTec's public offline loader on
2026-10-25. This module is our implementation, not copied evaluation code.
"""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

AD_CATEGORIES = {
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
}
AD2_CATEGORIES = {
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
}


def _inventory(root: Path, category: str, folder: str, dataset: str) -> list[dict]:
    source = root / category / folder / "good"
    paths = sorted(source.glob("*.png"))
    if not paths:
        raise ValueError(f"No normal PNG files in {category}/{folder}/good")
    records = []
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Dataset symlink points outside the declared root")
        raw = path.read_bytes()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            dimensions = f"{rgb.width},{rgb.height},RGB:".encode()
            pixels = hashlib.sha256(dimensions + rgb.tobytes()).hexdigest()
            records.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "split": "validation" if folder == "validation" else "train",
                    "image_id": path.relative_to(root).as_posix(),
                    "relative_path": path.relative_to(root).as_posix(),
                    "label": 0,
                    "mask": None,
                    "width": rgb.width,
                    "height": rgb.height,
                    "source_mode": image.mode,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "rgb_sha256": pixels,
                }
            )
    return records


def _holdout(records: list[dict], fraction: float, seed: int) -> None:
    groups = defaultdict(list)
    for record in records:
        groups[record["rgb_sha256"]].append(record)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("AD holdout needs at least two distinct RGB-content groups")
    random.Random(seed).shuffle(keys)
    target, selected = math.ceil(len(records) * fraction), 0
    for key in keys[:-1]:
        if selected >= target:
            break
        for record in groups[key]:
            record["split"] = "validation"
            selected += 1


def build_normal_manifest(
    root: Path, dataset: str, categories: list[str], *, seed: int = 0, val_fraction: float = 0.2
) -> tuple[list[dict], dict]:
    """Inventory normal data and reject exact RGB leakage across the final splits."""
    root = root.resolve(strict=True)
    if dataset not in {"mvtec_ad", "mvtec_ad2"}:
        raise ValueError("dataset must be mvtec_ad or mvtec_ad2")
    permitted = AD_CATEGORIES if dataset == "mvtec_ad" else AD2_CATEGORIES
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("Select distinct categories explicitly")
    if not set(categories) <= permitted:
        raise ValueError(f"Unknown categories: {sorted(set(categories) - permitted)}")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between 0 and 1")
    records = []
    for category in sorted(categories):
        train = _inventory(root, category, "train", dataset)
        if dataset == "mvtec_ad":
            _holdout(train, val_fraction, seed)
            records.extend(train)
        else:
            records.extend(train + _inventory(root, category, "validation", dataset))
    records.sort(key=lambda item: item["image_id"])
    content_splits = defaultdict(set)
    contents = Counter()
    for record in records:
        content_splits[record["rgb_sha256"]].add(record["split"])
        contents[record["rgb_sha256"]] += 1
    overlap = [key for key, splits in content_splits.items() if len(splits) > 1]
    if overlap:
        raise ValueError(f"Found {len(overlap)} identical RGB groups crossing train/validation")
    counts = Counter((r["category"], r["split"]) for r in records)
    audit = {
        "dataset": dataset,
        "categories": sorted(categories),
        "split_policy": "grouped_normal_holdout" if dataset == "mvtec_ad" else "official",
        "holdout_seed": seed if dataset == "mvtec_ad" else None,
        "requested_val_fraction": val_fraction if dataset == "mvtec_ad" else None,
        "records": len(records),
        "counts": {f"{cat}/{split}": n for (cat, split), n in sorted(counts.items())},
        "duplicate_rgb_groups": sum(count > 1 for count in contents.values()),
        "cross_split_rgb_groups": 0,
        "test_directories_read": False,
        "scope": "normal-data inventory only; not an acquisition-group or test-set audit",
    }
    return records, audit


def write_normal_manifest(records: list[dict], audit: dict, output: Path) -> None:
    """Write deterministic, portable JSONL and an adjacent audit; refuse replacement."""
    audit_path = output.with_suffix(".audit.json")
    if output.exists() or audit_path.exists():
        raise FileExistsError("Choose a new manifest name; existing manifests are not overwritten")
    encoded = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode()
    report = {**audit, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
