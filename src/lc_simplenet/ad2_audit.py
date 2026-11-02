"""Walk an on-disk MVTec AD 2 tree. Paper counts are expected, not labels."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from lc_simplenet.data_manifest import AD2_CATEGORIES

IMAGE_SUFFIXES = {".png"}
ROOT_DOCUMENTS = {"license.txt", "readme.txt"}
# Heckler-Kram et al., arXiv:2503.21622v1, Table IV. W×H. Private good/bad splits
# are paper-reported and are not recoverable from local private files.
PAPER_TABLE_IV = {
    "can": {
        "train": 412,
        "validation": 46,
        "test_public": 162,
        "test_public_good": 72,
        "test_public_bad": 90,
        "test_private": 321,
        "test_private_mixed": 321,
        "width": 2232,
        "height": 1024,
    },
    "fabric": {
        "train": 387,
        "validation": 43,
        "test_public": 156,
        "test_public_good": 66,
        "test_public_bad": 90,
        "test_private": 314,
        "test_private_mixed": 314,
        "width": 2448,
        "height": 2048,
    },
    "fruit_jelly": {
        "train": 263,
        "validation": 37,
        "test_public": 80,
        "test_public_good": 20,
        "test_public_bad": 60,
        "test_private": 255,
        "test_private_mixed": 255,
        "width": 2100,
        "height": 1520,
    },
    "rice": {
        "train": 313,
        "validation": 35,
        "test_public": 132,
        "test_public_good": 42,
        "test_public_bad": 90,
        "test_private": 277,
        "test_private_mixed": 277,
        "width": 2448,
        "height": 2048,
    },
    "sheet_metal": {
        "train": 137,
        "validation": 19,
        "test_public": 114,
        "test_public_good": 24,
        "test_public_bad": 90,
        "test_private": 142,
        "test_private_mixed": 142,
        "width": 4224,
        "height": 1056,
    },
    "vial": {
        "train": 291,
        "validation": 41,
        "test_public": 140,
        "test_public_good": 35,
        "test_public_bad": 105,
        "test_private": 276,
        "test_private_mixed": 276,
        "width": 1400,
        "height": 1900,
    },
    "wallplugs": {
        "train": 293,
        "validation": 33,
        "test_public": 150,
        "test_public_good": 60,
        "test_public_bad": 90,
        "test_private": 232,
        "test_private_mixed": 232,
        "width": 2448,
        "height": 2048,
    },
    "walnuts": {
        "train": 432,
        "validation": 48,
        "test_public": 150,
        "test_public_good": 60,
        "test_public_bad": 90,
        "test_private": 228,
        "test_private_mixed": 228,
        "width": 2448,
        "height": 2048,
    },
}
PUBLIC_IMAGE_FOLDERS = (
    ("train", "train/good"),
    ("validation", "validation/good"),
    ("test_public_good", "test_public/good"),
    ("test_public_bad", "test_public/bad"),
)
PRIVATE_IMAGE_FOLDERS = (
    ("test_private", "test_private"),
    ("test_private_mixed", "test_private_mixed"),
)
MASK_FOLDER = "test_public/ground_truth/bad"


class AuditError(ValueError):
    """On-disk AD 2 tree violates the audit contract."""


def _pngs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _symlink_entries(root: Path) -> list[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                target = os.readlink(path)
                found.append(f"{path.relative_to(root).as_posix()} -> {target}")
    return found


def _inspect_image(path: Path, root: Path) -> dict:
    if path.is_symlink():
        raise AuditError(f"Refusing symlink in the dataset tree: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise AuditError(f"Refusing path outside the dataset root: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(path) as image:
        image.load()
        frames = getattr(image, "n_frames", 1)
        return {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "frames": frames,
            "readable": True,
        }


def audit_mvtec_ad2(root: Path) -> dict:
    """Count files on disk and compare them with Table IV expected values."""
    root = root.resolve(strict=True)
    if root.is_symlink():
        raise AuditError(f"Refusing symlink dataset root: {root}")
    symlinks = _symlink_entries(root)
    if symlinks:
        preview = "; ".join(symlinks[:8])
        raise AuditError(f"Refusing symlink in the dataset tree: {preview}")
    extras = []
    documents = []
    for child in sorted(root.iterdir()):
        if child.is_file() and child.name.lower() in ROOT_DOCUMENTS:
            documents.append(child.name)
            continue
        if child.is_dir() and child.name not in AD2_CATEGORIES:
            extras.append(child.name)
        elif child.is_file():
            extras.append(child.name)
    categories = []
    missing = sorted(AD2_CATEGORIES - {p.name for p in root.iterdir() if p.is_dir()})
    present = sorted(name for name in AD2_CATEGORIES if (root / name).is_dir())
    duplicates = defaultdict(list)
    for category in present:
        source = root / category
        observed = {
            "category": category,
            "source": "files",
            "expected_source": "paper:arXiv:2503.21622v1:Table IV",
            "expected": PAPER_TABLE_IV[category],
            "folders": {},
            "images": [],
            "masks": [],
            "unreadable": [],
            "modes": Counter(),
            "shapes": Counter(),
        }
        for key, relative in PUBLIC_IMAGE_FOLDERS + PRIVATE_IMAGE_FOLDERS:
            folder = source / relative
            paths = _pngs(folder)
            observed["folders"][key] = {
                "relative_path": relative,
                "present": folder.is_dir(),
                "png_count": len(paths),
            }
            for path in paths:
                try:
                    record = _inspect_image(path, root)
                except OSError as error:
                    observed["unreadable"].append(
                        {"relative_path": path.relative_to(root).as_posix(), "error": str(error)}
                    )
                    continue
                record["kind"] = "image"
                record["private_label"] = None
                observed["images"].append(record)
                observed["modes"][record["mode"]] += 1
                observed["shapes"][f"{record['width']}x{record['height']}"] += 1
                duplicates[record["sha256"]].append(record["relative_path"])
        mask_dir = source / MASK_FOLDER
        mask_paths = _pngs(mask_dir)
        observed["folders"]["test_public_masks"] = {
            "relative_path": MASK_FOLDER,
            "present": mask_dir.is_dir(),
            "png_count": len(mask_paths),
        }
        for path in mask_paths:
            try:
                record = _inspect_image(path, root)
            except OSError as error:
                observed["unreadable"].append(
                    {"relative_path": path.relative_to(root).as_posix(), "error": str(error)}
                )
                continue
            record["kind"] = "public_mask"
            record["private_label"] = None
            observed["masks"].append(record)
            duplicates[record["sha256"]].append(record["relative_path"])
        counts = {
            "train": observed["folders"]["train"]["png_count"],
            "validation": observed["folders"]["validation"]["png_count"],
            "test_public_good": observed["folders"]["test_public_good"]["png_count"],
            "test_public_bad": observed["folders"]["test_public_bad"]["png_count"],
            "test_public": (
                observed["folders"]["test_public_good"]["png_count"]
                + observed["folders"]["test_public_bad"]["png_count"]
            ),
            "test_private": observed["folders"]["test_private"]["png_count"],
            "test_private_mixed": observed["folders"]["test_private_mixed"]["png_count"],
            "test_public_masks": observed["folders"]["test_public_masks"]["png_count"],
        }
        expected = PAPER_TABLE_IV[category]
        observed["counts"] = {
            key: {"observed": value, "source": "files"} for key, value in counts.items()
        }
        observed["comparisons"] = []
        for key in (
            "train",
            "validation",
            "test_public",
            "test_public_good",
            "test_public_bad",
            "test_private",
            "test_private_mixed",
        ):
            observed["comparisons"].append(
                {
                    "field": key,
                    "expected": expected[key],
                    "expected_source": "paper:arXiv:2503.21622v1:Table IV",
                    "observed": counts[key],
                    "observed_source": "files",
                    "match": counts[key] == expected[key],
                }
            )
        observed["private_labels_inferred"] = False
        observed["modes"] = dict(observed["modes"])
        observed["shapes"] = dict(observed["shapes"])
        categories.append(observed)
    duplicate_groups = [
        {"sha256": digest, "paths": paths} for digest, paths in duplicates.items() if len(paths) > 1
    ]
    return {
        "dataset": "mvtec_ad2",
        "root_documents": documents,
        "present_categories": present,
        "missing_categories": missing,
        "unexpected_entries": extras,
        "private_labels_inferred": False,
        "duplicate_sha256_groups": duplicate_groups,
        "categories": categories,
        "summary": {
            "categories_present": present,
            "image_pngs": sum(len(item["images"]) for item in categories),
            "mask_pngs": sum(len(item["masks"]) for item in categories),
            "unreadable": sum(len(item["unreadable"]) for item in categories),
        },
    }


def write_audit(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".md").exists():
        raise FileExistsError("Choose a new audit name; existing audits are not overwritten")
    serializable = json.loads(json.dumps(report))
    output.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MVTec AD 2 on-disk audit",
        "",
        "Expected counts are from Heckler-Kram et al., arXiv:2503.21622v1, Table IV.",
        "Observed counts are files on disk. Private test labels were not inferred.",
        "",
        f"- Present categories: {', '.join(report['present_categories']) or '(none)'}",
        f"- Missing categories: {', '.join(report['missing_categories']) or '(none)'}",
        f"- Unexpected entries: {', '.join(report['unexpected_entries']) or '(none)'}",
        f"- Image PNGs: {report['summary']['image_pngs']}",
        f"- Public masks: {report['summary']['mask_pngs']}",
        f"- Unreadable: {report['summary']['unreadable']}",
        "",
    ]
    for item in report["categories"]:
        lines.append(f"## {item['category']}")
        lines.append("")
        lines.append("| Field | Expected (paper) | Observed (files) | Match |")
        lines.append("|---|---:|---:|---|")
        for row in item["comparisons"]:
            lines.append(
                f"| {row['field']} | {row['expected']} | {row['observed']} | {row['match']} |"
            )
        lines.append("")
        lines.append(f"- Modes: {item['modes']}")
        lines.append(f"- Shapes: {item['shapes']}")
        lines.append(f"- Public masks: {item['counts']['test_public_masks']['observed']}")
        lines.append(f"- Unreadable: {len(item['unreadable'])}")
        lines.append("")
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
