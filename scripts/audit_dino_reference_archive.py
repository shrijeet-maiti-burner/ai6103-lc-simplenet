"""Read-only independent recount of the released DINO reference evidence."""

import argparse
import hashlib
import io
import json
import math
import re
import statistics
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

EXPECTED = "13e3ebe214a7ae350edcec7af638173f9b7b6cbe05802cb68d741403fed4260d"
CATEGORIES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]


def h(data):
    return hashlib.sha256(data).hexdigest()


def suffix(row):
    if row["label"] not in [0, 1]:
        raise ValueError("invalid reference evidence")
    parts = PurePosixPath(row["image_id"]).parts
    if not (
        len(parts) == 4
        and parts[1] == "test_public"
        and (parts[2] == ("bad" if row["label"] else "good"))
    ):
        raise ValueError("invalid reference evidence")
    match = re.fullmatch(
        "\\d+_(regular|underexposed|overexposed|shift_\\d+)", PurePosixPath(row["image_id"]).stem
    )
    if not match:
        raise ValueError("invalid reference evidence")
    return match[1]


def includes(condition, row):
    light = suffix(row)
    if condition == "all":
        return True
    if condition == "nonregular":
        return light != "regular"
    if condition == "exposure_changed":
        return light in ["underexposed", "overexposed"]
    if condition == "additional_light":
        return light.startswith("shift_")
    return light == condition


def calculate(rows, threshold):
    normal = [r for r in rows if r["label"] == 0]
    defect = [r for r in rows if r["label"] == 1]
    fp, tp = (
        sum(r["score"] > threshold for r in normal),
        sum(r["score"] > threshold for r in defect),
    )
    if not (normal and defect):
        raise ValueError("invalid reference evidence")
    wins = sum(
        (p["score"] > n["score"]) + 0.5 * (p["score"] == n["score"]) for p in defect for n in normal
    )
    return {
        "false_positives": fp,
        "normal_images": len(normal),
        "true_positives": tp,
        "defective_images": len(defect),
        "fpr": fp / len(normal),
        "recall": tp / len(defect),
        "image_auroc": wins / (len(normal) * len(defect)),
    }


def run(archive_path, output, expected_sha256):
    if output.exists():
        raise FileExistsError("preserve previous audit receipts")
    if not re.fullmatch("[a-f0-9]{64}", expected_sha256):
        raise ValueError("a full lowercase SHA256 is required")
    ARCHIVE = archive_path
    with ARCHIVE.open("rb") as stream:
        if not hashlib.file_digest(stream, "sha256").hexdigest() == expected_sha256:
            raise ValueError("invalid reference evidence")
    condition_rows, predictions, error = (0, 0, 0.0)
    summaries, script_drift, native = ({}, [], {})
    quantized_max_checks = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        if not (len(names) == len(set(names)) and archive.testzip() is None):
            raise ValueError("invalid reference evidence")
        if not all(
            not PurePosixPath(n).is_absolute()
            and ".." not in PurePosixPath(n).parts
            and (":" not in n)
            and ("\\" not in n)
            for n in names
        ):
            raise ValueError("invalid reference evidence")
        manifest = {}
        for line in archive.read("SHA256SUMS").decode().splitlines():
            expected, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            if not (re.fullmatch("[a-f0-9]{64}", expected) and name not in manifest):
                raise ValueError("invalid reference evidence")
            manifest[name] = expected
        if not set(manifest) == set(names) - {"SHA256SUMS"}:
            raise ValueError("invalid reference evidence")
        for name, expected in manifest.items():
            if not h(archive.read(name)) == expected:
                raise ValueError(name)
        for group, script in [
            ("knn", "scripts/dino_knn.py"),
            ("simplenet-v1", "scripts/simplenet_dino.py"),
            ("simplenet-v2", "scripts/simplenet_dino_v2.py"),
        ]:
            group_rows = []
            for category in CATEGORIES:
                data = json.loads(archive.read(f"{group}/{category}.json"))
                if data["script_sha256"] != h(archive.read(script)):
                    script_drift.append(
                        {
                            "group": group,
                            "category": category,
                            "recorded_sha256": data["script_sha256"],
                            "released_script_sha256": h(archive.read(script)),
                        }
                    )
                tags = sorted(data["test_rows"])
                expected_tags = (
                    ["R0", "RL"]
                    if group == "knn"
                    else [
                        f"{v}-seed{s}"
                        for v in (["B0", "B1"] if group.endswith("v1") else ["B0", "B1", "B1S"])
                        for s in range(3)
                    ]
                )
                if not set(tags) == set(expected_tags):
                    raise ValueError("invalid reference evidence")
                for tag in tags:
                    rows = data["test_rows"][tag]
                    if not (
                        len(rows) == data["test_images"]
                        and len({r["image_id"] for r in rows}) == len(rows)
                    ):
                        raise ValueError("invalid reference evidence")
                    if not all(
                        math.isfinite(r["score"]) and r["image_id"].startswith(category + "/")
                        for r in rows
                    ):
                        raise ValueError("invalid reference evidence")
                    run = (
                        data["evaluation"][tag]
                        if group == "knn"
                        else data["runs"][tag]["evaluation"]
                    )
                    if group.endswith("v2"):
                        run = run["max"]
                    threshold = run["calibration"]["threshold"]
                    if group == "knn":
                        val = data["validation_rows"][tag]
                        expected_threshold = sorted(r["score"] for r in val)[
                            math.ceil(0.95 * (len(val) - 1))
                        ]
                        if not threshold == expected_threshold:
                            raise ValueError("invalid reference evidence")
                    predictions += len(rows)
                    for condition, expected in run["conditions"].items():
                        calculated = calculate(
                            [r for r in rows if includes(condition, r)], threshold
                        )
                        for key, value in calculated.items():
                            difference = abs(value - expected[key])
                            if not difference < 1e-12:
                                raise ValueError((group, category, tag, condition, key, difference))
                            error = max(error, difference)
                        condition_rows += 1
                        group_rows.append(
                            {"category": category, "tag": tag, "condition": condition, **calculated}
                        )
                    map_name = f"{group}/{category}-{tag}-maps.npz"
                    with np.load(io.BytesIO(archive.read(map_name)), allow_pickle=False) as maps:
                        if not set(maps.files) == {r["image_id"].replace("/", "|") for r in rows}:
                            raise ValueError("invalid reference evidence")
                        for row in rows:
                            values = maps[row["image_id"].replace("/", "|")]
                            if not (
                                values.ndim == 2
                                and values.dtype == np.float16
                                and np.isfinite(values).all()
                            ):
                                raise ValueError("invalid reference evidence")
                            if group != "knn":
                                if not np.float16(row["score"]) == values.max():
                                    raise ValueError("invalid reference evidence")
                                quantized_max_checks += 1
                    native_name = f"{group}/native/{category}-{tag}.json"
                    nr = json.loads(archive.read(native_name))
                    if not (
                        nr["status"] == "complete"
                        and nr["images"] == len(rows)
                        and (nr["fpr_limit"] == 0.05)
                        and (nr["connectivity"] == 8)
                    ):
                        raise ValueError("invalid reference evidence")
                    native.setdefault((group, tag), []).append(
                        {
                            "category": category,
                            "pixel_auroc": nr["pixel_auroc"],
                            "au_pro": nr["au_pro"],
                        }
                    )
            group_summary = {}
            for tag in expected_tags:
                conditions = sorted({r["condition"] for r in group_rows})
                group_summary[tag] = {
                    condition: {
                        key: statistics.mean(
                            r[key]
                            for r in group_rows
                            if r["tag"] == tag and r["condition"] == condition
                        )
                        for key in ["fpr", "recall", "image_auroc"]
                    }
                    for condition in conditions
                }
                for condition in conditions:
                    available = sorted(
                        {
                            r["category"]
                            for r in group_rows
                            if r["tag"] == tag and r["condition"] == condition
                        }
                    )
                    group_summary[tag][condition]["available_categories"] = available
                    group_summary[tag][condition]["all_eight_categories"] = len(available) == 8
                group_summary[tag]["native"] = {
                    key: statistics.mean(r[key] for r in native[group, tag])
                    for key in ["pixel_auroc", "au_pro"]
                }
            summaries[group] = group_summary
    receipt = {
        "passed": True,
        "scope": "archive integrity and public metric recount; not model replay",
        "archive_sha256": expected_sha256,
        "files": len(names),
        "internal_sha256_verified": len(manifest),
        "prediction_rows": predictions,
        "condition_rows_recounted": condition_rows,
        "maximum_image_metric_error": error,
        "float16_rounded_image_max_bindings": quantized_max_checks,
        "script_identity_mismatches": script_drift,
        "limits": [
            "no trained heads retained; no cold checkpoint replay",
            "SimpleNet raw validation scores absent; calibration quantiles not reconstructed",
            "v2 top1 test rows not retained; top1 sensitivity metrics not independently recounted",
            "float16 maps; native metrics use different geometry/postprocessing from WRN",
            "native metadata/macros checked; native pixels not recomputed",
            "exposed public test; v2 follows v1 outcomes; no independent confirmation",
        ],
        "category_macros": summaries,
        "audit_source_sha256": h(Path(__file__).read_bytes()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ["category_macros", "script_identity_mismatches"]
            }
        )
    )
    print(json.dumps({"script_identity_mismatches": script_drift}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=EXPECTED)
    args = parser.parse_args()
    run(args.archive, args.out, args.expected_sha256)
