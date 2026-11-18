"""Validate a complete frozen native matrix and build category-macro report tables."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build(native: Path, scoring: Path, out: Path):
    protocol = read(native / "protocol.json")
    frozen = read(scoring / "frozen-protocol.json")
    status = read(native / "status.json")
    if protocol["frozen_image_protocol_sha256"] != digest(scoring / "frozen-protocol.json"):
        raise ValueError("native matrix does not bind this frozen image cohort")
    for key in ("categories", "methods", "seeds"):
        values = protocol[key]
        if not values or len(values) != len(set(values)) or values != frozen[key]:
            raise ValueError("native and image matrix scopes differ")
    if len(frozen["seeds"]) < 2:
        raise ValueError("sample SD requires at least two seeds")
    expected = set(itertools.product(frozen["categories"], frozen["methods"], frozen["seeds"]))
    completed = {tuple(item["identity"]): item for item in status["completed"]}
    if (
        status["state"] != "complete"
        or status["failed"]
        or status["active"]
        or status["expected"] != len(expected)
        or len(status["completed"]) != len(expected)
        or set(completed) != expected
    ):
        raise ValueError("native report requires every frozen category/method/seed exactly once")
    identities, populations, rows, resources = {}, {}, [], []
    inputs = {}
    for run_name, input_hashes in protocol["inputs"].items():
        if Path(run_name).name != run_name or "\\" in run_name:
            raise ValueError("unsafe run directory")
        run = native / "runs" / run_name
        scored = scoring / "runs" / run_name
        meta = read(scored / "test-scores.meta.json")
        method = meta["variant"]
        if method == "B2":
            method += f"@{meta['lambda_score']:g}"
        identity = (meta["category"], method, meta["seed"])
        if identity not in expected or identity in identities:
            raise ValueError("unexpected or duplicate scored identity")
        identities[identity] = run_name
        value = read(run / "native.json")
        before = read(run / "native.protocol.json")
        after = read(run / "native.provenance.json")
        output_hash = digest(run / "native.json")
        if (
            completed[identity]["output_sha256"] != output_hash
            or after["metric_output_sha256"] != output_hash
            or after["status"] != "complete"
            or before["sources"] != after["sources"]
            or not set(before["sources"].values()) <= set(protocol["source_sha256"].values())
            or before["fpr_limit"] != 0.05
            or not before["no_quantization"]
        ):
            raise ValueError("native output or evaluator provenance changed")
        bindings = {
            "test_scores_sha256": input_hashes["test-scores.json"],
            "test_maps_sha256": input_hashes["test-maps.npz"],
            "scoring_metadata_sha256": digest(scored / "test-scores.meta.json"),
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "run_config_sha256": meta["run_config_sha256"],
            "scoring_git_commit": frozen["scoring_git_commit"],
            "image_ids_sha256": meta["image_ids_sha256"],
            "images": meta["images"],
        }
        if (
            any(value.get(k) != v for k, v in bindings.items())
            or input_hashes["test-scores.meta.json"] != bindings["scoring_metadata_sha256"]
            or digest(scored / "test-scores.json") != input_hashes["test-scores.json"]
            or before["test_map_sha256"] != bindings["test_maps_sha256"]
            or before["test_score_sha256"] != bindings["test_scores_sha256"]
            or meta["scoring_git_commit"] != frozen["scoring_git_commit"]
            or value["fpr_limit"] != 0.05
            or value["connectivity"] != 8
            or value["status"] != "complete"
        ):
            raise ValueError("native metric does not bind the frozen scoring inputs/definition")
        counts = [value[k] for k in ("images", "normal_pixels", "defect_pixels", "regions")]
        if any(type(v) is not int or v <= 0 for v in counts):
            raise ValueError("native counts must be positive integers")
        if any(
            not isinstance(value[k], int | float)
            or not math.isfinite(value[k])
            or not 0 <= value[k] <= 1
            for k in ("pixel_auroc", "au_pro")
        ):
            raise ValueError("native metrics must be finite rates")
        population = [*counts, value["image_ids_sha256"], value["mask_sha256"]]
        if populations.setdefault(meta["category"], population) != population:
            raise ValueError("native images/masks/populations differ between compared heads")
        rows.append(
            dict(
                category=identity[0],
                method=identity[1],
                seed=identity[2],
                pixel_auroc=value["pixel_auroc"],
                au_pro_005=value["au_pro"],
            )
        )
        inputs[run_name] = output_hash
        resources.append(after)
    if set(identities) != expected:
        raise ValueError("native input records do not cover the complete matrix")
    scopes = {"all_categories": frozen["categories"]}
    if "sheet_metal" in frozen["categories"]:
        scopes["without_sheet_metal"] = [c for c in frozen["categories"] if c != "sheet_metal"]
    macros, tables = [], {}
    for scope, categories in scopes.items():
        table = [
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Model & Pixel AUROC (\%) & AU-PRO$_{.05}$ (\%) \\",
            r"\midrule",
        ]
        for method in frozen["methods"]:
            row = {"scope": scope, "method": method, "categories": categories}
            for metric in ("pixel_auroc", "au_pro_005"):
                seeds = [
                    statistics.mean(
                        r[metric]
                        for r in rows
                        if r["category"] in categories and r["method"] == method and r["seed"] == s
                    )
                    for s in frozen["seeds"]
                ]
                row[metric] = {
                    "mean": statistics.mean(seeds),
                    "sd": statistics.stdev(seeds),
                    "per_seed": seeds,
                }
            macros.append(row)
            label = method if "@" not in method else r"B2 ($\lambda=" + method.split("@")[1] + "$)"
            estimates = [
                f"{100 * row[k]['mean']:.2f}$\\pm${100 * row[k]['sd']:.2f}"
                for k in ("pixel_auroc", "au_pro_005")
            ]
            table.append(label + " & " + " & ".join(estimates) + r" \\")
        table.extend([r"\bottomrule", r"\end{tabular}"])
        tables[scope] = "\n".join(table) + "\n"
    evidence = {
        "scope": "frozen local public subset; not private-server results; no selection",
        "status": "complete",
        "runs": len(rows),
        "rows": sorted(rows, key=lambda r: (r["category"], r["method"], r["seed"])),
        "macros": macros,
        "aggregation": "category macro within seed, then mean/sample SD across seeds",
        "native_protocol_sha256": digest(native / "protocol.json"),
        "image_protocol_sha256": digest(scoring / "frozen-protocol.json"),
        "native_output_sha256": inputs,
        "population": {
            c: dict(
                zip(("images", "normal_pixels", "defect_pixels", "regions"), v[:4], strict=True)
            )
            for c, v in populations.items()
        },
        "max_worker_peak_rss_bytes": max(r["peak_rss_bytes"] for r in resources),
        "total_native_pixels_per_complete_sweep": sum(v[1] + v[2] for v in populations.values()),
    }
    out.mkdir(parents=True, exist_ok=True)
    for scope, table in tables.items():
        (out / f"ad2-native-{scope}.tex").write_text(table, encoding="utf-8")
    (out / "native-evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/report-evidence"))
    args = parser.parse_args()
    evidence = build(args.native, args.scoring, args.out)
    print(f"built native tables from {evidence['runs']} verified outputs")


if __name__ == "__main__":
    main()
