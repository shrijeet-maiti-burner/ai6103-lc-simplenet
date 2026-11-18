"""Local recorded evidence, retaining all cases; no fitting or result selection."""

import argparse
import base64
import hashlib
import io
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["B0", "B1", "B2", "L0", "L1", "L2", "L3", "R0", "R1", "R2", "P0", "P1"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(evidence_dir, destination):
    HERE = Path(evidence_dir)
    audit = read(HERE / "real-test-decision.json")
    protocol = read(HERE / "real-test-protocol.json")
    done = read(HERE / "real-test/complete.json")
    if (
        audit["all_image_decisions_independently_recounted"] != 59004
        or not audit["exploratory"]
        or audit["source_confirmation_all_gates_passed"]
        or audit["real_test_protocol_sha256"] != sha(HERE / "real-test-protocol.json")
    ):
        raise ValueError("complete unaltered real-matrix audit required")
    images, metrics, decisions = [], {}, 0
    for domain in protocol["domain_order"]:
        folder = HERE / done["domains"][domain]["path"]
        complete = read(folder / "complete.json")
        for name in ["image-scores.npz", "metrics.json"]:
            if sha(folder / name) != complete["artifacts"][name]["sha256"]:
                raise ValueError("recorded artifact changed")
        reported = read(folder / "metrics.json")
        population = protocol["populations"][domain]
        inputs = HERE / ("test-" + domain + "-inputs.zip")
        if sha(inputs) != population["input_sha256"]:
            raise ValueError("native presentation packet changed")
        labels = np.array([r["label"] for r in population["rows"]], dtype=bool)
        values = {}
        with np.load(folder / "image-scores.npz", allow_pickle=False) as z:
            for arm in ARMS:
                values[arm] = []
                for seed in [0, 1, 2]:
                    score = z[f"seed{seed}_{arm}"].reshape(-1)
                    q = protocol["thresholds"][str(seed)][arm]
                    flags = score > q
                    recounted = audit["metrics"][domain]["per_seed"][str(seed)][arm]
                    if (
                        len(score) != len(labels)
                        or not np.isfinite(score).all()
                        or int(flags[~labels].sum()) != recounted["false_positives"]
                        or int(flags[labels].sum()) != recounted["true_positives"]
                    ):
                        raise ValueError("gallery decisions differ from independent audit")
                    values[arm].append((score.copy(), q, flags.copy()))
        with zipfile.ZipFile(inputs) as archive:
            for i, row in enumerate(population["rows"]):
                raw = archive.read(row["image_id"])
                if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError("native presentation image changed")
                with Image.open(io.BytesIO(raw)) as original:
                    rgb = original.convert("RGB")
                    rgb_sha = hashlib.sha256(
                        f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()
                    ).hexdigest()
                    if rgb_sha != reported["native_rgb_sha256"][row["image_id"]]:
                        raise ValueError("presentation image differs from measured image")
                    thumbnail = rgb.resize((256, 256), Image.Resampling.BILINEAR)
                    encoded = io.BytesIO()
                    thumbnail.save(encoded, format="JPEG", quality=78, optimize=True)
                outputs = {
                    arm: [[float(s[i]), q, bool(f[i])] for s, q, f in values[arm]] for arm in ARMS
                }
                decisions += 36
                images.append(
                    {
                        "id": row["image_id"],
                        "domain": domain,
                        "label": row["label"],
                        "defect_type": row["defect_type"],
                        "outputs": outputs,
                        "native_rgb_sha256": rgb_sha,
                        "thumbnail": "data:image/jpeg;base64,"
                        + base64.b64encode(encoded.getvalue()).decode("ascii"),
                    }
                )
        metrics[domain] = {
            "means": audit["metrics"][domain]["means"],
            "normals": int((~labels).sum()),
            "defects": int(labels.sum()),
            "count": len(labels),
        }
    if len(images) != 1639 or decisions != 59004:
        raise ValueError("gallery omits a fixed test case")
    data = {
        "arms": ARMS,
        "images": images,
        "metrics": metrics,
        "qualified": audit["independent_real_test_qualified"],
        "intervals": audit["paired_differences_and_intervals"],
        "provenance": {
            "built utc": datetime.now(UTC).isoformat(),
            "audit sha256": sha(HERE / "real-test-decision.json"),
            "protocol sha256": sha(HERE / "real-test-protocol.json"),
            "source heads sha256": protocol["heads_sha256"],
        },
    }
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=False)
    template_path = ROOT / "demo/blade-gallery.template.html"
    template = template_path.read_text(encoding="utf-8")
    if template.count("__EVIDENCE_JSON__") != 1:
        raise ValueError("gallery template differs")
    serialized = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text(
        template.replace("__EVIDENCE_JSON__", serialized), encoding="utf-8"
    )
    shutil.copyfile(ROOT / "third_party/licenses/AeBAD-DATASET.txt", output / "DATASET-LICENSE.txt")
    receipt = {
        "all_images": len(images),
        "all_decisions": decisions,
        "thumbnail_format": "256x256 jpeg presentation copies; not used for inference",
        "selection": "all test cases; original sorted order within all four frozen domains",
        "script_sha256": sha(Path(__file__)),
        "template_sha256": sha(template_path),
        "audit_sha256": sha(HERE / "real-test-decision.json"),
        "files": {p.name: sha(p) for p in output.iterdir()},
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.evidence_dir, args.out)
