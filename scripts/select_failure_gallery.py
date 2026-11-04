"""Select failure-gallery cases from predictions.csv by score rank (no images are opened).

Per category and method, seed 0 only: up to two of the highest-scoring flagged normal images
(false positives) and up to two of the lowest-scoring unflagged defective images (false
negatives). Ties break on image_id in ascending order. A slot with no qualifying image stays
empty and is never filled with a correctly classified one.
Usage: python select_failure_gallery.py predictions.csv selection.json
"""

import csv
import json
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

out = []
for category in sorted({r["category"] for r in rows}):
    for method in ("B0", "B1"):
        sub = [
            r
            for r in rows
            if r["category"] == category and r["method"] == method and r["seed"] == "0"
        ]
        false_positives = [r for r in sub if r["label"] == "0" and r["flagged"] == "1"]
        false_negatives = [r for r in sub if r["label"] == "1" and r["flagged"] == "0"]
        false_positives.sort(key=lambda r: (-float(r["score"]), r["image_id"]))
        false_negatives.sort(key=lambda r: (float(r["score"]), r["image_id"]))
        for kind, picks in (
            ("false_positive", false_positives[:2]),
            ("false_negative", false_negatives[:2]),
        ):
            for rank, r in enumerate(picks, 1):
                out.append(
                    {
                        "category": category,
                        "method": method,
                        "seed": 0,
                        "kind": kind,
                        "rank": rank,
                        "image_id": r["image_id"],
                        "score": float(r["score"]),
                        "threshold": float(r["threshold"]),
                        "condition": r["condition"],
                    }
                )
with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as handle:
    json.dump(out, handle, indent=1)
