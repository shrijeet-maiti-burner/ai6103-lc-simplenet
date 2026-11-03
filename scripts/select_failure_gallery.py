"""Select failure-gallery cases from predictions.csv by score rank (no images are opened).

Per category and method, seed 0 only: the two highest-scoring flagged normal images (false positives) and the
two lowest-scoring unflagged defective images (false negatives). Ties break on image_id in ascending order.
Usage: python select_gallery.py predictions.csv gallery_selection.json
"""
import csv
import json
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
out = []
for category in sorted({r["category"] for r in rows}):
    for method in ("B0", "B1"):
        sub = [r for r in rows if r["category"] == category and r["method"] == method and r["seed"] == "0"]
        fps = sorted((r for r in sub if r["label"] == "0" and r["flagged"] == "1"), key=lambda r: (-float(r["score"]), r["image_id"]))[:2]
        fns = sorted((r for r in sub if r["label"] == "1" and r["flagged"] == "0"), key=lambda r: (float(r["score"]), r["image_id"]))[:2]
        for kind, picks in (("false_positive", fps), ("false_negative", fns)):
            for rank, r in enumerate(picks, 1):
                out.append({"category": category, "method": method, "seed": 0, "kind": kind, "rank": rank,
                            "image_id": r["image_id"], "score": float(r["score"]),
                            "threshold": float(r["threshold"]), "condition": r["condition"]})
json.dump(out, open(sys.argv[2], "w"), indent=1)
