"""Save real validation predictions for a labelled offline presentation fallback."""

import argparse
import itertools
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18763")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if urlsplit(args.url).hostname not in {"127.0.0.1", "localhost"}:
        parser.error("record from the local inspection server only")
    with urllib.request.urlopen(args.url + "/api/catalog") as response:
        catalog = json.load(response)
    if catalog["mode"] != "live inference":
        parser.error("recording needs the live model runtime")
    frames = []
    for category, item in catalog["categories"].items():
        seed, image_id = item["seeds"][0], item["images"][0]
        item["seeds"], item["images"] = [seed], [image_id]
        for weight, brightness, gamma in itertools.product(
            item["weights"], (0.8, 1.0, 1.2), (0.8, 1.0, 1.2)
        ):
            query = {
                "category": category,
                "seed": seed,
                "image_id": image_id,
                "weight": weight,
                "brightness": brightness,
                "gamma": gamma,
            }
            request = urllib.request.Request(
                args.url + "/api/frame",
                json.dumps(query).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                frames.append(json.load(response))
        print(f"recorded {category}: first normal validation image, seed {seed}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"schema": 1, "catalog": catalog, "frames": frames}, stream, allow_nan=False)
    print(f"saved {len(frames)} measured frames to {args.out}")


if __name__ == "__main__":
    main()
