"""Choose the B2 consistency weight from the tuning-seed search, on normal validation only."""

import argparse
import json
from pathlib import Path

from lc_simplenet.study import select_consistency_weight


def diagnostics(run: Path) -> dict:
    return json.loads((run / "normal-diagnostics.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True, help="Holds search/<run>/")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    seed, candidates = policy["weight_search"]["seed"], policy["weight_search"]["candidates"]
    search = args.study / "search"
    trials = []
    for category in policy["selected_categories"]:
        b1 = diagnostics(search / f"{category}-B1-seed{seed}")
        for weight in candidates:
            b2 = diagnostics(search / f"{category}-B2-lam{weight}-seed{seed}")
            trial = {"category": category, "candidate": weight}
            trials.append({**trial, "variant": "B1", "metrics": b1})
            trials.append({**trial, "variant": "B2", "metrics": b2})
    selection = select_consistency_weight(trials, candidates)
    selection["trials"] = trials
    text = json.dumps(selection, indent=2) + "\n"
    (args.study / "selection.json").write_text(text, encoding="utf-8")
    print(json.dumps({key: value for key, value in selection.items() if key != "trials"}))


if __name__ == "__main__":
    main()
