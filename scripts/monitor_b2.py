import json
import time
from pathlib import Path
from datetime import datetime

steps = {
    "can": 4160,
    "fabric": 3920,
    "fruit_jelly": 2640,
    "rice": 3200,
    "sheet_metal": 1440,
    "vial": 2960,
    "wallplugs": 2960,
    "walnuts": 4320,
}

root = Path("results/ad2-b2-full-lambda001")
completed = 0
active = 0

print("B2 LAMBDA 0.01 | FULL EXPERIMENT")
print("Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 85)

for category, total in steps.items():
    print("\n{} ({} steps per seed)".format(category.upper(), total))

    for seed in range(3):
        folder = root / category / "B2" / "seed-{}".format(seed)
        path = folder / "train.jsonl"
        result_path = folder / "result.json"

        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                if result.get("status") == "completed" and result.get("steps") == total:
                    print("  Seed {}: COMPLETE | {}/{} | 100.0%".format(seed, total, total))
                    completed += 1
                    continue
            except Exception:
                pass

        if not path.exists():
            print("  Seed {}: WAITING".format(seed))
            continue

        try:
            lines = path.read_text().splitlines()
            if not lines:
                print("  Seed {}: INITIALISING".format(seed))
                continue

            d = json.loads(lines[-1])
            step = int(d["step"])
            pct = min(100, 100 * step / total)

            print(
                "  Seed {}: {}/{} | {:.1f}% | Epoch {} | Loss {:.6f}".format(
                    seed, step, total, pct,
                    d.get("epoch", "?"), float(d.get("loss", 0))
                )
            )

            if step < total:
                active += 1

        except Exception:
            print("  Seed {}: WRITING / READING".format(seed))

print("\n" + "=" * 85)
print("VERIFIED COMPLETE: {}/24 runs".format(completed))
print("CURRENTLY IN PROGRESS: {} runs".format(active))
