import json
from datetime import datetime
from pathlib import Path

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
    print(f"\n{category.upper()} ({total} steps per seed)")

    for seed in range(3):
        folder = root / category / "B2" / f"seed-{seed}"
        path = folder / "train.jsonl"
        result_path = folder / "result.json"

        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                if result.get("status") == "completed" and result.get("steps") == total:
                    print(f"  Seed {seed}: COMPLETE | {total}/{total} | 100.0%")
                    completed += 1
                    continue
            except Exception:
                pass

        if not path.exists():
            print(f"  Seed {seed}: WAITING")
            continue

        try:
            lines = path.read_text().splitlines()
            if not lines:
                print(f"  Seed {seed}: INITIALISING")
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
            print(f"  Seed {seed}: WRITING / READING")

print("\n" + "=" * 85)
print(f"VERIFIED COMPLETE: {completed}/24 runs")
print(f"CURRENTLY IN PROGRESS: {active} runs")
