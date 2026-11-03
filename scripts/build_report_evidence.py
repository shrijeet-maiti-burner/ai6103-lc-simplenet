"""Build report tables/figures from released records; do not invent pending test results."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--ad-release", required=True, type=Path)
    parser.add_argument("--ad-checksum", type=Path, help="default: <ad-release>.sha256")
    parser.add_argument("--out", type=Path, default=Path("report/build/evidence"))
    args = parser.parse_args()
    checksum = args.ad_checksum or args.ad_release.with_suffix(args.ad_release.suffix + ".sha256")
    expected = checksum.read_text().split()[0].lower()
    actual = hashlib.sha256(args.ad_release.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("AD reproduction archive does not match its published SHA-256")
    args.out.mkdir(parents=True, exist_ok=True)
    grouped, source_hashes = {}, {}
    for directory in sorted(args.runs.iterdir()):
        if not (directory / "normal-diagnostics.json").exists():
            continue
        run = json.loads((directory / "run.json").read_text())
        config = run["config"]
        if config["seed"] not in (0, 1, 2) or config["image_size"] != [256, 256]:
            raise ValueError("report evidence requires the locked 256px main seeds")
        method = (
            config["variant"] if config["variant"] != "B2" else f"B2@{config['lambda_score']:g}"
        )
        diagnostic = directory / "normal-diagnostics.json"
        values = json.loads(diagnostic.read_text())
        seeds = grouped.setdefault((config["category"], method), {})
        if config["seed"] in seeds:
            raise ValueError("duplicate category/method/seed in report evidence")
        seeds[config["seed"]] = values
        source_hashes[str(directory.relative_to(args.runs))] = hashlib.sha256(
            diagnostic.read_bytes()
        ).hexdigest()
    categories = sorted({key[0] for key in grouped})
    methods = ["B0", "B1", "B2@0.01", "B2@0.1", "B2@1"]
    if set(categories) != {
        "can",
        "fabric",
        "fruit_jelly",
        "rice",
        "sheet_metal",
        "vial",
        "wallplugs",
        "walnuts",
    } or any(set(grouped.get((c, m), {})) != {0, 1, 2} for c in categories for m in methods):
        raise ValueError("report evidence needs all 120 paired runs")
    means = {
        key: {
            metric: float(np.mean([row[metric] for row in seeds.values()]))
            for metric in ("pair_mse", "patch_score_std", "synthetic_hinge")
        }
        for key, seeds in grouped.items()
    }
    ratios = [means[c, "B0"]["pair_mse"] / means[c, "B1"]["pair_mse"] for c in categories]
    summary = []
    tex = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Category & B0 MSE & B1 MSE & Reduction \\",
        r"\midrule",
    ]
    hinge = np.zeros((8, 3))
    for index, category in enumerate(categories):
        baseline, augmentation = means[category, "B0"], means[category, "B1"]
        label = category.replace("_", " ").title()
        tex.append(
            f"{label} & {baseline['pair_mse']:.3g} & {augmentation['pair_mse']:.3g} & "
            f"{ratios[index]:.1f}$\\times$ " + r"\\"
        )
        for column, method in enumerate(methods[2:]):
            hinge[index, column] = (
                means[category, method]["synthetic_hinge"] / augmentation["synthetic_hinge"]
            )
        summary.append(
            {
                "category": category,
                "reduction": ratios[index],
                "B2_hinge_ratios": hinge[index].tolist(),
            }
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (args.out / "ad2-normal.tex").write_text("\n".join(tex) + "\n")
    with zipfile.ZipFile(args.ad_release) as archive:
        ad = json.loads(archive.read("results.json"))
        seed_scores = {}
        for category in ad:
            for seed in range(3):
                path = f"runs/{category}-seed{seed}/summary.json"
                if path not in archive.namelist():
                    # The three grid diagnostic trajectories are retained separately.
                    candidates = [
                        n
                        for n in archive.namelist()
                        if n.endswith(f"{category}-seed{seed}/summary.json")
                    ]
                    if not candidates:
                        continue
                    path = candidates[0]
                row = json.loads(archive.read(path))["final"]
                seed_scores[category, seed] = row
        macro = {}
        for name, chosen in [("all15", list(ad)), ("without_grid", [c for c in ad if c != "grid"])]:
            macro[name] = {}
            for metric in ("image_auroc", "pixel_auroc"):
                if all((c, s) in seed_scores for c in chosen for s in range(3)):
                    per_seed = [
                        100 * np.mean([seed_scores[c, s][metric] for c in chosen]) for s in range(3)
                    ]
                    macro[name][metric] = {
                        "mean": float(np.mean(per_seed)),
                        "sd": float(np.std(per_seed, ddof=1)),
                    }
        ad_tex = [
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Scope & Image AUROC & Pixel AUROC \\",
            r"\midrule",
        ]
        for name, label in [("all15", "All 15"), ("without_grid", "14 without grid")]:
            if name not in macro or "image_auroc" not in macro[name]:
                raise ValueError(
                    "release lacks the complete 45-run fixed-final trajectory evidence"
                )
            image, pixel = [macro[name][k] for k in ("image_auroc", "pixel_auroc")]
            ad_tex.append(
                f"{label} & {image['mean']:.2f}$\\pm${image['sd']:.2f} & "
                f"{pixel['mean']:.2f}$\\pm${pixel['sd']:.2f} " + r"\\"
            )
        ad_tex += [r"\bottomrule", r"\end{tabular}"]
        (args.out / "ad-reproduction.tex").write_text("\n".join(ad_tex) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    figure, axes = plt.subplots(figsize=(3.4, 2.5))
    axes.barh(categories, ratios, color="#3d7b68", height=0.6)
    axes.set_xscale("log")
    axes.set_xlim(1, 230)
    axes.set_xlabel("normal pair MSE reduction, B0 / B1")
    axes.set_yticks(range(8), [c.replace("_", " ") for c in categories])
    for index, ratio in enumerate(ratios):
        axes.text(ratio * 1.06, index, f"{ratio:.1f}×", va="center", fontsize=8)
    axes.spines[["top", "right"]].set_visible(False)
    axes.invert_yaxis()
    figure.tight_layout(pad=0.4)
    figure.savefig(args.out / "pair-stability.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(figsize=(3.4, 2.8))
    view = axes.imshow(hinge, cmap="RdBu_r", norm=LogNorm(vmin=0.25, vmax=4), aspect="auto")
    axes.set_xticks(range(3), ["λ = 0.01", "λ = 0.1", "λ = 1.0"])
    axes.set_yticks(range(8), [c.replace("_", " ") for c in categories])
    for i in range(8):
        for j in range(3):
            value = hinge[i, j]
            axes.text(
                j,
                i,
                f"{value:.2f}" + ("*" if value > 1.25 else ""),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 2.3 or value < 0.45 else "black",
            )
    figure.colorbar(view, ax=axes, fraction=0.04, pad=0.03).set_label(
        "synthetic hinge ratio B2 / B1"
    )
    figure.tight_layout(pad=0.4)
    figure.savefig(args.out / "hinge-safeguard.pdf")
    plt.close(figure)
    evidence = {
        "scope": "normal-only AD 2 diagnostics and fixed-final AD reproduction",
        "AD2": summary,
        "AD": macro,
        "normal_diagnostic_sha256": source_hashes,
        "AD_release_sha256": hashlib.sha256(args.ad_release.read_bytes()).hexdigest(),
    }
    (args.out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"built evidence from {len(grouped) * 3} AD 2 runs and {len(seed_scores)} AD runs")


if __name__ == "__main__":
    main()
