"""Replay every calibration and evaluation score from the saved CUDA heads."""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc_simplenet.anchored_pilot import (  # noqa: E402
    cached,
    digest,
    new_heads,
    read,
    verified_cases,
    verify_protocol,
    write,
)
from lc_simplenet.pilot import seed_runtime, state_hash  # noqa: E402


@torch.no_grad()
def replay(out, role):
    protocol = verify_protocol(out)
    policy = protocol["policy"]
    seed_runtime(0)
    torch.set_num_threads(2)
    seeds = policy["seeds"] if role == "development" else [policy["confirmation_seed"]]
    manifest = read(out / "development-cache-manifest.json")
    folder = "cache" if role == "development" else "confirmation-cache"
    evaluation_manifest = (
        manifest if role == "development" else read(out / "confirmation-cache-manifest.json")
    )
    checked_scores, hashes = 0, {}
    for category, seed in itertools.product(policy["categories"], seeds):
        target = out / "training" / category / f"seed-{seed}"
        models = {}
        for arm in policy["arms"]:
            path = target / f"{arm}.pt"
            checkpoint = torch.load(path, map_location="cuda", weights_only=True)
            if (checkpoint["seed"], checkpoint["arm"], checkpoint["protocol_sha256"]) != (
                seed,
                arm,
                digest(out / "protocol.json"),
            ):
                raise AssertionError("checkpoint binding changed")
            model = new_heads(seed)
            for network, key in zip(model, ("adapter", "discriminator"), strict=True):
                network.load_state_dict(checkpoint[key], strict=True)
                network.eval()
            if (
                arm == "teacher"
                and [state_hash(t.state_dict()) for t in model]
                != checkpoint["teacher_state_hashes"]
            ):
                raise AssertionError("teacher checkpoint no longer matches the original teacher")
            models[arm] = model
            hashes[path.relative_to(out).as_posix()] = digest(path)
            del checkpoint
        calibration = read(target / "calibration.json")
        patches, hinges = {arm: [] for arm in models}, {arm: [] for arm in models}
        generator = torch.Generator().manual_seed(901600 + seed)
        for index in range(policy["calibration_parents"]):
            features = cached(
                out, manifest, out / "cache" / category / f"calibration-{index:02d}.npy"
            )
            noise = torch.randn((1024, 1536), generator=generator).to("cuda") * policy["noise_std"]
            for arm, (adapter, disc) in models.items():
                z = adapter(features.reshape(-1, 1536))
                patch = -disc(z).reshape(-1)
                if float(patch.max()) != calibration[arm]["scores"][index]:
                    raise AssertionError("calibration score did not replay bit-exactly")
                fake = disc(z + noise)
                hinges[arm].append(
                    float(
                        F.relu(policy["margin"] + patch).mean()
                        + F.relu(policy["margin"] + fake).mean()
                    )
                )
                patches[arm].append(patch.cpu().numpy())
        for arm in models:
            std = float(np.concatenate(patches[arm]).astype(np.float64).std())
            hinge = float(np.mean(hinges[arm]))
            if std != calibration[arm]["patch_std"] or hinge != calibration[arm]["gaussian_hinge"]:
                raise AssertionError("normal-calibration safeguards did not replay exactly")
        predictions = read(target / f"{role}-predictions.json")
        indexed = {(r["arm"], r["parent_id"], r["light_index"], r["kind"]): r for r in predictions}
        if len(indexed) != len(predictions):
            raise AssertionError("duplicate prediction keys")
        seen = set()
        for case in verified_cases(out, folder, category, role, protocol, evaluation_manifest):
            path = (
                out
                / folder
                / category
                / f"{role}-{case['parent_index']:02d}-light-{case['light_index']}.npy"
            )
            features = cached(out, evaluation_manifest, path)
            for arm, (adapter, disc) in models.items():
                patch = -disc(adapter(features.reshape(-1, 1536)))
                maxima = patch.reshape(4, 1024).max(dim=1).values.tolist()
                for kind, value in zip(("normal", "spot", "scratch", "blend"), maxima, strict=True):
                    key = arm, case["parent_id"], case["light_index"], kind
                    if value != indexed[key]["score"]:
                        raise AssertionError("evaluation score did not replay bit-exactly")
                    seen.add(key)
                    checked_scores += 1
        if seen != set(indexed):
            raise AssertionError("saved prediction population differs from full cache replay")
        del models, patches, hinges
        torch.cuda.empty_cache()
        print({"replayed": category, "seed": seed, "evaluation_scores": checked_scores}, flush=True)
    return {
        "scope": "saved-head CUDA replay; not a predictive improvement",
        "protocol_sha256": digest(out / "protocol.json"),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "calibration_and_safeguards_exact": True,
        "evaluation_scores_bit_exact": checked_scores,
        "checkpoint_sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--role", choices=("development", "confirmation"), default="development")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve replay attempts")
    write(args.out, replay(args.run, args.role))


if __name__ == "__main__":
    main()
