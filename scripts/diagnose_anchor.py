"""Describe fit/calibration score separation after the fixed pilot; never refit."""

import argparse
import hashlib
import itertools
import sys
import time
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
    verify_protocol,
    write,
)
from lc_simplenet.pilot import seed_runtime, state_hash  # noqa: E402


@torch.no_grad()
def diagnose(run, out):
    protocol = verify_protocol(run)
    policy = protocol["policy"]
    manifest = read(run / "development-cache-manifest.json")
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "protocol.json",
        {
            "scope": "post-hoc fit/calibration diagnosis, not a new candidate or causal test",
            "categories": policy["categories"],
            "seeds": policy["seeds"],
            "arms": policy["arms"],
            "roles": ["fit", "calibration"],
            "noise": "per-parent CPU Gaussian, anchor-generalization-v1:seed:image_id SHA256",
            "forward": "eval; duplicate clean view retains original four-group batch",
            "boundary": "no refitting, thresholds, confirmation, procedural changes or test input",
            "comparison": "calibration is optimizer-unseen, not an untouched development set",
            "original_protocol_sha256": digest(run / "protocol.json"),
            "script_sha256": digest(Path(__file__)),
        },
    )
    torch.set_num_threads(2)
    seed_runtime(0)
    rows, hashes = [], {}
    started = time.monotonic()
    for category, seed in itertools.product(policy["categories"], policy["seeds"]):
        models = {}
        for arm in policy["arms"]:
            path = run / "training" / category / f"seed-{seed}" / f"{arm}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if (checkpoint["protocol_sha256"], checkpoint["arm"], checkpoint["seed"]) != (
                digest(run / "protocol.json"),
                arm,
                seed,
            ):
                raise ValueError("checkpoint binding changed")
            model = new_heads(seed)
            for network, key in zip(model, ("adapter", "discriminator"), strict=True):
                network.load_state_dict(checkpoint[key], strict=True)
                network.eval()
            if (
                arm == "teacher"
                and [state_hash(n.state_dict()) for n in model]
                != checkpoint["teacher_state_hashes"]
            ):
                raise ValueError("teacher state changed")
            models[arm] = model
            hashes[path.relative_to(run).as_posix()] = digest(path)
        for role in ("fit", "calibration"):
            parents = protocol["roles"][category][role]
            if len(parents) != policy[f"{role}_parents"]:
                raise ValueError("parent count changed")
            for index, parent in enumerate(parents):
                features = cached(
                    run, manifest, run / "cache" / category / f"{role}-{index:02d}.npy"
                )[0]
                key = f"anchor-generalization-v1:{seed}:{parent['image_id']}"
                salt = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                generator = torch.Generator().manual_seed(salt)
                noise = (
                    torch.randn((2048, 1536), generator=generator).to("cuda") * policy["noise_std"]
                )
                inputs = torch.cat((features, features))
                values = {}
                for arm, (adapter, discriminator) in models.items():
                    adapted = adapter(inputs)
                    clean, _, gaussian, _ = discriminator(
                        torch.cat((adapted, adapted + noise))
                    ).chunk(4)
                    values[arm] = (clean, gaussian)
                teacher_normal, teacher_fake = values["teacher"]
                for arm, (normal, fake) in values.items():
                    normal_mse = F.mse_loss(normal, teacher_normal).item()
                    gaussian_mse = F.mse_loss(fake, teacher_fake).item()
                    normal_hinge = F.relu(policy["margin"] - normal).mean().item()
                    gaussian_hinge = F.relu(policy["margin"] + fake).mean().item()
                    rows.append(
                        {
                            "category": category,
                            "seed": seed,
                            "role": role,
                            "image_id": parent["image_id"],
                            "arm": arm,
                            "normal_teacher_mse": normal_mse,
                            "gaussian_teacher_mse": gaussian_mse,
                            "anchor_mse": (normal_mse + gaussian_mse) / 2,
                            "normal_hinge": normal_hinge,
                            "gaussian_hinge": gaussian_hinge,
                            "hinge_sum": normal_hinge + gaussian_hinge,
                            "normal_minus_gaussian_mean": (normal - fake).mean().item(),
                            "normal_above_gaussian_fraction": (normal > fake).float().mean().item(),
                        }
                    )
        print(f"analyzed {category} seed {seed}: {len(rows)} parent-arm rows", flush=True)
        del models, values
        torch.cuda.empty_cache()
    keys = [key for key in rows[0] if key not in {"category", "seed", "role", "image_id", "arm"}]
    aggregates = []
    for category, seed, role, arm in itertools.product(
        policy["categories"], policy["seeds"], ("fit", "calibration"), policy["arms"]
    ):
        group = [
            row
            for row in rows
            if (row["category"], row["seed"], row["role"], row["arm"])
            == (category, seed, role, arm)
        ]
        aggregates.append(
            {
                "category": category,
                "seed": seed,
                "role": role,
                "arm": arm,
                "parents": len(group),
                **{key: float(np.mean([row[key] for row in group])) for key in keys},
            }
        )
    write(out / "parents.json", rows)
    write(out / "aggregate.json", aggregates)
    write(
        out / "status.json",
        {
            "state": "complete",
            "parent_arm_rows": len(rows),
            "aggregate_rows": len(aggregates),
            "seconds": time.monotonic() - started,
            "checkpoint_sha256": hashes,
            "confirmation_decoded": False,
            "real_test_data_read": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run, args.out)
