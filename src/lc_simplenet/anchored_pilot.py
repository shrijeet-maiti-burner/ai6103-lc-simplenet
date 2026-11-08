"""Short fixed-teacher mechanism pilot on normal training parents only.

This is deliberately separate from the original study runners. Synthetic
qualification is not a real-defect improvement or a benchmark result.
"""

import copy
import hashlib
import itertools
import json
import math
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from lc_simplenet.anchored_consistency import (
    adaptation_mode,
    anchored_forward,
    frozen_teacher,
)
from lc_simplenet.evaluation import image_auroc
from lc_simplenet.photometric import relight
from lc_simplenet.pilot import PilotConfig, UpstreamBackend, seed_runtime, state_hash
from lc_simplenet.source_identity import UPSTREAM_SHA, source_record

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "configs/anchored-pilot-policy.json"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 2**20):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    paths = [
        POLICY,
        Path(__file__),
        ROOT / "scripts/anchored_pilot.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    paths += [
        ROOT / "src/lc_simplenet" / name
        for name in (
            "anchored_consistency.py",
            "paired_losses.py",
            "pilot.py",
            "photometric.py",
            "evaluation.py",
            "source_identity.py",
        )
    ]
    paths += [ROOT / "third_party/SimpleNet" / name for name in ("simplenet.py", "common.py")]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def verify_protocol(out):
    protocol = read(out / "protocol.json")
    if protocol["source_sha256"] != source_hashes() or protocol["policy"] != read(POLICY):
        raise ValueError("frozen pilot source or policy changed")
    return protocol


def normal_path(root, image_id):
    parts = PurePosixPath(image_id).parts
    if (
        len(parts) != 4
        or parts[1:3] != ("train", "good")
        or any(p in {".", ".."} or ":" in p or "\\" in p for p in parts)
    ):
        raise ValueError("pilot parents must be original training-normal paths")
    path = (root / image_id).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)):
        raise ValueError("parent escaped the dataset root")
    return path


def plan(out, normal_root, reused_roles, exclusions, rgb_audit):
    if out.exists():
        raise FileExistsError("preserve previous pilot attempts")
    policy, reused, excluded, rgb = (
        read(POLICY),
        read(reused_roles),
        read(exclusions),
        read(rgb_audit),
    )
    excluded_ids = set(excluded["excluded_image_ids"])
    if len(excluded_ids) != len(excluded["excluded_image_ids"]):
        raise ValueError("duplicate exclusion identities")
    training_rows = [row for row in rgb["rows"] if row["split"] == "train"]
    by_id = {row["image_id"]: row for row in training_rows}
    if len(by_id) != len(training_rows):
        raise ValueError("duplicate identities in the RGB audit")
    roles, available = {}, {}
    for category in policy["categories"]:
        selected = {role: [] for role in ("fit", "calibration", "development", "confirmation")}

        def row(image_id):
            path = normal_path(normal_root, image_id)
            original = by_id[image_id]
            if digest(path) != original["sha256"]:
                raise ValueError("RGB audit or source file changed")
            return {key: original[key] for key in ("image_id", "sha256", "decoded_rgb_sha256")}

        for role in ("fit", "calibration"):
            inputs = reused["images"][category][role]
            if len(inputs) != policy[f"{role}_parents"]:
                raise ValueError("incorrect reused fit/calibration count")
            selected[role] = [row(r["image_id"]) for r in inputs]
        candidates = [r for r in by_id if r.startswith(category + "/") and r not in excluded_ids]
        candidates.sort(key=lambda x: hashlib.sha256(("anchored-v1:" + x).encode()).hexdigest())
        available[category] = len(candidates)
        first, second = policy["development_parents"], policy["confirmation_parents"]
        if len(candidates) < first + second:
            raise ValueError("insufficient locally unscored parents; do not recycle a holdout")
        selected["development"] = [row(r) for r in candidates[:first]]
        selected["confirmation"] = [row(r) for r in candidates[first : first + second]]
        identifiers = [r["image_id"] for rows in selected.values() for r in rows]
        pixels = [r["decoded_rgb_sha256"] for rows in selected.values() for r in rows]
        if len(set(identifiers)) != len(identifiers) or len(set(pixels)) != len(pixels):
            raise ValueError("parent or decoded-RGB overlap between roles")
        roles[category] = selected
    identity = source_record(ROOT)
    if identity["git_dirty"] or identity["upstream_dirty"]:
        raise ValueError("commit source before freezing a pilot")
    if identity["upstream_commit"] != UPSTREAM_SHA:
        raise ValueError("unexpected upstream revision")
    out.mkdir(parents=True)
    write(
        out / "protocol.json",
        {
            "scope": "fixed normal-only teacher-anchored SimpleNet pilot; no real-defect claim",
            "policy": policy,
            "roles": roles,
            "locally_available_new_parents": available,
            "source": identity,
            "source_sha256": source_hashes(),
            "inputs": {
                "reused_roles_sha256": digest(reused_roles),
                "exclusions_sha256": digest(exclusions),
                "rgb_audit_sha256": digest(rgb_audit),
            },
            "exposure": (
                "development/confirmation exclude recorded local prototype/gradient parents; "
                "historical full-data fits and teammate exposure are not claimed absent"
            ),
            "confirmation_decoded": False,
            "fitting": (
                "32 fixed normal-image pairs, one deterministic uniform brightness/gamma "
                "transform per parent in [0.8,1.2]; short cached-feature experiment"
            ),
            "encoder_batch": (
                "always four 256px images, padding normal or normal/relit inputs; "
                "only required outputs retained"
            ),
            "selection": (
                "one fixed candidate; all original public results are exposed "
                "and cannot select this protocol"
            ),
            "status": "frozen_before_new_cache_decoding_and_model_fitting",
        },
    )


def rgb_image(root, row):
    path = normal_path(root, row["image_id"])
    if digest(path) != row["sha256"]:
        raise ValueError("normal parent bytes changed")
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1 or source.getexif().get(274, 1) != 1:
            raise ValueError("oriented/multiframe input needs an explicit protocol")
        rgb = source.convert("RGB")
        signature = hashlib.sha256(
            f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()
        ).hexdigest()
        if signature != row["decoded_rgb_sha256"]:
            raise ValueError("decoded RGB identity changed")
        return np.asarray(rgb.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255


def synthetic_changes(image, donor, image_id, role):
    seed = int(
        hashlib.sha256(("anchored-change-v1:" + role + ":" + image_id).encode()).hexdigest()[:8], 16
    )
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = rng.integers(0, h), rng.integers(0, w)
    radius = int(rng.integers(4, 11))
    spot = image.copy()
    spot[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = rng.uniform(0, 1, size=3)
    mask = Image.new("L", (w, h))
    x, y = rng.integers(0, w), rng.integers(0, h)
    angle, length = rng.uniform(0, 2 * np.pi), rng.uniform(32, 96)
    ImageDraw.Draw(mask).line(
        (
            int(x),
            int(y),
            int(np.clip(x + length * np.cos(angle), 0, w - 1)),
            int(np.clip(y + length * np.sin(angle), 0, h - 1)),
        ),
        fill=255,
        width=int(rng.integers(1, 3)),
    )
    scratch = image.copy()
    scratch[np.asarray(mask) > 0] = rng.uniform(0, 1, size=3)
    ph, pw = rng.integers(16, 41, size=2)
    y, x = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
    dy, dx = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
    blend = image.copy()
    blend[y : y + ph, x : x + pw] = (
        0.65 * image[y : y + ph, x : x + pw] + 0.35 * donor[dy : dy + ph, dx : dx + pw]
    )
    return [spot, scratch, blend]


@torch.no_grad()
def encode(backend, views):
    if not 1 <= len(views) <= 4:
        raise ValueError("fixed encoder batch supports one to four images")
    padded = [*views, *([views[0]] * (4 - len(views)))]
    images = torch.from_numpy(np.stack(padded)).permute(0, 3, 1, 2).to("cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    values, _ = backend.embed((images - mean) / std)
    values = values.reshape(4, 1024, 1536)[: len(views)].cpu().numpy()
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("invalid frozen features")
    return values


def backend_for(protocol):
    torch.set_num_threads(2)
    seed_runtime(0)
    if not torch.cuda.is_available():
        raise ValueError("the actual encoder pilot requires the CUDA runtime")
    config = PilotConfig("mvtec_ad2", "can", digest(POLICY), image_size=(256, 256))
    backend = UpstreamBackend(config, torch.device("cuda"))
    return backend


def prepare(out, normal_root, role="development"):
    protocol = verify_protocol(out)
    if role == "confirmation":
        decision = read(out / "screening-decision.json")
        if not decision["passed"]:
            write(
                out / "confirmation-status.json",
                {
                    "state": "not started: complete screening gate failed",
                    "confirmation_decoded": False,
                    "decision_sha256": digest(out / "screening-decision.json"),
                },
            )
            return
        if qualify(read(out / "screening-metrics.json"), protocol["policy"], [0, 1]) != decision:
            raise ValueError("screening decision does not replay")
    cache = out / ("cache" if role == "development" else "confirmation-cache")
    cache.mkdir(exist_ok=False)
    backend = backend_for(protocol)
    encoder_before = state_hash(backend.encoder.state_dict())
    manifest = {}
    policy = protocol["policy"]
    for category in policy["categories"]:
        target = cache / category
        target.mkdir()
        selected = protocol["roles"][category]
        if role == "development":
            for subset in ("fit", "calibration"):
                for index, row in enumerate(selected[subset]):
                    image = rgb_image(normal_root, row)
                    views = [image]
                    if subset == "fit":
                        salt = int(
                            hashlib.sha256(
                                ("anchor-pair-v1:" + row["image_id"]).encode()
                            ).hexdigest()[:8],
                            16,
                        )
                        brightness, gamma = np.random.default_rng(salt).uniform(0.8, 1.2, size=2)
                        views.append(
                            relight(image, brightness=float(brightness), gamma=float(gamma))
                        )
                    path = target / f"{subset}-{index:02d}.npy"
                    np.save(path, encode(backend, views), allow_pickle=False)
                    manifest[path.relative_to(out).as_posix()] = digest(path)
        parents = [rgb_image(normal_root, row) for row in selected[role]]
        metadata = []
        for index, (row, image) in enumerate(zip(selected[role], parents, strict=True)):
            changed = synthetic_changes(
                image, parents[(index + 1) % len(parents)], row["image_id"], role
            )
            for light_index, (brightness, gamma) in enumerate(policy["lights"]):
                views = [relight(v, brightness=brightness, gamma=gamma) for v in [image, *changed]]
                path = target / f"{role}-{index:02d}-light-{light_index}.npy"
                np.save(path, encode(backend, views), allow_pickle=False)
                manifest[path.relative_to(out).as_posix()] = digest(path)
                metadata.append(
                    {
                        "parent_id": row["image_id"],
                        "parent_index": index,
                        "light_index": light_index,
                        "changed_pixels": [
                            0,
                            *[int(np.any(v != views[0], axis=2).sum()) for v in views[1:]],
                        ],
                    }
                )
        write(target / "cases.json", metadata)
        manifest[(target / "cases.json").relative_to(out).as_posix()] = digest(
            target / "cases.json"
        )
        verify_protocol(out)
        write(
            out / f"{role}-preparation-status.json",
            {"category": category, "cache_files": len(manifest)},
        )
        print(
            json.dumps({"prepared": category, "role": role, "cache_files": len(manifest)}),
            flush=True,
        )
        if shutil.disk_usage(out).free < 24 * 2**30:
            raise OSError("24 GiB reserve reached; stop without deleting data")
    if state_hash(backend.encoder.state_dict()) != encoder_before:
        raise AssertionError("feature extractor changed during preparation")
    write(out / f"{role}-cache-manifest.json", manifest)
    write(
        out / f"{role}-encoder.json",
        {
            "encoder_hash": encoder_before,
            "torch": str(torch.__version__),
            "precision": "fp32; TF32 and AMP off; fixed batch four",
            "protocol_sha256": digest(out / "protocol.json"),
        },
    )


def cached(out, manifest, path):
    if digest(path) != manifest[path.relative_to(out).as_posix()]:
        raise ValueError("cached feature identity changed")
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("nonfinite or wrong-precision feature cache")
    return torch.from_numpy(values).to("cuda")


def verified_cases(out, folder, category, role, protocol, manifest):
    path = out / folder / category / "cases.json"
    if digest(path) != manifest[path.relative_to(out).as_posix()]:
        raise ValueError("cached evaluation metadata changed")
    cases = read(path)
    parents = protocol["roles"][category][role]
    expected = set(itertools.product(range(len(parents)), range(len(protocol["policy"]["lights"]))))
    actual = {(c["parent_index"], c["light_index"]) for c in cases}
    if len(cases) != len(expected) or actual != expected:
        raise ValueError("complete unique parent/light evaluation cases required")
    for case in cases:
        if case["parent_id"] != parents[case["parent_index"]]["image_id"]:
            raise ValueError("cached parent identity changed")
        counts = case["changed_pixels"]
        if (
            len(counts) != 4
            or counts[0] != 0
            or any(type(v) is not int or not 0 <= v <= 256**2 for v in counts)
        ):
            raise ValueError("invalid change counts; erased changes must remain present")
    return cases


def optimizers(adapter, discriminator, policy):
    values = policy["optimizer"]
    return (
        torch.optim.AdamW(
            adapter.parameters(),
            lr=values["adapter_lr"],
            weight_decay=values["adapter_weight_decay"],
        ),
        torch.optim.Adam(
            discriminator.parameters(),
            lr=values["discriminator_lr"],
            weight_decay=values["discriminator_weight_decay"],
        ),
    )


def new_heads(seed):
    upstream = ROOT / "third_party/SimpleNet"
    sys.path.insert(0, str(upstream))
    from simplenet import Discriminator, Projection

    seed_runtime(901100 + seed)
    return Projection(1536, 1536, n_layers=1, layer_type=0).to("cuda"), Discriminator(
        1536, n_layers=2, hidden=1024
    ).to("cuda")


def train_heads(out, category, seed, protocol, manifest):
    policy = protocol["policy"]
    target = out / "training" / category / f"seed-{seed}"
    target.mkdir(parents=True, exist_ok=False)
    fit = torch.stack(
        [cached(out, manifest, out / "cache" / category / f"fit-{i:02d}.npy") for i in range(32)]
    )
    adapter, disc = new_heads(seed)
    teacher_opts = optimizers(adapter, disc, policy)
    sampler = torch.Generator().manual_seed(901200 + seed)
    noise_rng = torch.Generator().manual_seed(901300 + seed)
    order = None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    teacher_started = time.monotonic()
    with (target / "teacher-train.jsonl").open("x", encoding="utf-8") as log:
        for step in range(policy["teacher_steps"]):
            if step % 32 == 0:
                order = torch.randperm(32, generator=sampler).tolist()
            clean = fit[order[step % 32], 0]
            noise = torch.randn((2048, 1536), generator=noise_rng).to("cuda") * policy["noise_std"]
            adapter.train()
            disc.train()
            for optimizer in teacher_opts:
                optimizer.zero_grad(set_to_none=True)
            z = adapter(torch.cat((clean, clean)))
            normal, fake = disc(torch.cat((z, z + noise))).chunk(2)
            loss = F.relu(policy["margin"] - normal).mean() + F.relu(policy["margin"] + fake).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite teacher loss")
            loss.backward()
            for optimizer in teacher_opts:
                optimizer.step()
            if step % 64 == 0:
                log.write(json.dumps({"step": step + 1, "loss": loss.item()}) + "\n")
    torch.cuda.synchronize()
    teacher_seconds = time.monotonic() - teacher_started
    teacher_a, teacher_d = frozen_teacher(adapter, disc)
    teacher_hash = [state_hash(t.state_dict()) for t in (teacher_a, teacher_d)]
    models = {"teacher": (teacher_a, teacher_d)}
    opts = {}
    for arm in policy["arms"][1:]:
        models[arm] = copy.deepcopy(adapter), copy.deepcopy(disc)
        adaptation_mode(*models[arm])
        opts[arm] = optimizers(*models[arm], policy)
    if any([state_hash(t.state_dict()) for t in models[arm]] != teacher_hash for arm in opts):
        raise AssertionError("student initializations differ from the fitted teacher")
    write(
        target / "initialization.json",
        {
            "teacher_state_hashes": teacher_hash,
            "arms": list(models),
            "adaptation_optimizer": "fresh identical states; teacher optimizers are not inherited",
        },
    )
    sampler.manual_seed(901400 + seed)
    noise_rng.manual_seed(901500 + seed)
    torch.cuda.synchronize()
    adaptation_started = time.monotonic()
    with (target / "adaptation-train.jsonl").open("x", encoding="utf-8") as log:
        for step in range(policy["adaptation_steps"]):
            if step % 32 == 0:
                order = torch.randperm(32, generator=sampler).tolist()
            clean, light = fit[order[step % 32]]
            noise = torch.randn((2048, 1536), generator=noise_rng).to("cuda") * policy["noise_std"]
            for arm, (student_a, student_d) in models.items():
                if arm == "teacher":
                    continue
                adaptation_mode(student_a, student_d)
                for optimizer in opts[arm]:
                    optimizer.zero_grad(set_to_none=True)
                terms = anchored_forward(
                    student_a,
                    student_d,
                    teacher_a,
                    teacher_d,
                    clean,
                    clean if arm == "identity" else light,
                    noise,
                    margin=policy["margin"],
                    lambda_score=policy["lambda_score"] if arm == "candidate" else 0,
                    lambda_anchor=policy["lambda_anchor"] if arm in {"anchor", "candidate"} else 0,
                )
                if not torch.isfinite(terms["total"]):
                    raise FloatingPointError("nonfinite student loss")
                terms["total"].backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for t in (student_a, student_d)
                    for p in t.parameters()
                ):
                    raise FloatingPointError("nonfinite student gradient")
                for optimizer in opts[arm]:
                    optimizer.step()
                if step % 64 == 0:
                    log.write(
                        json.dumps(
                            {
                                "step": step + 1,
                                "arm": arm,
                                **{k: float(v.detach()) for k, v in terms.items()},
                            }
                        )
                        + "\n"
                    )
            if step % 128 == 0:
                log.flush()
                print(
                    json.dumps({"category": category, "seed": seed, "adaptation_step": step + 1}),
                    flush=True,
                )
    torch.cuda.synchronize()
    write(
        target / "timing.json",
        {
            "teacher_steps": policy["teacher_steps"],
            "teacher_seconds": teacher_seconds,
            "adaptation_steps_per_arm": policy["adaptation_steps"],
            "adaptation_arms": len(opts),
            "joint_adaptation_seconds": time.monotonic() - adaptation_started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scope": "measured training time; not compute-matched with the original study",
        },
    )
    if teacher_hash != [state_hash(t.state_dict()) for t in (teacher_a, teacher_d)]:
        raise AssertionError("teacher parameters or buffers changed")
    for arm, (model_a, model_d) in models.items():
        model_a.eval()
        model_d.eval()
        torch.save(
            {
                "adapter": model_a.state_dict(),
                "discriminator": model_d.state_dict(),
                "teacher_state_hashes": teacher_hash,
                "arm": arm,
                "seed": seed,
                "optimizer_states": [o.state_dict() for o in opts[arm]] if arm in opts else [],
                "sampler_state": sampler.get_state(),
                "noise_state": noise_rng.get_state(),
                "protocol_sha256": digest(out / "protocol.json"),
            },
            target / f"{arm}.pt",
        )
    return models, target


@torch.no_grad()
def scores(model, features):
    adapter, disc = model
    adapter.eval()
    disc.eval()
    values = -disc(adapter(features.reshape(-1, 1536)))
    return values.reshape(len(features), 1024)


def calibrate(out, category, seed, protocol, manifest, models, target):
    result = {arm: {"scores": [], "patches": [], "hinges": []} for arm in models}
    noise_rng = torch.Generator().manual_seed(901600 + seed)
    for index in range(32):
        features = cached(out, manifest, out / "cache" / category / f"calibration-{index:02d}.npy")
        noise = (
            torch.randn((1024, 1536), generator=noise_rng).to("cuda")
            * protocol["policy"]["noise_std"]
        )
        for arm, model in models.items():
            with torch.no_grad():
                patch = scores(model, features)[0]
                result[arm]["scores"].append(float(patch.max()))
                result[arm]["patches"].append(patch.cpu().numpy())
                z = model[0](features[0])
                fake = model[1](z + noise)
                result[arm]["hinges"].append(
                    float(
                        F.relu(protocol["policy"]["margin"] + patch).mean()
                        + F.relu(protocol["policy"]["margin"] + fake).mean()
                    )
                )
            if index == 0:
                state = torch.load(target / f"{arm}.pt", map_location="cuda", weights_only=True)
                reloaded = copy.deepcopy(model[0]), copy.deepcopy(model[1])
                reloaded[0].load_state_dict(state["adapter"], strict=True)
                reloaded[1].load_state_dict(state["discriminator"], strict=True)
                if not torch.equal(patch, scores(reloaded, features)[0]):
                    raise AssertionError("strict checkpoint patch replay failed")
                del reloaded, state
    records = {
        arm: {
            "threshold": float(
                np.quantile(r["scores"], 1 - protocol["policy"]["target_fpr"], method="higher")
            ),
            "scores": r["scores"],
            "patch_std": float(np.concatenate(r["patches"]).astype(np.float64).std()),
            "gaussian_hinge": float(np.mean(r["hinges"])),
            "replay_bit_exact": True,
            "seed": seed,
        }
        for arm, r in result.items()
    }
    write(target / "calibration.json", records)
    return records


def evaluate_cached(out, category, seed, role, protocol, manifest, models, calibration, target):
    folder = "cache" if role == "development" else "confirmation-cache"
    cases = verified_cases(out, folder, category, role, protocol, manifest)
    records = []
    kinds = ("normal", "spot", "scratch", "blend")
    for case in cases:
        path = (
            out
            / folder
            / category
            / f"{role}-{case['parent_index']:02d}-light-{case['light_index']}.npy"
        )
        features = cached(out, manifest, path)
        for arm, model in models.items():
            prediction = scores(model, features).max(dim=1).values.tolist()
            for kind, score, changed in zip(kinds, prediction, case["changed_pixels"], strict=True):
                threshold = calibration[arm]["threshold"]
                records.append(
                    {
                        "category": category,
                        "seed": seed,
                        "arm": arm,
                        "role": role,
                        "parent_id": case["parent_id"],
                        "light_index": case["light_index"],
                        "kind": kind,
                        "label": int(kind != "normal"),
                        "changed_pixels": changed,
                        "score": score,
                        "threshold": threshold,
                        "flagged": score > threshold,
                    }
                )
    write(target / f"{role}-predictions.json", records)
    metrics = []
    for arm, condition in itertools.product(models, ("identity", "corners")):
        selected = [
            r
            for r in records
            if r["arm"] == arm and (r["light_index"] == 0) == (condition == "identity")
        ]
        normal = [r for r in selected if r["label"] == 0]
        altered = [r for r in selected if r["label"] == 1]
        metrics.append(
            {
                "category": category,
                "seed": seed,
                "arm": arm,
                "condition": condition,
                "normal": len(normal),
                "synthetic_changes": len(altered),
                "false_positives": sum(r["flagged"] for r in normal),
                "true_positives": sum(r["flagged"] for r in altered),
                "image_auroc": image_auroc(
                    [r["score"] for r in selected], [r["label"] for r in selected]
                ),
                "erased_changes": sum(r["changed_pixels"] == 0 for r in altered),
                "patch_std": calibration[arm]["patch_std"],
                "gaussian_hinge": calibration[arm]["gaussian_hinge"],
            }
        )
    write(target / f"{role}-metrics.json", metrics)
    return metrics


def qualify(rows, policy, seeds):
    expected = set(
        itertools.product(policy["categories"], seeds, policy["arms"], ("identity", "corners"))
    )
    indexed = {(r["category"], r["seed"], r["arm"], r["condition"]): r for r in rows}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("complete unique category/seed/arm/condition metrics required")
    parents = policy[
        "confirmation_parents" if seeds == [policy["confirmation_seed"]] else "development_parents"
    ]
    for row in rows:
        multiplier = 1 if row["condition"] == "identity" else len(policy["lights"]) - 1
        if (
            row["normal"] != parents * multiplier
            or row["synthetic_changes"] != 3 * parents * multiplier
        ):
            raise ValueError(
                "expected every parent and every synthetic change, including erased ones"
            )
        for key in ("normal", "synthetic_changes", "false_positives", "true_positives"):
            if type(row[key]) is not int:
                raise ValueError("raw counts must be integers")
        if not all(math.isfinite(row[k]) and row[k] >= 0 for k in ("patch_std", "gaussian_hinge")):
            raise ValueError("invalid normal-calibration safeguards")
        if row["patch_std"] == 0:
            raise ValueError("collapsed calibration scores cannot qualify")
    rule = policy["qualification"]

    def fraction(row, key):
        numerator, denominator = (
            ("false_positives", "normal")
            if key == "fpr"
            else ("true_positives", "synthetic_changes")
        )
        if not 0 <= row[numerator] <= row[denominator] or row[denominator] <= 0:
            raise ValueError("invalid raw count")
        return Fraction(row[numerator], row[denominator])

    per_seed = []
    for seed in seeds:
        comparisons = {}
        eligible = True
        for control in ("teacher", "identity", "augmentation", "anchor"):
            changes = {}
            for condition in ("identity", "corners"):
                pairs = [
                    (indexed[c, seed, "candidate", condition], indexed[c, seed, control, condition])
                    for c in policy["categories"]
                ]
                changes[condition] = {
                    k: sum((fraction(a, k) - fraction(b, k) for a, b in pairs), Fraction())
                    / len(pairs)
                    for k in ("fpr", "recall")
                }
            drops = [
                fraction(indexed[c, seed, "candidate", "corners"], "recall")
                - fraction(indexed[c, seed, control, "corners"], "recall")
                for c in policy["categories"]
            ]
            gain = (
                rule["corner_fpr_drop_vs_anchor"]
                if control == "anchor"
                else rule["corner_fpr_drop_vs_identity_and_augmentation"]
            )
            passed = (
                changes["corners"]["fpr"] <= -Fraction(str(gain))
                and changes["corners"]["recall"] >= 0
                and changes["identity"]["fpr"] <= Fraction(str(rule["clean_fpr_increase_max"]))
                and changes["identity"]["recall"] >= -Fraction(str(rule["clean_recall_drop_max"]))
                and min(drops) >= -Fraction(str(rule["worst_category_corner_recall_drop_max"]))
            )
            comparisons[control] = {
                "deltas": {
                    c: {k: float(v) for k, v in values.items()} for c, values in changes.items()
                },
                "passed": passed,
            }
        positive = sum(
            fraction(indexed[c, seed, "candidate", "corners"], "fpr")
            < fraction(indexed[c, seed, "augmentation", "corners"], "fpr")
            for c in policy["categories"]
        )
        for category in policy["categories"]:
            a = indexed[category, seed, "candidate", "identity"]
            b = indexed[category, seed, "augmentation", "identity"]
            eligible &= (
                a["patch_std"]
                >= rule["minimum_calibration_patch_std_ratio_vs_augmentation"] * b["patch_std"]
            )
            eligible &= (
                a["gaussian_hinge"]
                <= rule["maximum_calibration_gaussian_hinge_ratio_vs_augmentation"]
                * b["gaussian_hinge"]
            )
        per_seed.append(
            {
                "seed": seed,
                "comparisons": comparisons,
                "eligible": bool(eligible),
                "categories_fpr_improved": positive,
                "passed": bool(
                    eligible
                    and all(r["passed"] for r in comparisons.values())
                    and positive >= rule["categories_corner_fpr_improved_vs_augmentation_min"]
                ),
            }
        )
    return {
        "passed": all(r["passed"] for r in per_seed),
        "per_seed": per_seed,
        "scope": "synthetic qualification only; no real-defect or SOTA conclusion",
    }


def train(out, role="development"):
    protocol = verify_protocol(out)
    policy = protocol["policy"]
    if role == "confirmation":
        decision = qualify(read(out / "screening-metrics.json"), policy, policy["seeds"])
        if not decision["passed"]:
            write(
                out / "confirmation-status.json",
                {"state": "not started: screening gate failed", "confirmation_decoded": False},
            )
            return
    seed_runtime(0)
    torch.set_num_threads(2)
    manifest = read(out / "development-cache-manifest.json")
    evaluation_manifest = (
        manifest if role == "development" else read(out / "confirmation-cache-manifest.json")
    )
    seeds = policy["seeds"] if role == "development" else [policy["confirmation_seed"]]
    all_rows = []
    started = time.monotonic()
    for category, seed in itertools.product(policy["categories"], seeds):
        verify_protocol(out)
        models, target = train_heads(out, category, seed, protocol, manifest)
        calibration = calibrate(out, category, seed, protocol, manifest, models, target)
        all_rows.extend(
            evaluate_cached(
                out,
                category,
                seed,
                role,
                protocol,
                evaluation_manifest,
                models,
                calibration,
                target,
            )
        )
        del models
        torch.cuda.empty_cache()
        write(
            out / f"{role}-training-status.json",
            {
                "category": category,
                "seed": seed,
                "metric_rows": len(all_rows),
                "seconds": time.monotonic() - started,
            },
        )
    verify_protocol(out)
    prefix = "screening" if role == "development" else "confirmation"
    write(out / f"{prefix}-metrics.json", all_rows)
    decision = qualify(all_rows, policy, seeds)
    write(out / f"{prefix}-decision.json", decision)
    print(json.dumps(decision), flush=True)
    if role == "development" and not decision["passed"]:
        write(
            out / "confirmation-status.json",
            {"state": "not started: complete screening gate failed", "confirmation_decoded": False},
        )
