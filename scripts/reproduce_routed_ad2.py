"""Reproduce the fixed R0 training recipe from its checked execution snapshot.

This launcher does not choose a method, epoch, seed or threshold from test data.
Data and ImageNet weights must be acquired separately; see docs/routed-reproduction.md.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import zipfile
from datetime import UTC
from pathlib import Path, PurePosixPath

SOURCE_SHA256 = "74f2c9adb729e432237a6879ae82373a3dcee032e5517e7daabb515b5096cb8a"
FROZEN_PROTOCOL = "751098d10d402f10b10f61d6e5b39a768a74dad04e59af0c9df917063dd57c08"
BACKBONE_SHA256 = "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"
ENCODER_SHA256 = "aa373db6792ab7d661690bd10e1897c76de2abfb7039145c871709ac02ab3509"
ARMS = ["B0", "B1", "B2", "L0", "L1", "L2", "R0"]
CATEGORIES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


@contextlib.contextmanager
def execution_source(archive):
    if sha(archive) != SOURCE_SHA256:
        raise ValueError("execution source archive hash differs")
    with tempfile.TemporaryDirectory(prefix="routed-ad2-source-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive) as source:
            manifest = json.loads(source.read("SOURCE-MANIFEST.json"))
            if set(source.namelist()) != {*manifest["files"], "SOURCE-MANIFEST.json"}:
                raise ValueError("execution source inventory differs")
            for name, expected in manifest["files"].items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                    raise ValueError("unsafe source member")
                raw = source.read(name)
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError("execution source file changed")
                destination = root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
        yield root


def policy(root):
    return read(root / "configs/ad2-study-policy.json")


def texture_manifest(root):
    return read(root / "results/published-research-20261109/dtd-texture-subset-001/manifest.json")[
        "files"
    ]


def safe_input(root, relative):
    path = (Path(root) / relative).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError("input escapes its declared root")
    return path


def add_zip(archive, name, raw):
    info = zipfile.ZipInfo(name, date_time=(2026, 11, 16, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, raw)


def prepare_into(args, root):
    import numpy as np
    import PIL
    from PIL import Image

    if (np.__version__, PIL.__version__) != ("1.26.4", "11.3.0"):
        raise ValueError("use the locked numpy/pillow versions for exact staging")
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / (args.category + "-normals.json")
    archive_path = output / (args.category + "-normals-256.zip")
    if metadata_path.exists() or archive_path.exists():
        raise ValueError(
            "normal staging already exists; use another output or reuse the existing inputs"
        )
    manifest = root / f"references/ad2-normal-manifests/{args.category}.json"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    counts = {s: sum(r["split"] == s for r in records) for s in ["train", "validation"]}
    if list(counts.values()) != policy(root)["category_counts_train_validation"][args.category]:
        raise ValueError("original source counts differ")
    staged = []
    with zipfile.ZipFile(archive_path, "x") as archive:
        for record in records:
            if record["label"] != 0 or record["mask"] is not None:
                raise ValueError("only original normal sources may enter staging")
            path = safe_input(args.data_root, record["relative_path"])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("original normal bytes differ: " + record["image_id"])
            with Image.open(io.BytesIO(raw)) as image:
                rgb = image.convert("RGB")
                actual = hashlib.sha256(
                    f"{rgb.width},{rgb.height},RGB:".encode() + rgb.tobytes()
                ).hexdigest()
                if actual != record["rgb_sha256"]:
                    raise ValueError("original normal pixels differ")
                resized = rgb.resize((256, 256), Image.Resampling.BILINEAR)
                buffer = io.BytesIO()
                resized.save(buffer, format="PNG")
                encoded = buffer.getvalue()
                pixel_sha = hashlib.sha256(np.asarray(resized).tobytes()).hexdigest()
            add_zip(archive, record["image_id"], encoded)
            staged.append(
                {
                    **record,
                    "native_sha256": record["sha256"],
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "resized_rgb_sha256": pixel_sha,
                }
            )
    actual = [{k: r[k] for k in ["image_id", "sha256", "resized_rgb_sha256"]} for r in staged]
    expected = read(root / "references/routed-input-fingerprints.json")[args.category]
    if actual != expected:
        raise ValueError("staged bytes or pixel fingerprints differ from the frozen inputs")
    write(
        metadata_path,
        {
            "category": args.category,
            "manifest_sha256": sha(manifest),
            "rows": staged,
            "counts": counts,
            "archive_sha256": sha(archive_path),
            "geometry": "native RGB, full-frame Pillow bilinear 256x256, lossless staged PNG",
        },
    )
    textures = output / "textures.zip"
    donors = texture_manifest(root)
    if not textures.exists():
        with zipfile.ZipFile(textures, "x") as archive:
            for row in donors:
                raw = safe_input(args.textures_root, row["path"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError("official texture bytes differ: " + row["path"])
                add_zip(archive, row["path"], raw)
    with zipfile.ZipFile(textures) as archive:
        if set(archive.namelist()) != {r["path"] for r in donors}:
            raise ValueError("texture population differs")
        for row in donors:
            if hashlib.sha256(archive.read(row["path"])).hexdigest() != row["sha256"]:
                raise ValueError("texture packet differs")
    return counts


def prepare(args, root):
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=args.out) as temporary:
        staged = argparse.Namespace(**{**vars(args), "out": Path(temporary)})
        counts = prepare_into(staged, root)
        files = list(Path(temporary).iterdir())
        for path in files:
            destination = args.out / path.name
            if destination.exists() and sha(destination) != sha(path):
                raise ValueError("existing staged file differs: " + path.name)
        for path in files:
            destination = args.out / path.name
            if not destination.exists():
                path.replace(destination)
    print(
        json.dumps({"prepared": args.category, "counts": counts, "test_images_opened": False}),
        flush=True,
    )


def bootstrap(root):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path[:0] = [str(root / "src"), str(root / "results/recovery/ad2-local-global-20261115")]
    guard = types.ModuleType("metrics")
    guard.__file__ = "<excluded-author-test-metrics>"

    def forbidden(name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError("the upstream training/test metrics are excluded: " + name)

    guard.__getattr__ = forbidden
    sys.modules["metrics"] = guard
    import numpy as np
    import PIL
    import torch
    import torchvision

    if (
        torch.__version__.split("+")[0],
        torchvision.__version__.split("+")[0],
        np.__version__,
        PIL.__version__,
    ) != ("2.6.0", "0.21.0", "1.26.4", "11.3.0"):
        raise ValueError("use the repository's locked torch/torchvision/numpy/pillow environment")
    torch.set_num_threads(4)
    return torch, np


def load_inputs(root, inputs, category, *, mechanics=False):
    import numpy as np
    from ad2_train import StagedDataset
    from PIL import Image

    metadata = read(inputs / (category + "-normals.json"))
    expected_manifest = root / f"references/ad2-normal-manifests/{category}.json"
    if metadata["category"] != category or metadata["manifest_sha256"] != sha(expected_manifest):
        raise ValueError("normal source manifest differs")
    counts = policy(root)["category_counts_train_validation"][category]
    if [metadata["counts"][s] for s in ["train", "validation"]] != counts:
        raise ValueError("original normal counts differ")
    expected_rows = read(root / "references/routed-input-fingerprints.json")[category]
    actual_rows = [
        {k: r[k] for k in ["image_id", "sha256", "resized_rgb_sha256"]} for r in metadata["rows"]
    ]
    if actual_rows != expected_rows:
        raise ValueError("original staged input fingerprint/order differs")
    native_rows = [json.loads(line) for line in expected_manifest.read_text().splitlines()]
    for row, original in zip(metadata["rows"], native_rows, strict=True):
        if (
            any(row[k] != value for k, value in original.items() if k != "sha256")
            or row["native_sha256"] != original["sha256"]
        ):
            raise ValueError("original normal roles or identity changed")
    archive = inputs / (category + "-normals-256.zip")
    if sha(archive) != metadata["archive_sha256"]:
        raise ValueError("staged normal packet changed")
    data = StagedDataset(archive, metadata, "train")
    validation = StagedDataset(archive, metadata, "validation")
    textures = []
    with zipfile.ZipFile(inputs / "textures.zip") as source:
        rows = texture_manifest(root)
        if set(source.namelist()) != {r["path"] for r in rows}:
            raise ValueError("texture population differs")
        for row in rows:
            raw = source.read(row["path"])
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("texture source changed")
            with Image.open(io.BytesIO(raw)) as image:
                textures.append(
                    np.asarray(
                        image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
                        dtype=np.float32,
                    )
                    / np.float32(255)
                )
    if mechanics:
        data.records, data.images = data.records[:1], data.images[:1]
        validation.records, validation.images = validation.records[:2], validation.images[:2]
    return data, validation, metadata, textures


def new_trainer(args, root):
    torch, np = bootstrap(root)
    from ad2_train import Trainer

    data, validation, metadata, textures = load_inputs(
        root, args.inputs, args.category, mechanics=args.mechanics_only
    )
    trainer = Trainer(data, metadata, textures, args.seed, "R0", device=args.device)
    weights = Path(torch.hub.get_dir()) / "checkpoints/wide_resnet50_2-95faca4d.pth"
    if sha(weights) != BACKBONE_SHA256 or trainer.encoder.raw.encoder_hash != ENCODER_SHA256:
        raise ValueError("frozen pretrained encoder identity differs")
    return trainer, validation, torch, np


def train(args, root):
    from datetime import datetime

    trainer, validation, torch, np = new_trainer(args, root)
    from ad2_runtime import calibrate, preview

    protocol = (
        FROZEN_PROTOCOL
        if not args.mechanics_only
        else hashlib.sha256((FROZEN_PROTOCOL + ":mechanics-first1").encode()).hexdigest()
    )
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    binding = {
        "category": args.category,
        "seed": args.seed,
        "source_archive_sha256": SOURCE_SHA256,
        "protocol_sha256": protocol,
        "mechanics_only": args.mechanics_only,
        "normal_metadata_sha256": sha(args.inputs / (args.category + "-normals.json")),
        "textures_sha256": sha(args.inputs / "textures.zip"),
        "required_updates": trainer.end,
        "same_gpu_as_original_run": args.device.startswith("cuda")
        and torch.cuda.get_device_name() == "NVIDIA L40S",
    }
    run_path = output / "run.json"
    if run_path.exists():
        if not args.resume or read(run_path) != binding:
            raise ValueError(
                "use --resume with matching inputs, device and output; existing run differs"
            )
    else:
        write(run_path, binding)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        trainer.restore(state, protocol)
        if preview(trainer) != state["next_preview"]:
            raise ValueError("resume next-input/feature fingerprint differs")
        del state
    end = min(trainer.end, trainer.step + 2) if args.mechanics_only else trainer.end
    if args.stop_after is not None:
        end = min(end, args.stop_after)
    if not trainer.step < end <= trainer.end:
        raise ValueError("requested endpoint must advance the saved state")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    with (output / ("steps-" + stamp + ".jsonl")).open("x", encoding="utf-8") as log:
        while trainer.step < end:
            value = trainer.train_step()
            log.write(json.dumps(value, allow_nan=False) + "\n")
            log.flush()
            if trainer.step % 32 == 0 or trainer.step == end:
                print(
                    json.dumps({"step": trainer.step, "required_updates": trainer.end}), flush=True
                )
            if trainer.step % 256 == 0 or trainer.step == end:
                state = trainer.state(protocol)
                state["next_preview"] = preview(trainer)
                path = output / f"checkpoint-{trainer.step:05d}.pt"
                if path.exists():
                    raise ValueError("refuse to overwrite a checkpoint")
                temporary = path.with_suffix(".tmp")
                torch.save(state, temporary)
                temporary.replace(path)
                latest = output / "latest.json"
                pending = latest.with_suffix(".tmp")
                pending.write_text(
                    json.dumps({"path": path.name, "sha256": sha(path), "step": trainer.step}),
                    encoding="utf-8",
                )
                pending.replace(latest)
                del state
    if trainer.step == trainer.end and not args.mechanics_only:
        metrics, patches, ids = calibrate(trainer, validation)
        write(
            output / "calibration.json",
            {
                "metrics": metrics,
                "validation_ids": ids,
                "checkpoint_sha256": sha(path),
                "run_sha256": sha(run_path),
                "threshold_rule": "higher q95; strict >",
            },
        )
        np.savez_compressed(output / "calibration-patch-scores.npz", **patches)
        write(
            output / "complete.json",
            {
                "completed": True,
                "checkpoint": path.name,
                "checkpoint_sha256": sha(path),
                "calibration_sha256": sha(output / "calibration.json"),
                "patches_sha256": sha(output / "calibration-patch-scores.npz"),
                "run_sha256": sha(run_path),
            },
        )
    print(
        json.dumps(
            {
                "step": trainer.step,
                "required_updates": trainer.end,
                "full_training_complete": trainer.step == trainer.end and not args.mechanics_only,
                "mechanics_only": args.mechanics_only,
            }
        ),
        flush=True,
    )


def score(args, root):
    binding = read(args.run / "run.json")
    if binding["mechanics_only"] or binding["source_archive_sha256"] != SOURCE_SHA256:
        raise ValueError("a completed full-budget run is required for target scoring")
    complete = read(args.run / "complete.json")
    if not complete["completed"]:
        raise ValueError("a completed full-budget run is required for target scoring")
    for name, key in [
        (complete["checkpoint"], "checkpoint_sha256"),
        ("calibration.json", "calibration_sha256"),
        ("calibration-patch-scores.npz", "patches_sha256"),
        ("run.json", "run_sha256"),
    ]:
        if sha(args.run / name) != complete[key]:
            raise ValueError("completed run artifact changed")
    args.category, args.seed, args.mechanics_only = binding["category"], binding["seed"], False
    trainer, validation, torch, np = new_trainer(args, root)
    from ad2_runtime import calibrate

    from lc_simplenet.evaluation import evaluate_run
    from lc_simplenet.public_scoring import condition_from_name, list_public_test, read_resized

    state = torch.load(args.run / complete["checkpoint"], map_location="cpu", weights_only=True)
    trainer.restore(state, FROZEN_PROTOCOL)
    if trainer.step != trainer.end or trainer.end != binding["required_updates"]:
        raise ValueError("nonfinal checkpoint cannot be scored")
    metrics, calibration, ids = calibrate(trainer, validation)
    saved = read(args.run / "calibration.json")
    with np.load(args.run / "calibration-patch-scores.npz", allow_pickle=False) as expected:
        if (
            metrics != saved["metrics"]
            or ids != saved["validation_ids"]
            or any(not np.array_equal(calibration[a], expected[a]) for a in ARMS)
        ):
            raise ValueError("cold calibration replay differs; do not refit thresholds")
    args.out.mkdir(parents=True, exist_ok=False)
    rows, patches = [], {a: [] for a in ARMS}
    normalize = trainer.baselines["B1"].normalize
    with torch.no_grad():
        for image in list_public_test(args.data_root, args.category):
            pixels, _ = read_resized(image.path, (256, 256))
            trainer.encoder.clear()
            f, _ = trainer.encoder.embed(
                normalize(
                    torch.from_numpy(pixels.copy())[None]
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .to(trainer.device)
                )
            )
            for arm, (adapter, discriminator) in trainer.models.items():
                value = (-discriminator(adapter(f))).reshape(1024).cpu().numpy()
                if not np.isfinite(value).all():
                    raise FloatingPointError("nonfinite target score")
                patches[arm].append(value)
            rows.append(
                {
                    "image_id": image.image_id,
                    "label": image.label,
                    "condition": condition_from_name(image.image_id),
                }
            )
    arrays = {a: np.stack(v) for a, v in patches.items()}
    np.savez_compressed(args.out / "patch-scores.npz", **arrays)
    results = {}
    for arm in ARMS:
        cal_rows = [
            {"image_id": image_id, "label": 0, "score": float(v.max())}
            for image_id, v in zip(ids, calibration[arm], strict=True)
        ]
        target_rows = [
            {**r, "score": float(v.max())} for r, v in zip(rows, arrays[arm], strict=True)
        ]
        value = evaluate_run(
            cal_rows,
            target_rows,
            0.05,
            conditions={r["image_id"]: r["condition"] for r in rows},
            lighting_aggregates=True,
        )
        if value["calibration"]["threshold"] != metrics[arm]["threshold"]:
            raise ValueError("frozen normal threshold recount differs")
        results[arm] = value
    write(
        args.out / "predictions.json",
        {
            "rows": rows,
            "scores": {a: arrays[a].max(axis=1).tolist() for a in ARMS},
            "thresholds": {a: metrics[a]["threshold"] for a in ARMS},
        },
    )
    write(
        args.out / "metrics.json",
        {
            "category": args.category,
            "seed": args.seed,
            "methods": results,
            "cold_calibration_replay_bit_exact": True,
            "thresholds_refitted": False,
            "scope": (
                "all public images; no method/epoch/seed selection; original all8 qualification "
                "requires both baseline anchors and an independent full-matrix audit"
            ),
        },
    )
    print(
        json.dumps({"scored_images": len(rows), "methods": ARMS, "thresholds_refitted": False}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    prep = commands.add_parser("prepare")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--textures-root", type=Path, required=True)
    prep.add_argument("--category", choices=CATEGORIES, required=True)
    prep.add_argument("--out", type=Path, required=True)
    training = commands.add_parser("train")
    training.add_argument("--category", choices=CATEGORIES, required=True)
    training.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    training.add_argument("--stop-after", type=int)
    training.add_argument("--mechanics-only", action="store_true")
    scoring = commands.add_parser("score")
    scoring.add_argument("--run", type=Path, required=True)
    scoring.add_argument("--data-root", type=Path, required=True)
    scoring.add_argument("--out", type=Path, required=True)
    for command in [training, scoring]:
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with execution_source(args.source_archive) as root:
        if args.command == "verify":
            print(json.dumps({"source_sha256": SOURCE_SHA256, "source_inventory_verified": True}))
        else:
            {"prepare": prepare, "train": train, "score": score}[args.command](args, root)


if __name__ == "__main__":
    main()
