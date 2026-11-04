"""Train the upstream single-view SimpleNet baseline on one MVTec AD or AD 2 category.

No test images are read. The new run directory receives the final checkpoint,
raw normal-validation scores/maps with their crop geometry and a summary with hashes,
runtime, peak VRAM and reload differences. AD 2 sources are resized full-frame to
--resize (aspect ratio changes) and not cropped; MVTec AD keeps upstream resize and crop.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from lc_simplenet.baseline import (
    ROOT,
    UPSTREAM_AD,
    UPSTREAM_AD_INPUT,
    UpstreamInput,
    build_model,
    input_record,
    load_trained_state,
    predict,
    train_fixed,
    training_checkpoint,
    upstream_module,
)
from lc_simplenet.data import NormalManifestDataset
from lc_simplenet.provenance import capture_rng_state, file_sha256, runtime_record, state_sha256
from lc_simplenet.source_identity import source_record


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mvtec_ad", "mvtec_ad2"], default="mvtec_ad")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--meta-epochs", type=int, default=UPSTREAM_AD["meta_epochs"])
    parser.add_argument("--resize", type=int, default=UPSTREAM_AD_INPUT["resize"])
    parser.add_argument("--imagesize", type=int, default=UPSTREAM_AD_INPUT["imagesize"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    stretch = args.dataset == "mvtec_ad2"
    if stretch and args.resize != args.imagesize:
        parser.error("AD 2 is resized full-frame without a crop: set --resize equal to --imagesize")
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    upstream_module("utils").fix_seeds(args.seed, with_cuda=cuda)
    initial_rng = capture_rng_state(cuda=cuda)
    settings = {**UPSTREAM_AD, "meta_epochs": args.meta_epochs}
    resize, imagesize = args.resize, args.imagesize
    record = input_record(args.dataset, resize, imagesize)
    geometry = record["geometry"]

    def loader(split: str, shuffle: bool):
        source = NormalManifestDataset(
            args.manifest,
            args.root,
            dataset=args.dataset,
            category=args.category,
            split=split,
            expected_sha256=args.manifest_sha256,
            output_size=(resize, resize),
        )
        return torch.utils.data.DataLoader(
            UpstreamInput(source, imagesize, stretch=stretch),
            batch_size=UPSTREAM_AD_INPUT["batch_size"],
            shuffle=shuffle,
            num_workers=args.num_workers,
        )

    def new_model():
        backbone = upstream_module("backbones").load("wideresnet50")
        return build_model(backbone, device, imagesize, settings)

    train, validation = loader("train", True), loader("validation", False)
    model = new_model()
    encoder_hash = state_sha256(model.backbone.state_dict())
    weight_path = Path(torch.hub.get_dir()) / "checkpoints/wide_resnet50_2-95faca4d.pth"
    sources = [
        Path(__file__),
        ROOT / "src/lc_simplenet/baseline.py",
        ROOT / "src/lc_simplenet/provenance.py",
        ROOT / "src/lc_simplenet/data.py",
        ROOT / "src/lc_simplenet/data_manifest.py",
        ROOT / "third_party/SimpleNet/simplenet.py",
        ROOT / "third_party/SimpleNet/common.py",
    ]
    run = {
        "schema": 2,
        "seed": args.seed,
        "category": args.category,
        "settings": settings,
        **record,
        "checkpoint_policy": "fixed_final",
        "manifest_sha256": args.manifest_sha256,
        "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
        **source_record(ROOT),
        "runtime": runtime_record(device),
        "backbone_after_probe_sha256": encoder_hash,
        "pretrained_weight_sha256": file_sha256(weight_path),
        "source_sha256": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sources},
    }
    torch.save(initial_rng, args.out / "initial-rng.pt")
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    train_fixed(model, train, args.out / "logs")
    if cuda:
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None
    checkpoint = args.out / "checkpoint.pt"
    state = training_checkpoint(model, settings, args.seed)
    state["provenance"] = run
    torch.save(state, checkpoint)
    if state_sha256(model.backbone.state_dict()) != encoder_hash:
        raise RuntimeError("Frozen baseline backbone changed during training")
    scores = predict(model, validation)
    arrays = {name: np.asarray(value) for name, value in geometry.items()}
    np.savez_compressed(args.out / "validation_normal.npz", **scores, **arrays)

    reloaded = new_model()
    load_trained_state(
        reloaded, torch.load(checkpoint, map_location="cpu", weights_only=True)["state"]
    )
    again = predict(reloaded, validation)
    summary = {
        "scope": "fixed-final upstream single-view baseline; normal validation only, no test data",
        "category": args.category,
        "seed": args.seed,
        "settings": settings,
        **record,
        "train_images": len(train.dataset),
        "validation_images": len(validation.dataset),
        "manifest_sha256": args.manifest_sha256,
        "checkpoint_sha256": _sha256(checkpoint),
        "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
        **source_record(ROOT),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if cuda else "cpu",
        "train_seconds": round(train_seconds, 1),
        "train_peak_vram_gib": None if peak is None else round(peak, 2),
        "reload_max_abs_score_diff": float(
            np.abs(again["image_score"] - scores["image_score"]).max()
        ),
        "reload_max_abs_map_diff": float(np.abs(again["map"] - scores["map"]).max()),
        "checkpoint_schema": 2,
        "run_metadata_sha256": _sha256(args.out / "run.json"),
        "initial_rng_sha256": _sha256(args.out / "initial-rng.pt"),
        "pretrained_weight_sha256": run["pretrained_weight_sha256"],
        "backbone_after_probe_sha256": encoder_hash,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
