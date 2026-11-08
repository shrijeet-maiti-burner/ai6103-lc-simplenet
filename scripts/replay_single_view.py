"""Capture original-probe buffers or replay a complete saved normal population."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc_simplenet.backbone_state import capture_backbone_state, restore_backbone_state  # noqa: E402
from lc_simplenet.baseline import (  # noqa: E402
    UpstreamInput,
    build_model,
    load_trained_state,
    upstream_module,
)
from lc_simplenet.data import NormalManifestDataset  # noqa: E402
from lc_simplenet.provenance import file_sha256, runtime_record  # noqa: E402
from lc_simplenet.reference_inference import predict_reference_batch  # noqa: E402
from lc_simplenet.source_identity import source_record  # noqa: E402


def original_model(run, device):
    summary = json.loads((run / "summary.json").read_text())
    path = run / "checkpoint.pt"
    if file_sha256(path) != summary["checkpoint_sha256"]:
        raise ValueError("reference checkpoint differs from its summary")
    upstream_module("utils").fix_seeds(summary["seed"], with_cuda=device.type == "cuda")
    backbone = upstream_module("backbones").load("wideresnet50")
    model = build_model(backbone, device, summary["input"]["imagesize"], summary["settings"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["settings"] != summary["settings"]:
        raise ValueError("checkpoint settings differ")
    load_trained_state(model, state["state"])
    weights = Path(torch.hub.get_dir()) / "checkpoints/wide_resnet50_2-95faca4d.pth"
    return model, summary, file_sha256(weights)


def replay(args):
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, summary, weights_sha = original_model(args.run, device)
    shape = (3, summary["input"]["imagesize"], summary["input"]["imagesize"])
    if args.command == "capture":
        actual = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        if actual != summary["device"]:
            raise ValueError(
                "capture on the original device; matching parameters alone is insufficient"
            )
        snapshot = capture_backbone_state(
            model.backbone, image_shape=shape, pretrained_sha256=weights_sha
        )
        snapshot["provenance"] = {
            "scope": "fresh original-runtime reconstruction; requires saved-normal replay",
            "source": source_record(ROOT),
            "runtime": runtime_record(device),
            "checkpoint_sha256": summary["checkpoint_sha256"],
        }
        with (args.out / "backbone-state.pt").open("xb") as stream:
            torch.save(snapshot, stream)
        return {
            "captured": True,
            "normal_replay_required": True,
            "backbone_sha256": snapshot["backbone_sha256"],
        }
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    restore_backbone_state(
        model.backbone, snapshot, image_shape=shape, pretrained_sha256=weights_sha
    )
    source = NormalManifestDataset(
        args.manifest,
        args.data,
        dataset=summary["dataset"],
        category=summary["category"],
        split="validation",
        expected_sha256=args.manifest_sha256,
        output_size=(summary["input"]["resize"], summary["input"]["resize"]),
    )
    source = UpstreamInput(source, shape[-1], stretch=summary["dataset"] == "mvtec_ad2")
    saved_path = args.run / "validation_normal.npz"
    with np.load(saved_path, allow_pickle=False) as saved:
        identifiers = [r.image_id for r in source.source.records]
        if identifiers != saved["image_id"].tolist():
            raise ValueError("saved normal population/order differs; no test scoring permitted")
        loader = torch.utils.data.DataLoader(
            source, batch_size=summary["input"]["batch_size"], shuffle=False, num_workers=0
        )
        scores, maps = [], []
        for batch in loader:
            score, mask = predict_reference_batch(model, batch["image"])
            scores.extend(score)
            maps.extend(mask)
        scores, maps = np.asarray(scores), np.stack(maps)
        score_pass = bool(np.allclose(scores, saved["image_score"], atol=1e-4, rtol=1e-4))
        map_pass = bool(np.allclose(maps, saved["map"], atol=1e-4, rtol=1e-4))
        threshold = float(np.quantile(saved["image_score"], 0.95, method="higher"))
        record = {
            "scope": "complete normal-only replay; no test access or predictive improvement",
            "category": summary["category"],
            "seed": summary["seed"],
            "images": len(scores),
            "score_max_error": float(np.abs(scores - saved["image_score"]).max()),
            "map_max_error": float(np.abs(maps - saved["map"]).max()),
            "score_passed": score_pass,
            "map_passed": map_pass,
            "passed": score_pass and map_pass,
            "guard": "np.allclose score and complete map; atol1e-4 rtol1e-4, unchanged",
            "saved_q95_threshold": threshold,
            "normal_flag_differences": int(
                np.count_nonzero((scores > threshold) != (saved["image_score"] > threshold))
            ),
            "snapshot_sha256": file_sha256(args.snapshot),
            "manifest_sha256": args.manifest_sha256,
            "saved_normal_sha256": file_sha256(saved_path),
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "source": source_record(ROOT),
            "runtime": runtime_record(device),
            "memory_adaptation": (
                "one auxiliary feature channel; original score/map branch unchanged"
            ),
        }
    np.savez_compressed(
        args.out / "replayed-normal.npz",
        image_id=np.array(identifiers),
        image_score=scores,
        map=maps,
    )
    (args.out / "replay.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "replay"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.command == "replay" and any(
        value is None for value in (args.snapshot, args.manifest, args.manifest_sha256, args.data)
    ):
        parser.error("replay requires snapshot, manifest, manifest-sha256 and data")
    record = replay(args)
    print(json.dumps(record))
    if not record.get("passed", record.get("captured", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
