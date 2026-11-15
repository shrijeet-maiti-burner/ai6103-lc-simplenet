"""Score a released original reference only after its complete normal replay passes."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc_simplenet.backbone_state import restore_backbone_state  # noqa: E402
from lc_simplenet.baseline import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from lc_simplenet.provenance import file_sha256  # noqa: E402
from lc_simplenet.public_scoring import (  # noqa: E402
    condition_from_name,
    list_public_test,
    read_resized,
)
from lc_simplenet.reference_inference import predict_reference_batch  # noqa: E402
from lc_simplenet.source_identity import source_record  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "original_replay_cli", ROOT / "scripts/replay_single_view.py"
)
normal_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normal_cli)


def cpu_image(path, resize, imagesize):
    pixels, original = read_resized(path, (resize, resize))
    tensor = TF.center_crop(torch.from_numpy(pixels).permute(2, 0, 1), [imagesize, imagesize])
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD), original


def score(args):
    args.out.mkdir(parents=True, exist_ok=False)
    guard_args = SimpleNamespace(**vars(args))
    guard_args.command = "replay"
    guard_args.out = args.out / "normal-replay"
    guard = normal_cli.replay(guard_args)
    if (
        not guard["passed"]
        or guard["score_max_error"] != 0
        or guard["map_max_error"] != 0
        or guard["normal_flag_differences"] != 0
    ):
        raise ValueError(
            "complete saved-normal replay is not bit-exact; public input remains closed"
        )
    bindings = {
        "snapshot_sha256": args.snapshot,
        "checkpoint_sha256": args.run / "checkpoint.pt",
        "saved_normal_sha256": args.run / "validation_normal.npz",
    }
    if any(file_sha256(path) != guard[key] for key, path in bindings.items()):
        raise ValueError(
            "reference inputs changed after normal replay; public input remains closed"
        )
    model, summary, weight_sha = normal_cli.original_model(args.run, torch.device(args.device))
    resize, imagesize = summary["input"]["resize"], summary["input"]["imagesize"]
    if summary["dataset"] != "mvtec_ad2" or resize != imagesize:
        raise ValueError("public scoring requires the full-frame AD 2 original reference")
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    restored = restore_backbone_state(
        model.backbone,
        snapshot,
        image_shape=(3, imagesize, imagesize),
        pretrained_sha256=weight_sha,
    )
    images = list_public_test(args.public_root, summary["category"])
    conditions = {i.image_id: condition_from_name(i.image_id) for i in images}
    rows, maps, inputs = [], [], []
    batch_size = summary["input"]["batch_size"]
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        tensors, sizes = zip(*(cpu_image(i.path, resize, imagesize) for i in batch), strict=True)
        scores, masks = predict_reference_batch(model, torch.stack(tensors))
        for image, value, mask, original in zip(batch, scores, masks, sizes, strict=True):
            rows.append(
                {
                    "image_id": image.image_id,
                    "label": image.label,
                    "score": float(value),
                    "original_size": original,
                    "resized_size": [resize, resize],
                    "image_size": [imagesize, imagesize],
                    "crop_box_xyxy": None,
                    "map_coordinates": "full_frame",
                }
            )
            maps.append(mask)
            inputs.append({"image_id": image.image_id, "sha256": file_sha256(image.path)})
    shutil.copyfile(args.run / "checkpoint.pt", args.out / "checkpoint.pt")
    if file_sha256(args.out / "checkpoint.pt") != guard["checkpoint_sha256"]:
        raise ValueError("copied checkpoint differs from the normal-replay binding")
    with np.load(args.run / "validation_normal.npz", allow_pickle=False) as normal:
        validation = [
            {"image_id": i, "label": 0, "score": float(s)}
            for i, s in zip(normal["image_id"].tolist(), normal["image_score"], strict=True)
        ]
    identity = source_record(ROOT)
    meta = {
        "scope": "frozen original reference on exposed annotated public data; no method selection",
        "category": summary["category"],
        "variant": "original-single-view",
        "seed": summary["seed"],
        "images": len(rows),
        "normal": sum(row["label"] == 0 for row in rows),
        "defective": sum(row["label"] == 1 for row in rows),
        "image_ids_sha256": hashlib.sha256(
            "\n".join(row["image_id"] for row in rows).encode()
        ).hexdigest(),
        "checkpoint_file": "checkpoint.pt",
        "checkpoint_sha256": file_sha256(args.out / "checkpoint.pt"),
        "run_config_sha256": file_sha256(args.run / "summary.json"),
        "snapshot_sha256": file_sha256(args.snapshot),
        "normal_replay": guard,
        "restored_backbone": restored,
        "source": identity,
        "scoring_git_commit": identity["git_commit"],
        "calibration": "saved original normal q95; higher quantile, strict score > threshold",
        "memory_adaptation": (
            "omit independent dense feature visualization; score/map branch unchanged"
        ),
    }
    outputs = {
        "test-scores.json": rows,
        "validation-scores.json": validation,
        "test-scores.meta.json": meta,
        "test-inputs.json": inputs,
        "test-conditions.json": conditions,
    }
    for name, value in outputs.items():
        with (args.out / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    np.savez_compressed(
        args.out / "test-maps.npz", maps=np.stack(maps), image_ids=[r["image_id"] for r in rows]
    )
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "snapshot", "manifest", "data", "public-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    print(json.dumps(score(parser.parse_args())))
