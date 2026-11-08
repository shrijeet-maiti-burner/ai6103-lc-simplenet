"""Measure a single FP32 graph forward with ImageNet E and random anomaly heads."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc_simplenet.baseline import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from lc_simplenet.pilot import PilotConfig, UpstreamBackend, seed_runtime  # noqa: E402
from lc_simplenet.provenance import file_sha256  # noqa: E402
from lc_simplenet.public_scoring import read_resized  # noqa: E402
from lc_simplenet.reference_inference import predict_reference_batch  # noqa: E402
from lc_simplenet.source_identity import source_record  # noqa: E402


def run(args):
    protocol_path = args.out.with_suffix(".protocol.json")
    if args.out.exists() or protocol_path.exists():
        raise FileExistsError("preserve existing memory-smoke records")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image_sha = file_sha256(args.image)
    scope = "single graph/shape/memory check; ImageNet E, random A/D; no trained-detector accuracy"
    protocol = {
        "scope": scope,
        "image_sha256": image_sha,
        "input_size": args.size,
        "batch": 1,
        "precision": "FP32; TF32 off; deterministic algorithms",
        "source": source_record(ROOT),
        "helper_sha256": file_sha256(ROOT / "src/lc_simplenet/reference_inference.py"),
        "timing": "one operational forward including transfers; construction/decoding excluded",
        "original_auxiliary_bytes": 1536 * args.size**2 * 4,
        "replacement_auxiliary_bytes": args.size**2 * 4,
    }
    with protocol_path.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)
    torch.set_num_threads(2)
    seed_runtime(0)
    # These names configure the graph geometry; no dataset loader is constructed.
    config = PilotConfig("mvtec_ad2", "can", image_sha, image_size=(args.size, args.size))
    backend = UpstreamBackend(config, torch.device("cuda"))
    pixels, _ = read_resized(args.image, (args.size, args.size))
    tensor = TF.normalize(torch.from_numpy(pixels).permute(2, 0, 1), IMAGENET_MEAN, IMAGENET_STD)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    scores, maps = predict_reference_batch(backend.model, tensor[None])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if scores.shape != (1,) or maps.shape != (1, args.size, args.size):
        raise ValueError("graph output shape differs")
    if not np.isfinite(scores).all() or not np.isfinite(maps).all():
        raise FloatingPointError("nonfinite graph output")
    result = {
        "scope": scope,
        "passed": True,
        "map_shape": list(maps.shape),
        "single_forward_seconds": elapsed,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "protocol_sha256": file_sha256(protocol_path),
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(256, 512, 1024), default=1024)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(run(parser.parse_args())))
