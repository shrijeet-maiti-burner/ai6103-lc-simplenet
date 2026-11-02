"""Portable RNG snapshots and runtime records for normal-only training."""

import hashlib
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}:".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def capture_rng_state(*, cuda: bool = False) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python_rng": random.getstate(),
        "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if cuda else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python_rng"])
    numpy_state = state["numpy_rng"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])


def runtime_record(device: torch.device) -> dict:
    packages = {}
    for name in (
        "numpy",
        "pillow",
        "torch",
        "torchvision",
        "timm",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "opencv-python",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
