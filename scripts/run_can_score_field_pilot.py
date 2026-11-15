#!/usr/bin/env python3
"""Validation-only can pilot: six regular-light checks, then seven relights.

Scores ``can-B0`` and ``can-B1``, seeds 0–2, from released ``heads.pt`` files.
Paths come from the command line. Released heads and score files are read-only.
The selector is not run, and no test split is opened.

The regular-light gate is all-or-nothing. Every run is scored under the original
light. The seven frozen relights are scored only after all six reproductions
are within 1e-3. A failure writes the check record and no map file.

Five safeguards, matching the approved TC2 pilot:

1. Locked stack: CPython 3.12.15, uv 0.13.0, torch 2.6.0+cu124,
   torchvision 0.21.0+cu124, CUDA 12.4, cuDNN 90100, one NVIDIA A40, and
   ``CUBLAS_WORKSPACE_CONFIG=:4096:8``. The allocated GPU is checked before a
   head is loaded.
2. NumPy 1.26.4 and Pillow 11.3.0.
3. ``wide_resnet50_2-95faca4d.pth`` matches the study SHA-256 before the
   backbone is constructed. A missing or different file is not downloaded.
4. The checkpoint encoder hash and the loaded encoder both equal the study hash.
5. Each ``heads.pt`` matches its archive SHA-256 and its formal 256 B0/B1 config.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import torch

from lc_simplenet.score_field import RELIGHT_NAMES, refuse_test_path, relight_validation_image
from lc_simplenet.validation_maps import (
    REGULAR_ATOL,
    _one_map,
    assert_formal_study_head,
    check_regular_reproduction,
    check_validation_rows,
    save_validation_maps,
)

CAN_RUNS = (
    ("B0", 0),
    ("B0", 1),
    ("B0", 2),
    ("B1", 0),
    ("B1", 1),
    ("B1", 2),
)
CAN_RUN_NAMES = tuple(f"can-{variant}-seed{seed}" for variant, seed in CAN_RUNS)
EXPECTED_IMAGES = 46
WEIGHT_FILENAME = "wide_resnet50_2-95faca4d.pth"
WEIGHT_SHA256 = "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"
ENCODER_SHA256 = "aa373db6792ab7d661690bd10e1897c76de2abfb7039145c871709ac02ab3509"
HEAD_SHA256 = {
    "can-B0-seed0": "1bb5721d3bdcc7454d55f37b164145911ee71b1f8612332e94e3622de94b6993",
    "can-B0-seed1": "8ae9ae438c82a2ebf4af064ee05452592a4fbb55048e5a7bce79c41827d6cc3b",
    "can-B0-seed2": "27cdbde73db92930e2c7618d6ef2b07bc89594d171a7fad57477eae4a6bdeb71",
    "can-B1-seed0": "69c7e9cb79a6535b55dd801d159d81614829d0a5034fc0964e894aba401aa196",
    "can-B1-seed1": "adb07e345119724ed08a8d367613c97597edaa19b3405e60e4bc2ef81fc2382b",
    "can-B1-seed2": "b1642125cb3fbb3d08f8ae550e0d2cc3a564c96d74f1a792a64a609414e5cd27",
}
LOCKED_PYTHON = "3.12.15"
LOCKED_UV = "0.13.0"
LOCKED_TORCH = "2.6.0+cu124"
LOCKED_TORCHVISION = "0.21.0+cu124"
LOCKED_CUDA = "12.4"
LOCKED_CUDNN = 90100
LOCKED_GPU = "NVIDIA A40"
LOCKED_NUMPY = "1.26.4"
LOCKED_PILLOW = "11.3.0"
LOCKED_CUBLAS = ":4096:8"
STUDY_THREADS = 3


class PilotFailure(Exception):
    """A safeguard or regular-light check stopped the pilot before relighting."""


@dataclass
class PreparedRun:
    name: str
    variant: str
    seed: int
    rows: list[dict]
    images: list[np.ndarray]
    predict: object


def _generator():
    path = Path(__file__).with_name("generate_validation_score_maps.py")
    spec = importlib.util.spec_from_file_location(
        "generate_validation_score_maps_for_can_pilot", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _generator()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    """Return a JSON-safe copy, with filesystem paths rendered as strings."""
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, dict):
        return {
            os.fspath(key) if isinstance(key, Path) else key: json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_can_run(name: str) -> tuple[str, int]:
    for variant, seed in CAN_RUNS:
        if name == f"can-{variant}-seed{seed}":
            return variant, seed
    raise PilotFailure(f"Unexpected run {name}; the can pilot accepts only the six can runs")


def locked_runtime_record() -> dict:
    """The runtime the formal can runs recorded. Used to test the checker."""
    return {
        "python_implementation": "CPython",
        "python": LOCKED_PYTHON,
        "uv": LOCKED_UV,
        "torch": LOCKED_TORCH,
        "torchvision": LOCKED_TORCHVISION,
        "cuda_runtime": LOCKED_CUDA,
        "cudnn": LOCKED_CUDNN,
        "cuda_available": True,
        "gpu_count": 1,
        "gpu": LOCKED_GPU,
        "numpy": LOCKED_NUMPY,
        "pillow": LOCKED_PILLOW,
        "cublas_workspace_config": LOCKED_CUBLAS,
        "threads": STUDY_THREADS,
    }


def parse_uv_version(text: str) -> str:
    parts = text.strip().split()
    if len(parts) < 2 or parts[0] != "uv":
        raise PilotFailure(f"Unreadable uv version: {text.strip()}")
    return parts[1]


def read_uv_version() -> str:
    try:
        text = subprocess.check_output(["uv", "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotFailure("uv 0.13.0 is required and was not available") from exc
    return parse_uv_version(text)


def collect_runtime(threads: int) -> dict:
    import torchvision

    cuda_available = bool(torch.cuda.is_available())
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "uv": read_uv_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "gpu_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else "",
        "numpy": np.__version__,
        "pillow": package_version("pillow"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "threads": threads,
    }


def assert_locked_runtime(info: dict) -> None:
    """Safeguards 1 and 2: the recorded TC2 stack, including NumPy and Pillow."""
    problems = []
    if info.get("python_implementation") != "CPython" or info.get("python") != LOCKED_PYTHON:
        problems.append(f"python={info.get('python_implementation')} {info.get('python')}")
    if info.get("uv") != LOCKED_UV:
        problems.append(f"uv={info.get('uv')}")
    if info.get("torch") != LOCKED_TORCH:
        problems.append(f"torch={info.get('torch')}")
    if info.get("torchvision") != LOCKED_TORCHVISION:
        problems.append(f"torchvision={info.get('torchvision')}")
    if info.get("cuda_runtime") != LOCKED_CUDA:
        problems.append(f"cuda={info.get('cuda_runtime')}")
    if info.get("cudnn") != LOCKED_CUDNN:
        problems.append(f"cudnn={info.get('cudnn')}")
    if not info.get("cuda_available") or int(info.get("gpu_count") or 0) < 1:
        problems.append("cuda_unavailable")
    if info.get("gpu") != LOCKED_GPU:
        problems.append(f"gpu={info.get('gpu')}")
    if info.get("numpy") != LOCKED_NUMPY:
        problems.append(f"numpy={info.get('numpy')}")
    if info.get("pillow") != LOCKED_PILLOW:
        problems.append(f"pillow={info.get('pillow')}")
    if info.get("cublas_workspace_config") != LOCKED_CUBLAS:
        problems.append(f"cublas={info.get('cublas_workspace_config')}")
    if info.get("threads") != STUDY_THREADS:
        problems.append(f"threads={info.get('threads')}")
    if problems:
        raise PilotFailure("Locked TC2 runtime mismatch: " + "; ".join(problems))


def verify_provenance(source_root: Path, manifest_path: Path, sidecar_path: Path) -> dict:
    """Check the manifest sidecar, then every listed source file, before scoring."""
    payload = manifest_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    recorded = sidecar_path.read_text(encoding="utf-8").strip().split()
    if not recorded or recorded[0] != digest:
        raise PilotFailure("Provenance manifest hash does not match the sidecar")
    if len(recorded) > 1 and Path(recorded[1]).name != manifest_path.name:
        raise PilotFailure("Provenance sidecar names a different manifest")
    manifest = json.loads(payload)
    head = manifest.get("git_head")
    if not isinstance(head, str) or len(head) != 40:
        raise PilotFailure("Provenance manifest has no git HEAD")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise PilotFailure("Provenance manifest has no file hashes")
    root = source_root.resolve()
    for relative, expected in files.items():
        if not isinstance(relative, str) or relative.startswith("/") or "\\" in relative:
            raise PilotFailure(f"Provenance path is not a relative source path: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise PilotFailure(f"Provenance path escapes the source root: {relative}")
        if not path.is_file() or sha256_file(path) != expected:
            raise PilotFailure(f"Provenance hash mismatch: {relative}")
    return {"sha256": digest, "git_head": head, "manifest": manifest}


def assert_weight_file(path: Path) -> str:
    """Safeguard 3. Hash the local backbone file and do not fetch a replacement."""
    if path.name != WEIGHT_FILENAME:
        raise PilotFailure(f"Backbone file must be named {WEIGHT_FILENAME}")
    if not path.is_file():
        raise PilotFailure(f"Backbone file is missing: {path}")
    digest = sha256_file(path)
    if digest != WEIGHT_SHA256:
        raise PilotFailure("Backbone checksum does not match the formal can study")
    return digest


def stage_verified_weights(weight_path: Path, hub_root: Path) -> Path:
    """Put the already verified file where torchvision will load it locally."""
    digest = assert_weight_file(weight_path)
    source = weight_path.resolve()
    destination_dir = hub_root / "checkpoints"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / WEIGHT_FILENAME
    if destination.resolve() != source:
        if destination.exists() and sha256_file(destination) != digest:
            raise PilotFailure("Torch hub already holds a different backbone file")
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        if sha256_file(destination) != digest:
            raise PilotFailure("Staged backbone checksum changed")
    torch.hub.set_dir(str(hub_root))
    return destination


def assert_encoder_hash(encoder_hash: str) -> None:
    """Safeguard 4 for the hash stored in the checkpoint and the loaded encoder."""
    if encoder_hash != ENCODER_SHA256:
        raise PilotFailure("Frozen encoder hash differs from the formal can study")


def assert_head_file(path: Path, name: str) -> str:
    """Safeguard 5, file bytes. The formal archive checksum, read-only."""
    parse_can_run(name)
    if not path.is_file():
        raise PilotFailure(f"{name} is missing heads.pt")
    digest = sha256_file(path)
    if digest != HEAD_SHA256[name]:
        raise PilotFailure(f"{name} heads.pt checksum does not match the formal archive")
    return digest


def assert_released_head_config(state_config: dict, run_config: dict, name: str) -> None:
    """Safeguard 5, configuration. Formal full-frame 256 B0/B1 can head only."""
    variant, seed = parse_can_run(name)
    try:
        released = GENERATOR.config_from(state_config)
        recorded = GENERATOR.config_from(run_config)
        assert_formal_study_head(released)
    except (TypeError, ValueError) as exc:
        raise PilotFailure(f"{name} configuration is not a formal can head: {exc}") from exc
    if released != recorded:
        raise PilotFailure(f"{name} heads.pt configuration differs from run.json")
    if released.category != "can" or released.variant != variant or released.seed != seed:
        raise PilotFailure(f"{name} does not match its directory name")
    if released.weights != "imagenet_v1":
        raise PilotFailure(f"{name} does not request ImageNet V1 weights")


def assert_can_score_rows(grouped: dict[str, list[dict]]) -> None:
    """Forty-six shared ``can/validation/good`` ids. No test id is accepted."""
    if set(grouped) != set(CAN_RUN_NAMES):
        raise PilotFailure("Validation scores must cover exactly the six can runs")
    identities = []
    for name in CAN_RUN_NAMES:
        try:
            image_ids = check_validation_rows("can", grouped[name])
        except (TypeError, ValueError, KeyError) as exc:
            raise PilotFailure(f"{name} validation scores are not usable: {exc}") from exc
        if len(image_ids) != EXPECTED_IMAGES:
            raise PilotFailure(
                f"{name} has {len(image_ids)} validation images; expected {EXPECTED_IMAGES}"
            )
        identities.append(tuple(image_ids))
    if len(set(identities)) != 1:
        raise PilotFailure("The six can runs do not share the same validation image ids")


def assert_heads_layout(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise PilotFailure("Heads root is not a directory")
    refuse_test_path(root.as_posix())
    expected = set(CAN_RUN_NAMES)
    children = list(root.iterdir())
    directories = {path.name for path in children if path.is_dir()}
    unexpected = sorted(path.name for path in children if path.name not in expected)
    missing = sorted(expected - directories)
    if missing or unexpected or directories != expected:
        raise PilotFailure(
            "Heads root must contain exactly the six can runs; "
            f"missing={missing} unexpected={unexpected}"
        )
    return {name: root / name for name in CAN_RUN_NAMES}


def assert_fresh_output(out_dir: Path, heads_root: Path) -> None:
    refuse_test_path(out_dir.as_posix())
    out_resolved = out_dir.resolve()
    heads_resolved = heads_root.resolve()
    overlaps = (
        out_resolved == heads_resolved
        or out_resolved.is_relative_to(heads_resolved)
        or heads_resolved.is_relative_to(out_resolved)
    )
    if overlaps:
        raise PilotFailure("Output directory overlaps the released heads")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise PilotFailure("Output directory is not empty")


def max_abs_difference(saved_scores, maps: np.ndarray) -> float | None:
    got = np.asarray(maps, dtype=np.float64).reshape(len(maps), -1).max(axis=1)
    saved = np.asarray(saved_scores, dtype=np.float64)
    if saved.shape != got.shape or not np.isfinite(saved).all() or not np.isfinite(got).all():
        return None
    return float(np.max(np.abs(got - saved)))


def execute(runs: list[PreparedRun], out_dir: Path, report: dict) -> None:
    """Score six regular views, then relight only if every reproduction passes."""
    if len(runs) != 6 or {run.name for run in runs} != set(CAN_RUN_NAMES):
        raise PilotFailure("Can pilot requires all six runs before any score is computed")
    by_name = {run.name: run for run in runs}
    if len(by_name) != 6:
        raise PilotFailure("Duplicate can run")
    for name in CAN_RUN_NAMES:
        run = by_name[name]
        variant, seed = parse_can_run(name)
        if run.variant != variant or type(run.seed) is not int or run.seed != seed:
            raise PilotFailure(f"{name} identity does not match the prepared run")
        if len(run.images) != len(run.rows):
            raise PilotFailure(f"{name} images and validation rows differ in length")
        try:
            check_validation_rows("can", run.rows)
        except (TypeError, ValueError, KeyError) as exc:
            raise PilotFailure(f"{name} validation rows were refused: {exc}") from exc

    started = time.perf_counter()
    regular_maps = {}
    checks = []
    for name in CAN_RUN_NAMES:
        run = by_name[name]
        maps = np.stack([_one_map(image, run.predict) for image in run.images])
        saved = [row["score"] for row in run.rows]
        difference = max_abs_difference(saved, maps)
        try:
            check_regular_reproduction(saved, maps)
            passed = True
        except ValueError:
            passed = False
        checks.append(
            {
                "name": name,
                "images": len(run.rows),
                "max_abs_diff": difference,
                "atol": REGULAR_ATOL,
                "passed": passed,
            }
        )
        regular_maps[name] = maps
        rendered = "non-finite" if difference is None else f"{difference:.6e}"
        print(f"{name} regular max_abs_diff={rendered} passed={passed}", flush=True)
    report["regular_seconds"] = time.perf_counter() - started
    report["maps_written"] = 0
    all_passed = all(check["passed"] for check in checks)
    write_json(
        out_dir / "regular-checks.json",
        {"atol": REGULAR_ATOL, "passed": all_passed, "runs": checks},
    )
    if not all_passed:
        raise PilotFailure("Regular-light score reproduction failed the 1e-3 guard")

    relight_started = time.perf_counter()
    payloads = []
    for name in CAN_RUN_NAMES:
        run = by_name[name]
        relit = {relight_name: [] for relight_name in RELIGHT_NAMES}
        for image in run.images:
            changed = relight_validation_image(np.asarray(image, dtype=np.float32))
            if tuple(changed) != RELIGHT_NAMES:
                raise PilotFailure("Relight names drifted from the frozen list")
            for relight_name in RELIGHT_NAMES:
                relit[relight_name].append(_one_map(changed[relight_name], run.predict))
        stacked = {relight_name: np.stack(views) for relight_name, views in relit.items()}
        for relight_name, views in stacked.items():
            if views.shape != regular_maps[name].shape:
                raise PilotFailure(f"{name} {relight_name} map shape differs from regular light")
        payloads.append(
            (
                name,
                {
                    "category": "can",
                    "seed": run.seed,
                    "variant": run.variant,
                    "image_ids": [row["image_id"] for row in run.rows],
                    "regular": regular_maps[name],
                    "relit": stacked,
                },
            )
        )
    report["relight_seconds"] = time.perf_counter() - relight_started
    publish_maps(out_dir, payloads)
    report["maps_written"] = len(payloads)
    print(f"wrote {len(payloads)} can validation-map files", flush=True)


def publish_maps(out_dir: Path, payloads: list[tuple[str, dict]]) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=".maps-", dir=out_dir))
    published = False
    try:
        for name, payload in payloads:
            save_validation_maps(temporary / f"{name}.npz", payload)
        destination = out_dir / "maps"
        if destination.exists():
            raise PilotFailure("Refusing to overwrite maps")
        os.rename(temporary, destination)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
    names = sorted(path.name for path in (out_dir / "maps").iterdir())
    expected = sorted(f"{name}.npz" for name in CAN_RUN_NAMES)
    if names != expected:
        shutil.rmtree(out_dir / "maps")
        raise PilotFailure("Map publication did not contain exactly the six can runs")


def load_checkpoint_state(run_dir: Path, name: str) -> tuple[dict, dict, list[dict]]:
    assert_head_file(run_dir / "heads.pt", name)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows = json.loads((run_dir / "validation-scores.json").read_text(encoding="utf-8"))
    state = torch.load(run_dir / "heads.pt", map_location="cpu", weights_only=True)
    assert_released_head_config(state["config"], run["config"], name)
    assert_encoder_hash(state.get("encoder_hash"))
    return run, state, rows


def prepare_runs(heads_root: Path, data_root: Path, device: torch.device) -> list[PreparedRun]:
    """Load six heads. The backbone file must already have passed its checksum."""
    layout = assert_heads_layout(heads_root)
    loaded = {name: load_checkpoint_state(path, name) for name, path in layout.items()}
    assert_can_score_rows({name: rows for name, (_, _, rows) in loaded.items()})
    refuse_test_path(data_root.as_posix())
    prepared = []
    for name in CAN_RUN_NAMES:
        run_dir = layout[name]
        _run, state, rows = loaded[name]
        try:
            images = [GENERATOR.read_validation_image(data_root, row["image_id"]) for row in rows]
        except (OSError, ValueError) as exc:
            raise PilotFailure(f"{name} validation image was refused: {exc}") from exc
        for image_id, image in zip((row["image_id"] for row in rows), images, strict=True):
            if image.shape != (256, 256, 3):
                raise PilotFailure(f"{image_id} did not load as a 256x256 RGB validation image")
        try:
            _config, predict = GENERATOR.load_study_head(run_dir, device)
        except ValueError as exc:
            raise PilotFailure(f"{name} checkpoint load failed: {exc}") from exc
        assert_encoder_hash(state["encoder_hash"])
        variant, seed = parse_can_run(name)
        prepared.append(
            PreparedRun(
                name=name,
                variant=variant,
                seed=seed,
                rows=rows,
                images=images,
                predict=predict,
            )
        )
    return prepared


def gpu_peaks() -> tuple[int | None, int | None]:
    if not torch.cuda.is_available():
        return None, None
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "regular_seconds": None,
        "relight_seconds": None,
        "maps_written": 0,
        "status": "started",
    }
    environment = {"safeguards": {}}
    writable = False
    started = time.perf_counter()
    code = 1
    try:
        if args.threads != STUDY_THREADS:
            raise PilotFailure(f"The formal can runs used {STUDY_THREADS} threads")
        assert_fresh_output(args.out_dir, args.heads_root)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        writable = True
        # Identity and the allocated GPU are settled before any head is loaded.
        verified = verify_provenance(args.source_root, args.provenance, args.provenance_sha256)
        environment["provenance_sha256"] = verified["sha256"]
        environment["git_head"] = verified["git_head"]
        submodule = verified["manifest"].get("submodule") or {}
        environment["submodule_commit"] = submodule.get("commit")
        runtime = collect_runtime(args.threads)
        environment["runtime"] = runtime
        assert_locked_runtime(runtime)
        environment["safeguards"]["locked_runtime"] = "pass"
        environment["safeguards"]["numpy_pillow"] = "pass"
        staged = stage_verified_weights(args.weights, args.out_dir / "torch-hub")
        environment["safeguards"]["backbone_sha256"] = sha256_file(staged)
        torch.set_num_threads(args.threads)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        runs = prepare_runs(args.heads_root, args.data_root, torch.device(args.device))
        environment["safeguards"]["encoder_sha256"] = ENCODER_SHA256
        environment["safeguards"]["head_checksums"] = {
            name: HEAD_SHA256[name] for name in CAN_RUN_NAMES
        }
        execute(runs, args.out_dir, report)
        report["status"] = "completed"
        code = 0
    except PilotFailure as exc:
        report["status"] = f"failed: {exc}"
        print(report["status"], file=sys.stderr, flush=True)
        code = 1
    except Exception as exc:
        report["status"] = f"failed: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        code = 1
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["elapsed_seconds"] = time.perf_counter() - started
        allocated, reserved = gpu_peaks()
        report["peak_gpu_memory_allocated_bytes"] = allocated
        report["peak_gpu_memory_reserved_bytes"] = reserved
        if writable:
            environment["status"] = report["status"]
            write_json(args.out_dir / "environment.json", environment)
            write_json(args.out_dir / "runtime.json", report)
    return code


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--provenance-sha256", type=Path, required=True)
    parser.add_argument("--heads-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=STUDY_THREADS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
