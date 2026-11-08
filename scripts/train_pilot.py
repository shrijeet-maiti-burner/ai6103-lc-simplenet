"""Fixed-step, normal-only development run; outputs remain local."""

import argparse
import ctypes
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from lc_simplenet.data import CachedSource, NormalManifestDataset
from lc_simplenet.geometry import restore_score_map
from lc_simplenet.metrics import image_counts, normal_validation_threshold
from lc_simplenet.normal_validation import paired_normal_metrics
from lc_simplenet.pilot import (
    ROOT,
    PilotConfig,
    PilotTrainer,
    assert_same_state,
    config_hash,
    prediction_geometry,
    state_hash,
    validate_input_geometry,
)
from lc_simplenet.source_identity import source_record


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def truncate_train_log(path: Path, max_step: int) -> list[dict]:
    """Keep records with ``step <= max_step``. Create the file if missing.

    Blank lines are ignored. The only tolerated parse failure is an incomplete
    final line with no trailing newline, as left by an interrupted write.
    """
    if type(max_step) is not int or max_step < 0:
        raise ValueError("max_step must be a nonnegative integer")
    kept = []
    if path.exists():
        text = path.read_text(encoding="utf-8")
        unterminated_last = bool(text) and not text.endswith("\n")
        previous_step = 0
        lines = text.splitlines()
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if unterminated_last and index == len(lines):
                    break
                raise ValueError(f"{path}:{index}: malformed JSON") from error
            if type(record) is not dict:
                raise ValueError(f"{path}:{index}: log record must be a JSON object")
            step = record.get("step")
            if type(step) is not int or step < 1:
                raise ValueError(f"{path}:{index}: step must be a positive integer")
            if step <= previous_step:
                raise ValueError(f"{path}:{index}: steps must be strictly increasing")
            previous_step = step
            if step <= max_step:
                kept.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(record, allow_nan=False) + "\n" for record in kept)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return kept


def should_write_periodic_checkpoint(step: int, interval: int) -> bool:
    return interval > 0 and step > 0 and step % interval == 0


def periodic_checkpoint_path(output: Path, step: int) -> Path:
    return output / f"checkpoint-{step:06d}.pt"


def final_checkpoint_path(output: Path) -> Path:
    return output / "final.pt"


def maybe_save_periodic_checkpoint(trainer, output: Path, interval: int) -> Path | None:
    if not should_write_periodic_checkpoint(trainer.step, interval):
        return None
    path = periodic_checkpoint_path(output, trainer.step)
    trainer.save(path)
    return path


def save_final_checkpoint(trainer, output: Path) -> Path:
    path = final_checkpoint_path(output)
    trainer.save(path)
    return path


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative; 0 disables periodic checkpoints")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Local PilotConfig JSON")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory below results/; must be new unless --resume",
    )
    parser.add_argument("--steps", type=int, required=True, help="Total fixed optimizer steps")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=1.0)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--checkpoint-every",
        type=nonnegative_int,
        default=0,
        help="Save a checkpoint every N optimizer steps; 0 disables periodic checkpoints",
    )
    parser.add_argument("--verify-replay", action="store_true")
    parser.add_argument("--normal-diagnostics", action="store_true")
    parser.add_argument(
        "--cache-images",
        action="store_true",
        help="Keep decoded, resized images after the first hash-checked read (same results)",
    )
    return parser


def validate_resume_checkpoint(resume: Path, output: Path) -> Path:
    resume = resume.resolve()
    output = output.resolve()
    if not resume.is_file():
        raise FileNotFoundError(f"resume checkpoint is not a file: {resume}")
    if not resume.is_relative_to(output):
        raise ValueError("resume checkpoint must belong to the --output directory")
    return resume


def prepare_output_directory(output: Path, *, resume: Path | None, results_root: Path) -> Path:
    output = output.resolve()
    results_root = results_root.resolve()
    if not output.is_relative_to(results_root) or output == results_root:
        raise ValueError("output must be a directory below this checkout's ignored results/")
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        return output
    if not output.is_dir():
        raise FileNotFoundError("resume requires an existing --output directory")
    return output


def peak_rss_bytes():
    if sys.platform != "win32":
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1 if sys.platform == "darwin" else 1024))

    # PROCESS_MEMORY_COUNTERS: DWORD sizes/flags followed by eight SIZE_Ts.
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in ("peak", "working", "qpp", "qpu", "qnpp", "qnpu", "page", "peak_page")
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    api = ctypes.windll.psapi.GetProcessMemoryInfo
    api.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not api(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.peak)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error("steps/threads must be positive; GPU memory fraction must be in (0,1]")
    if not 0 < args.target_fpr < 1:
        parser.error("target-fpr must be in (0,1)")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("image_size", "brightness", "gamma"):
        if key in raw:
            raw[key] = tuple(raw[key])
    config = PilotConfig(**raw)
    results_root = (ROOT / "results").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(results_root) or output == results_root:
        parser.error("output must be a directory below this checkout's ignored results/")
    if args.resume is not None:
        try:
            args.resume = validate_resume_checkpoint(args.resume, output)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this environment")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    datasets = {
        split: NormalManifestDataset(
            args.manifest,
            args.root,
            dataset=config.dataset,
            category=config.category,
            split=split,
            expected_sha256=config.manifest_sha256,
            output_size=config.load_size,
        )
        for split in ("train", "validation")
    }
    for source in datasets.values():
        validate_input_geometry(config, source)
    if args.cache_images:
        datasets = {split: CachedSource(source) for split, source in datasets.items()}
    try:
        prepare_output_directory(output, resume=args.resume, results_root=results_root)
    except FileExistsError:
        parser.error("output directory already exists; pass --resume to continue a run")
    except FileNotFoundError as error:
        parser.error(str(error))
    except ValueError as error:
        parser.error(str(error))
    metadata = {
        "scope": "development pilot; no defect or benchmark results",
        "config": asdict(config),
        "config_sha256": config_hash(config),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **source_record(ROOT),
        "lock_sha256": sha256(ROOT / "uv.lock"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "split_counts": {key: len(value) for key, value in datasets.items()},
        "target_fpr": args.target_fpr,
        "checkpoint_policy": (
            "periodic optimizer-step checkpoints and fixed final step"
            if args.checkpoint_every > 0
            else "fixed final step"
        ),
        "checkpoint_every": args.checkpoint_every,
        "runner_sha256": sha256(__file__),
        "postprocessing": "raw -D patch maps; image=max patch; bilinear align_corners=False "
        "to input, Gaussian sigma=4 input pixels; "
        + ("retain crop coordinates" if config.crop_box is not None else "PIL bilinear to native"),
    }
    if config.weights == "imagenet_v1":
        # The published V1 URL has this filename; hash full bytes after factory loading.
        metadata["weight_file"] = "wide_resnet50_2-95faca4d.pth"
    write_json(output / "run.json", metadata)
    started = time.perf_counter()
    try:
        trainer = PilotTrainer(config, datasets["train"], args.device)
        metadata["continuation_contract"] = trainer.continuation_contract()
        metadata["encoder_sha256"] = trainer.backend.encoder_hash
        if config.weights == "imagenet_v1":
            metadata["weight_file_sha256"] = sha256(
                Path(torch.hub.get_dir()) / "checkpoints" / metadata["weight_file"]
            )
        if args.resume:
            trainer.load(args.resume)
            metadata["resume_checkpoint_sha256"] = sha256(args.resume)
            truncate_train_log(output / "train.jsonl", trainer.step)
        if trainer.step >= args.steps:
            raise ValueError("Total steps must exceed the resumed step")
        metadata["start_step"] = trainer.step
        metadata["initial_trainable_sha256"] = trainer.trainable_hash()
        metadata["initialization_seconds"] = time.perf_counter() - started
        write_json(output / "run.json", metadata)
        timings = []
        log_path = output / "train.jsonl"
        log_mode = "a" if args.resume else "x"
        with log_path.open(log_mode, encoding="utf-8") as log:
            while trainer.step < args.steps:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                record = trainer.train_step()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                record["seconds_including_load"] = time.perf_counter() - begin
                timings.append(record["seconds_including_load"])
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                periodic_checkpoint = maybe_save_periodic_checkpoint(
                    trainer, output, args.checkpoint_every
                )
                if periodic_checkpoint is not None:
                    print(
                        json.dumps(
                            {
                                "event": "checkpoint_saved",
                                "step": trainer.step,
                                "path": str(periodic_checkpoint),
                            }
                        ),
                        flush=True,
                    )
        checkpoint = save_final_checkpoint(trainer, output)
        replay = None
        if args.verify_replay:
            first = trainer.train_step()
            # Serialize tensors so subsequent updates cannot mutate the expected result.
            temporary = output / "replay-reference.pt"
            trainer.save(temporary)
            trainer.load(checkpoint)
            second = trainer.train_step()
            assert_same_state(first, second)
            expected = torch.load(temporary, map_location=trainer.device, weights_only=True)
            assert_same_state(expected, trainer.checkpoint())
            replay = {
                "next_step": trainer.step,
                "exact_metrics_and_full_state": True,
                "reference_sha256": sha256(temporary),
            }
            trainer.load(checkpoint)
        # Predict normal validation only after the fixed training horizon.
        if args.normal_diagnostics:
            write_json(
                output / "normal-diagnostics.json",
                paired_normal_metrics(trainer, datasets["validation"]),
            )
        maps, rows = [], []
        native_shape = None
        crop_shape = None
        for index in range(len(datasets["validation"])):
            sample = datasets["validation"][index]
            patch_map = trainer.predict([sample["image"]])[0]
            if not np.isfinite(patch_map).all():
                raise FloatingPointError("Non-finite raw prediction")
            maps.append(patch_map)
            rows.append(
                {
                    "image_id": sample["image_id"],
                    "label": 0,
                    "score": float(patch_map.max()),
                    **prediction_geometry(config, sample),
                }
            )
            if index == 0:
                resized = F.interpolate(
                    torch.from_numpy(patch_map)[None, None],
                    size=config.image_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy()
                smoothed = gaussian_filter(resized, sigma=4)
                if config.crop_box is None:
                    native = restore_score_map(
                        smoothed,
                        original_size=sample["original_size"],
                        image_size=config.image_size,
                    )
                    native_shape = list(native.shape)
                    np.save(output / "first-validation-native-map.npy", native, allow_pickle=False)
                else:
                    crop_shape = list(smoothed.shape)
                    np.save(output / "first-validation-crop-map.npy", smoothed, allow_pickle=False)
        np.savez_compressed(output / "validation-patch-maps.npz", scores=np.stack(maps))
        write_json(output / "validation-scores.json", rows)
        scores = [row["score"] for row in rows]
        threshold = normal_validation_threshold(scores, args.target_fpr)
        counts = image_counts(scores, [0] * len(scores), threshold)
        if state_hash(trainer.backend.encoder.state_dict()) != trainer.backend.encoder_hash:
            raise RuntimeError("Frozen backbone state changed, including BatchNorm buffers")
        result = {
            "status": "completed",
            "scope": metadata["scope"],
            "steps": trainer.step,
            "trainable_sha256": trainer.trainable_hash(),
            "checkpoint_sha256": sha256(checkpoint),
            "validation_scores_sha256": sha256(output / "validation-scores.json"),
            "calibration": {
                "method": "higher empirical quantile; strict score > threshold",
                "target_fpr": args.target_fpr,
                "threshold": threshold,
                **counts,
            },
            "native_map_shape": native_shape,
            "crop_map_shape": crop_shape,
            "replay": replay,
            "training_timing": {
                "steps": len(timings),
                "warmup_steps": min(2, len(timings)),
                "median_seconds_after_warmup": float(np.median(timings[2:]))
                if len(timings) > 2
                else None,
                "total_seconds_including_load": sum(timings),
            },
            "process_peak_rss_bytes": peak_rss_bytes(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if args.device == "cuda"
            else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
            if args.device == "cuda"
            else None,
            "total_seconds": time.perf_counter() - started,
        }
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
