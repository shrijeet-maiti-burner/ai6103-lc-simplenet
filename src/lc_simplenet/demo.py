"""Local, normal-validation inspection with frozen heads and calibration."""

import base64
import copy
import hashlib
import io
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from lc_simplenet.evaluation import calibrate
from lc_simplenet.pilot import PilotConfig, UpstreamBackend, seed_runtime


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def png_data(array):
    pixels = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def heatmap_data(scores, limits):
    """Fixed validation-derived display scale; never per-image min/max."""
    low, high = limits
    levels = np.clip((scores - low) / (high - low), 0, 1)
    positions = np.array([0, 0.25, 0.5, 0.75, 1])
    colors = np.array(
        [
            [0.10, 0.18, 0.25],
            [0.16, 0.38, 0.46],
            [0.24, 0.65, 0.60],
            [0.86, 0.78, 0.37],
            [0.95, 0.33, 0.20],
        ]
    )
    rgb = np.stack([np.interp(levels, positions, colors[:, channel]) for channel in range(3)], -1)
    return png_data(rgb)


def parse_config(raw):
    raw = dict(raw)
    for key in ("image_size", "brightness", "gamma"):
        if key in raw:
            raw[key] = tuple(raw[key])
    settings = PilotConfig(**raw)
    if settings.dataset != "mvtec_ad2" or settings.ad_resize is not None:
        raise ValueError("the demo currently accepts full-frame ad 2 study runs only")
    return settings


def validate_request(request, catalog):
    if not isinstance(request, dict) or set(request) - {
        "category",
        "seed",
        "image_id",
        "brightness",
        "gamma",
        "weight",
    }:
        raise ValueError("invalid inspection request")
    category = request.get("category")
    seed = request.get("seed", 0)
    if (
        not isinstance(category, str)
        or category not in catalog
        or type(seed) is not int
        or seed not in catalog[category]["seeds"]
    ):
        raise ValueError("choose an available category and seed")
    image_id = request.get("image_id")
    if image_id not in catalog[category]["images"]:
        raise ValueError("choose a listed normal-validation image")
    values = {}
    for name in ("brightness", "gamma"):
        value = request.get(name, 1.0)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("lighting values must be numbers")
        if not math.isfinite(value) or not 0.8 <= value <= 1.2:
            raise ValueError("lighting controls are limited to the studied range [0.8,1.2]")
        values[name] = float(value)
    weight = request.get("weight", 0.1)
    if type(weight) not in (float, int) or weight not in catalog[category]["weights"]:
        raise ValueError("choose an available consistency ablation")
    return {
        "category": category,
        "seed": seed,
        "image_id": image_id,
        "weight": float(weight),
        **values,
    }


class InspectionRuntime:
    def __init__(self, runs: Path, data: Path, device="cpu"):
        import torch

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("cuda is unavailable")
        self.data = data.resolve(strict=True)
        self.entries, self.catalog = {}, {}
        self.backend = None
        self.loaded = OrderedDict()
        self.frames = OrderedDict()
        self.lock = threading.RLock()
        for path in sorted(runs.rglob("run.json")):
            checkpoint = path.parent / "heads.pt"
            if not checkpoint.exists():
                checkpoint = path.parent / "final.pt"
            if not checkpoint.exists() or not (path.parent / "validation-scores.json").exists():
                continue
            run = json.loads(path.read_text())
            if run.get("config", {}).get("dataset") != "mvtec_ad2":
                continue
            settings = parse_config(run["config"])
            if settings.variant not in {"B0", "B1", "B2"}:
                continue
            rows = json.loads((path.parent / "validation-scores.json").read_text())
            if not rows or any(row["label"] != 0 for row in rows):
                raise ValueError("demo images must be normal validation only")
            for row in rows:
                parts = PurePosixPath(row["image_id"]).parts
                if (
                    len(parts) != 4
                    or parts[:3] != (settings.category, "validation", "good")
                    or any(p in {".", ".."} or ":" in p or "\\" in p for p in parts)
                ):
                    raise ValueError("unsafe normal-validation image ID")
            method = (
                settings.variant if settings.variant != "B2" else f"B2@{settings.lambda_score:g}"
            )
            key = settings.category, settings.seed, method
            if key in self.entries:
                raise ValueError(f"duplicate model identity: {key}")
            calibration = calibrate(rows, run["target_fpr"])
            values = np.array([row["score"] for row in rows])
            maps_path = path.parent / "validation-patch-maps.npz"
            with np.load(maps_path, allow_pickle=False) as archive:
                maps = archive["scores"]
                if not np.isfinite(maps).all() or len(maps) != len(rows):
                    raise ValueError("invalid saved validation maps")
                limits = np.quantile(maps, [0.02, 0.98]).tolist()
            if limits[0] == limits[1]:
                limits[1] = limits[0] + 1e-6
            self.entries[key] = {
                "config": settings,
                "run": run,
                "path": path.parent,
                "checkpoint": checkpoint,
                "calibration": calibration,
                "rows": {row["image_id"]: row for row in rows},
                "score_std": float(values.std()),
                "limits": limits,
            }
        for category in sorted({key[0] for key in self.entries}):
            entries = [v for k, v in self.entries.items() if k[0] == category]
            images = sorted(entries[0]["rows"])
            if any(sorted(row["rows"]) != images for row in entries):
                raise ValueError("models do not share validation image identities")
            seeds = sorted({key[1] for key in self.entries if key[0] == category})
            weights = sorted(
                {row["config"].lambda_score for row in entries if row["config"].variant == "B2"}
            )
            complete_seeds = [
                seed
                for seed in seeds
                if all(
                    (category, seed, method) in self.entries
                    for method in ["B0", "B1", *[f"B2@{weight:g}" for weight in weights]]
                )
            ]
            if complete_seeds and weights:
                self.catalog[category] = {
                    "images": images,
                    "seeds": complete_seeds,
                    "weights": weights,
                }
        if not self.catalog:
            raise ValueError("no complete released b0/b1/b2 validation models found")

    def describe(self):
        import torch

        return {
            "categories": self.catalog,
            "mode": "live inference",
            "device": str(self.device),
            "torch": str(torch.__version__),
            "scope": "normal-validation inspection, not held-out defect evaluation",
            "transform": "clip(brightness * pixels ** gamma, 0, 1)",
            "threshold_policy": "normal-validation 5% target; strict score > threshold",
            "heatmap_policy": "per-model 2nd/98th percentiles of saved validation patch scores",
            "selection": "no consistency weight passed; all b2 weights are ablations",
        }

    def _load(self, entry):
        import torch

        path = entry["checkpoint"]
        identity = str(path)
        if identity in self.loaded:
            self.loaded.move_to_end(identity)
            return self.loaded[identity]
        settings = entry["config"]
        seed_runtime(settings.seed)
        if self.backend is None:
            self.backend = UpstreamBackend(settings, self.device)
            self.input_size = settings.image_size
        if settings.image_size != self.input_size:
            raise ValueError("comparison models must share input geometry")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if parse_config(state["config"]) != settings:
            raise ValueError("head configuration does not match run.json")
        if state["encoder_hash"] != self.backend.encoder_hash:
            raise ValueError("head uses another frozen backbone")
        adapter = copy.deepcopy(self.backend.adapter)
        discriminator = copy.deepcopy(self.backend.discriminator)
        adapter.load_state_dict(state["adapter"])
        discriminator.load_state_dict(state["discriminator"])
        adapter.eval()
        discriminator.eval()
        item = (adapter, discriminator, sha256(path))
        self.loaded[identity] = item
        if len(self.loaded) > 6:
            self.loaded.popitem(last=False)
        return item

    def inspect(self, request):
        import torch
        from scipy.ndimage import gaussian_filter
        from torch.nn import functional as F

        query = validate_request(request, self.catalog)
        with self.lock:
            source = (self.data / query["image_id"]).resolve(strict=True)
            if not source.is_relative_to(self.data):
                raise ValueError("image escapes the dataset root")
            raw = source.read_bytes()
            image_hash = hashlib.sha256(raw).hexdigest()
            cache_key = (*query.items(), ("image_sha256", image_hash))
            if cache_key in self.frames:
                self.frames.move_to_end(cache_key)
                return {
                    **self.frames[cache_key],
                    "mode": "cached inference",
                    "timing_note": "time of the original measured computation",
                }
            methods = ["B0", "B1", f"B2@{query['weight']:g}"]
            entries = [self.entries[query["category"], query["seed"], method] for method in methods]
            settings = entries[0]["config"]
            controls = asdict(settings)
            for field in ("variant", "lambda_score"):
                controls.pop(field)
            for entry in entries:
                other = asdict(entry["config"])
                for field in ("variant", "lambda_score"):
                    other.pop(field)
                if other != controls or entry["calibration"]["target_fpr"] != 0.05:
                    raise ValueError("comparison models must share controls and 5% calibration")
            heads = [self._load(e) for e in entries]
            with Image.open(io.BytesIO(raw)) as image:
                original_size = [image.height, image.width]
                original = np.asarray(
                    image.convert("RGB").resize(
                        (settings.image_size[1], settings.image_size[0]), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                ) / np.float32(255)
            pixels = torch.from_numpy(original).permute(2, 0, 1)[None].to(self.device)
            altered = query["brightness"] * pixels.pow(query["gamma"])
            clipped = ((altered > 1) & (pixels < 1)).float().mean().item()
            altered = altered.clamp(0, 1)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            mean = pixels.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = pixels.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            outputs = []
            with torch.inference_mode():
                reference, grids = self.backend.embed((pixels - mean) / std)
                changed = (
                    reference
                    if query["brightness"] == query["gamma"] == 1
                    else (self.backend.embed((altered - mean) / std)[0])
                )
                for entry, (adapter, discriminator, checksum), method in zip(
                    entries, heads, methods, strict=True
                ):
                    base_map = -discriminator(adapter(reference)).reshape(*grids[0])
                    patch_map = -discriminator(adapter(changed)).reshape(*grids[0])
                    base, score = float(base_map.max()), float(patch_map.max())
                    recorded = entry["rows"][query["image_id"]]["score"]
                    error = abs(base - recorded)
                    if error > 1e-3:
                        raise ValueError(f"validation replay differs for {method}: {error:.6g}")
                    threshold = entry["calibration"]["threshold"]
                    resized = (
                        F.interpolate(
                            patch_map[None, None],
                            size=settings.image_size,
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                        .cpu()
                        .numpy()
                    )
                    smoothed = gaussian_filter(resized, sigma=4)
                    outputs.append(
                        {
                            "method": method,
                            "score": score,
                            "reference_score": base,
                            "score_change": score - base,
                            "threshold": threshold,
                            "flagged": score > threshold,
                            "reference_flagged": base > threshold,
                            "margin": score - threshold,
                            "validation_score_std": entry["score_std"],
                            "standardized_margin": (score - threshold) / entry["score_std"]
                            if entry["score_std"] > 1e-12
                            else None,
                            "pair_mse": float((patch_map - base_map).square().mean()),
                            "heatmap": heatmap_data(smoothed, entry["limits"]),
                            "heatmap_limits": entry["limits"],
                            "checkpoint_sha256": checksum,
                            "validation_replay_max_score_error": error,
                            "calibration": entry["calibration"],
                        }
                    )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            response = {
                "schema": 1,
                "mode": "live inference",
                "query": query,
                "scope": "normal-validation diagnostic; not held-out robustness or defect recall",
                "image_sha256": image_hash,
                "label": 0,
                "original_size": original_size,
                "model_input_size": list(settings.image_size),
                "original": png_data(original),
                "altered": png_data(altered[0].permute(1, 2, 0).cpu().numpy()),
                "newly_clipped_fraction": clipped,
                "models": outputs,
                "encoder_sha256": self.backend.encoder_hash,
                "milliseconds": (time.perf_counter() - started) * 1000,
                "timing_scope": "two shared encoder passes, three heads and map rendering; "
                "decoding and lazy model loading excluded",
                "device": str(self.device),
                "torch": str(torch.__version__),
            }
            # All numbers are measured; the cache is keyed by the complete request.
            json.dumps(response, allow_nan=False)
            self.frames[cache_key] = response
            if len(self.frames) > 64:
                self.frames.popitem(last=False)
            return response

    def sweep(self, request):
        query = validate_request(request, self.catalog)
        rows = []
        for gamma in (0.8, 1.0, 1.2):
            for brightness in (0.8, 1.0, 1.2):
                frame = self.inspect({**query, "gamma": gamma, "brightness": brightness})
                rows.append(
                    {
                        "brightness": brightness,
                        "gamma": gamma,
                        "models": [
                            {
                                k: m[k]
                                for k in ("method", "score", "threshold", "flagged", "pair_mse")
                            }
                            for m in frame["models"]
                        ],
                    }
                )
        return {
            "query": query,
            "scope": "nine synthetic lighting settings on one normal image",
            "selection": "diagnostic only; no hyperparameter selection",
            "rows": rows,
        }


class RecordedRuntime:
    """Serve previously measured outputs explicitly labelled as recorded."""

    def __init__(self, path):
        bundle = json.loads(Path(path).read_text())
        self.description = bundle["catalog"]
        self.description["mode"] = "recorded predictions"
        self.description["recorded_device"] = self.description["device"]
        self.description["device"] = "offline playback"
        self.description["lighting_step"] = 0.2
        self.catalog = self.description["categories"]
        self.frames = {self._key(frame["query"]): frame for frame in bundle["frames"]}

    @staticmethod
    def _key(query):
        return tuple((key, query[key]) for key in sorted(query))

    def describe(self):
        return self.description

    def inspect(self, request):
        query = validate_request(request, self.catalog)
        key = self._key(query)
        if key not in self.frames:
            raise ValueError(
                "that lighting setting was not recorded; use the fixed 3-point controls"
            )
        return {
            **self.frames[key],
            "mode": "recorded predictions",
            "timing_note": "time of the original measured computation",
        }

    sweep = InspectionRuntime.sweep
