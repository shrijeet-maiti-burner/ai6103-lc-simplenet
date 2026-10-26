"""Manifest-enforced normal images, before relighting and model normalization."""

import hashlib
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from lc_simplenet.data_manifest import AD2_CATEGORIES, AD_CATEGORIES


@dataclass(frozen=True)
class NormalRecord:
    image_id: str
    category: str
    split: str
    width: int
    height: int
    source_mode: str
    sha256: str
    rgb_sha256: str


class NormalManifestDataset:
    """One category/split from a pinned normal manifest; no directory discovery.

    ``image`` is float32 HWC RGB in [0, 1]. Geometry is native unless output_size
    (height, width) explicitly requests full-image bilinear resizing. No crop,
    relighting or normalization is implicit. Torch's default collation yields
    BHWC; the trainer must convert to BCHW after relighting, then normalize.
    This source is not a drop-in replacement for upstream's training interface.
    """

    def __init__(
        self,
        manifest: Path,
        root: Path,
        *,
        dataset: str,
        category: str,
        split: str,
        expected_sha256: str,
        output_size: tuple[int, int] | None = None,
    ):
        if dataset not in {"mvtec_ad", "mvtec_ad2"}:
            raise ValueError("Unknown dataset")
        allowed = AD_CATEGORIES if dataset == "mvtec_ad" else AD2_CATEGORIES
        if category not in allowed or split not in {"train", "validation"}:
            raise ValueError("Choose a known category and train/validation split")
        if output_size is not None and (
            len(output_size) != 2
            or any(type(value) is not int or value <= 0 for value in output_size)
        ):
            raise ValueError("output_size must be positive integer (height, width)")
        self.root = root.resolve(strict=True)
        raw = manifest.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if self.manifest_sha256 != expected_sha256:
            raise ValueError("Manifest SHA-256 differs from the declared pin")
        entries = []
        identities = set()
        rgb_splits = defaultdict(set)
        for line in raw.decode("utf-8").splitlines():
            row = json.loads(line)
            path = row["relative_path"]
            parts = PurePosixPath(path).parts
            source_split = "train" if dataset == "mvtec_ad" else row["split"]
            if (
                row["dataset"] != dataset
                or row["category"] not in allowed
                or row["split"] not in {"train", "validation"}
                or type(row["label"]) is not int
                or row["label"] != 0
                or row["mask"] is not None
                or row["image_id"] != path
                or path in identities
                or len(parts) != 4
                or parts[:3] != (row["category"], source_split, "good")
                or not parts[3].endswith(".png")
                or PurePosixPath(path).as_posix() != path
                or any(p in {".", ".."} or ":" in p or "\\" in p or "\0" in p for p in parts)
                or any(type(row[k]) is not int or row[k] <= 0 for k in ("width", "height"))
                or row["source_mode"] not in {"RGB", "L"}
                or any(not re.fullmatch(r"[0-9a-f]{64}", row[k]) for k in ("sha256", "rgb_sha256"))
            ):
                raise ValueError("Manifest violates the normal-image/split/path contract")
            identities.add(path)
            rgb_splits[row["rgb_sha256"]].add(row["split"])
            if row["category"] == category and row["split"] == split:
                resolved = (self.root / path).resolve(strict=True)
                if not resolved.is_relative_to(self.root) or not resolved.is_file():
                    raise ValueError("Selected image escapes the dataset root or is not a file")
                entries.append(
                    NormalRecord(**{k: row[k] for k in NormalRecord.__dataclass_fields__})
                )
        if any(len(splits) > 1 for splits in rgb_splits.values()):
            raise ValueError("Exact RGB content crosses train/validation")
        if not entries:
            raise ValueError("The selected category/split is empty")
        self.records = tuple(entries)
        self.dataset, self.category, self.split = dataset, category, split
        self.output_size = output_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = (self.root / record.image_id).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("Selected image escapes the dataset root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.sha256:
            raise ValueError("Image bytes changed after the manifest was frozen")
        with Image.open(io.BytesIO(raw)) as source:
            if (
                source.size != (record.width, record.height)
                or source.mode != record.source_mode
                or getattr(source, "n_frames", 1) != 1
            ):
                raise ValueError("Image dimensions/mode/frame count violate the manifest")
            image = source.convert("RGB")
            if source.getexif().get(274, 1) != 1:
                raise ValueError("Non-default EXIF orientation needs an explicit geometry rule")
            if self.output_size is not None:
                height, width = self.output_size
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32) / np.float32(255)
        return {
            "image": pixels,
            "image_id": record.image_id,
            "dataset": self.dataset,
            "category": record.category,
            "split": record.split,
            "label": 0,
            "original_size": np.array([record.height, record.width], dtype=np.int64),
            "image_size": np.array(pixels.shape[:2], dtype=np.int64),
        }
