"""Exact native metrics with disk-backed float32 radix histograms."""

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from lc_simplenet.localization import METRIC_VERSION, integrate_to


class ExternalNativeMetrics:
    def __init__(self, work, fpr_limit=0.05, max_histogram_bytes=64 * 2**20, prefix_bits=12):
        if not math.isfinite(fpr_limit) or not 0 < fpr_limit <= 1:
            raise ValueError("invalid FPR limit")
        if type(prefix_bits) is not int or not 8 <= prefix_bits <= 24:
            raise ValueError("invalid prefix width")
        if type(max_histogram_bytes) is not int or max_histogram_bytes <= 0:
            raise ValueError("histogram budget must be a positive integer")
        self.shift = 32 - prefix_bits
        self.bins = 1 << self.shift
        self.histogram_bytes = self.bins * 24
        if self.histogram_bytes > max_histogram_bytes:
            raise MemoryError("fine histograms exceed declared persistent budget")
        self.work = Path(work).resolve()
        self.work.mkdir(parents=True, exist_ok=False)
        self.limit = fpr_limit
        self.images = self.regions = self.normal = self.defect = self.work_bytes = 0
        self.files, self.handles = {}, OrderedDict()
        self.cached = None
        self.failed = False

    def _handle(self, kind, prefix):
        key = (kind, prefix)
        if key in self.handles:
            self.handles.move_to_end(key)
            return self.handles[key]
        path = self.work / f"{kind}_{prefix:06x}.bin"
        self.files[key] = path
        handle = path.open("ab", buffering=0)
        self.handles[key] = handle
        if len(self.handles) > 128:
            self.handles.popitem(last=False)[1].close()
        return handle

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def _append(self, kind, keys, weights=None):
        if not keys.size:
            return
        prefix = keys >> self.shift
        order = np.argsort(prefix, kind="stable")
        ordered = keys[order]
        prefixes = prefix[order]
        boundaries = np.r_[0, np.flatnonzero(prefixes[1:] != prefixes[:-1]) + 1, len(keys)]
        if weights is not None:
            weights = weights[order]
        low = ordered & np.uint32(self.bins - 1)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            handle = self._handle(kind, int(prefixes[start]))
            if weights is None:
                low[start:stop].astype("<u4", copy=False).tofile(handle)
                self.work_bytes += 4 * int(stop - start)
            else:
                record = np.empty(stop - start, dtype=[("key", "<u4"), ("weight", "<f8")])
                record["key"], record["weight"] = low[start:stop], weights[start:stop]
                record.tofile(handle)
                self.work_bytes += 12 * len(record)

    def add(self, scores, truth):
        from scipy.ndimage import label

        if self.cached is not None or self.failed:
            raise ValueError("accumulation already sealed")
        scores, truth = np.asarray(scores), np.asarray(truth)
        if (
            scores.dtype != np.float32
            or scores.ndim != 2
            or not scores.size
            or scores.shape != truth.shape
            or not np.isfinite(scores).all()
            or not np.isin(truth, [0, 1]).all()
        ):
            raise ValueError("expected finite float32 maps and aligned binary masks")
        connected, count = label(truth, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(connected.ravel())
        bits = scores.ravel().view(np.uint32)
        keys = np.where(bits >> 31, ~bits, bits ^ np.uint32(0x80000000))
        keys = np.where(scores.ravel() == 0, np.uint32(0x80000000), keys)
        normal = connected.ravel() == 0
        # A failed append leaves temporary records; never reuse that accumulator.
        self.failed = True
        self._append("n", keys[normal])
        defect = ~normal
        if defect.any():
            regions = connected.ravel()[defect]
            self._append("d", keys[defect], 1.0 / sizes[regions])
        self.images += 1
        self.regions += int(count)
        self.normal += int(sizes[0])
        self.defect += int(scores.size - sizes[0])
        self.failed = False

    def _histograms(self, prefix):
        normal = np.zeros(self.bins, dtype=np.uint64)
        defect = np.zeros(self.bins, dtype=np.uint64)
        weights = np.zeros(self.bins, dtype=np.float64)
        for kind, dtype in [
            ("n", np.dtype("<u4")),
            ("d", np.dtype([("key", "<u4"), ("weight", "<f8")])),
        ]:
            path = self.files.get((kind, prefix))
            if path is None:
                continue
            with path.open("rb") as handle:
                while True:
                    chunk = np.fromfile(handle, dtype=dtype, count=2**18)
                    if not len(chunk):
                        break
                    keys = chunk if kind == "n" else chunk["key"]
                    counts = np.bincount(keys, minlength=self.bins).astype(np.uint64)
                    if kind == "n":
                        normal += counts
                    else:
                        defect += counts
                        weights += np.bincount(keys, weights=chunk["weight"], minlength=self.bins)
        occupied = np.flatnonzero((normal != 0) | (defect != 0))[::-1]
        return normal[occupied], defect[occupied], weights[occupied]

    def result(self):
        if self.failed:
            raise ValueError("a failed disk append invalidated this accumulator")
        if self.cached is not None:
            return dict(self.cached)
        self.close()
        if not self.images:
            raise ValueError("no images accumulated")
        result = {
            "metric_version": METRIC_VERSION,
            "algorithm": "exact external radix histogram",
            "images": self.images,
            "normal_pixels": self.normal,
            "defect_pixels": self.defect,
            "regions": self.regions,
            "connectivity": 8,
            "fpr_limit": self.limit,
            "histogram_bytes": self.histogram_bytes,
            "temporary_disk_bytes": self.work_bytes,
            "pixel_auroc": None,
            "au_pro": None,
            "normalization": "trapezoid integral to FPR limit, divided by that limit",
        }
        if not self.normal or not self.defect or not self.regions:
            result["status"] = "unavailable: missing normal pixels or defect regions"
            self.cached = result
            return dict(result)
        fp = tp = 0
        overlap = pixel_area = pro_area = last_x = last_y = last_pro = 0.0
        pro_complete = False
        for prefix in sorted({p for _, p in self.files}, reverse=True):
            normal, defect, changes = self._histograms(prefix)
            false = fp + np.cumsum(normal, dtype=np.uint64)
            true = tp + np.cumsum(defect, dtype=np.uint64)
            pro = overlap + np.cumsum(changes, dtype=np.float64)
            x = np.r_[last_x, false / self.normal]
            y = np.r_[last_y, true / self.defect]
            region_y = np.r_[last_pro, np.clip(pro / self.regions, 0, 1)]
            pixel_area += integrate_to(x, y, x[-1])
            if not pro_complete:
                pro_area += integrate_to(x, region_y, min(self.limit, x[-1]))
                pro_complete = x[-1] >= self.limit
            fp, tp, overlap = int(false[-1]), int(true[-1]), float(pro[-1])
            last_x, last_y, last_pro = x[-1], y[-1], region_y[-1]
        if (
            fp != self.normal
            or tp != self.defect
            or not math.isclose(overlap, self.regions, abs_tol=1e-7)
        ):
            raise AssertionError("external histogram population or region mass changed")
        result.update(pixel_auroc=pixel_area, au_pro=pro_area / self.limit, status="complete")
        self.cached = result
        return dict(result)

    def cleanup(self):
        """Remove only files created by this instance, after the caller saves evidence."""
        self.close()
        for path in self.files.values():
            resolved = path.resolve(strict=True)
            if resolved.parent != self.work or not resolved.name.endswith(".bin"):
                raise ValueError("temporary file escaped its owned directory")
            resolved.unlink()
        self.work.rmdir()
