"""Exact float32 score histograms for native maps, processed one image at a time."""

import math

import numpy as np

from lc_simplenet.localization import METRIC_VERSION, integrate_to


class NativeMetrics:
    """Accumulate exact score ties without retaining or sorting a whole category.

    Float32 values are indexed by their ordered bit representation. Sparse
    Blocks address 65,536 score values but store only their occupied bins.
    This quantizes no scores. The budget covers persistent histogram arrays;
    one image's labels/sort and temporary arrays require additional memory.
    """

    BINS = 65536

    def __init__(self, fpr_limit=0.05, max_histogram_bytes=2**31):
        if not math.isfinite(fpr_limit) or not 0 < fpr_limit <= 1:
            raise ValueError("fpr_limit must be finite and in (0,1]")
        if type(max_histogram_bytes) is not int or max_histogram_bytes < 26:
            raise ValueError("histogram budget must hold at least one occupied bin")
        self.limit, self.budget = fpr_limit, max_histogram_bytes
        self.blocks = {}
        self.histogram_bytes = 0
        self.images = self.regions = self.normal = self.defect = 0

    def add(self, scores, truth):
        from scipy.ndimage import label

        scores, truth = np.asarray(scores), np.asarray(truth)
        if (
            scores.dtype != np.float32
            or scores.ndim != 2
            or not scores.size
            or scores.shape != truth.shape
            or not np.isfinite(scores).all()
            or not np.isin(truth, [0, 1]).all()
        ):
            raise ValueError("expected finite float32 2D scores and aligned binary masks")
        connected, regions = label(truth, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(connected.ravel())
        weights = np.zeros(len(sizes), dtype=np.float64)
        weights[1:] = 1.0 / sizes[1:]
        bits = scores.ravel().view(np.uint32)
        keys = np.where(bits >> 31, ~bits, bits ^ np.uint32(0x80000000))
        keys = np.where(scores.ravel() == 0, np.uint32(0x80000000), keys)  # merge signed zero
        order = np.argsort(keys, kind="stable")
        ordered = keys[order]
        prefixes = ordered >> 16
        boundaries = np.r_[0, np.flatnonzero(prefixes[1:] != prefixes[:-1]) + 1, len(order)]
        labels = connected.ravel()[order]
        pending = {}
        estimated = self.histogram_bytes
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            prefix = int(prefixes[start])
            low = (ordered[start:end] & np.uint32(65535)).astype(np.uint16)
            region = labels[start:end]
            starts = np.r_[0, np.flatnonzero(low[1:] != low[:-1]) + 1]
            keys = low[starts]
            counts = np.diff(np.r_[starts, len(low)]).astype(np.uint64)
            positives = np.add.reduceat((region != 0).astype(np.uint64), starts)
            contribution = np.add.reduceat(weights[region], starts)
            replacement = (keys, counts - positives, positives, contribution)
            if prefix in self.blocks:
                previous = self.blocks[prefix]
                union = np.union1d(previous[0], keys)
                merged = (
                    union,
                    np.zeros(len(union), dtype=np.uint64),
                    np.zeros(len(union), dtype=np.uint64),
                    np.zeros(len(union), dtype=np.float64),
                )
                old_indices, new_indices = (
                    np.searchsorted(union, previous[0]),
                    np.searchsorted(union, keys),
                )
                for i in range(1, 4):
                    merged[i][old_indices] = previous[i]
                    merged[i][new_indices] += replacement[i]
                estimated -= sum(value.nbytes for value in previous)
                replacement = merged
            estimated += sum(value.nbytes for value in replacement)
            if estimated > self.budget:
                raise MemoryError("exact score histogram exceeds its declared memory budget")
            pending[prefix] = replacement
        self.blocks.update(pending)
        self.histogram_bytes = estimated
        self.images += 1
        self.regions += regions
        self.normal += int(sizes[0])
        self.defect += int(truth.size - sizes[0])

    def result(self):
        if not self.images:
            raise ValueError("no images accumulated")
        result = {
            "metric_version": METRIC_VERSION,
            "algorithm": "exact sparse float32 histogram",
            "images": self.images,
            "normal_pixels": self.normal,
            "defect_pixels": self.defect,
            "regions": self.regions,
            "connectivity": 8,
            "fpr_limit": self.limit,
            "histogram_bytes": self.histogram_bytes,
            "pixel_auroc": None,
            "au_pro": None,
            "normalization": "trapezoid integral to FPR limit, divided by that limit",
        }
        if not self.normal or not self.defect or not self.regions:
            result["status"] = "unavailable: missing normal pixels or defect regions"
            return result
        fp = tp = 0
        overlap = pixel_area = pro_area = 0.0
        last_fpr = last_tpr = last_pro = 0.0
        pro_complete = False
        for prefix in sorted(self.blocks, reverse=True):
            _, normal, defect, changes = self.blocks[prefix]
            selected = np.arange(len(normal) - 1, -1, -1)
            if not len(selected):
                continue
            false = fp + np.cumsum(normal[selected], dtype=np.uint64)
            true = tp + np.cumsum(defect[selected], dtype=np.uint64)
            pro = overlap + np.cumsum(changes[selected], dtype=np.float64)
            x = np.r_[last_fpr, false / self.normal]
            y = np.r_[last_tpr, true / self.defect]
            region_y = np.r_[last_pro, np.clip(pro / self.regions, 0, 1)]
            pixel_area += integrate_to(x, y, x[-1])
            if not pro_complete:
                end = min(self.limit, x[-1])
                pro_area += integrate_to(x, region_y, end)
                pro_complete = x[-1] >= self.limit
            fp, tp, overlap = int(false[-1]), int(true[-1]), float(pro[-1])
            last_fpr, last_tpr, last_pro = x[-1], y[-1], region_y[-1]
        result.update(pixel_auroc=pixel_area, au_pro=pro_area / self.limit, status="complete")
        return result
