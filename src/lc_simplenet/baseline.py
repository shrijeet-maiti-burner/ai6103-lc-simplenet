"""Upstream single-view SimpleNet with a fixed final checkpoint and no test data.

Upstream ``SimpleNet.train`` evaluates test labels after every meta-epoch and keeps
the best test AUROC. Here every meta-epoch only calls the unchanged upstream
``_train_discriminator`` and the final state is kept. Settings follow upstream
``run.sh`` for MVTec AD at commit 351a2b8.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
UPSTREAM_AD = {
    "layers_to_extract_from": ["layer2", "layer3"],
    "pretrain_embed_dimension": 1536,
    "target_embed_dimension": 1536,
    "patchsize": 3,
    "meta_epochs": 40,
    "embedding_size": 256,
    "gan_epochs": 4,
    "noise_std": 0.015,
    "dsc_hidden": 1024,
    "dsc_layers": 2,
    "dsc_margin": 0.5,
    "pre_proj": 1,
}
UPSTREAM_AD_INPUT = {"batch_size": 8, "resize": 329, "imagesize": 288}


def upstream_module(name: str):
    """Import a module from the pinned, unmodified upstream checkout.

    Bytecode writing is off during the import. Upstream ignores only its top-level
    ``__pycache__``, so new caches in ``datasets/`` would show the submodule as modified
    and every later run from the checkout would record ``git_dirty``.
    """
    path = str(ROOT / "third_party/SimpleNet")
    if path not in sys.path:
        sys.path.insert(0, path)
    writes = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module(name)
    finally:
        sys.dont_write_bytecode = writes


class UpstreamInput(torch.utils.data.Dataset):
    """Square-resized HWC samples to center-cropped, ImageNet-normalized CHW.

    With ``output_size=(resize, resize)``, the source applies the same PIL bilinear
    resize as upstream ``transforms.Resize(resize)``, but only for square images.
    """

    def __init__(self, source, imagesize: int):
        size = source.output_size
        if size is None or size[0] != size[1] or size[0] < imagesize:
            raise ValueError("Source needs a square output_size of at least imagesize")
        if any(record.width != record.height for record in source.records):
            raise ValueError("Upstream Resize(resize) is a square resize only for square images")
        self.source, self.imagesize = source, imagesize

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        image = torch.from_numpy(sample["image"]).permute(2, 0, 1)
        image = TF.center_crop(image, [self.imagesize, self.imagesize])
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
        return {"image": image, "image_id": sample["image_id"]}


def center_crop_box(resize: int, imagesize: int) -> list[int]:
    """[top, left, bottom, right] of the torchvision center crop in the resized square."""
    top = int(round((resize - imagesize) / 2.0))
    return [top, top, top + imagesize, top + imagesize]


def build_model(backbone, device, imagesize: int, settings: dict = UPSTREAM_AD):
    model = upstream_module("simplenet").SimpleNet(device)
    model.load(backbone=backbone, device=device, input_shape=(3, imagesize, imagesize), **settings)
    return model


def train_fixed(model, loader, log_dir: Path) -> None:
    """Run all meta-epochs without evaluation; upstream logs to TensorBoard under log_dir."""
    model.set_model_dir(str(log_dir), "upstream")
    for _ in range(model.meta_epochs):
        model._train_discriminator(loader)


def trained_state(model) -> dict:
    state = {"discriminator": model.discriminator.state_dict()}
    if model.pre_proj > 0:
        state["pre_projection"] = model.pre_projection.state_dict()
    return {name: {k: v.detach().cpu() for k, v in s.items()} for name, s in state.items()}


def load_trained_state(model, state: dict) -> None:
    model.discriminator.load_state_dict(state["discriminator"])
    if model.pre_proj > 0:
        model.pre_projection.load_state_dict(state["pre_projection"])


def predict(model, loader) -> dict:
    """Raw upstream image scores and smoothed maps at model-input size; no normalization."""
    ids, scores, maps = [], [], []
    for batch in loader:
        image_scores, image_maps, _ = model._predict(batch["image"])
        ids.extend(batch["image_id"])
        scores.extend(image_scores)
        maps.extend(image_maps)
    return {
        "image_id": np.array(ids),
        "image_score": np.asarray(scores, dtype=np.float32),
        "map": np.stack(maps).astype(np.float32),
    }
