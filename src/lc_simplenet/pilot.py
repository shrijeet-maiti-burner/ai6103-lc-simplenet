"""Matched paired-view SimpleNet pilot; no test loader or test-driven selection."""

import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lc_simplenet.losses import patch_score_consistency

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PilotConfig:
    dataset: str
    category: str
    manifest_sha256: str
    variant: str = "B0"
    lambda_score: float = 0.0
    seed: int = 0
    batch_size: int = 2
    image_size: tuple[int, int] = (288, 288)
    noise_std: float = 0.015
    margin: float = 0.5
    adapter_lr: float = 0.0001
    discriminator_lr: float = 0.0002
    brightness: tuple[float, float] = (0.8, 1.2)
    gamma: tuple[float, float] = (0.8, 1.2)
    weights: str = "imagenet_v1"

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 - 3:
            raise ValueError("seed must be an integer in [0, 2**32-3)")
        if self.variant not in {"B0", "B1", "B2"}:
            raise ValueError("variant must be B0/B1/B2")
        if not np.isfinite(self.lambda_score) or self.lambda_score < 0:
            raise ValueError("lambda_score must be finite and nonnegative")
        if self.variant != "B2" and self.lambda_score != 0:
            raise ValueError("Only B2 can use a consistency weight")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if len(self.image_size) != 2 or any(type(x) is not int or x < 32 for x in self.image_size):
            raise ValueError("image_size must contain two integers >=32")
        for value in (self.noise_std, self.margin, self.adapter_lr, self.discriminator_lr):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Noise, margin and learning rates must be finite and positive")
        for limits in (self.brightness, self.gamma):
            if len(limits) != 2 or not 0 < limits[0] <= limits[1] or not np.isfinite(limits).all():
                raise ValueError("Transform ranges must be ordered, finite and positive")
        if self.weights not in {"imagenet_v1", "random"}:
            raise ValueError("weights must be imagenet_v1 or explicit random smoke weights")


def seed_runtime(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{key}:{array.dtype}:{array.shape}:".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class UpstreamBackend:
    """Reuse frozen upstream embedding/heads, never its train/evaluate methods."""

    def __init__(self, config: PilotConfig, device: torch.device):
        import torchvision

        upstream = ROOT / "third_party/SimpleNet"
        sys.path.insert(0, str(upstream))
        from simplenet import SimpleNet

        weights = (
            torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            if config.weights == "imagenet_v1"
            else None
        )
        self.encoder = torchvision.models.wide_resnet50_2(weights=weights).eval()
        self.encoder.requires_grad_(False)
        self.model = SimpleNet(device)
        self.model.load(
            backbone=self.encoder,
            layers_to_extract_from=["layer2", "layer3"],
            device=device,
            input_shape=(3, *config.image_size),
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536,
            pre_proj=1,
            dsc_layers=2,
            dsc_hidden=1024,
            patchsize=3,
            train_backbone=False,
            meta_epochs=160,
            aed_meta_epochs=0,
        )
        self.adapter = self.model.pre_projection
        self.discriminator = self.model.discriminator
        self.encoder_hash = state_hash(self.encoder.state_dict())

    def embed(self, normalized):
        return self.model._embed(normalized, evaluation=True)


class BatchOrder:
    def __init__(self, count: int, batch_size: int, seed: int):
        if count < 1:
            raise ValueError("Empty training data")
        self.count, self.batch_size = count, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.cursor, self.epoch = None, 0, -1

    def next(self) -> list[int]:
        if self.order is None or self.cursor == self.count:
            self.order = torch.randperm(self.count, generator=self.generator)
            self.cursor, self.epoch = 0, self.epoch + 1
        end = min(self.cursor + self.batch_size, self.count)
        indices = self.order[self.cursor : end].tolist()
        self.cursor = end
        return indices

    def state_dict(self):
        return {
            "count": self.count,
            "batch_size": self.batch_size,
            "order": self.order,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "rng": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        if (state["count"], state["batch_size"]) != (self.count, self.batch_size):
            raise ValueError("Checkpoint batch order does not match the data")
        self.order = state["order"].cpu() if state["order"] is not None else None
        self.cursor, self.epoch = state["cursor"], state["epoch"]
        self.generator.set_state(state["rng"].cpu())


class PilotTrainer:
    def __init__(
        self, config: PilotConfig, train_data, device="cpu", backend_factory=UpstreamBackend
    ):
        self.config, self.data, self.device = config, train_data, torch.device(device)
        if (
            train_data.dataset,
            train_data.category,
            train_data.split,
            train_data.manifest_sha256,
            train_data.output_size,
        ) != (config.dataset, config.category, "train", config.manifest_sha256, config.image_size):
            raise ValueError("Training source does not match the pinned configuration")
        seed_runtime(config.seed)
        self.backend = backend_factory(config, self.device)
        self.order = BatchOrder(len(train_data), config.batch_size, config.seed + 1)
        self.transform_rng = torch.Generator().manual_seed(config.seed + 2)
        self.noise_rng = torch.Generator().manual_seed(config.seed + 3)
        self.adapter_opt = torch.optim.AdamW(
            self.backend.adapter.parameters(), lr=config.adapter_lr, weight_decay=0.01
        )
        self.discriminator_opt = torch.optim.Adam(
            self.backend.discriminator.parameters(), lr=config.discriminator_lr, weight_decay=1e-5
        )
        self.step = 0

    def continuation_contract(self):
        paths = [
            Path(__file__),
            ROOT / "src/lc_simplenet/losses.py",
            ROOT / "src/lc_simplenet/data.py",
            ROOT / "third_party/SimpleNet/simplenet.py",
            ROOT / "third_party/SimpleNet/common.py",
        ]
        return {
            "sources": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "python": sys.version,
            "threads": torch.get_num_threads(),
            "device": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else "cpu",
        }

    def normalize(self, pixels):
        mean = pixels.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = pixels.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        return (pixels - mean) / std

    def paired_views(self, pixels):
        count = len(pixels)
        # Draw even for B0; its identity view does not shift any other RNG stream.
        draws = torch.rand((count, 2), generator=self.transform_rng).to(self.device)
        brightness = self.config.brightness[0] + draws[:, 0] * (
            self.config.brightness[1] - self.config.brightness[0]
        )
        gamma = self.config.gamma[0] + draws[:, 1] * (self.config.gamma[1] - self.config.gamma[0])
        transformed = brightness[:, None, None, None] * pixels.pow(gamma[:, None, None, None])
        newly_clipped = ((transformed > 1) & (pixels < 1)).float().mean().item()
        relit = pixels if self.config.variant == "B0" else transformed.clamp(0, 1)
        return torch.cat([pixels, relit]), newly_clipped

    def train_step(self):
        indices = self.order.next()
        samples = [self.data[index] for index in indices]
        pixels = torch.from_numpy(np.stack([sample["image"] for sample in samples]))
        pixels = pixels.permute(0, 3, 1, 2).contiguous().to(self.device)
        self.backend.adapter.train()
        self.backend.discriminator.train()
        self.adapter_opt.zero_grad(set_to_none=True)
        self.discriminator_opt.zero_grad(set_to_none=True)
        views, clipped = self.paired_views(pixels)
        features, _ = self.backend.embed(self.normalize(views))
        features = self.backend.adapter(features)
        # Independent per-view Gaussian perturbations after the adapter, as upstream.
        noise = torch.randn(features.shape, generator=self.noise_rng) * self.config.noise_std
        synthetic = features + noise.to(self.device)
        # Joint BN composition: normal original/relit, then synthetic original/relit.
        scores = self.backend.discriminator(torch.cat([features, synthetic]))
        normal, fake = scores.split(len(features))
        original, relit = normal.chunk(2)
        normal_hinge = F.relu(self.config.margin - normal).mean()
        fake_hinge = F.relu(self.config.margin + fake).mean()
        consistency = patch_score_consistency(original, relit)
        total = normal_hinge + fake_hinge + self.config.lambda_score * consistency
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite training objective")
        total.backward()
        for module in (self.backend.adapter, self.backend.discriminator):
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()
            ):
                raise FloatingPointError("Non-finite trainable gradient")
        if any(p.grad is not None for p in self.backend.encoder.parameters()):
            raise RuntimeError("Frozen encoder received a gradient")
        self.adapter_opt.step()
        self.discriminator_opt.step()
        self.step += 1
        return {
            "step": self.step,
            "epoch": self.order.epoch,
            "image_ids": [sample["image_id"] for sample in samples],
            "loss": total.item(),
            "normal_hinge": normal_hinge.item(),
            "synthetic_hinge": fake_hinge.item(),
            "consistency": consistency.item(),
            "normal_score_mean": normal.detach().mean().item(),
            "normal_score_std": normal.detach().std(unbiased=False).item(),
            "synthetic_score_mean": fake.detach().mean().item(),
            "newly_clipped_fraction": clipped if self.config.variant != "B0" else 0.0,
        }

    def predict(self, images):
        self.backend.adapter.eval()
        self.backend.discriminator.eval()
        pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().to(self.device)
        with torch.inference_mode():
            features, grids = self.backend.embed(self.normalize(pixels))
            scores = -self.backend.discriminator(self.backend.adapter(features))
            maps = scores.reshape(len(images), *grids[0])
        return maps.cpu().numpy()

    def checkpoint(self):
        numpy_state = np.random.get_state()
        return {
            "schema": 1,
            "config": asdict(self.config),
            "step": self.step,
            "continuation_contract": self.continuation_contract(),
            "device_type": self.device.type,
            "encoder_hash": self.backend.encoder_hash,
            "adapter": self.backend.adapter.state_dict(),
            "discriminator": self.backend.discriminator.state_dict(),
            "adapter_optimizer": self.adapter_opt.state_dict(),
            "discriminator_optimizer": self.discriminator_opt.state_dict(),
            "order": self.order.state_dict(),
            "transform_rng": self.transform_rng.get_state(),
            "noise_rng": self.noise_rng.get_state(),
            "python_rng": random.getstate(),
            "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        }

    def save(self, path: Path):
        if path.exists() or path.with_suffix(path.suffix + ".part").exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
        temporary = path.with_suffix(path.suffix + ".part")
        torch.save(self.checkpoint(), temporary)
        temporary.replace(path)

    def load(self, path: Path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state["schema"] != 1 or state["config"] != asdict(self.config):
            raise ValueError("Checkpoint configuration mismatch")
        if state["device_type"] != self.device.type:
            raise ValueError("Exact continuation requires the same device type")
        if state["continuation_contract"] != self.continuation_contract():
            raise ValueError("Continuation requires matching sources and runtime")
        if state["encoder_hash"] != self.backend.encoder_hash:
            raise ValueError("Frozen encoder differs from the checkpoint")
        self.backend.adapter.load_state_dict(state["adapter"])
        self.backend.discriminator.load_state_dict(state["discriminator"])
        self.adapter_opt.load_state_dict(state["adapter_optimizer"])
        self.discriminator_opt.load_state_dict(state["discriminator_optimizer"])
        self.order.load_state_dict(state["order"])
        self.transform_rng.set_state(state["transform_rng"].cpu())
        self.noise_rng.set_state(state["noise_rng"].cpu())
        random.setstate(state["python_rng"])
        numpy_state = state["numpy_rng"]
        np.random.set_state(
            (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
        )
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        self.step = state["step"]

    def trainable_hash(self):
        return state_hash(
            {
                **{f"adapter.{k}": v for k, v in self.backend.adapter.state_dict().items()},
                **{
                    f"discriminator.{k}": v
                    for k, v in self.backend.discriminator.state_dict().items()
                },
            }
        )


def config_hash(config: PilotConfig):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def assert_same_state(left, right):
    """Exact recursive equality, including optimizers, sampler, BN and RNGs."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("State dictionary keys differ")
        for key in left:
            assert_same_state(left[key], right[key])
    elif isinstance(left, tuple | list):
        if len(left) != len(right):
            raise AssertionError("State sequence lengths differ")
        for first, second in zip(left, right, strict=True):
            assert_same_state(first, second)
    else:
        if left != right:
            raise AssertionError(f"State values differ: {left!r} != {right!r}")
