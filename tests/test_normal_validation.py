from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lc_simplenet.normal_validation import paired_normal_metrics  # noqa: E402
from lc_simplenet.pilot import PilotConfig, PilotTrainer, state_hash  # noqa: E402


class Data:
    dataset, category, manifest_sha256 = "mvtec_ad", "bottle", "a" * 64
    split, output_size = "train", (32, 32)
    records = (SimpleNamespace(width=32, height=32),) * 2

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "image": np.random.default_rng(index).uniform(0.1, 0.8, (32, 32, 3)).astype(np.float32),
            "image_id": f"bottle/train/good/{index:03}.png",
        }


class Backend:
    def __init__(self, config, device):
        self.encoder = torch.nn.Conv2d(3, 4, 1).to(device).eval().requires_grad_(False)
        self.adapter = torch.nn.Linear(4, 4).to(device)
        self.discriminator = torch.nn.Linear(4, 1).to(device)
        self.encoder_hash = state_hash(self.encoder.state_dict())

    def embed(self, image):
        feature = self.encoder(torch.nn.functional.adaptive_avg_pool2d(image, (2, 2)))
        return feature.permute(0, 2, 3, 1).reshape(-1, 4), [(2, 2)]


def test_normal_diagnostics_preserve_training_rng_and_frozen_encoder():
    config = PilotConfig("mvtec_ad", "bottle", "a" * 64, image_size=(32, 32), variant="B1")
    engine = PilotTrainer(config, Data(), backend_factory=Backend)
    validation = Data()
    validation.split = "validation"
    transform_rng = engine.transform_rng.get_state().clone()
    noise_rng = engine.noise_rng.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    result = paired_normal_metrics(engine, validation)
    assert result["images"] == 2 and len(result["lighting_corners"]) == 4
    assert result["pair_mse"] > 0 and result["patch_score_std"] > 0
    assert state_hash(engine.backend.encoder.state_dict()) == engine.backend.encoder_hash
    for before, after in (
        (transform_rng, engine.transform_rng.get_state()),
        (noise_rng, engine.noise_rng.get_state()),
        (global_rng, torch.get_rng_state()),
    ):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    validation.split = "train"
    with pytest.raises(ValueError, match="normal-validation"):
        paired_normal_metrics(engine, validation)
