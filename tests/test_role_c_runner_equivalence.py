"""Compare the paired helper with the released runner's train_step."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from test_pilot import TinyBackend, TinyData  # noqa: E402

from lc_simplenet.paired_training import paired_discriminator_loss  # noqa: E402
from lc_simplenet.pairs import make_pair  # noqa: E402
from lc_simplenet.pilot import PilotConfig, PilotTrainer  # noqa: E402


@pytest.mark.parametrize("weight", [0.0, 0.01, 0.1, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_helper_matches_actual_train_step(weight, seed):
    torch.set_num_threads(2)
    data = TinyData()
    config = PilotConfig(
        data.dataset,
        data.category,
        data.manifest_sha256,
        variant="B1" if weight == 0 else "B2",
        lambda_score=weight,
        seed=seed,
        image_size=data.output_size,
    )
    trainer = PilotTrainer(config, data, backend_factory=TinyBackend)
    reference = copy.deepcopy(trainer.backend.discriminator)
    captured = {}

    def capture(module, args, output):
        captured["features"] = args[0].detach().clone()

    hook = trainer.backend.discriminator.register_forward_hook(capture)
    record = trainer.train_step()
    hook.remove()
    a, b, fa, fb = captured["features"].chunk(4)
    terms, _ = paired_discriminator_loss(
        reference, a, b, fa - a, fb - b, margin=config.margin, lambda_score=weight
    )
    assert terms["total"].item() == pytest.approx(record["loss"], rel=1e-6, abs=1e-6)
    assert terms["discrimination"].item() == pytest.approx(
        record["normal_hinge"] + record["synthetic_hinge"], rel=1e-6, abs=1e-6
    )
    assert terms["consistency"].item() == pytest.approx(record["consistency"], abs=1e-6)
    terms["total"].backward()
    for actual, expected in zip(
        trainer.backend.discriminator.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(
        trainer.backend.discriminator.buffers(), reference.buffers(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_same_draws_give_same_relighting_at_256():
    data = TinyData()
    data.output_size = (256, 256)
    config = PilotConfig(
        data.dataset,
        data.category,
        data.manifest_sha256,
        variant="B1",
        image_size=data.output_size,
    )
    trainer = PilotTrainer(config, data, backend_factory=TinyBackend)
    image = data[0]["image"]
    pixels = trainer.prepare_pixels([image])
    generator = torch.Generator()
    generator.set_state(trainer.transform_rng.get_state())
    draw = torch.rand((1, 2), generator=generator)
    brightness, gamma = (0.8 + draw[0] * 0.4).tolist()
    pair = make_pair(image, brightness=brightness, gamma=gamma)
    views, _ = trainer.paired_views(pixels)
    np.testing.assert_allclose(views[1].permute(1, 2, 0).numpy(), pair.relit, atol=2e-7)
