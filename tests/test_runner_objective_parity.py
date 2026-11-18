"""Compare the reference objective with the paired runner's complete update."""

import pytest

torch = pytest.importorskip("torch")

from test_pilot import config, trainer  # noqa: E402

from lc_simplenet.paired_training import paired_discriminator_loss  # noqa: E402
from lc_simplenet.pilot import assert_same_state  # noqa: E402


def helper_step(model):
    indices = model.order.next()
    samples = [model.data[index] for index in indices]
    pixels = model.prepare_pixels([sample["image"] for sample in samples])
    model.backend.adapter.train()
    model.backend.discriminator.train()
    model.adapter_opt.zero_grad(set_to_none=True)
    model.discriminator_opt.zero_grad(set_to_none=True)
    views, _ = model.paired_views(pixels)
    features, _ = model.backend.embed(model.normalize(views))
    original, relit = model.backend.adapter(features).chunk(2)
    noise = torch.randn((2 * len(original), original.shape[1]), generator=model.noise_rng)
    first_noise, second_noise = (noise * model.config.noise_std).chunk(2)
    terms, _ = paired_discriminator_loss(
        model.backend.discriminator,
        original,
        relit,
        first_noise,
        second_noise,
        margin=model.config.margin,
        lambda_score=model.config.lambda_score,
    )
    terms["total"].backward()
    model.adapter_opt.step()
    model.discriminator_opt.step()
    model.step += 1
    return terms


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize(
    "variant,weight", [("B0", 0), ("B1", 0), ("B2", 0), ("B2", 0.1), ("B2", 1)]
)
def test_helper_matches_actual_runner_losses_gradients_bn_updates_and_random_streams(
    seed, variant, weight
):
    torch.set_num_threads(2)
    settings = config(variant=variant, lambda_score=weight, seed=seed)
    actual, helper = trainer(settings), trainer(settings)
    # Includes a partial final batch and a new epoch permutation.
    for _ in range(3):
        row, terms = actual.train_step(), helper_step(helper)
        assert row["loss"] == pytest.approx(terms["total"].item(), rel=1e-6, abs=1e-7)
        assert row["consistency"] == pytest.approx(terms["consistency"].item(), rel=1e-6, abs=1e-7)
        assert row["normal_hinge"] + row["synthetic_hinge"] == pytest.approx(
            terms["discrimination"].item(), rel=1e-6, abs=1e-7
        )
        # Different float32 reduction order permits bounded rounding differences.
        for name in ("adapter", "discriminator"):
            first, second = getattr(actual.backend, name), getattr(helper.backend, name)
            torch.testing.assert_close(
                first.state_dict(), second.state_dict(), rtol=1e-5, atol=1e-7
            )
            for a, b in zip(first.parameters(), second.parameters(), strict=True):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-7)
        for name in ("adapter_opt", "discriminator_opt"):
            torch.testing.assert_close(
                getattr(actual, name).state_dict(),
                getattr(helper, name).state_dict(),
                rtol=1e-4,
                atol=1e-7,
            )
        assert_same_state(actual.order.state_dict(), helper.order.state_dict())
        for name in ("transform_rng", "noise_rng"):
            assert_same_state(getattr(actual, name).get_state(), getattr(helper, name).get_state())
        assert all(p.grad is None for p in actual.backend.encoder.parameters())
        assert all(p.grad is None for p in helper.backend.encoder.parameters())
