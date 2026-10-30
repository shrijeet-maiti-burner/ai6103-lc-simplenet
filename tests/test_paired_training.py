import numpy as np
import pytest

torch = pytest.importorskip("torch")
from lc_simplenet.paired_training import (  # noqa: E402
    normalized_pair,
    paired_discriminator_loss,
)
from lc_simplenet.pairs import make_pair  # noqa: E402


def test_normalization_follows_relighting_and_keeps_alignment():
    image = np.full((3, 4, 3), 0.5, dtype=np.float32)
    pair = make_pair(image, brightness=0.8, gamma=1)
    a, b = normalized_pair(pair)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    assert a.shape == b.shape == (3, 3, 4)
    torch.testing.assert_close(a * std + mean, torch.full_like(a, 0.5))
    torch.testing.assert_close(b * std + mean, torch.full_like(b, 0.4))


def test_joint_forward_order_and_diagnostics():
    class Recorder(torch.nn.Module):
        def forward(self, x):
            self.calls = getattr(self, "calls", 0) + 1
            self.seen = x.detach().clone()
            return x.sum(dim=1, keepdim=True)

    d = Recorder()
    a = torch.ones(2, 3, requires_grad=True)
    b = torch.full((2, 3), 2.0, requires_grad=True)
    noise_a, noise_b = torch.full_like(a, 0.1), torch.full_like(b, 0.2)
    terms, stats = paired_discriminator_loss(
        d, a, b, noise_a, noise_b, margin=0.5, lambda_score=0.1
    )
    assert d.calls == 1
    torch.testing.assert_close(d.seen, torch.cat([a, b, a + noise_a, b + noise_b]))
    assert all(not value.requires_grad for value in stats.values())
    assert stats["normal_synthetic_gap"].item() == pytest.approx(-0.45)
    terms["total"].backward()
    assert a.grad is not None and b.grad is not None


def test_invalid_feature_shape_fails_before_forward():
    a = torch.ones(2, 3)
    with pytest.raises(ValueError):
        paired_discriminator_loss(
            torch.nn.Linear(3, 1),
            a,
            a,
            torch.ones(1, 3),
            a,
            margin=0.5,
            lambda_score=0,
        )


def test_b1_b2_match_batchnorm_and_scores_with_only_loss_weight_changed():
    import copy

    torch.manual_seed(10)
    b1 = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.LeakyReLU(), torch.nn.Linear(4, 1)
    )
    b2 = copy.deepcopy(b1)
    a, b, na, nb = [torch.randn(8, 3) for _ in range(4)]
    first, stats1 = paired_discriminator_loss(b1, a, b, na, nb, margin=0.5, lambda_score=0)
    second, stats2 = paired_discriminator_loss(b2, a, b, na, nb, margin=0.5, lambda_score=0.2)
    torch.testing.assert_close(second["total"] - first["total"], 0.2 * second["consistency"])
    for key in stats1:
        torch.testing.assert_close(stats1[key], stats2[key], rtol=0, atol=0)
    torch.testing.assert_close(b1[1].running_mean, b2[1].running_mean, rtol=0, atol=0)
    torch.testing.assert_close(b1[1].running_var, b2[1].running_var, rtol=0, atol=0)


@pytest.mark.parametrize("margin,weight", [(0, 0.1), (float("nan"), 0.1), (0.5, -1)])
def test_invalid_settings_do_not_mutate_batchnorm(margin, weight):
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 1))
    before = {key: value.clone() for key, value in model.named_buffers()}
    values = [torch.ones(2, 3) for _ in range(4)]
    with pytest.raises(ValueError):
        paired_discriminator_loss(model, *values, margin=margin, lambda_score=weight)
    for key, value in model.named_buffers():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4, 3), dtype=np.float32),
        np.full((3, 4, 3), 1.1, dtype=np.float32),
        np.full((3, 4, 3), np.nan, dtype=np.float32),
        np.zeros((3, 4, 3), dtype=np.uint8),
    ],
)
def test_normalization_rejects_invalid_public_pair(bad):
    from lc_simplenet.pairs import LightingPair

    pair = LightingPair(np.zeros((3, 4, 3), dtype=np.float32), bad, 1, 1, 0)
    with pytest.raises(ValueError):
        normalized_pair(pair)
