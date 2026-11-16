"""synthetic mining isolation and the complete consistency gradient path."""

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from lc_simplenet.local_global_losses import (  # noqa: E402
    local_focal_loss,
    local_global_objective,
    mine_feature_noise,
)


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        yield


def heads():
    return nn.Linear(4, 4), nn.Sequential(
        nn.Linear(4, 8), nn.BatchNorm1d(8), nn.LeakyReLU(0.2), nn.Linear(8, 1)
    )


def test_mining_changes_only_offsets_and_preserves_classifier_state():
    _, discriminator = heads()
    discriminator[2].eval()
    modes = [m.training for m in discriminator.modules()]
    saved = copy.deepcopy(discriminator.state_dict())
    features = torch.randn(12, 4, requires_grad=True)
    before = features.detach().clone()
    noise = 0.015 * torch.randn_like(features)
    delta = mine_feature_noise(discriminator, features, noise)
    assert not delta.requires_grad
    assert torch.equal(features, before)
    assert not torch.equal(delta, noise)
    assert torch.all(delta.norm(dim=1) >= 0.5 - 1e-6)
    assert torch.all(delta.norm(dim=1) <= 1.0 + 1e-6)
    assert features.grad is None and all(p.grad is None for p in discriminator.parameters())
    assert [m.training for m in discriminator.modules()] == modes
    assert all(torch.equal(discriminator.state_dict()[k], v) for k, v in saved.items())
    assert torch.equal(mine_feature_noise(discriminator, features, noise, steps=0), noise)


def test_routing_preserves_values_and_classification_gradients():
    reference = heads()
    variants = [copy.deepcopy(reference) for _ in range(3)]
    inputs = [torch.randn(6, 4) for _ in range(4)]
    noise = 0.015 * torch.randn(12, 4)
    masks = torch.tensor([[True, False, True, False, True, False]] * 2)
    results = []
    for nets, weight, route in zip(variants, [0, 0.1, 0.1], [True, True, False], strict=True):
        result = local_global_objective(
            *nets, *inputs, noise, masks, lambda_score=weight, route_consistency=route
        )
        result["total"].backward()
        results.append(result)
        assert nets[1][1].num_batches_tracked.item() == 1
        assert all(m.training for net in nets for m in net.modules())
    assert all(torch.equal(results[1][k], results[2][k]) for k in results[1])
    for lhs, rhs in zip(variants[0][1].parameters(), variants[1][1].parameters(), strict=True):
        assert torch.equal(lhs.grad, rhs.grad)
    assert any(
        not torch.equal(a.grad, b.grad)
        for a, b in zip(variants[0][0].parameters(), variants[1][0].parameters(), strict=True)
    )
    assert any(
        not torch.equal(a.grad, b.grad)
        for a, b in zip(variants[1][1].parameters(), variants[2][1].parameters(), strict=True)
    )
    assert all(
        torch.isfinite(p.grad).all() for nets in variants for net in nets for p in net.parameters()
    )


def test_focal_ties_keep_both_classes_with_the_normality_sign():
    score = torch.zeros(4, 1, requires_grad=True)
    labels = torch.tensor([True, False, True, False])
    loss = local_focal_loss(score, labels)
    expected = 0.25 * F.binary_cross_entropy_with_logits(-score[:, 0], labels.float())
    assert torch.equal(loss, expected)
    loss.backward()
    assert torch.all(score.grad[labels] > 0)
    assert torch.all(score.grad[~labels] < 0)


def test_invalid_loss_weight_fails_before_classifier_state_changes():
    adapter, discriminator = heads()
    x = torch.ones(4, 4)
    with pytest.raises(ValueError, match="lambda_score"):
        local_global_objective(
            adapter,
            discriminator,
            x,
            x,
            x,
            x,
            torch.ones(8, 4),
            torch.zeros(2, 4, dtype=torch.bool),
            lambda_score=float("nan"),
        )
    assert discriminator[1].num_batches_tracked.item() == 0
