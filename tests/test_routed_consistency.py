"""gradient routing and state isolation for the auxiliary classifier forward."""

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from lc_simplenet.routed_consistency import adapter_only_score_consistency  # noqa: E402


def classifier():
    return nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.LeakyReLU(0.2), nn.Linear(8, 1))


def test_value_and_adapter_gradient_match_without_classifier_gradients():
    torch.manual_seed(17)
    adapter, discriminator = nn.Linear(4, 4), classifier().eval()
    features = adapter(torch.randn(24, 4))
    score = discriminator(features)
    reference = F.mse_loss(score[:6], score[6:12])
    routed = adapter_only_score_consistency(discriminator, features, patches_per_view=6)
    assert torch.equal(reference, routed)
    expected = torch.autograd.grad(reference, tuple(adapter.parameters()), retain_graph=True)
    actual = torch.autograd.grad(routed, tuple(adapter.parameters()), retain_graph=True)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert all(torch.isfinite(g).all() for g in actual)
    assert sum(g.abs().sum().item() for g in actual) > 0
    assert all(
        g is None
        for g in torch.autograd.grad(
            routed, tuple(discriminator.parameters()), allow_unused=True, retain_graph=True
        )
    )
    gradient = torch.autograd.grad(routed, features)[0]
    assert torch.count_nonzero(gradient[12:]) == 0


def test_classifier_update_matches_classification_only_and_bn_updates_once():
    torch.manual_seed(23)
    reference, actual = classifier(), None
    actual = copy.deepcopy(reference)
    features = torch.randn(24, 4, requires_grad=True)
    ref_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(actual.parameters(), lr=0.001)
    reference(features).square().mean().backward()
    classification = actual(features).square().mean()
    before = {k: v.clone() for k, v in actual.named_buffers()}
    consistency = adapter_only_score_consistency(actual, features, patches_per_view=6)
    (classification + 0.1 * consistency).backward()
    assert actual.training and all(m.training for m in actual.modules())
    assert actual[1].num_batches_tracked.item() == 1
    assert all(torch.equal(before[k], v) for k, v in actual.named_buffers())
    assert all(
        torch.equal(a.grad, b.grad)
        for a, b in zip(actual.parameters(), reference.parameters(), strict=True)
    )
    optimizer.step()
    ref_optimizer.step()
    assert all(torch.equal(actual.state_dict()[k], v) for k, v in reference.state_dict().items())


def test_buffers_and_mixed_modes_survive_a_forward_error():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Dropout()
            self.register_buffer("calls", torch.tensor(0))

        def forward(self, features):
            self.calls.add_(1)
            raise RuntimeError("deliberate forward failure")

    module = Broken().train()
    module.child.eval()
    with pytest.raises(RuntimeError, match="deliberate forward failure"):
        adapter_only_score_consistency(module, torch.ones(8, 4), patches_per_view=2)
    assert module.training and not module.child.training
    assert module.calls.item() == 0
