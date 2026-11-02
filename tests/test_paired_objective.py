import pytest

torch = pytest.importorskip("torch")
from lc_simplenet.paired_losses import paired_objective, patch_score_consistency  # noqa: E402


def test_identical_views_have_zero_consistency():
    x = torch.tensor([0.2, 0.4], requires_grad=True)
    loss = patch_score_consistency(x, x.clone())
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))


def test_lambda_zero_matches_independent_two_view_hinge_and_gradients():
    scores = [torch.tensor([0.1, -0.2], requires_grad=True) for _ in range(4)]
    terms = paired_objective(*scores, margin=0.5, lambda_score=0)
    expected = (
        sum(
            torch.clamp(0.5 + sign * score, min=0).mean()
            for sign, score in zip([-1, -1, 1, 1], scores, strict=True)
        )
        / 2
    )
    torch.testing.assert_close(terms["total"], expected)
    actual_grad = torch.autograd.grad(terms["total"], scores, retain_graph=True)
    reference_grad = torch.autograd.grad(expected, scores)
    for actual, reference in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(actual, reference)


def test_consistency_does_not_change_synthetic_score_gradients():
    scores = [torch.tensor([0.1 * i, -0.2], requires_grad=True) for i in range(4)]
    zero = paired_objective(*scores, margin=0.5, lambda_score=0)
    positive = paired_objective(*scores, margin=0.5, lambda_score=2)
    before = torch.autograd.grad(zero["total"], scores, retain_graph=True)
    after = torch.autograd.grad(positive["total"], scores)
    for i in [2, 3]:
        torch.testing.assert_close(before[i], after[i])
    assert not torch.equal(before[0], after[0])
    assert not torch.equal(before[1], after[1])


def test_joint_forward_reaches_adapter_discriminator_but_not_frozen_extractor():
    torch.manual_seed(2)
    extractor = torch.nn.Linear(3, 4).requires_grad_(False).eval()
    adapter = torch.nn.Linear(4, 4)
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        torch.nn.BatchNorm1d(5),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(5, 1),
    )
    with torch.no_grad():
        x = extractor(torch.randn(6, 3))
        y = extractor(torch.randn(6, 3))
    a, b = adapter(x), adapter(y)
    joint = torch.cat([a, b, a + 0.1, b + 0.1])
    scores = discriminator(joint).chunk(4)
    terms = paired_objective(*scores, margin=0.5, lambda_score=0.1)
    terms["total"].backward()
    assert all(p.grad is None for p in extractor.parameters())
    for module in [adapter, discriminator]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum().item() for p in module.parameters()) > 0


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_weight_is_rejected(weight):
    scores = [torch.ones(2) for _ in range(4)]
    with pytest.raises(ValueError):
        paired_objective(*scores, margin=0.5, lambda_score=weight)


def test_shape_mismatch_rejected_before_broadcast():
    with pytest.raises(ValueError):
        paired_objective(
            torch.ones(2),
            torch.ones(2),
            torch.ones(2, 1),
            torch.ones(2),
            margin=0.5,
            lambda_score=1,
        )


@pytest.mark.parametrize("weight", [0.0, 0.1])
def test_half_precision_loss_reductions_remain_finite(weight):
    a = torch.tensor([300.0], dtype=torch.float16, requires_grad=True)
    b = torch.tensor([-300.0], dtype=torch.float16, requires_grad=True)
    fake = torch.zeros(1, dtype=torch.float16, requires_grad=True)
    terms = paired_objective(a, b, fake, fake, margin=0.5, lambda_score=weight)
    assert terms["consistency"].dtype == torch.float32
    assert terms["consistency"].item() == 360000
    assert all(torch.isfinite(value).all() for value in terms.values())
    if weight == 0:
        torch.testing.assert_close(terms["total"], terms["discrimination"], rtol=0, atol=0)
    terms["total"].backward()
    assert all(torch.isfinite(value.grad).all() for value in (a, b, fake))
