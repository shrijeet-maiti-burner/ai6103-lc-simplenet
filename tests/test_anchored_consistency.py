import copy

import pytest

torch = pytest.importorskip("torch")

from lc_simplenet.anchored_consistency import (  # noqa: E402
    adaptation_mode,
    anchored_forward,
    anchored_objective,
    frozen_teacher,
)
from lc_simplenet.paired_losses import paired_objective  # noqa: E402


def networks():
    torch.manual_seed(61)
    adapter = torch.nn.Linear(4, 4).double()
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(4, 6),
        torch.nn.BatchNorm1d(6),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(6, 1),
    ).double()
    return adapter, discriminator


def test_zero_anchor_matches_original_optimizer_and_batchnorm_state():
    adapter, disc = networks()
    other_adapter, other_disc = copy.deepcopy(adapter), copy.deepcopy(disc)
    teacher_a, teacher_d = frozen_teacher(adapter, disc)
    old_teacher = copy.deepcopy((teacher_a.state_dict(), teacher_d.state_dict()))
    first = torch.optim.AdamW([*adapter.parameters(), *disc.parameters()], lr=0.001)
    second = torch.optim.AdamW([*other_adapter.parameters(), *other_disc.parameters()], lr=0.001)
    rng = torch.Generator().manual_seed(84)
    for _ in range(4):
        clean = torch.randn((5, 4), generator=rng, dtype=torch.float64)
        light = clean * 0.9
        noise = torch.randn((10, 4), generator=rng, dtype=torch.float64) * 0.015
        adaptation_mode(adapter, disc)
        adaptation_mode(other_adapter, other_disc)
        first.zero_grad(set_to_none=True)
        second.zero_grad(set_to_none=True)
        terms = anchored_forward(
            adapter,
            disc,
            teacher_a,
            teacher_d,
            clean,
            light,
            noise,
            margin=0.5,
            lambda_score=0.1,
            lambda_anchor=0,
        )
        z = other_adapter(torch.cat((clean, light)))
        scores = other_disc(torch.cat((z, z + noise))).chunk(4)
        original = paired_objective(*scores, margin=0.5, lambda_score=0.1)
        torch.testing.assert_close(terms["total"], original["total"], atol=0, rtol=0)
        terms["total"].backward()
        original["total"].backward()
        first.step()
        second.step()
    for one, two in ((adapter, other_adapter), (disc, other_disc)):
        for name, tensor in one.state_dict().items():
            torch.testing.assert_close(tensor, two.state_dict()[name], atol=0, rtol=0)
    for original, teacher in zip(old_teacher, (teacher_a, teacher_d), strict=True):
        for name, tensor in teacher.state_dict().items():
            torch.testing.assert_close(tensor, original[name], atol=0, rtol=0)
        assert all(p.grad is None for p in teacher.parameters())


def test_anchor_gradients_only_reach_clean_student_scores():
    values = [
        torch.tensor([[v]], dtype=torch.float64, requires_grad=True)
        for v in (0.2, 0.4, -0.1, -0.3, 0.7, -0.5)
    ]
    terms = anchored_objective(*values, margin=0.5, lambda_score=0.1, lambda_anchor=1)
    gradients = torch.autograd.grad(terms["anchor"], values, allow_unused=True)
    assert gradients[1] is None and gradients[3] is None
    assert gradients[4] is None and gradients[5] is None
    torch.testing.assert_close(gradients[0], torch.tensor([[-0.5]], dtype=torch.float64))
    torch.testing.assert_close(gradients[2], torch.tensor([[0.4]], dtype=torch.float64))


def test_full_objective_gradient_matches_finite_differences():
    point = torch.tensor([0.21, 0.37, -0.12, -0.31], dtype=torch.float64, requires_grad=True)
    teacher = (
        torch.tensor([0.72], dtype=torch.float64),
        torch.tensor([-0.61], dtype=torch.float64),
    )

    def objective(values):
        return anchored_objective(
            *[x.reshape(1) for x in values],
            *teacher,
            margin=0.5,
            lambda_score=0.1,
            lambda_anchor=0.7,
        )["total"]

    actual = torch.autograd.grad(objective(point), point)[0]
    numeric = []
    for index in range(4):
        delta = torch.zeros(4, dtype=torch.float64)
        delta[index] = 1e-6
        numeric.append(
            (objective(point.detach() + delta) - objective(point.detach() - delta)) / 2e-6
        )
    torch.testing.assert_close(actual, torch.stack(numeric), atol=1e-9, rtol=0)


def test_teacher_copy_is_independent_and_anchor_starts_at_zero():
    adapter, disc = networks()
    teacher_a, teacher_d = frozen_teacher(adapter, disc)
    assert adapter.training and disc.training
    adaptation_mode(adapter, disc)
    clean = torch.randn(5, 4, dtype=torch.float64)
    terms = anchored_forward(
        adapter,
        disc,
        teacher_a,
        teacher_d,
        clean,
        clean * 0.9,
        torch.zeros(10, 4, dtype=torch.float64),
        margin=0.5,
        lambda_score=0.1,
        lambda_anchor=1,
    )
    assert terms["anchor"].item() == 0
    for p, q in zip(adapter.parameters(), teacher_a.parameters(), strict=True):
        assert p.data_ptr() != q.data_ptr()
    terms["total"].backward()
    assert any(p.grad is not None for p in adapter.parameters())
    assert any(p.grad is not None for p in disc.parameters())
    assert all(p.grad is None for p in [*teacher_a.parameters(), *teacher_d.parameters()])


def test_unfrozen_teacher_rejected_before_student_bn_changes():
    adapter, disc = networks()
    teacher_a, teacher_d = frozen_teacher(adapter, disc)
    teacher_d[1].train()
    before = copy.deepcopy(disc.state_dict())
    with pytest.raises(ValueError, match="forward modes"):
        anchored_forward(
            adapter,
            disc,
            teacher_a,
            teacher_d,
            torch.ones(2, 4, dtype=torch.float64),
            torch.ones(2, 4, dtype=torch.float64),
            torch.zeros(4, 4, dtype=torch.float64),
            margin=0.5,
            lambda_score=0.1,
            lambda_anchor=1,
        )
    for name, value in disc.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("coefficient", [-1, float("nan"), float("inf")])
def test_invalid_anchor_coefficient_rejected(coefficient):
    values = [torch.ones(2, 1) for _ in range(6)]
    with pytest.raises(ValueError, match="lambda_anchor"):
        anchored_objective(*values, margin=0.5, lambda_score=0.1, lambda_anchor=coefficient)


def test_teacher_shape_cannot_broadcast():
    values = [torch.ones(2, 1) for _ in range(5)] + [torch.ones(1)]
    with pytest.raises(ValueError, match="teacher scores"):
        anchored_objective(*values, margin=0.5, lambda_score=0.1, lambda_anchor=1)


def test_inference_mode_teacher_targets_are_valid_detached_targets():
    with torch.inference_mode():
        teacher = [torch.ones(2, 1) for _ in range(2)]
    values = [torch.zeros(2, 1, requires_grad=True) for _ in range(4)]
    anchored_objective(*values, *teacher, margin=0.5, lambda_score=0.1, lambda_anchor=1)[
        "total"
    ].backward()
    assert all(value.grad is not None for value in values)
