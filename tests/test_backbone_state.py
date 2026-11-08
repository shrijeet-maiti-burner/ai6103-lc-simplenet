import copy

import pytest

torch = pytest.importorskip("torch")

from lc_simplenet.backbone_state import capture_backbone_state, restore_backbone_state  # noqa: E402

WEIGHT = "a" * 64
SHAPE = (3, 32, 32)


def model():
    torch.manual_seed(84)
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))


def capture(network):
    return capture_backbone_state(network, image_shape=SHAPE, pretrained_sha256=WEIGHT)


def restore(network, snapshot):
    return restore_backbone_state(network, snapshot, image_shape=SHAPE, pretrained_sha256=WEIGHT)


def test_different_shape_probes_need_buffers_and_restore_predictions(tmp_path):
    first, second = model(), model()
    first(torch.ones(6, 4))
    second(torch.ones(6, 4) * 7)
    first.eval()
    second.eval()
    image = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert not torch.equal(first(image), second(image))
    snapshot = capture(first)
    torch.save(snapshot, tmp_path / "buffers.pt")
    loaded = torch.load(tmp_path / "buffers.pt", weights_only=True)
    restore(second, loaded)
    assert torch.equal(first(image), second(image))
    assert not first.training and not second.training


def test_capture_does_not_alias_live_cpu_buffers():
    network = model()
    snapshot = capture(network)
    before = {k: v.clone() for k, v in snapshot["buffers"].items()}
    network(torch.ones(6, 4) * 8)
    assert all(torch.equal(v, before[k]) for k, v in snapshot["buffers"].items())


@pytest.mark.parametrize(
    "damage",
    ["schema", "geometry", "weight", "parameter", "shape", "dtype", "nan", "bytes", "fingerprint"],
)
def test_invalid_restore_rejected_before_any_mutation(damage):
    network = model()
    snapshot = capture(network)
    before = copy.deepcopy(network.state_dict())
    key = "1.running_mean"
    if damage == "schema":
        snapshot["schema"] = 99
    elif damage == "geometry":
        snapshot["image_shape"] = [3, 64, 64]
    elif damage == "weight":
        snapshot["pretrained_sha256"] = "b" * 64
    elif damage == "parameter":
        snapshot["parameters_sha256"] = "b" * 64
    elif damage == "shape":
        snapshot["buffers"][key] = torch.zeros(3)
    elif damage == "dtype":
        snapshot["buffers"][key] = snapshot["buffers"][key].double()
    elif damage == "nan":
        snapshot["buffers"][key][0] = float("nan")
    elif damage == "bytes":
        snapshot["buffers"][key][0] += 1
    else:
        snapshot["backbone_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        restore(network, snapshot)
    assert all(torch.equal(value, before[k]) for k, value in network.state_dict().items())
