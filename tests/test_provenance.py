import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lc_simplenet.provenance import (  # noqa: E402
    capture_rng_state,
    restore_rng_state,
    runtime_record,
)


def test_rng_snapshot_is_weights_only_safe_and_replays_mixed_draws(tmp_path):
    saved = capture_rng_state()
    expected = (random.random(), np.random.standard_normal(7), torch.rand(7))
    path = tmp_path / "rng.pt"
    torch.save(saved, path)
    random.seed(55)
    np.random.seed(55)
    torch.manual_seed(55)
    restore_rng_state(torch.load(path, map_location="cpu", weights_only=True))
    assert random.random() == expected[0]
    np.testing.assert_array_equal(np.random.standard_normal(7), expected[1])
    torch.testing.assert_close(torch.rand(7), expected[2], atol=0, rtol=0)


def test_runtime_record_does_not_consume_rng_or_change_numerical_flags():
    before = capture_rng_state()
    flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
    record = runtime_record(torch.device("cpu"))
    after = capture_rng_state()
    assert before["python_rng"] == after["python_rng"]
    assert before["numpy_rng"] == after["numpy_rng"]
    torch.testing.assert_close(before["torch_rng"], after["torch_rng"], atol=0, rtol=0)
    assert flags == (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
    assert record["device"] == "cpu"
    assert record["packages"]["torch"] == str(torch.__version__)
