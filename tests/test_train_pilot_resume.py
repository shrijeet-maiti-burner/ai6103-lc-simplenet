import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def load_train_pilot():
    key = "ad2_scripts.train_pilot"
    if key in sys.modules:
        del sys.modules[key]
    path = ROOT / "scripts" / "train_pilot.py"
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def required_args(output: Path) -> list[str]:
    return [
        "--config",
        "config.json",
        "--manifest",
        "manifest.jsonl",
        "--root",
        "data",
        "--output",
        str(output),
        "--steps",
        "10",
    ]


class FakeTrainer:
    def __init__(self, start_step=0):
        self.step = start_step
        self.saved = []

    def train_step(self):
        self.step += 1
        return {"step": self.step}

    def save(self, path):
        self.saved.append(Path(path))


def write_log(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_truncate_train_log_keeps_records_up_to_checkpoint(tmp_path):
    train_pilot = load_train_pilot()
    path = tmp_path / "train.jsonl"
    records = [{"step": step, "loss": 1.0 / step} for step in range(1, 301)]
    write_log(path, "".join(json.dumps(record) + "\n" for record in records))
    kept = train_pilot.truncate_train_log(path, 250)
    assert [record["step"] for record in kept] == list(range(1, 251))
    restored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert restored == kept
    assert 251 not in {record["step"] for record in restored}


def test_truncate_train_log_allows_blank_lines(tmp_path):
    train_pilot = load_train_pilot()
    path = tmp_path / "train.jsonl"
    write_log(
        path,
        "\n".join(
            [
                json.dumps({"step": 1, "loss": 0.9}),
                "",
                "   ",
                json.dumps({"step": 2, "loss": 0.8}),
                "",
                json.dumps({"step": 3, "loss": 0.7}),
                "",
            ]
        )
        + "\n",
    )
    kept = train_pilot.truncate_train_log(path, 2)
    assert [record["step"] for record in kept] == [1, 2]


def test_truncate_train_log_rejects_malformed_interior_json(tmp_path):
    train_pilot = load_train_pilot()
    path = tmp_path / "train.jsonl"
    original = (
        "\n".join(
            [
                json.dumps({"step": 1, "loss": 0.9}),
                "{not json}",
                json.dumps({"step": 2, "loss": 0.8}),
            ]
        )
        + "\n"
    )
    write_log(path, original)
    with pytest.raises(ValueError, match=r":2: malformed JSON"):
        train_pilot.truncate_train_log(path, 10)
    assert path.read_text(encoding="utf-8") == original


def test_truncate_train_log_rejects_invalid_records_and_steps(tmp_path):
    train_pilot = load_train_pilot()
    path = tmp_path / "train.jsonl"
    cases = [
        (json.dumps(["not", "a", "record"]) + "\n", r":1: log record must be a JSON object"),
        (json.dumps({"loss": 0.1}) + "\n", r":1: step must be a positive integer"),
        (json.dumps({"step": True}) + "\n", r":1: step must be a positive integer"),
        (json.dumps({"step": 0}) + "\n", r":1: step must be a positive integer"),
        (json.dumps({"step": -3}) + "\n", r":1: step must be a positive integer"),
        (json.dumps({"step": 1.5}) + "\n", r":1: step must be a positive integer"),
    ]
    for payload, pattern in cases:
        write_log(path, payload)
        with pytest.raises(ValueError, match=pattern):
            train_pilot.truncate_train_log(path, 10)
        assert path.read_text(encoding="utf-8") == payload


def test_truncate_train_log_rejects_duplicate_and_out_of_order_steps(tmp_path):
    train_pilot = load_train_pilot()
    path = tmp_path / "train.jsonl"
    duplicate = (
        json.dumps({"step": 1})
        + "\n"
        + json.dumps({"step": 2})
        + "\n"
        + json.dumps({"step": 2})
        + "\n"
    )
    write_log(path, duplicate)
    with pytest.raises(ValueError, match=r":3: steps must be strictly increasing"):
        train_pilot.truncate_train_log(path, 10)
    assert path.read_text(encoding="utf-8") == duplicate

    disordered = (
        json.dumps({"step": 1})
        + "\n"
        + json.dumps({"step": 3})
        + "\n"
        + json.dumps({"step": 2})
        + "\n"
    )
    write_log(path, disordered)
    with pytest.raises(ValueError, match=r":3: steps must be strictly increasing"):
        train_pilot.truncate_train_log(path, 10)
    assert path.read_text(encoding="utf-8") == disordered


def test_truncate_train_log_tolerates_incomplete_final_line_only(tmp_path):
    train_pilot = load_train_pilot()
    path = tmp_path / "train.jsonl"
    write_log(path, json.dumps({"step": 1, "loss": 0.9}) + "\n" + '{"step": 2, "lo')
    kept = train_pilot.truncate_train_log(path, 10)
    assert [record["step"] for record in kept] == [1]

    complete_unterminated = json.dumps({"step": 1}) + "\n" + json.dumps({"step": 2})
    write_log(path, complete_unterminated)
    kept = train_pilot.truncate_train_log(path, 10)
    assert [record["step"] for record in kept] == [1, 2]

    interior_and_tail = json.dumps({"step": 1}) + "\n" + "{not json}\n" + '{"step": 2, "lo'
    write_log(path, interior_and_tail)
    with pytest.raises(ValueError, match=r":2: malformed JSON"):
        train_pilot.truncate_train_log(path, 10)
    assert path.read_text(encoding="utf-8") == interior_and_tail


def test_truncate_train_log_creates_missing_file(tmp_path):
    train_pilot = load_train_pilot()
    path = tmp_path / "missing" / "train.jsonl"
    assert not path.exists()
    kept = train_pilot.truncate_train_log(path, 250)
    assert kept == []
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == ""


def test_resume_does_not_duplicate_steps(tmp_path):
    train_pilot = load_train_pilot()
    path = tmp_path / "train.jsonl"
    records = [{"step": step, "loss": 1.0 / step} for step in range(1, 301)]
    write_log(path, "".join(json.dumps(record) + "\n" for record in records))
    kept = train_pilot.truncate_train_log(path, 250)
    assert [record["step"] for record in kept] == list(range(1, 251))
    with path.open("a", encoding="utf-8") as log:
        for step in range(251, 256):
            log.write(json.dumps({"step": step, "loss": 0.1}) + "\n")
    steps = [json.loads(line)["step"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert steps == list(range(1, 256))
    assert len(steps) == len(set(steps))


def test_periodic_checkpointing_disabled_by_default():
    train_pilot = load_train_pilot()
    parser = train_pilot.build_parser()
    args = parser.parse_args(required_args(Path("results/run")))
    assert args.checkpoint_every == 0
    assert train_pilot.should_write_periodic_checkpoint(250, args.checkpoint_every) is False
    assert train_pilot.should_write_periodic_checkpoint(0, 250) is False


def test_periodic_checkpointing_enabled_at_interval(tmp_path):
    train_pilot = load_train_pilot()
    parser = train_pilot.build_parser()
    args = parser.parse_args([*required_args(Path("results/run")), "--checkpoint-every", "250"])
    assert args.checkpoint_every == 250
    trainer = FakeTrainer()
    while trainer.step < 500:
        trainer.train_step()
        train_pilot.maybe_save_periodic_checkpoint(trainer, tmp_path, args.checkpoint_every)
    assert [path.name for path in trainer.saved] == ["checkpoint-000250.pt", "checkpoint-000500.pt"]


def test_final_checkpoint_saved_when_periodic_disabled(tmp_path):
    train_pilot = load_train_pilot()
    trainer = FakeTrainer()
    steps = 7
    while trainer.step < steps:
        trainer.train_step()
        train_pilot.maybe_save_periodic_checkpoint(trainer, tmp_path, 0)
    final = train_pilot.save_final_checkpoint(trainer, tmp_path)
    assert trainer.saved == [tmp_path / "final.pt"]
    assert final == tmp_path / "final.pt"
    assert not train_pilot.should_write_periodic_checkpoint(steps, 0)


def test_negative_checkpoint_every_is_rejected():
    train_pilot = load_train_pilot()
    parser = train_pilot.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*required_args(Path("results/run")), "--checkpoint-every", "-1"])


def test_fresh_run_rejects_existing_output_directory(tmp_path):
    train_pilot = load_train_pilot()
    results_root = tmp_path / "results"
    output = results_root / "run"
    results_root.mkdir()
    train_pilot.prepare_output_directory(output, resume=None, results_root=results_root)
    with pytest.raises(FileExistsError):
        train_pilot.prepare_output_directory(output, resume=None, results_root=results_root)


def test_resume_requires_checkpoint_inside_output(tmp_path):
    train_pilot = load_train_pilot()
    results_root = tmp_path / "results"
    output = results_root / "run"
    other = results_root / "other"
    results_root.mkdir()
    output.mkdir()
    other.mkdir()
    inside = output / "checkpoint-000250.pt"
    outside = other / "checkpoint-000250.pt"
    inside.write_bytes(b"ckpt")
    outside.write_bytes(b"ckpt")
    assert train_pilot.validate_resume_checkpoint(inside, output) == inside.resolve()
    with pytest.raises(ValueError, match="belong to the --output directory"):
        train_pilot.validate_resume_checkpoint(outside, output)
    missing = results_root / "missing"
    with pytest.raises(FileNotFoundError, match="existing --output directory"):
        train_pilot.prepare_output_directory(
            missing,
            resume=inside,
            results_root=results_root,
        )
    assert (
        train_pilot.prepare_output_directory(
            output,
            resume=inside,
            results_root=results_root,
        )
        == output.resolve()
    )
