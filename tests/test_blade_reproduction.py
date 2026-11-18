"""Guard execution provenance, incomplete runs and the worker byte protocol."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reproduce_routed_blade.py"
spec = importlib.util.spec_from_file_location("blade_reproduction", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unknown_source_is_rejected_before_extraction(tmp_path):
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"unverified source")
    with pytest.raises(ValueError, match="archive hash differs"), module.execution_source(archive):
        pytest.fail("unverified source must not execute")


def test_partial_pack_cannot_open_targets(tmp_path):
    (tmp_path / "pack.json").write_text(
        json.dumps(
            {
                "source_archive_sha256": module.SOURCE_SHA256,
                "heads": 36,
                "steps": 128,
                "primary": "R0",
            }
        )
    )
    with pytest.raises(ValueError, match="completed full-budget"):
        module.score(argparse.Namespace(pack=tmp_path), tmp_path / "no-source-or-target-files")


def test_missing_training_cases_cannot_be_packed(tmp_path):
    pytest.importorskip("torch")
    source = tmp_path / "source"
    source.mkdir()
    (source / "recipe.json").write_text(
        json.dumps({"families": {"base": {}, "routed": {}, "pure": {}}})
    )
    with pytest.raises(ValueError, match="all nine completed"):
        module.pack(argparse.Namespace(runs=tmp_path / "missing"), source)


@pytest.mark.parametrize("mode", ["valid", "unsafe_name", "bad_hash", "worker_error"])
def test_real_subprocess_protocol_fails_closed(tmp_path, mode):
    weights = tmp_path / "weights"
    weights.write_bytes(b"fixture weights, no pretrained model")
    (tmp_path / "recipe.json").write_text(json.dumps({"backbone_sha256": module.sha(weights)}))
    (tmp_path / "runtime.zip").write_bytes(b"fixture source")
    inputs = tmp_path / "inputs.zip"
    inputs.write_bytes(b"fixture inputs")
    payload = b'{"fixture": true}'
    name = "../escaped.json" if mode == "unsafe_name" else "result.json"
    digest = "0" * 64 if mode == "bad_hash" else hashlib.sha256(payload).hexdigest()
    worker = tmp_path / "worker.py"
    code = """import base64, json, sys
def send(raw):
    sys.stdout.buffer.write(str(len(raw)).encode()+b"\\n")
    if raw: sys.stdout.buffer.write(base64.b64encode(raw)+b"\\n")
    sys.stdout.buffer.flush()
def receive():
    size=int(sys.stdin.buffer.readline())
    raw=bytearray()
    while len(raw)<size: raw.extend(base64.b64decode(sys.stdin.buffer.readline()))
    return bytes(raw)
send(json.dumps({"state":"ready"}).encode())
for _ in range(4): receive()
"""
    if mode == "worker_error":
        code += 'send(json.dumps({"state":"worker_error","message":"fixture"}).encode())\n'
    else:
        event = {"state": "artifact", "name": name, "bytes": len(payload), "sha256": digest}
        code += f"send({json.dumps(event).encode()!r})\n"
        code += f"send({payload!r})\n"
        code += 'send(json.dumps({"state":"complete"}).encode())\n'
    worker.write_text(code)
    output = tmp_path / "output"
    if mode == "valid":
        receipt = module.packet(tmp_path, worker.name, {}, inputs, output, weights)
        assert receipt["completed"] and (output / "result.json").read_bytes() == payload
    else:
        kind = RuntimeError if mode == "worker_error" else ValueError
        with pytest.raises(kind):
            module.packet(tmp_path, worker.name, {}, inputs, output, weights)
        assert not (output / "receipt.json").exists()
        assert not (tmp_path / "escaped.json").exists()
