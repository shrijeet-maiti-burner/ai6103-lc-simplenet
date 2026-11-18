"""Run the frozen blade workers locally on an allocated L40S; no SSH operator."""

import argparse
import base64
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

SOURCE_SHA256 = "023a644655348d1a0b610e2e7406cced3cae4fbe997ff0e5055573722a047436"
MAX_FRAME = 600 * 2**20


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


@contextlib.contextmanager
def execution_source(archive):
    if sha(archive) != SOURCE_SHA256:
        raise ValueError("blade execution archive hash differs")
    cache = Path.cwd() / "results" / "blade-source-temporary"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="checked-", dir=cache) as temporary:
        root = Path(temporary).resolve()
        if not root.is_relative_to(cache.resolve()):
            raise ValueError("temporary source escapes its workspace")
        with zipfile.ZipFile(archive) as source:
            manifest = json.loads(source.read("SOURCE-MANIFEST.json"))
            if set(source.namelist()) != {*manifest["files"], "SOURCE-MANIFEST.json"}:
                raise ValueError("source inventory differs")
            for name, expected in manifest["files"].items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                    raise ValueError("unsafe source member")
                raw = source.read(name)
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError("source member differs")
                path = root.joinpath(*relative.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
        yield root


def send(stream, data):
    if len(data) > MAX_FRAME:
        raise ValueError("oversized frame")
    stream.write(str(len(data)).encode() + b"\n")
    for offset in range(0, len(data), 48 * 2**10):
        stream.write(base64.b64encode(data[offset : offset + 48 * 2**10]) + b"\n")
    stream.flush()


def receive(stream):
    header = stream.readline(32)
    if not header.endswith(b"\n"):
        raise EOFError("missing frame header")
    size = int(header.strip())
    if not 0 <= size <= MAX_FRAME:
        raise ValueError("oversized frame")
    result = bytearray()
    while len(result) < size:
        line = stream.readline(65539)
        if not line.endswith(b"\n"):
            raise EOFError("incomplete frame")
        result.extend(base64.b64decode(line.rstrip(b"\r\n"), validate=True))
        if len(result) > size:
            raise ValueError("frame length exceeded")
    return bytes(result)


def packet(root, worker, launch, inputs, out, weights, resume=None, heads=None):
    config = read(root / "recipe.json")
    if sha(weights) != config["backbone_sha256"]:
        raise ValueError("ImageNet weights differ")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "launch.json", launch)
    env = dict(os.environ)
    env.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    artifacts, last_step, finished = {}, launch.get("start_step", 0), False
    with (out / "stderr.log").open("wb") as errors:
        process = subprocess.Popen(
            [sys.executable, "-u", str(root / worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            env=env,
        )
        try:
            ready = json.loads(receive(process.stdout))
            if ready["state"] != "ready":
                raise ValueError("worker is not ready")
            write(out / "runtime.json", ready)
            send(process.stdin, json.dumps(launch).encode())
            send(process.stdin, (root / "runtime.zip").read_bytes())
            if ready.get("backbone_cached_sha256") != config["backbone_sha256"]:
                send(process.stdin, Path(weights).read_bytes())
            if heads is not None:
                send(process.stdin, Path(heads).read_bytes())
            send(process.stdin, Path(inputs).read_bytes())
            if resume is not None:
                send(process.stdin, Path(resume).read_bytes())
            with (out / "events.jsonl").open("x", encoding="utf-8") as events:
                while True:
                    event = json.loads(receive(process.stdout))
                    events.write(json.dumps(event) + "\n")
                    events.flush()
                    print(json.dumps(event), flush=True)
                    if event["state"] == "worker_error":
                        raise RuntimeError("frozen worker: " + json.dumps(event))
                    if event["state"] == "artifact":
                        name = event["name"]
                        if (
                            name in ("", ".", "..")
                            or PurePosixPath(name).name != name
                            or "\\" in name
                            or ":" in name
                        ):
                            raise ValueError("unsafe artifact name")
                        raw = receive(process.stdout)
                        if (
                            len(raw) != event["bytes"]
                            or hashlib.sha256(raw).hexdigest() != event["sha256"]
                        ):
                            raise ValueError("worker artifact differs")
                        with (out / name).open("xb") as stream:
                            stream.write(raw)
                        artifacts[name] = {"sha256": event["sha256"], "bytes": len(raw)}
                        if name.startswith("checkpoint-"):
                            last_step = int(event["step"])
                    if event["state"] == "complete":
                        finished = True
                        break
            process.stdin.close()
            code = process.wait(timeout=30)
            if code != 0 or not finished:
                raise ValueError("worker did not finish successfully")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
            for stream in (process.stdin, process.stdout):
                with contextlib.suppress(BrokenPipeError, OSError):
                    stream.close()
    receipt = {
        "completed": True,
        "exit_code": code,
        "source_archive_sha256": SOURCE_SHA256,
        "launch_sha256": sha(out / "launch.json"),
        "artifacts": artifacts,
        "completed_step": last_step,
    }
    write(out / "receipt.json", receipt)
    return receipt


def prepare(args, root):
    config = read(root / "recipe.json")
    args.out.mkdir(parents=True, exist_ok=False)
    roles = config["roles"]
    source_rows = [r for role in ("fit", "calibration", "development") for r in roles[role]]
    with zipfile.ZipFile(args.aebad_archive) as native:
        populations = {"source": source_rows, "replay": roles["development"]}
        if args.include_tests:
            populations.update({"test-" + d: rows for d, rows in config["test_rows"].items()})
        outputs = {}
        for name, rows in populations.items():
            path = args.out / (name + ".zip")
            with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as output:
                for row in rows:
                    raw = native.read(row["image_id"])
                    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                        raise ValueError("native image bytes differ: " + row["image_id"])
                    output.writestr(row["image_id"], raw)
                if name == "source":
                    for donor in config["textures"]:
                        texture_root = args.textures_root.resolve(strict=True)
                        path_donor = (texture_root / donor["path"]).resolve(strict=True)
                        if (
                            not path_donor.is_relative_to(texture_root)
                            or sha(path_donor) != donor["sha256"]
                        ):
                            raise ValueError("texture bytes/path differ")
                        output.writestr("textures/" + donor["path"], path_donor.read_bytes())
            outputs[name] = {"file": path.name, "sha256": sha(path)}
    write(args.out / "prepared.json", {"source_archive_sha256": SOURCE_SHA256, "files": outputs})


def input_path(folder, name):
    metadata = read(folder / "prepared.json")
    if metadata["source_archive_sha256"] != SOURCE_SHA256:
        raise ValueError("prepared inputs belong to another source")
    record = metadata["files"][name]
    if Path(record["file"]).name != record["file"]:
        raise ValueError("unsafe input filename")
    path = folder / record["file"]
    if sha(path) != record["sha256"]:
        raise ValueError("prepared input changed")
    return path


def train(args, root):
    import torch

    config = read(root / "recipe.json")
    family = config["families"][args.family]
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["protocol_sha256"] != family["protocol_sha256"] or state["seed"] != args.seed:
            raise ValueError("resume belongs to another execution or seed")
        start = state["step"]
        del state
    if not 0 <= start < args.end_step <= 5120 or start % 32 or args.end_step % 32:
        raise ValueError("invalid fixed training budget")
    source = input_path(args.inputs, "source")
    launch = {
        "seed": args.seed,
        "start_step": start,
        "end_step": args.end_step,
        "protocol_sha256": family["protocol_sha256"],
        "policy": family["policy"],
        "runtime_sha256": config["runtime_sha256"],
        "runtime_files": config["runtime_files"],
        "backbone_sha256": config["backbone_sha256"],
        "input_sha256": sha(source),
    }
    if args.resume:
        launch["resume_sha256"] = sha(args.resume)
    receipt = packet(root, family["worker"], launch, source, args.out, args.weights, args.resume)
    write(
        args.out / "case.json",
        {
            "family": args.family,
            "seed": args.seed,
            "full_budget_complete": receipt["completed_step"] == 5120,
        },
    )


def tensor_hash(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(f"{key}:{array.dtype}:{array.shape}:".encode())
        result.update(memoryview(array).cast("B"))
    return result.hexdigest()


def pack(args, root):
    import numpy as np
    import torch

    config = read(root / "recipe.json")
    cases = {}
    for metadata in args.runs.rglob("case.json"):
        case = read(metadata)
        if not case["full_budget_complete"]:
            continue
        key = (case["family"], case["seed"])
        if key in cases:
            raise ValueError("ambiguous duplicate completed case")
        cases[key] = metadata.parent
    if set(cases) != {(f, s) for f in config["families"] for s in range(3)}:
        raise ValueError("all nine completed full-budget cases are required")
    models, thresholds, signatures, references = (
        {str(s): {} for s in range(3)},
        {str(s): {} for s in range(3)},
        {str(s): {} for s in range(3)},
        {},
    )
    encoders, features = set(), set()
    for (family, seed), folder in cases.items():
        receipt, launch = read(folder / "receipt.json"), read(folder / "launch.json")
        expected = config["families"][family]
        if (
            not receipt["completed"]
            or receipt["exit_code"] != 0
            or receipt["completed_step"] != 5120
            or receipt["source_archive_sha256"] != SOURCE_SHA256
            or receipt["launch_sha256"] != sha(folder / "launch.json")
            or launch["protocol_sha256"] != expected["protocol_sha256"]
            or launch["seed"] != seed
            or launch["policy"] != expected["policy"]
        ):
            raise ValueError("completed training lineage differs")
        for name, record in receipt["artifacts"].items():
            if Path(name).name != name or sha(folder / name) != record["sha256"]:
                raise ValueError("training artifact changed")
        state = torch.load(folder / "checkpoint-05120.pt", map_location="cpu", weights_only=True)
        if state["step"] != 5120 or state["protocol_sha256"] != expected["protocol_sha256"]:
            raise ValueError("incomplete or mismatched final checkpoint")
        encoders.add(state["encoder_sha256"])
        features.add(state["feature_sha256"])
        screen = read(folder / "source-screen.json")
        with np.load(folder / "source-patch-scores.npz", allow_pickle=False) as patches:
            for arm in expected["policy"]["arms"]:
                calibration = np.stack([patches[f"calibration_{arm}_{i:03d}"] for i in range(64)])
                q = float(np.quantile(calibration.max(axis=-1), 0.95, method="higher"))
                if q != screen["metrics"][arm]["threshold"]:
                    raise ValueError("source calibration differs")
                key = str(seed)
                models[key][arm] = state["models"][arm]
                signatures[key][arm] = [tensor_hash(s) for s in state["models"][arm]]
                thresholds[key][arm] = q
                references[f"seed{seed}_{arm}"] = np.stack(
                    [patches[f"development_{arm}_{i:03d}"] for i in range(64)]
                )
        del state
    if len(encoders) != 1 or len(features) != 1:
        raise ValueError("encoder or source feature populations differ")
    arms = ["B0", "B1", "B2", "L0", "L1", "L2", "L3", "R0", "R1", "R2", "P0", "P1"]
    args.out.mkdir(parents=True, exist_ok=False)
    value = {
        "protocol_sha256": config["study_protocol_sha256"],
        "steps": 5120,
        "source_protocols": config["source_protocols"],
        "encoder_sha256": encoders.pop(),
        "arms": arms,
        "seeds": [0, 1, 2],
        "models": models,
        "thresholds": thresholds,
        "head_tensor_sha256": signatures,
        "primary": "R0",
    }
    with (args.out / "heads.pt").open("xb") as stream:
        torch.save(value, stream)
    np.savez_compressed(args.out / "development-reference.npz", **references)
    write(
        args.out / "pack.json",
        {
            "source_archive_sha256": SOURCE_SHA256,
            "heads": 36,
            "steps": 5120,
            "primary": "R0",
            "thresholds": thresholds,
            "heads_sha256": sha(args.out / "heads.pt"),
            "reference_sha256": sha(args.out / "development-reference.npz"),
        },
    )


def score(args, root):
    metadata = read(args.pack / "pack.json")
    if (
        metadata.get("source_archive_sha256") != SOURCE_SHA256
        or metadata.get("heads") != 36
        or metadata.get("steps") != 5120
        or metadata.get("primary") != "R0"
    ):
        raise ValueError("completed full-budget blade pack required")
    config = read(root / "recipe.json")
    if (
        sha(args.pack / "heads.pt") != metadata["heads_sha256"]
        or sha(args.pack / "development-reference.npz") != metadata["reference_sha256"]
    ):
        raise ValueError("head/reference pack differs")
    import numpy as np
    from scipy.stats import rankdata

    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for domain in ["replay", *config["test_rows"]]:
        source = input_path(args.inputs, domain if domain == "replay" else "test-" + domain)
        launch = {
            "mode": "replay" if domain == "replay" else "test",
            "domain": domain,
            "primary": "R0",
            "source_protocols": config["source_protocols"],
            "study_protocol_sha256": config["study_protocol_sha256"],
            "real_test_protocol_sha256": config["score_protocol_sha256"],
            "rows": config["roles"]["development"]
            if domain == "replay"
            else config["test_rows"][domain],
            "runtime_files": config["runtime_files"],
            "runtime_sha256": config["runtime_sha256"],
            "backbone_sha256": config["backbone_sha256"],
            "heads_sha256": metadata["heads_sha256"],
            "input_sha256": sha(source),
        }
        folder = args.out / domain
        packet(
            root,
            "workers/score.py",
            launch,
            source,
            folder,
            args.weights,
            heads=args.pack / "heads.pt",
        )
        with np.load(folder / "patch-scores.npz", allow_pickle=False) as patches:
            if domain == "replay":
                with np.load(
                    args.pack / "development-reference.npz", allow_pickle=False
                ) as reference:
                    if set(patches.files) != set(reference.files) or any(
                        not np.array_equal(patches[k], reference[k]) for k in reference.files
                    ):
                        raise ValueError("strict source replay differs; targets stay closed")
                write(args.out / "source-replay.json", {"bit_exact": True, "heads": 36})
                continue
            labels = np.array([r["label"] for r in launch["rows"]], dtype=bool)
            n1, n0 = int(labels.sum()), int((~labels).sum())
            for seed in range(3):
                for arm, threshold in metadata["thresholds"][str(seed)].items():
                    values = patches[f"seed{seed}_{arm}"].reshape(len(labels), -1).max(axis=1)
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite predictions")
                    flags = values > threshold
                    auc = (rankdata(values, method="average")[labels].sum() - n1 * (n1 + 1) / 2) / (
                        n1 * n0
                    )
                    results.append(
                        {
                            "domain": domain,
                            "method": arm,
                            "seed": seed,
                            "normal_count": n0,
                            "defect_count": n1,
                            "false_positives": int(flags[~labels].sum()),
                            "true_positives": int(flags[labels].sum()),
                            "fpr": float(flags[~labels].mean()),
                            "recall": float(flags[labels].mean()),
                            "image_auroc": float(auc),
                            "threshold": threshold,
                        }
                    )
    with (args.out / "domain-seed-metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write(
        args.out / "complete.json",
        {
            "completed": True,
            "source_archive_sha256": SOURCE_SHA256,
            "rows": len(results),
            "csv_sha256": sha(args.out / "domain-seed-metrics.csv"),
            "scope": (
                "new reproduction; not a relabelled original result or source-qualification claim"
            ),
        },
    )
    audit(argparse.Namespace(scored=args.out, pack=args.pack), root)


def audit(args, root):
    import numpy as np

    config = read(root / "recipe.json")
    pack_metadata = read(args.pack / "pack.json")
    completed = read(args.scored / "complete.json")
    if (
        completed["source_archive_sha256"] != SOURCE_SHA256
        or not completed["completed"]
        or completed["csv_sha256"] != sha(args.scored / "domain-seed-metrics.csv")
        or pack_metadata["source_archive_sha256"] != SOURCE_SHA256
    ):
        raise ValueError("complete scoring evidence required")
    spec = importlib.util.spec_from_file_location("blade_audit_helpers", root / "audit_helpers.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    means, comparisons, source_overlap, duplicates, conflicts = {}, {}, [], {}, {}
    source_rgb = {r["decoded_rgb_sha256"] for rows in config["roles"].values() for r in rows}
    rng = np.random.default_rng(19317)
    decisions = 0
    for domain, rows in config["test_rows"].items():
        folder = args.scored / domain
        receipt, launch, reported = (
            read(folder / "receipt.json"),
            read(folder / "launch.json"),
            read(folder / "metrics.json"),
        )
        if (
            not receipt["completed"]
            or receipt["exit_code"] != 0
            or receipt["source_archive_sha256"] != SOURCE_SHA256
            or receipt["launch_sha256"] != sha(folder / "launch.json")
            or launch["rows"] != rows
            or launch["study_protocol_sha256"] != config["study_protocol_sha256"]
            or launch["real_test_protocol_sha256"] != config["score_protocol_sha256"]
            or launch["heads_sha256"] != pack_metadata["heads_sha256"]
        ):
            raise ValueError("scoring lineage differs")
        for name, record in receipt["artifacts"].items():
            if Path(name).name != name or sha(folder / name) != record["sha256"]:
                raise ValueError("scoring artifact changed")
        labels = np.array([r["label"] for r in rows], dtype=bool)
        hashes = [reported["native_rgb_sha256"][r["image_id"]] for r in rows]
        groups = {}
        for row, identity in zip(rows, hashes, strict=True):
            groups.setdefault(identity, []).append(row)
            if identity in source_rgb:
                source_overlap.append(row["image_id"])
        duplicates[domain] = {h: len(v) for h, v in groups.items() if len(v) > 1}
        conflicts[domain] = [h for h, v in groups.items() if len({r["label"] for r in v}) > 1]
        flags, means[domain] = {}, {}
        with np.load(folder / "patch-scores.npz", allow_pickle=False) as patches:
            arms = list(pack_metadata["thresholds"]["0"])
            if set(patches.files) != {f"seed{s}_{a}" for s in range(3) for a in arms}:
                raise ValueError("incomplete raw prediction matrix")
            for arm in arms:
                predictions = []
                for seed in range(3):
                    values = patches[f"seed{seed}_{arm}"].reshape(len(rows), -1).max(axis=1)
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite raw predictions")
                    predictions.append(values > pack_metadata["thresholds"][str(seed)][arm])
                    decisions += len(rows)
                flags[arm] = np.stack(predictions)
                means[domain][arm] = {
                    "fpr": float(flags[arm][:, ~labels].mean()),
                    "recall": float(flags[arm][:, labels].mean()),
                }
        comparisons[domain] = {}
        for control in arms:
            if control == "R0":
                continue
            comparisons[domain][control] = {}
            for metric, mask in (("fpr", ~labels), ("recall", labels)):
                difference = (flags["R0"][:, mask].astype(float) - flags[control][:, mask]).mean(
                    axis=0
                )
                comparisons[domain][control][metric] = helpers.paired_interval(
                    difference,
                    [h for h, keep in zip(hashes, mask, strict=True) if keep],
                    rng,
                    10000,
                )
    a, b = means["illumination"], means["same"]
    gates = {
        "illumination_fpr": a["B0"]["fpr"] - a["R0"]["fpr"] >= 0.05 - 1e-12,
        "illumination_recall": a["R0"]["recall"] - a["B0"]["recall"] >= -0.05 - 1e-12,
        "same_fpr": b["R0"]["fpr"] - b["B0"]["fpr"] <= 0.02 + 1e-12,
        "same_recall": b["R0"]["recall"] - b["B0"]["recall"] >= -0.05 - 1e-12,
    }
    if decisions != 59004:
        raise ValueError("incomplete real-image decision count")
    write(
        args.scored / "audit.json",
        {
            "independent_raw_recount": True,
            "image_decisions": decisions,
            "means": means,
            "comparisons": comparisons,
            "fixed_mean_gates": gates,
            "mean_targets_met": all(gates.values()),
            "source_rgb_overlap": source_overlap,
            "duplicate_rgb_groups": duplicates,
            "conflicting_label_groups": conflicts,
            "source_archive_sha256": SOURCE_SHA256,
            "scope": (
                "fresh reproduction, three-seed means; original unsuccessful source "
                "screen/confirmation unchanged; no new-seed or physical-specimen claim"
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    p = sub.add_parser("prepare")
    p.add_argument("--aebad-archive", type=Path, required=True)
    p.add_argument("--textures-root", type=Path, required=True)
    p.add_argument("--include-tests", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("train")
    p.add_argument("--family", choices=["base", "routed", "pure"], required=True)
    p.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--end-step", type=int, default=5120)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("pack")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("audit")
    p.add_argument("--scored", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    args = parser.parse_args()
    with execution_source(args.source_archive) as root:
        if args.command == "verify":
            print(json.dumps({"source_verified": True, "sha256": SOURCE_SHA256}))
        else:
            {"prepare": prepare, "train": train, "pack": pack, "score": score, "audit": audit}[
                args.command
            ](args, root)


if __name__ == "__main__":
    main()
