"""Inspect the scaffold without downloading datasets or pretrained weights."""

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

UPSTREAM_SHA = "351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b"
ROOT = Path(__file__).resolve().parents[2]


def doctor(require_cuda: bool) -> int:
    info = {"python": platform.python_version(), "executable": sys.executable}
    ok = sys.version_info[:2] == (3, 12)
    submodule = ROOT / "third_party/SimpleNet"
    result = subprocess.run(
        ["git", "-C", str(submodule), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    info["upstream_sha"] = result.stdout.strip() if result.returncode == 0 else None
    ok &= info["upstream_sha"] == UPSTREAM_SHA
    if importlib.util.find_spec("torch"):
        import torch
        import torchvision

        info.update(torch=torch.__version__, torchvision=torchvision.__version__)
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
            x = torch.ones((16, 16), device="cuda", requires_grad=True)
            (x @ x).mean().backward()
            torch.cuda.synchronize()
            info["cuda_backward"] = bool(torch.isfinite(x.grad).all().item())
            ok &= info["cuda_backward"]
    else:
        info.update(torch="not installed; use --extra cpu or --extra cuda", cuda_available=False)
    if require_cuda:
        ok &= info["cuda_available"]
    info["ok"] = bool(ok)
    print(json.dumps(info, indent=2))
    return 0 if ok else 1


def plan(path: Path) -> int:
    study = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(study, indent=2))
    jobs = sum(len(cats) for cats in study["datasets"].values())
    jobs *= len(study["seeds"]) * len(study["variants"])
    print(f"Planned method/category/seed runs: {jobs}")
    print("Draft only: categories and hyperparameters are unresolved; training is not implemented.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostics = commands.add_parser("doctor")
    diagnostics.add_argument("--require-cuda", action="store_true")
    planning = commands.add_parser("plan")
    planning.add_argument("--config", type=Path, default=ROOT / "configs/study.json")
    args = parser.parse_args()
    return doctor(args.require_cuda) if args.command == "doctor" else plan(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
