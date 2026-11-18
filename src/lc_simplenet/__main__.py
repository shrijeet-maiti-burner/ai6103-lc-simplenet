"""Check the Python runtime, source identity and optional GPU."""

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

from lc_simplenet.source_identity import UPSTREAM_SHA, source_record

ROOT = Path(__file__).resolve().parents[2]


def doctor(require_cuda: bool) -> int:
    info = {"python": platform.python_version(), "executable": sys.executable}
    ok = sys.version_info[:2] == (3, 12)
    try:
        source = source_record(ROOT)
        info.update(source=source, upstream_sha=source["upstream_commit"])
        ok &= info["upstream_sha"] == UPSTREAM_SHA and not source["upstream_dirty"]
    except (ValueError, OSError) as error:
        info.update(source_error=str(error), upstream_sha=None)
        ok = False
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostics = commands.add_parser("doctor")
    diagnostics.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    return doctor(args.require_cuda)


if __name__ == "__main__":
    raise SystemExit(main())
