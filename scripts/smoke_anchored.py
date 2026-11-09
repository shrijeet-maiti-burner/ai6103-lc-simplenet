"""Check actual-size CUDA heads, zero-weight parity and saved-state continuation."""

import argparse
import copy
import io
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lc_simplenet.anchored_consistency import (  # noqa: E402
    adaptation_mode,
    anchored_forward,
    frozen_teacher,
)
from lc_simplenet.anchored_pilot import new_heads, optimizers, read, write  # noqa: E402
from lc_simplenet.paired_losses import paired_objective  # noqa: E402
from lc_simplenet.pilot import seed_runtime, state_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve smoke attempts")
    if not torch.cuda.is_available():
        raise ValueError("this smoke requires CUDA")
    seed_runtime(0)
    torch.set_num_threads(2)
    policy = read(ROOT / "configs/anchored-pilot-policy.json")
    adapter, disc = new_heads(0)
    direct = copy.deepcopy(adapter), copy.deepcopy(disc)
    teacher = frozen_teacher(adapter, disc)
    initial_teacher = [state_hash(t.state_dict()) for t in teacher]
    opts = optimizers(adapter, disc, policy)
    direct_opts = optimizers(*direct, policy)
    clean = torch.randn(1024, 1536, device="cuda")
    light = clean * 0.9
    noise = torch.randn(2048, 1536, device="cuda") * policy["noise_std"]

    def update(model, optimizers_, *, anchored, reference=False):
        adaptation_mode(*model)
        for opt in optimizers_:
            opt.zero_grad(set_to_none=True)
        if reference:
            z = model[0](torch.cat((clean, light)))
            parts = model[1](torch.cat((z, z + noise))).chunk(4)
            loss = paired_objective(*parts, margin=0.5, lambda_score=0.1)["total"]
        else:
            loss = anchored_forward(
                *model,
                *teacher,
                clean,
                light,
                noise,
                margin=0.5,
                lambda_score=0.1,
                lambda_anchor=1 if anchored else 0,
            )["total"]
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for net in model
            for p in net.parameters()
        ):
            raise AssertionError("nonfinite actual-head gradient")
        for opt in optimizers_:
            opt.step()

    for _ in range(3):
        update((adapter, disc), opts, anchored=False)
        update(direct, direct_opts, anchored=False, reference=True)
    if [state_hash(t.state_dict()) for t in direct] != [
        state_hash(t.state_dict()) for t in (adapter, disc)
    ]:
        raise AssertionError("CUDA zero-weight control differs from the original objective")
    del direct, direct_opts
    memory = io.BytesIO()
    torch.save(
        {
            "models": [t.state_dict() for t in (adapter, disc)],
            "optimizers": [o.state_dict() for o in opts],
        },
        memory,
    )
    memory.seek(0)
    checkpoint = torch.load(memory, map_location="cuda", weights_only=True)
    restored = new_heads(99)
    restored_opts = optimizers(*restored, policy)
    for module, state in zip(restored, checkpoint["models"], strict=True):
        module.load_state_dict(state, strict=True)
    for opt, state in zip(restored_opts, checkpoint["optimizers"], strict=True):
        opt.load_state_dict(state)
    with torch.no_grad():
        for net in (*restored, adapter, disc):
            net.eval()
        if not torch.equal(disc(adapter(clean)), restored[1](restored[0](clean))):
            raise AssertionError("CUDA checkpoint prediction replay differs")
    update((adapter, disc), opts, anchored=True)
    update(restored, restored_opts, anchored=True)
    if [state_hash(t.state_dict()) for t in restored] != [
        state_hash(t.state_dict()) for t in (adapter, disc)
    ]:
        raise AssertionError("saved-state optimizer continuation differs")
    del restored, restored_opts, checkpoint, memory
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(8):
        update((adapter, disc), opts, anchored=True)
    torch.cuda.synchronize()
    if initial_teacher != [state_hash(t.state_dict()) for t in teacher]:
        raise AssertionError("teacher parameters or buffers changed")
    receipt = {
        "scope": "random-feature CUDA contract smoke; no data or predictive result",
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "zero_weight_updates_bit_exact": True,
        "checkpoint_prediction_bit_exact": True,
        "optimizer_continuation_bit_exact": True,
        "teacher_unchanged": True,
        "anchor_steps": 8,
        "seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write(args.out, receipt)
    print(receipt)


if __name__ == "__main__":
    main()
