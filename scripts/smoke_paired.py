"""Actual SimpleNet paired optimizer-step check, random weights and synthetic inputs only."""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

from lc_simplenet.paired_training import normalized_pair, paired_discriminator_loss
from lc_simplenet.pairs import sample_pair

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-size", type=int, help="Override configured size for a quick check")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/paired-lighting.json").read_text())
    height, width = config["input_size"]
    if args.image_size is not None:
        if args.image_size < 32:
            parser.error("--image-size must be at least 32")
        height = width = args.image_size
    torch.set_num_threads(2)
    torch.manual_seed(6103)
    rng = np.random.default_rng(6103)
    sys.path.insert(0, str(ROOT / "third_party/SimpleNet"))
    from simplenet import SimpleNet

    backbone = torchvision.models.wide_resnet50_2(weights=None).requires_grad_(False).eval()
    model = SimpleNet(torch.device("cpu"))
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=torch.device("cpu"),
        input_shape=(3, height, width),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        pre_proj=1,
        dsc_hidden=1024,
        dsc_layers=2,
        train_backbone=False,
    )
    pairs = [
        sample_pair(
            rng.uniform(0.1, 0.9, (height, width, 3)).astype(np.float32),
            rng=rng,
            brightness_range=tuple(config["brightness_range"]),
            gamma_range=tuple(config["gamma_range"]),
        )
        for _ in range(2)
    ]
    views = [normalized_pair(pair) for pair in pairs]
    images = torch.cat([torch.stack([v[0] for v in views]), torch.stack([v[1] for v in views])])
    features, _ = model._embed(images)
    a, b = model.pre_projection(features).chunk(2)
    model.discriminator.train()
    model.proj_opt.zero_grad()
    model.dsc_opt.zero_grad()
    # One Gaussian component for this smoke check only; the trainer owns the study policy.
    terms, diagnostics = paired_discriminator_loss(
        model.discriminator,
        a,
        b,
        torch.randn_like(a) * 0.05,
        torch.randn_like(b) * 0.05,
        margin=model.dsc_margin,
        lambda_score=0.1,
    )
    terms["total"].backward()
    adapter_grad = sum(p.grad.abs().sum().item() for p in model.pre_projection.parameters())
    discriminator_grad = sum(p.grad.abs().sum().item() for p in model.discriminator.parameters())
    frozen = all(p.grad is None for p in backbone.parameters())
    assert frozen and adapter_grad > 0 and discriminator_grad > 0
    assert all(torch.isfinite(value) for value in terms.values())
    before = next(model.pre_projection.parameters()).detach().clone()
    model.proj_opt.step()
    model.dsc_opt.step()
    changed = not torch.equal(before, next(model.pre_projection.parameters()).detach())
    assert changed
    model.pre_projection.eval()
    model.discriminator.eval()
    with torch.no_grad():
        expected = model.discriminator(model.pre_projection(features))
    checkpoint = io.BytesIO()
    torch.save(
        {
            "adapter": model.pre_projection.state_dict(),
            "discriminator": model.discriminator.state_dict(),
        },
        checkpoint,
    )
    with torch.no_grad():
        for parameter in model.pre_projection.parameters():
            parameter.add_(1)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    model.pre_projection.load_state_dict(state["adapter"])
    model.discriminator.load_state_dict(state["discriminator"])
    with torch.no_grad():
        restored = model.discriminator(model.pre_projection(features))
    torch.testing.assert_close(expected, restored, rtol=0, atol=0)
    report = {
        "scope": "CPU random-weight SimpleNet; synthetic inputs; not benchmark results",
        "input_size": [height, width],
        "patches_per_view": len(a),
        "seed": 6103,
        "smoke_lambda_only": 0.1,
        "losses": {k: v.detach().item() for k, v in terms.items()},
        "diagnostics": {k: v.item() for k, v in diagnostics.items()},
        "adapter_gradient_l1": adapter_grad,
        "discriminator_gradient_l1": discriminator_grad,
        "backbone_frozen": frozen,
        "adapter_updated": changed,
        "checkpoint_scores_exact": True,
        "pairs": [
            {"brightness": p.brightness, "gamma": p.gamma, "clipped_fraction": p.clipped_fraction}
            for p in pairs
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
