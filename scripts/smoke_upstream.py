"""Random-weight architecture/gradient check, without data or pretrained downloads."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision

from lc_simplenet.losses import patch_score_consistency

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "third_party/SimpleNet"))
    from simplenet import SimpleNet

    torch.manual_seed(0)
    device = torch.device(args.device)
    backbone = torchvision.models.wide_resnet50_2(weights=None)
    backbone.requires_grad_(False)
    model = SimpleNet(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, 64, 64),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        pre_proj=1,
        dsc_hidden=1024,
        dsc_layers=2,
    )
    image = torch.rand(2, 3, 64, 64, device=device)
    paired = torch.cat([image, image.pow(1.1)])
    features, _ = model._embed(paired)
    features = model.pre_projection(features)
    scores = model.discriminator(features)
    original, relit = scores.chunk(2)
    loss = patch_score_consistency(original, relit)
    loss.backward()
    adapter_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.pre_projection.parameters()
    )
    discriminator_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.discriminator.parameters()
    )
    frozen = all(p.grad is None for p in backbone.parameters())
    if not (adapter_grad and discriminator_grad and frozen and torch.isfinite(loss).item()):
        raise SystemExit("Failed architecture/gradient check")
    print(
        json.dumps(
            {
                "device": str(device),
                "input": list(paired.shape),
                "scores": list(scores.shape),
                "adapter_gradient": adapter_grad,
                "discriminator_gradient": discriminator_grad,
                "backbone_frozen": frozen,
                "scope": "random weights, 64px, no data; not a reproduction or training result",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
