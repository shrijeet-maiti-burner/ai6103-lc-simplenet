import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lc_simplenet.baseline import UPSTREAM_AD, build_model  # noqa: E402
from lc_simplenet.reference_inference import predict_reference_batch  # noqa: E402


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1, stride=2), torch.nn.BatchNorm2d(4)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, 3, padding=1, stride=2), torch.nn.BatchNorm2d(8)
        )

    def forward(self, images):
        return self.layer3(self.layer2(images))


def test_memory_adaptation_matches_original_scores_and_complete_masks():
    torch.manual_seed(51)
    settings = {
        **UPSTREAM_AD,
        "target_embed_dimension": 16,
        "pretrain_embed_dimension": 16,
        "dsc_hidden": 8,
        "meta_epochs": 1,
        "gan_epochs": 1,
    }
    model = build_model(TinyBackbone(), torch.device("cpu"), 32, settings)
    images = torch.randn(3, 3, 32, 32)
    original_scores, original_masks, _ = model._predict(images)
    scores, masks = predict_reference_batch(model, images)
    np.testing.assert_array_equal(scores, original_scores)
    np.testing.assert_array_equal(masks, original_masks)
