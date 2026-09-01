import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


HERE = Path(__file__).resolve()
LINEAR_DIR = HERE.parents[1]
DINOV3_DIR = HERE.parents[2] / "DINOv3_BBCH"
for path in (str(DINOV3_DIR), str(LINEAR_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from infer_single_image import build_metadata, build_model_from_checkpoint, format_prediction
from linear_phenology import (
    DINOv3LinearClassifier,
    LINEAR_METADATA_COLUMNS,
    collate_tiled_images,
    configure_partial_finetuning,
    create_backbone_extractor,
)
from multiscale_phenology import HybridOrdinalLoss, generate_loso_train_val_test_folds
from run_linear_training import run_epoch


class FakeBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class FakeBackbone(nn.Module):
    def __init__(self, dim=8, blocks=3):
        super().__init__()
        self.layer = nn.ModuleList([FakeBlock(dim) for _ in range(blocks)])
        self.norm = nn.LayerNorm(dim)


class FakeExtractor(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.out_dim = dim
        self.backbone = FakeBackbone(dim)
        self.preprocess_image_size = 16
        self.preprocess_mean = [0.5, 0.5, 0.5]
        self.preprocess_std = [0.25, 0.25, 0.25]
        self.patch_size = 4
        self.num_register_tokens = 2

    def forward_dense(self, images, grid_size=2, include_cls=True):
        x = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1).repeat(1, self.out_dim)
        for block in self.backbone.layer:
            x = block(x)
        x = self.backbone.norm(x)
        tokens = grid_size**2 + int(include_cls)
        offsets = torch.arange(tokens, device=x.device, dtype=x.dtype).view(1, tokens, 1) / 100
        return x.unsqueeze(1) + offsets


class LegacyExtractor:
    """Matches the shared extractor API before local_config was introduced."""

    def __init__(self, backbone_name, pretrained=True):
        self.backbone_name = backbone_name
        self.pretrained = pretrained


def make_model():
    return DINOv3LinearClassifier(
        backbone_name="fake",
        num_classes=3,
        dense_grid_size=2,
        dense_include_cls=True,
        metadata_columns=LINEAR_METADATA_COLUMNS,
        dropout=0.0,
        pretrained=False,
        extractor=FakeExtractor(),
    )


class LinearPipelineTests(unittest.TestCase):
    def test_legacy_extractor_api_is_compatible_for_training(self):
        extractor = create_backbone_extractor(
            "legacy-model",
            pretrained=False,
            backbone_config=None,
            extractor_cls=LegacyExtractor,
        )
        self.assertEqual(extractor.backbone_name, "legacy-model")
        self.assertFalse(extractor.pretrained)

        with self.assertRaisesRegex(RuntimeError, "older version without 'local_config' support"):
            create_backbone_extractor(
                "legacy-model",
                pretrained=False,
                backbone_config={"hidden_size": 8},
                extractor_cls=LegacyExtractor,
            )

    def test_model_is_single_day_and_has_no_temporal_transformer(self):
        model = make_model()
        self.assertFalse(any(isinstance(module, nn.TransformerEncoder) for module in model.modules()))
        tiles = torch.randn(2, 3, 3, 16, 16)
        mask = torch.ones(2, 3, dtype=torch.bool)
        metadata = torch.zeros(2, len(LINEAR_METADATA_COLUMNS))
        self.assertEqual(tuple(model(tiles, mask, metadata).shape), (2, 3))
        self.assertEqual(model.classifier.in_features, 8 + len(LINEAR_METADATA_COLUMNS))

    def test_collate_and_attention_mask_support_unequal_tile_counts(self):
        items = [
            {
                "tiles": torch.ones(2, 3, 16, 16),
                "metadata": torch.zeros(4),
                "target": torch.tensor([1.0, 0.0, 0.0]),
                "path": "a",
                "station_year": "a",
                "station_code": "01.01",
                "date": "2020-01-01",
                "error": None,
            },
            {
                "tiles": torch.ones(1, 3, 16, 16),
                "metadata": torch.zeros(4),
                "target": torch.tensor([0.0, 1.0, 0.0]),
                "path": "b",
                "station_year": "b",
                "station_code": "02.02",
                "date": "2020-01-02",
                "error": None,
            },
        ]
        batch = collate_tiled_images(items)
        self.assertEqual(tuple(batch["tiles"].shape), (2, 2, 3, 16, 16))
        self.assertEqual(batch["tile_mask"].tolist(), [[True, True], [True, False]])
        model = make_model().eval()
        with torch.no_grad():
            first = model(batch["tiles"], batch["tile_mask"], batch["metadata"])
            batch["tiles"][1, 1].fill_(999)
            second = model(batch["tiles"], batch["tile_mask"], batch["metadata"])
        torch.testing.assert_close(first[1], second[1])

    def test_metadata_is_directly_concatenated_and_changes_logits(self):
        model = make_model().eval()
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()
            model.classifier.weight[0, -len(LINEAR_METADATA_COLUMNS)] = 1.0
        tiles = torch.zeros(1, 1, 3, 16, 16)
        mask = torch.ones(1, 1, dtype=torch.bool)
        low = torch.zeros(1, len(LINEAR_METADATA_COLUMNS))
        high = low.clone()
        high[0, 0] = 1.0
        with torch.no_grad():
            low_logits = model(tiles, mask, low)
            high_logits = model(tiles, mask, high)
        self.assertNotEqual(float(low_logits[0, 0]), float(high_logits[0, 0]))

    def test_inference_metadata_builds_cumulative_gdd_and_location(self):
        with tempfile.TemporaryDirectory() as directory:
            weather_path = Path(directory) / "weather.csv"
            pd.DataFrame({
                "station_family": ["02", "02"],
                "date": ["2020-01-01", "2020-01-02"],
                "tavg": [10.0, 12.0],
                "tmin": [5.0, 6.0],
                "tmax": [15.0, 18.0],
                "prcp": [0.0, 1.0],
            }).to_csv(weather_path, index=False)
            metadata, values = build_metadata(
                "2020-01-02",
                "2020-01-01",
                "02.02",
                str(weather_path),
            )
        self.assertEqual(tuple(metadata.shape), (len(LINEAR_METADATA_COLUMNS),))
        self.assertAlmostEqual(values["weather_gdd_cum_norm"], 22.0 / 2500.0, places=6)
        self.assertTrue(torch.isfinite(metadata).all())

    def test_only_configured_backbone_block_receives_gradient(self):
        model = make_model()
        info = configure_partial_finetuning(model.extractor, 1, True)
        self.assertEqual(info["unfrozen_blocks"], [2])
        logits = model(
            torch.randn(2, 2, 3, 16, 16),
            torch.ones(2, 2, dtype=torch.bool),
            torch.zeros(2, len(LINEAR_METADATA_COLUMNS)),
        )
        logits.sum().backward()
        self.assertIsNone(model.extractor.backbone.layer[0].linear.weight.grad)
        self.assertIsNotNone(model.extractor.backbone.layer[-1].linear.weight.grad)

    def test_loso_splits_are_station_disjoint(self):
        frame = pd.DataFrame({
            "station_code": np.repeat(["01.01", "02.02", "03.03", "04.04", "05.05"], 2)
        })
        for train_idx, val_idx, test_idx in generate_loso_train_val_test_folds(
            frame, group_col="station_code", n_val=2, random_state=42
        ):
            train = set(frame.iloc[train_idx]["station_code"])
            val = set(frame.iloc[val_idx]["station_code"])
            test = set(frame.iloc[test_idx]["station_code"])
            self.assertFalse(train & val)
            self.assertFalse(train & test)
            self.assertFalse(val & test)

    def test_synthetic_train_eval_and_checkpoint_reconstruction(self):
        model = make_model()
        configure_partial_finetuning(model.extractor, 1, True)
        batch = {
            "tiles": torch.randn(4, 2, 3, 16, 16),
            "tile_mask": torch.ones(4, 2, dtype=torch.bool),
            "metadata": torch.randn(4, len(LINEAR_METADATA_COLUMNS)),
            "target": torch.eye(3)[torch.tensor([0, 1, 2, 1])],
            "path": ["a", "b", "c", "d"],
            "station_year": ["s"] * 4,
            "station_code": ["01.01"] * 4,
            "date": ["2020-01-01"] * 4,
        }
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        metrics, _ = run_epoch(
            model,
            [batch],
            HybridOrdinalLoss(2, 0.5),
            torch.device("cpu"),
            ["a", "b", "c"],
            optimizer=optimizer,
            scaler=scaler,
        )
        self.assertTrue(np.isfinite(metrics["loss"]))
        eval_metrics, _ = run_epoch(
            model,
            [batch],
            HybridOrdinalLoss(2, 0.5),
            torch.device("cpu"),
            ["a", "b", "c"],
        )
        self.assertTrue(np.isfinite(eval_metrics["loss"]))
        checkpoint = {
            "model": model.state_dict(),
            "model_config": model.checkpoint_config(),
            "classes": ["a", "b", "c"],
        }
        restored = build_model_from_checkpoint(
            checkpoint,
            torch.device("cpu"),
            extractor=FakeExtractor(),
        )
        restored.eval()
        with torch.no_grad():
            logits = restored(
                batch["tiles"],
                batch["tile_mask"],
                batch["metadata"],
            )
        result = format_prediction(logits[:1], checkpoint["classes"])
        self.assertEqual(len(result["probabilities"]), 3)
        self.assertAlmostEqual(
            sum(item["probability"] for item in result["probabilities"]),
            1.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
