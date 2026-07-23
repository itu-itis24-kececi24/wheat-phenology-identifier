import importlib.util
import os
import sys
import unittest

import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODULE_PATH = os.path.join(ROOT, "multiscale_phenology.py")
SPEC = importlib.util.spec_from_file_location("dinov3_bbch_model", MODULE_PATH)
DINOV3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DINOV3)
FINETUNE_PATH = os.path.join(ROOT, "finetune_dinov3_backbone.py")
FINETUNE_SPEC = importlib.util.spec_from_file_location("dinov3_bbch_finetune", FINETUNE_PATH)
FINETUNE = importlib.util.module_from_spec(FINETUNE_SPEC)
FINETUNE_SPEC.loader.exec_module(FINETUNE)
MASTER_PATH = os.path.join(ROOT, "run_finetuned_loso_pipeline.py")
MASTER_SPEC = importlib.util.spec_from_file_location("dinov3_bbch_finetune_master", MASTER_PATH)
MASTER = importlib.util.module_from_spec(MASTER_SPEC)
MASTER_SPEC.loader.exec_module(MASTER)


class DINOv3BBCHTests(unittest.TestCase):
    def test_master_quadratic_kappa_is_one_for_exact_predictions(self):
        true = torch.tensor([0, 1, 2, 3]).numpy()
        self.assertAlmostEqual(MASTER.quadratic_weighted_kappa(true, true.copy(), 4), 1.0)

    def test_partial_finetuning_unfreezes_only_requested_tail_blocks(self):
        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Linear(4, 4)
                self.layer = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])
                self.layernorm = torch.nn.LayerNorm(4)

        class DummyExtractor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = DummyBackbone()

        extractor = DummyExtractor()
        info = FINETUNE.configure_partial_finetuning(extractor, 2, True)

        self.assertEqual(info["block_path"], "layer")
        self.assertEqual(info["unfrozen_blocks"], [2, 3])
        self.assertFalse(any(parameter.requires_grad for parameter in extractor.backbone.embedding.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in extractor.backbone.layer[0].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in extractor.backbone.layer[2].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in extractor.backbone.layernorm.parameters()))

    def test_loso_folds_test_every_station_once_and_balance_validation(self):
        stations = [f"{station:02d}.01" for station in range(1, 16)]
        frame = pd.DataFrame(
            {
                "station_code": stations,
                "station_year": [f"{station}_2020" for station in stations],
            },
            index=range(100, 115),
        )

        folds = DINOV3.generate_loso_train_val_test_folds(
            frame,
            group_col="station_code",
            n_val=2,
            random_state=42,
        )

        self.assertEqual(len(folds), 15)
        test_counts = {station: 0 for station in stations}
        val_counts = {station: 0 for station in stations}
        for train_idx, val_idx, test_idx in folds:
            train = set(frame.iloc[train_idx]["station_code"])
            val = set(frame.iloc[val_idx]["station_code"])
            test = set(frame.iloc[test_idx]["station_code"])
            self.assertEqual((len(train), len(val), len(test)), (12, 2, 1))
            self.assertFalse(train & val)
            self.assertFalse(train & test)
            self.assertFalse(val & test)
            for station in val:
                val_counts[station] += 1
            for station in test:
                test_counts[station] += 1

        self.assertEqual(set(test_counts.values()), {1})
        self.assertEqual(set(val_counts.values()), {2})

    def test_dense_token_extraction_excludes_register_tokens(self):
        hidden = torch.zeros(1, 7, 3)
        hidden[:, 0] = 100.0
        hidden[:, 1:3] = -100.0
        hidden[:, 3:] = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

        dense = DINOV3.compact_dense_tokens(
            hidden,
            pixel_height=32,
            pixel_width=32,
            patch_size=16,
            num_register_tokens=2,
            grid_size=2,
            include_cls=True,
        )

        self.assertEqual(tuple(dense.shape), (1, 5, 3))
        torch.testing.assert_close(dense[:, 0], hidden[:, 0])
        torch.testing.assert_close(dense[:, 1:], hidden[:, 3:])

    def test_dense_embedding_dataset_pads_tiles_and_builds_mask(self):
        path = os.path.join(ROOT, "dense_test.jpeg")
        frame = pd.DataFrame(
            {
                "station_year": ["01.01_2020"],
                "date": [pd.Timestamp("2020-01-01")],
                "label": ["BBCH0"],
                "micro_path": [path],
                "macro_path": [None],
                "target": [[1.0, 0.0]],
                "date_score": [[1.0, 0.0]],
            }
        )
        cache = {
            "feature_dim": 8,
            "tiling": {
                "tile_pooling": "attention",
                "micro_tile_counts": {path: 3},
                "macro_tile_counts": {},
            },
            "dense_features": {
                "enabled": True,
                "streams": ["micro"],
                "tokens_per_tile": 5,
            },
            "micro": {path: torch.randn(2, 5, 8)},
            "macro": {},
        }
        config = DINOV3.WindowConfig(
            window_days=1,
            center_offset=0,
            classes=("BBCH0", "BBCH1"),
            stream="micro",
            temporal_feature_columns=tuple(),
        )

        sample = DINOV3.MultiScaleEmbeddingWindowDataset(frame, config, cache)[0]

        self.assertEqual(tuple(sample["micro"].shape), (1, 3, 5, 8))
        self.assertEqual(sample["micro_tile_mask"].tolist(), [[True, True, False]])

    def test_hierarchical_pooler_is_finite_with_empty_days(self):
        torch.manual_seed(4)
        features = torch.randn(2, 3, 4, 5, 16, requires_grad=True)
        mask = torch.ones(2, 3, 4, dtype=torch.bool)
        mask[:, 1] = False
        pooler = DINOV3.HierarchicalDenseTilePooler(16, dense_tokens_per_tile=5)

        output = pooler(features, mask)
        output.sum().backward()

        self.assertEqual(tuple(output.shape), (2, 3, 16))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_dense_cached_temporal_model_forward(self):
        batch, days, tiles, tokens, dim = 2, 5, 3, 5, 16
        macro = torch.zeros(batch, days, tiles, tokens, dim)
        micro = torch.randn(batch, days, tiles, tokens, dim)
        tile_mask = torch.ones(batch, days, tiles, dtype=torch.bool)
        temporal_mask = torch.ones(batch, days, dtype=torch.bool)
        model = DINOV3.SingleStreamEmbeddingTemporalTransformer(
            feature_dim=dim,
            stream="micro",
            num_classes=8,
            embed_dim=32,
            temporal_layers=1,
            temporal_heads=4,
            dense_tokens_per_tile=tokens,
        ).eval()

        logits = model(
            macro,
            micro,
            temporal_mask,
            tile_mask,
            tile_mask,
        )

        self.assertEqual(tuple(logits.shape), (batch, 8))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
