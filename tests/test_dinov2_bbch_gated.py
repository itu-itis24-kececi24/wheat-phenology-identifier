import importlib.util
import os
import unittest

import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(ROOT, "DINOv2_BBCH_Gated", "multiscale_phenology.py")
SPEC = importlib.util.spec_from_file_location("dinov2_bbch_gated_model", MODULE_PATH)
GATED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATED)


class DINOv2BBCHGatedTests(unittest.TestCase):
    def test_neutral_gate_starts_as_projected_average(self):
        fusion = GATED.GatedMultiviewFusion(
            input_dim=4,
            embed_dim=4,
            hidden_dim=8,
            dropout=0.0,
        ).eval()
        macro = torch.randn(2, 3, 4)
        micro = torch.randn(2, 3, 4)

        actual = fusion(macro, micro)
        expected = fusion.post(0.5 * fusion.macro_proj(macro) + 0.5 * fusion.micro_proj(micro))

        torch.testing.assert_close(actual, expected)

    def test_missing_camera_is_hard_masked(self):
        fusion = GATED.GatedMultiviewFusion(
            input_dim=4,
            embed_dim=4,
            hidden_dim=8,
            dropout=0.0,
        ).eval()
        macro = torch.randn(2, 3, 4)
        micro = torch.randn(2, 3, 4)
        macro_valid = torch.zeros(2, 3, dtype=torch.bool)
        micro_valid = torch.ones(2, 3, dtype=torch.bool)

        actual = fusion(macro, micro, macro_valid=macro_valid, micro_valid=micro_valid)
        changed_macro = fusion(macro * 1000.0, micro, macro_valid=macro_valid, micro_valid=micro_valid)
        expected = fusion.post(fusion.micro_proj(micro))

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(changed_macro, expected)

    def test_modality_dropout_never_removes_only_available_camera(self):
        fusion = GATED.GatedMultiviewFusion(
            input_dim=4,
            embed_dim=4,
            hidden_dim=8,
            dropout=0.0,
            modality_dropout=0.99,
        ).train()
        macro = torch.randn(2, 3, 4)
        micro = torch.randn(2, 3, 4)
        macro_valid = torch.zeros(2, 3, dtype=torch.bool)
        micro_valid = torch.ones(2, 3, dtype=torch.bool)

        actual = fusion(macro, micro, macro_valid=macro_valid, micro_valid=micro_valid)
        expected = fusion.post(fusion.micro_proj(micro))

        torch.testing.assert_close(actual, expected)

    def test_both_stream_dataset_accepts_micro_only_target(self):
        micro_path = os.path.join(ROOT, "synthetic_micro.jpeg")
        frame = pd.DataFrame(
            {
                "station_year": ["01.01_2020"],
                "date": [pd.Timestamp("2020-01-01")],
                "label": ["BBCH0"],
                "macro_path": [None],
                "micro_path": [micro_path],
                "target": [[1.0, 0.0]],
                "date_score": [[1.0, 0.0]],
            }
        )
        cache = {
            "feature_dim": 4,
            "macro": {},
            "micro": {os.path.abspath(micro_path): torch.randn(2, 4)},
            "tiling": {
                "tile_pooling": "attention",
                "macro_tile_counts": {},
                "micro_tile_counts": {os.path.abspath(micro_path): 2},
            },
        }
        config = GATED.WindowConfig(
            window_days=1,
            center_offset=0,
            classes=("BBCH0", "BBCH1"),
            stream="both",
            temporal_feature_columns=(),
        )

        dataset = GATED.MultiScaleEmbeddingWindowDataset(frame, config, cache)
        sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertFalse(bool(sample["macro_valid"][0]))
        self.assertTrue(bool(sample["micro_valid"][0]))
        self.assertTrue(bool(sample["mask"][0]))

    def test_cached_gated_model_runs_with_partially_missing_views(self):
        model = GATED.MultiScaleEmbeddingTemporalTransformer(
            feature_dim=8,
            num_classes=3,
            embed_dim=8,
            temporal_layers=1,
            temporal_heads=2,
            dropout=0.0,
            target_index=2,
            temporal_feature_dim=1,
            gate_hidden_dim=4,
        ).eval()
        macro = torch.randn(2, 3, 2, 8)
        micro = torch.randn(2, 3, 2, 8)
        macro_tile_mask = torch.ones(2, 3, 2, dtype=torch.bool)
        micro_tile_mask = torch.ones(2, 3, 2, dtype=torch.bool)
        macro_tile_mask[:, 1] = False
        micro_tile_mask[:, 0] = False
        temporal_mask = macro_tile_mask.any(-1) | micro_tile_mask.any(-1)
        temporal_features = torch.randn(2, 3, 1)

        with torch.no_grad():
            logits = model(
                macro,
                micro,
                temporal_mask,
                macro_tile_mask,
                micro_tile_mask,
                temporal_features,
            )

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
