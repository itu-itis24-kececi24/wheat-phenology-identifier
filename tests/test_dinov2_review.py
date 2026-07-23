import importlib.util
import os
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DINOV2 = load_module("dinov2_review_model", os.path.join("DINOv2", "multiscale_phenology.py"))
GATED = load_module("dinov2_gated_review_model", os.path.join("DINOv2_gated", "multiscale_phenology.py"))


class DINOv2ReviewTests(unittest.TestCase):
    def test_cumulative_gdd_includes_dates_without_images(self):
        daily = pd.DataFrame(
            {
                "station_year": ["02.02_2020", "02.02_2020"],
                "group_id": [1, 1],
                "station_code": [2.02, 2.02],
                "date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
                "planting_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            }
        )
        weather = pd.DataFrame(
            {
                "station_family": ["02", "02", "02"],
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "tavg": [10.0, 10.0, 10.0],
                "tmin": [5.0, 5.0, 5.0],
                "tmax": [15.0, 15.0, 15.0],
                "prcp": [0.0, 0.0, 0.0],
            }
        )
        original = DINOV2.build_or_load_meteostat_weather_cache
        DINOV2.build_or_load_meteostat_weather_cache = lambda *args, **kwargs: weather.copy()
        try:
            result = DINOV2.add_weather_metadata(daily, cache_path=None, gdd_base_temp=0.0)
        finally:
            DINOV2.build_or_load_meteostat_weather_cache = original
        self.assertEqual(result["weather_gdd_cum_raw"].tolist(), [10.0, 30.0])

    def test_station_group_folds_do_not_split_repeated_station_years(self):
        frame = pd.DataFrame(
            {
                "station_code": ["02.03", "02.03", "06.01", "11.03", "27.05"],
                "group_id": [1, 2, 3, 4, 5],
            }
        )
        folds = DINOV2.generate_group_train_val_test_folds(
            frame,
            group_col="station_code",
            n_train=2,
            n_val=1,
            n_test=1,
            num_folds=1,
            random_state=7,
        )
        train_idx, val_idx, test_idx = folds[0]
        train = set(frame.iloc[train_idx]["station_code"])
        val = set(frame.iloc[val_idx]["station_code"])
        test = set(frame.iloc[test_idx]["station_code"])
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)

    def test_mean_cache_mask_reports_missing_embedding(self):
        dataset = object.__new__(DINOV2.MultiScaleEmbeddingWindowDataset)
        dataset.tile_attention = False
        dataset.macro_embeddings = {}
        dataset.micro_embeddings = {DINOV2._path_key("present.jpeg"): torch.ones(8)}
        self.assertFalse(dataset._tile_mask("missing.jpeg", "micro").item())
        self.assertTrue(dataset._tile_mask("present.jpeg", "micro").item())

    def test_gated_fusion_ignores_a_stream_marked_missing(self):
        torch.manual_seed(3)
        fusion = GATED.GatedMultiviewFusion(input_dim=8, embed_dim=8, dropout=0.0).eval()
        macro = torch.randn(2, 4, 8)
        micro_a = torch.randn(2, 4, 8)
        micro_b = micro_a + 1000.0
        macro_valid = torch.ones(2, 4, dtype=torch.bool)
        micro_valid = torch.zeros(2, 4, dtype=torch.bool)
        first = fusion(macro, micro_a, macro_valid=macro_valid, micro_valid=micro_valid)
        second = fusion(macro, micro_b, macro_valid=macro_valid, micro_valid=micro_valid)
        torch.testing.assert_close(first, second)

    def test_hybrid_ordinal_loss_is_finite(self):
        logits = torch.tensor([[1.0, 0.5, -0.5]], requires_grad=True)
        target = torch.tensor([[0.0, 1.0, 0.0]])
        loss = DINOV2.HybridOrdinalLoss(power=2, cross_entropy_weight=0.5)(logits, target)
        self.assertTrue(np.isfinite(loss.item()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_cached_models_forward_with_pre_norm(self):
        batch, days, tiles, feature_dim = 2, 5, 3, 16
        macro = torch.randn(batch, days, tiles, feature_dim)
        micro = torch.randn(batch, days, tiles, feature_dim)
        macro_tile_mask = torch.ones(batch, days, tiles, dtype=torch.bool)
        micro_tile_mask = torch.ones(batch, days, tiles, dtype=torch.bool)
        temporal_mask = torch.ones(batch, days, dtype=torch.bool)
        temporal_features = torch.randn(batch, days, 1)

        single = DINOV2.SingleStreamEmbeddingTemporalTransformer(
            feature_dim=feature_dim,
            stream="micro",
            num_classes=8,
            temporal_layers=1,
            temporal_heads=8,
            temporal_feature_dim=1,
            temporal_norm_first=True,
        ).eval()
        single_logits = single(
            macro,
            micro,
            temporal_mask,
            macro_tile_mask,
            micro_tile_mask,
            temporal_features,
        )
        self.assertEqual(tuple(single_logits.shape), (batch, 8))

        gated = GATED.MultiScaleEmbeddingTemporalTransformer(
            feature_dim=feature_dim,
            num_classes=8,
            temporal_layers=1,
            temporal_heads=8,
            temporal_feature_dim=1,
            temporal_norm_first=True,
        ).eval()
        micro_tile_mask[:, 2] = False
        gated_logits = gated(
            macro,
            micro,
            temporal_mask,
            macro_tile_mask,
            micro_tile_mask,
            temporal_features,
        )
        self.assertEqual(tuple(gated_logits.shape), (batch, 8))


if __name__ == "__main__":
    unittest.main()
