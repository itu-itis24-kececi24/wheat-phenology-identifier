import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(ROOT, "DINOv2_BBCH", "multiscale_phenology.py")
SPEC = importlib.util.spec_from_file_location("dinov2_bbch_model", MODULE_PATH)
BBCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BBCH)

TRAINING_PATH = os.path.join(ROOT, "DINOv2_BBCH", "run_multiscale_training.py")
sys.path.insert(0, os.path.dirname(TRAINING_PATH))
TRAINING_SPEC = importlib.util.spec_from_file_location("dinov2_bbch_training", TRAINING_PATH)
TRAINING = importlib.util.module_from_spec(TRAINING_SPEC)
TRAINING_SPEC.loader.exec_module(TRAINING)


class DINOv2BBCHTests(unittest.TestCase):
    def test_class_order_matches_bbch_intervals(self):
        self.assertEqual(
            BBCH.BASE_CLASSES,
            [
                "OffSeason",
                "BBCH0",
                "BBCH1",
                "BBCH2",
                "BBCH3",
                "BBCH5",
                "BBCH6_7",
                "BBCH8",
            ],
        )
        self.assertEqual(BBCH.PHENOLOGY_BOUNDARY_OFFSET, 0)

    def test_csv_and_xlsx_stage_dates_match(self):
        csv_table = BBCH._load_phenology_excel(
            os.path.join(ROOT, "labeling_bbch_iso_dates.csv")
        )
        xlsx_table = BBCH._load_phenology_excel(
            os.path.join(ROOT, "labeling_bbch_iso_dates.xlsx")
        )
        pd.testing.assert_frame_equal(
            csv_table[BBCH.STAGE_COLUMNS],
            xlsx_table[BBCH.STAGE_COLUMNS],
            check_dtype=False,
        )

    def test_supplied_durations_match_milestone_dates(self):
        table = pd.read_csv(os.path.join(ROOT, "labeling_bbch_iso_dates.csv"))
        dates = table[BBCH.STAGE_COLUMNS].apply(pd.to_datetime)
        duration_columns = [
            "BBCH 0",
            "BBCH 1",
            "BBCH 2",
            "BBCH 3",
            "BBCH 5",
            "BBCH 6_7",
            "BBCH 8",
        ]
        for index, duration_column in enumerate(duration_columns):
            calculated = (
                dates[BBCH.STAGE_COLUMNS[index + 1]]
                - dates[BBCH.STAGE_COLUMNS[index]]
            ).dt.days
            self.assertTrue(calculated.equals(table[duration_column]))

    def test_boundary_date_prefers_new_interval_with_soft_target(self):
        table = BBCH._load_phenology_excel(
            os.path.join(ROOT, "labeling_bbch_iso_dates.csv")
        )
        boundaries = [pd.Timestamp(value) for value in BBCH._stage_boundaries(table.iloc[0])]
        class_to_idx = {label: index for index, label in enumerate(BBCH.BASE_CLASSES)}
        target = BBCH._soft_interval_label(
            boundaries[1],
            boundaries,
            class_to_idx,
            transition_days=2,
        )
        self.assertEqual(BBCH.BASE_CLASSES[int(np.argmax(target))], "BBCH1")
        self.assertAlmostEqual(float(target[class_to_idx["BBCH0"]]), 0.4)
        self.assertAlmostEqual(float(target[class_to_idx["BBCH1"]]), 0.6)

    @staticmethod
    def _coverage_frame(labels):
        dates = pd.date_range("2020-01-01", periods=len(labels), freq="D")
        return pd.DataFrame(
            {
                "station_year": "01.01_2020",
                "date": dates,
                "label": labels,
                "micro_path": [f"image_{index}.jpeg" for index in range(len(labels))],
                "macro_path": [None] * len(labels),
                "target": [[1.0]] * len(labels),
                "date_score": [[1.0]] * len(labels),
            }
        )

    def test_stage_support_filter_only_removes_training_candidates_below_threshold(self):
        frame = self._coverage_frame(["BBCH0"] * 10 + ["BBCH1"] * 20)
        config = BBCH.WindowConfig(
            window_days=5,
            center_offset=4,
            classes=("BBCH0", "BBCH1"),
            stream="micro",
        )
        unfiltered = BBCH.MultiScaleWindowDataset(
            frame,
            config,
            transform=lambda _: torch.zeros(3, 224, 224),
        )
        filtered = BBCH.MultiScaleWindowDataset(
            frame,
            config,
            transform=lambda _: torch.zeros(3, 224, 224),
            min_stage_support_days=12,
        )
        self.assertEqual(len(unfiltered), 30)
        self.assertEqual(len(filtered), 20)
        self.assertEqual(filtered.filter_summary["excluded_stage_support_samples"], 10)

    def test_offseason_rows_count_as_context_without_becoming_targets(self):
        frame = self._coverage_frame(["OffSeason"] * 5 + ["BBCH0"] * 5)
        config = BBCH.WindowConfig(
            window_days=5,
            center_offset=4,
            classes=("BBCH0",),
            stream="micro",
        )
        dataset = BBCH.MultiScaleWindowDataset(
            frame,
            config,
            transform=lambda _: torch.zeros(3, 224, 224),
            min_window_coverage_days=5,
        )
        self.assertEqual(len(dataset), 5)
        self.assertTrue(all(station_date[1] >= pd.Timestamp("2020-01-06") for station_date in dataset.samples))
        self.assertTrue(
            all(value["window_coverage_days"] == 5 for value in dataset.sample_coverage.values())
        )

    def test_weather_missing_flags_are_recorded_before_interpolation(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        daily = pd.DataFrame(
            {
                "station_year": ["01.01_2020"] * 3,
                "group_id": [1] * 3,
                "station_code": ["01.01"] * 3,
                "date": dates,
                "planting_date": [dates[0]] * 3,
            }
        )
        weather = pd.DataFrame(
            {
                "station_family": ["01", "01"],
                "date": dates[:2],
                "tavg": [10.0, np.nan],
                "tmin": [5.0, 6.0],
                "tmax": [15.0, 16.0],
                "prcp": [0.0, np.nan],
            }
        )
        with patch.object(BBCH, "build_or_load_meteostat_weather_cache", return_value=weather):
            result = BBCH.add_weather_metadata(daily, cache_path=None)

        self.assertEqual(float(result.loc[0, "weather_tavg_missing"]), 0.0)
        self.assertEqual(float(result.loc[1, "weather_tavg_missing"]), 1.0)
        self.assertEqual(float(result.loc[1, "weather_prcp_missing"]), 1.0)
        self.assertEqual(float(result.loc[2, "weather_tmin_missing"]), 1.0)
        self.assertTrue(np.isfinite(result.loc[1, "weather_tavg_norm"]))

    def test_integer_inferred_weather_family_matches_zero_padded_station(self):
        date = pd.Timestamp("2020-01-01")
        daily = pd.DataFrame(
            {
                "station_year": ["01.01_2020"],
                "group_id": [1],
                "station_code": ["01.01"],
                "date": [date],
                "planting_date": [date],
            }
        )
        # pandas infers a CSV column containing 01 as integer 1 unless its
        # dtype is explicitly preserved.
        cached_weather = pd.DataFrame(
            {
                "station_family": [1],
                "date": [date],
                "tmin": [10.0],
                "tmax": [20.0],
                "prcp": [2.0],
            }
        )
        with patch.object(
            BBCH,
            "build_or_load_meteostat_weather_cache",
            return_value=cached_weather,
        ):
            result = BBCH.add_weather_metadata(daily, cache_path=None)

        self.assertAlmostEqual(float(result.loc[0, "weather_tavg_norm"]), 15.0 / BBCH.WEATHER_TEMP_SCALE)
        self.assertAlmostEqual(float(result.loc[0, "weather_tmin_norm"]), 10.0 / BBCH.WEATHER_TEMP_SCALE)
        self.assertAlmostEqual(float(result.loc[0, "weather_tmax_norm"]), 20.0 / BBCH.WEATHER_TEMP_SCALE)
        self.assertEqual(float(result.loc[0, "weather_tmin_missing"]), 0.0)
        self.assertEqual(float(result.loc[0, "weather_tmax_missing"]), 0.0)
        self.assertEqual(float(result.loc[0, "weather_prcp_missing"]), 0.0)

    def test_missing_weather_defaults_to_flagged_not_observed(self):
        features = BBCH._temporal_features_for_date(
            pd.Timestamp("2020-01-01"),
            planting_date=None,
            row=None,
            feature_columns=BBCH.WEATHER_TEMPORAL_FEATURE_COLUMNS,
        )
        value_count = len(BBCH.WEATHER_VALUE_FEATURE_COLUMNS)
        self.assertTrue(torch.equal(features[:value_count], torch.zeros(value_count)))
        self.assertTrue(
            torch.equal(
                features[value_count:],
                torch.ones(len(BBCH.WEATHER_MISSING_FEATURE_COLUMNS)),
            )
        )

    def test_weather_feature_sets_default_to_cumulative_gdd_only(self):
        self.assertEqual(
            BBCH.WEATHER_FEATURE_SETS["cumulative"],
            ("weather_gdd_cum_norm",),
        )
        self.assertEqual(
            BBCH.WEATHER_FEATURE_SETS["daily_cumulative"],
            ("weather_gdd_norm", "weather_gdd_cum_norm"),
        )

    def test_gated_calendar_and_weather_branches_receive_gradients(self):
        model = BBCH.SingleStreamEmbeddingTemporalTransformer(
            feature_dim=8,
            embed_dim=16,
            num_classes=3,
            target_index=2,
            temporal_layers=1,
            temporal_heads=4,
            temporal_feature_dim=2,
            temporal_feature_hidden_dim=8,
            weather_feature_dim=1,
            temporal_feature_fusion="gated",
            temporal_feature_gate_init=0.1,
            weather_feature_gate_init=0.1,
            temporal_aggregation="mean",
        )
        features = torch.randn(2, 3, 2, 8)
        mask = torch.ones(2, 3, dtype=torch.bool)
        tile_mask = torch.ones(2, 3, 2, dtype=torch.bool)
        # Column order is calendar metadata first, then selected weather data.
        metadata = torch.tensor(
            [
                [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
                [[0.4, 0.1], [0.5, 0.2], [0.6, 0.3]],
            ]
        )
        logits = model(
            macro=torch.zeros_like(features),
            micro=features,
            mask=mask,
            macro_tile_mask=torch.zeros_like(tile_mask),
            micro_tile_mask=tile_mask,
            temporal_features=metadata,
        )
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertAlmostEqual(
            torch.sigmoid(model.temporal_feature_gate_logit.detach()).item(),
            0.1,
            places=6,
        )
        self.assertAlmostEqual(
            torch.sigmoid(model.weather_feature_gate_logit.detach()).item(),
            0.1,
            places=6,
        )
        logits.sum().backward()
        self.assertGreater(float(model.temporal_feature_mlp[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.weather_feature_mlp[0].weight.grad.abs().sum()), 0.0)

    def test_legacy_metadata_fusion_keeps_old_checkpoint_parameter_names(self):
        model = BBCH.SingleStreamEmbeddingTemporalTransformer(
            feature_dim=8,
            embed_dim=16,
            num_classes=3,
            target_index=2,
            temporal_layers=1,
            temporal_heads=4,
            temporal_feature_dim=2,
            temporal_feature_hidden_dim=8,
            temporal_feature_fusion="legacy",
        )
        state_keys = set(model.state_dict())
        self.assertIn("temporal_feature_fusion.0.weight", state_keys)
        self.assertIn("temporal_feature_fusion.1.weight", state_keys)
        self.assertFalse(any(key.startswith("weather_feature_mlp") for key in state_keys))

    def test_station_location_features_are_normalized_from_city_family(self):
        features = BBCH.station_location_features("02.06", strict=True)
        expected = torch.tensor(
            [
                (37.76 - BBCH.LOCATION_LATITUDE_CENTER) / BBCH.LOCATION_LATITUDE_SCALE,
                (38.2761 - BBCH.LOCATION_LONGITUDE_CENTER) / BBCH.LOCATION_LONGITUDE_SCALE,
                (669.0 - BBCH.LOCATION_ELEVATION_CENTER) / BBCH.LOCATION_ELEVATION_SCALE,
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(features, expected)

    def test_location_metadata_rejects_unknown_station_in_strict_mode(self):
        daily = pd.DataFrame(
            {
                "station_code": ["02.06", "99.01"],
                "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            }
        )
        with self.assertRaisesRegex(ValueError, "99"):
            BBCH.add_location_metadata(daily, strict=True)

    def test_location_mlp_fuses_static_features_and_receives_gradients(self):
        model = BBCH.SingleStreamEmbeddingTemporalTransformer(
            feature_dim=8,
            embed_dim=16,
            num_classes=3,
            target_index=2,
            temporal_layers=1,
            temporal_heads=4,
            temporal_feature_dim=0,
            location_feature_dim=3,
            location_feature_hidden_dim=8,
            location_gate_init=0.1,
            temporal_aggregation="mean",
        )
        features = torch.randn(2, 3, 2, 8)
        mask = torch.ones(2, 3, dtype=torch.bool)
        tile_mask = torch.ones(2, 3, 2, dtype=torch.bool)
        location = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.8]])
        logits = model(
            macro=torch.zeros_like(features),
            micro=features,
            mask=mask,
            macro_tile_mask=torch.zeros_like(tile_mask),
            micro_tile_mask=tile_mask,
            temporal_features=torch.zeros(2, 3, 0),
            location_features=location,
        )
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(logits).all())
        learned_gate = torch.sigmoid(model.location_gate_logit.detach()).item()
        self.assertAlmostEqual(learned_gate, 0.1, places=6)
        logits.sum().backward()
        self.assertIsNotNone(model.location_feature_mlp[0].weight.grad)
        self.assertGreater(float(model.location_feature_mlp[0].weight.grad.abs().sum()), 0.0)

    def test_location_disabled_preserves_legacy_state_dict_shape(self):
        model = BBCH.SingleStreamEmbeddingTemporalTransformer(
            feature_dim=8,
            embed_dim=16,
            num_classes=3,
            target_index=2,
            temporal_layers=1,
            temporal_heads=4,
            location_feature_dim=0,
        )
        self.assertFalse(any(key.startswith("location_") for key in model.state_dict()))

    def test_monotonic_viterbi_decoder_never_moves_backward(self):
        probabilities = [
            [0.90, 0.09, 0.01],
            [0.05, 0.90, 0.05],
            [0.80, 0.15, 0.05],
            [0.01, 0.09, 0.90],
        ]
        decoded = TRAINING.monotonic_viterbi_decode(probabilities)
        self.assertEqual(decoded, [0, 1, 1, 2])
        self.assertTrue(all(next_stage >= stage for stage, next_stage in zip(decoded, decoded[1:])))
        self.assertTrue(all(next_stage - stage <= 1 for stage, next_stage in zip(decoded, decoded[1:])))

    def test_monotonic_decoder_can_cross_stages_after_image_gap(self):
        probabilities = [
            [0.99, 0.005, 0.005],
            [0.005, 0.005, 0.99],
        ]
        self.assertEqual(
            TRAINING.monotonic_viterbi_decode(probabilities, max_advances=[2]),
            [0, 2],
        )


if __name__ == "__main__":
    unittest.main()
