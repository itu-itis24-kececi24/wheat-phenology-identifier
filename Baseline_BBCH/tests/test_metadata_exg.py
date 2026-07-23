import concurrent.futures
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "run_metadata_exg_training.py"
sys.path.insert(0, str(SCRIPT.parent))
import run_metadata_exg_training as MODULE


class MetadataExgTests(unittest.TestCase):
    def test_exg_is_high_for_green_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "green.png"
            Image.fromarray(np.full((20, 20, 3), [20, 220, 20], dtype=np.uint8)).save(path)
            features = MODULE.extract_exg_features(str(path))
        self.assertGreater(features["exg_mean"], 0.5)
        self.assertEqual(features["exg_missing"], 0.0)

    def test_causal_rolling_feature_does_not_see_future(self):
        rows = []
        for offset, value in enumerate([0.1, 0.2, 9.0]):
            row = {
                "station_year": "01.01_2014",
                "date": pd.Timestamp("2014-01-01") + pd.Timedelta(days=offset),
                "planting_date": pd.Timestamp("2013-10-01"),
            }
            for column in MODULE.EXG_COLUMNS:
                row[column] = value if column == "exg_mean" else 0.0
            for column in MODULE.WEATHER_TEMPORAL_FEATURE_COLUMNS:
                row[column] = 0.0
            for column in MODULE.LOCATION_FEATURE_COLUMNS:
                row[column] = 0.0
            rows.append(row)
        featured, _ = MODULE.add_causal_features(pd.DataFrame(rows), window_days=21)
        self.assertAlmostEqual(featured.iloc[1]["exg_mean_causal_21d_mean"], 0.15, places=6)

    def test_corrupt_image_is_marked_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jpeg"
            path.write_text("not an image", encoding="utf-8")
            features = MODULE.extract_exg_features(str(path))
        self.assertEqual(features["exg_missing"], 1.0)

    def test_parallel_exg_cache_extracts_unique_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, color in enumerate(([20, 220, 20], [180, 80, 20])):
                path = root / f"{index}.png"
                Image.fromarray(np.full((20, 20, 3), color, dtype=np.uint8)).save(path)
                paths.append(str(path))
            frame = pd.DataFrame({"micro_path": [paths[0], paths[1], paths[0]]})
            featured = MODULE.add_exg_features(
                frame,
                str(root / "cache.csv"),
                workers=2,
                max_side=32,
            )
            cache = pd.read_csv(root / "cache.csv")
        self.assertEqual(len(cache), 2)
        self.assertEqual(len(featured), 3)
        self.assertEqual(featured.iloc[0]["exg_mean"], featured.iloc[2]["exg_mean"])

    def test_fold_worker_runs_in_a_process(self):
        x = np.arange(140 * 4, dtype=np.float32).reshape(140, 4) / 100.0
        y = np.tile(np.arange(7), 20)
        task = (
            1,
            np.arange(0, 98),
            np.arange(98, 119),
            np.arange(119, 140),
            x,
            y,
            [f"class_{i}" for i in range(7)],
            {
                "max_iter_candidates": [2],
                "learning_rate": 0.1,
                "max_leaf_nodes": 5,
                "l2_regularization": 1.0,
                "seed": 42,
            },
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            result = executor.submit(MODULE._fit_fold_task, task).result(timeout=30)
        self.assertEqual(result["fold_id"], 1)
        self.assertEqual(len(result["test_pred"]), 21)


if __name__ == "__main__":
    unittest.main()
