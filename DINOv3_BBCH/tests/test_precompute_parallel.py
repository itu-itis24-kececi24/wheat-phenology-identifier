import os
import sys
import tempfile
import unittest

import torch
from PIL import Image


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import precompute_multiscale_embeddings as precompute


class DummyDenseModel(torch.nn.Module):
    def forward_dense(self, images, grid_size=2, include_cls=True):
        values = images.mean(dim=(1, 2, 3)).reshape(-1, 1, 1)
        tokens = grid_size**2 + int(include_cls)
        return values.expand(-1, tokens, 3)


class ParallelPrecomputeTests(unittest.TestCase):
    def test_batches_images_and_restores_variable_tile_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            image_specs = (
                ("dark.png", 32, (448, 224)),
                ("light.png", 224, (672, 224)),
            )
            for name, value, size in image_specs:
                path = os.path.join(directory, name)
                Image.new("RGB", size, color=(value, value, value)).save(path)
                paths.append(path)

            encoded, tile_counts, failures = precompute.encode_tiled_paths(
                DummyDenseModel(),
                paths,
                batch_size=4,
                num_workers=2,
                device=torch.device("cpu"),
                tile_size=224,
                tile_stride=224,
                image_size=224,
                output_dtype=torch.float32,
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                image_batch_size=2,
                prefetch_factor=2,
            )

            normalized_paths = [precompute.path_key(path) for path in paths]
            self.assertFalse(failures)
            self.assertEqual([tile_counts[path] for path in normalized_paths], [2, 3])
            self.assertEqual(tuple(encoded[normalized_paths[0]].shape), (2, 5, 3))
            self.assertEqual(tuple(encoded[normalized_paths[1]].shape), (3, 5, 3))
            self.assertLess(encoded[normalized_paths[0]].mean(), encoded[normalized_paths[1]].mean())

    def test_rejects_invalid_batch_sizes(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            precompute.encode_tiled_paths(
                DummyDenseModel(),
                [],
                batch_size=0,
                num_workers=0,
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "image_batch_size"):
            precompute.encode_tiled_paths(
                DummyDenseModel(),
                [],
                batch_size=1,
                image_batch_size=0,
                num_workers=0,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
