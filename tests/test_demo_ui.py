import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
import demo_ui


class DemoUiTests(unittest.TestCase):
    def test_default_pipeline_is_dinov3_and_command_uses_selected_inputs(self):
        command = demo_ui.build_inference_command(
            python_executable="python",
            pipeline="DINOv3 (default)",
            checkpoint="weights.pt",
            image_path="target.jpeg",
            image_dir="temporal",
            embedding_cache="cache.pt",
            backbone_source="backbone.pt",
            device="cpu",
        )
        self.assertIn(str(demo_ui.PIPELINES["DINOv3 (default)"]), command)
        self.assertEqual(command[command.index("--checkpoint") + 1], "weights.pt")
        self.assertEqual(command[command.index("--image-dir") + 1], "temporal")
        self.assertEqual(command[command.index("--image-backbone-weights") + 1], "backbone.pt")
        self.assertNotIn("--image-backbone", command)
        self.assertEqual(command[command.index("--device") + 1], "cpu")

    def test_command_requires_demo_inputs(self):
        with self.assertRaisesRegex(ValueError, "temporal weights"):
            demo_ui.build_inference_command(
                python_executable="python",
                pipeline="DINOv3 (default)",
                checkpoint="",
                image_path="target.jpeg",
                image_dir="temporal",
            )

    def test_temporal_folder_is_optional_for_single_image_mode(self):
        command = demo_ui.build_inference_command(
            python_executable="python",
            pipeline="DINOv3 (default)",
            checkpoint="temporal.pt",
            image_path="target.jpeg",
            image_dir="",
        )
        self.assertNotIn("--image-dir", command)

    def test_blank_backbone_source_uses_pipeline_default(self):
        command = demo_ui.build_inference_command(
            python_executable="python",
            pipeline="DINOv3 (default)",
            checkpoint="temporal.pt",
            image_path="target.jpeg",
            image_dir="temporal",
            backbone_source="",
        )
        self.assertEqual(
            command[command.index("--image-backbone") + 1],
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
        )

    def test_json_result_allows_warning_prefix(self):
        expected = {"prediction": "BBCH3", "top_k": []}
        output = "A harmless warning\n" + json.dumps(expected, indent=2)
        self.assertEqual(demo_ui.parse_json_result(output), expected)


if __name__ == "__main__":
    unittest.main()
