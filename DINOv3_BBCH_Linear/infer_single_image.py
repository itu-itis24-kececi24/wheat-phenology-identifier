"""Single-image inference for the non-temporal DINOv3 BBCH linear model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DINOV3_DIR = PROJECT_ROOT / "DINOv3_BBCH"
for path in (str(DINOV3_DIR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from linear_phenology import DINOv3LinearClassifier, LINEAR_METADATA_COLUMNS  # noqa: E402
from multiscale_phenology import (  # noqa: E402
    WEATHER_MISSING_FEATURE_COLUMNS,
    add_weather_metadata,
    station_location_features,
    _extract_date,
)
from precompute_multiscale_embeddings import (  # noqa: E402
    build_image_transform,
    select_tiles,
    tile_boxes,
)


def resolve_target_date(image_path: str, target_date: Optional[str]) -> pd.Timestamp:
    if target_date:
        return pd.Timestamp(target_date).normalize()
    parsed = _extract_date(image_path)
    if parsed is None:
        raise ValueError("Provide --target-date when the image filename does not contain a date")
    return pd.Timestamp(parsed).normalize()


def build_metadata(
    target_date,
    planting_date,
    station_code,
    weather_cache,
    gdd_base_temp=0.0,
    allow_missing_weather=False,
):
    target_date = pd.Timestamp(target_date).normalize()
    planting_date = pd.Timestamp(planting_date).normalize()
    if target_date < planting_date:
        raise ValueError("Target date cannot be earlier than planting date")
    dates = pd.date_range(planting_date, target_date, freq="D")
    frame = pd.DataFrame({
        "station_year": "inference",
        "group_id": 0,
        "station_code": str(station_code),
        "date": dates,
        "planting_date": planting_date,
    })
    frame = add_weather_metadata(
        frame,
        weather_cache,
        force_refresh=False,
        gdd_base_temp=gdd_base_temp,
    )
    if (
        not allow_missing_weather
        and frame[list(WEATHER_MISSING_FEATURE_COLUMNS)].to_numpy(dtype=float).mean() >= 0.999
    ):
        raise RuntimeError(
            "Weather is unavailable. Supply the training weather cache or use "
            "--allow-missing-weather only for a diagnostic run."
        )
    location = station_location_features(station_code, strict=True)
    target = frame.iloc[-1]
    values = torch.tensor(
        [float(target["weather_gdd_cum_norm"]), *location.tolist()],
        dtype=torch.float32,
    )
    return values, {
        "weather_gdd_cum_norm": float(target["weather_gdd_cum_norm"]),
        "location_latitude_norm": float(location[0]),
        "location_longitude_norm": float(location[1]),
        "location_elevation_norm": float(location[2]),
    }


def build_model_from_checkpoint(checkpoint: Dict, device, extractor=None):
    config = dict(checkpoint["model_config"])
    config.pop("preprocess", None)
    config["pretrained"] = False
    config["extractor"] = extractor
    model = DINOv3LinearClassifier(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_image_tiles(image_path, checkpoint, device):
    tiling = checkpoint["tiling"]
    preprocess = checkpoint["model_config"]["preprocess"]
    transform = build_image_transform(
        image_size=int(tiling["vit_image_size"]),
        augment=False,
        mean=preprocess["mean"],
        std=preprocess["std"],
    )
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        boxes = tile_boxes(
            image.width,
            image.height,
            int(tiling["tile_size"]),
            int(tiling["tile_stride"]),
        )
        boxes = select_tiles(boxes, int(tiling["max_tiles"]))
        tiles = torch.stack([transform(image.crop(box)) for box in boxes])
    return tiles.unsqueeze(0).to(device), torch.ones(1, len(boxes), dtype=torch.bool, device=device)


def format_prediction(logits, classes, metadata_values=None):
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()
    order = torch.argsort(probabilities, descending=True)
    return {
        "prediction": classes[int(order[0])],
        "confidence": float(probabilities[int(order[0])]),
        "probabilities": [
            {"label": classes[int(index)], "probability": float(probabilities[int(index)])}
            for index in order
        ],
        "metadata": metadata_values or {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--planting-date", required=True)
    parser.add_argument("--station-code", required=True)
    parser.add_argument("--weather-cache", default=None)
    parser.add_argument("--allow-missing-weather", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("metadata_columns", ())) != tuple(LINEAR_METADATA_COLUMNS):
        raise ValueError("Checkpoint metadata columns do not match this inference implementation")
    model = build_model_from_checkpoint(checkpoint, device)
    target_date = resolve_target_date(args.image_path, args.target_date)
    metadata, metadata_values = build_metadata(
        target_date,
        args.planting_date,
        args.station_code,
        args.weather_cache,
        gdd_base_temp=float(checkpoint.get("gdd_base_temp", 0.0)),
        allow_missing_weather=args.allow_missing_weather,
    )
    tiles, tile_mask = load_image_tiles(args.image_path, checkpoint, device)
    with torch.no_grad():
        logits = model(tiles, tile_mask, metadata.unsqueeze(0).to(device))
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "image_path": os.path.abspath(args.image_path),
        "target_date": str(target_date.date()),
        "planting_date": str(pd.Timestamp(args.planting_date).date()),
        "station_code": args.station_code,
        **format_prediction(logits, checkpoint["classes"], metadata_values),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
