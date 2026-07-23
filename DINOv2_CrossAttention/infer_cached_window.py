import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image

from multiscale_phenology import (
    BASE_CLASSES,
    DINO_DEFAULT_BACKBONE,
    DEFAULT_TEMPORAL_FEATURE_COLUMNS,
    IMAGE_EXTS,
    MultiScaleEmbeddingTemporalTransformer,
    SingleStreamEmbeddingTemporalTransformer,
    WEATHER_TEMPORAL_FEATURE_COLUMNS,
    ViTBackboneFeatureExtractor,
    add_weather_metadata,
    _extract_date,
    _temporal_features_for_date,
)
from precompute_multiscale_embeddings import TileTransform, path_key, select_tiles, tile_boxes


INFERENCE_CURRENT_DATA_RE = re.compile(
    r"(?P<station>\d{2}[_.,]\d{2})-(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
)
INFERENCE_DATE_RE = re.compile(r"(?P<year>\d{4})[_-](?P<month>\d{2})[_-](?P<day>\d{2})")


def parse_image_date(path_or_name: str) -> Optional[pd.Timestamp]:
    name = os.path.basename(path_or_name)
    match = INFERENCE_CURRENT_DATA_RE.search(name)
    if match is None:
        match = INFERENCE_DATE_RE.search(name)
    if match is None:
        return _extract_date(path_or_name)
    try:
        return pd.Timestamp(
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
        ).normalize()
    except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
        return _extract_date(path_or_name)


def parse_station_code(path_or_name: Optional[str]) -> Optional[str]:
    if not path_or_name:
        return None
    match = INFERENCE_CURRENT_DATA_RE.search(os.path.basename(path_or_name))
    if match is None:
        return None
    return match.group("station").replace("_", ".").replace(",", ".")


def image_matches_stream(path: str, stream: str) -> bool:
    lowered = os.path.normpath(path).lower()
    filename = os.path.basename(lowered)
    parts = lowered.split(os.sep)
    if stream == "micro":
        return "10x" in parts or "-10x" in filename or "_10x" in filename
    if stream == "macro":
        return "1x" in parts or "-1x" in filename or "_1x" in filename
    return True


def find_images_by_date(image_dir: str, stream: str = "micro") -> Dict[pd.Timestamp, str]:
    if os.path.isfile(image_dir):
        image_dir = os.path.dirname(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    by_date = {}
    for dirpath, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() not in IMAGE_EXTS:
                continue
            path = os.path.join(dirpath, filename)
            if not image_matches_stream(path, stream):
                continue
            date = parse_image_date(path)
            if date is None:
                continue
            date = date.normalize()
            if date in by_date:
                current = path_key(by_date[date])
                candidate = path_key(path)
                if candidate < current:
                    by_date[date] = path
                continue
            by_date[date] = path
    return by_date


def infer_date(args) -> pd.Timestamp:
    if args.target_date:
        return pd.Timestamp(args.target_date).normalize()
    if args.image_path:
        date = parse_image_date(args.image_path)
        if date is not None:
            return date.normalize()
    raise ValueError("Provide --target-date, or use an --image-path whose filename contains a date.")


def load_tiling_config(checkpoint: Dict, embedding_cache: Optional[Dict], args) -> Dict:
    tiling = {}
    if embedding_cache is not None:
        tiling.update(embedding_cache.get("tiling", {}))
    tiling.setdefault("tile_pooling", args.tile_pooling)
    tiling.setdefault("tile_size", args.tile_size)
    tiling.setdefault("tile_stride", args.tile_stride)
    tiling.setdefault("max_tiles", args.max_tiles)
    tiling.setdefault("vit_image_size", args.vit_image_size)
    return tiling


@torch.no_grad()
def encode_one_image(
    model,
    image_path: str,
    device: torch.device,
    tile_size: int,
    tile_stride: int,
    max_tiles: int,
    image_size: int,
    tile_pooling: str,
    tile_batch_size: int,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> torch.Tensor:
    transform = TileTransform(image_size=image_size, mean=mean, std=std)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        boxes = select_tiles(tile_boxes(image.width, image.height, tile_size, tile_stride), max_tiles)
        features = []
        for start in range(0, len(boxes), tile_batch_size):
            batch_boxes = boxes[start:start + tile_batch_size]
            batch = torch.stack([transform(image.crop(box)) for box in batch_boxes])
            batch = batch.to(device, non_blocking=True)
            features.append(model(batch).detach().cpu())
    tile_features = torch.cat(features, dim=0).float()
    return tile_features if tile_pooling == "attention" else tile_features.mean(dim=0)


def build_window_paths(
    image_path: str,
    image_dir: Optional[str],
    target_date: pd.Timestamp,
    window_days: int,
    stream: str,
) -> List[Optional[str]]:
    by_date = find_images_by_date(image_dir, stream=stream) if image_dir else {}
    if image_path:
        by_date[target_date.normalize()] = path_key(image_path)

    start_date = target_date - pd.Timedelta(days=window_days - 1)
    return [by_date.get(start_date + pd.Timedelta(days=offset)) for offset in range(window_days)]


def pad_tile_features(features: List[torch.Tensor], feature_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_tiles = max([feat.shape[0] for feat in features if feat.ndim == 2], default=1)
    padded, masks = [], []
    for feat in features:
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        out = torch.zeros(max_tiles, feature_dim, dtype=torch.float32)
        mask = torch.zeros(max_tiles, dtype=torch.bool)
        n_tiles = min(feat.shape[0], max_tiles)
        out[:n_tiles] = feat[:n_tiles].float()
        mask[:n_tiles] = True
        padded.append(out)
        masks.append(mask)
    return torch.stack(padded), torch.stack(masks)


def build_temporal_features(
    target_date: pd.Timestamp,
    planting_date: Optional[str],
    window_days: int,
    feature_columns: List[str],
    weather_cache: Optional[str] = None,
    station_code: Optional[str] = None,
    gdd_base_temp: float = 0.0,
) -> torch.Tensor:
    if not feature_columns:
        return torch.zeros(window_days, 0)
    start_date = target_date.normalize() - pd.Timedelta(days=window_days - 1)
    dates = [start_date + pd.Timedelta(days=offset) for offset in range(window_days)]
    planting = pd.Timestamp(planting_date).normalize() if planting_date is not None else None
    rows = pd.DataFrame(
        {
            "station_year": "inference",
            "group_id": 0,
            "station_code": station_code or "",
            "date": dates,
            "planting_date": planting,
        }
    )
    needs_weather = any(col in WEATHER_TEMPORAL_FEATURE_COLUMNS for col in feature_columns)
    if needs_weather and weather_cache and station_code:
        rows = add_weather_metadata(rows, weather_cache, force_refresh=False, gdd_base_temp=gdd_base_temp)
    elif needs_weather:
        print("Warning: checkpoint expects weather metadata but --weather-cache/--station-code was not supplied; weather features are zero.", flush=True)
    rows = rows.set_index("date")
    return torch.stack(
        [
            _temporal_features_for_date(
                date,
                planting,
                row=rows.loc[date] if date in rows.index else None,
                feature_columns=feature_columns,
            )
            for date in dates
        ]
    )


def temporal_feature_columns_from_checkpoint(checkpoint: Dict) -> List[str]:
    columns = checkpoint.get("temporal_feature_columns")
    if columns is not None:
        return list(columns)
    columns = []
    if checkpoint.get("use_days_since_planting", int(checkpoint.get("temporal_feature_dim", 0)) > 0):
        columns.extend(DEFAULT_TEMPORAL_FEATURE_COLUMNS)
    if checkpoint.get("use_weather_metadata", False):
        columns.extend(WEATHER_TEMPORAL_FEATURE_COLUMNS)
    return columns


def build_model(checkpoint: Dict, feature_dim: int, classes: List[str], stream: str, device: torch.device):
    target_index = checkpoint.get("target_index")
    temporal_aggregation = checkpoint.get("temporal_aggregation", "target")
    temporal_model = checkpoint.get("temporal_model", "transformer")
    temporal_layers = int(checkpoint.get("temporal_layers", 4))
    temporal_heads = int(checkpoint.get("temporal_heads", 8))
    temporal_feature_dim = int(checkpoint.get("temporal_feature_dim", 0))
    temporal_feature_hidden_dim = int(checkpoint.get("temporal_feature_hidden_dim", 0))
    if stream == "both":
        model = MultiScaleEmbeddingTemporalTransformer(
            feature_dim=feature_dim,
            num_classes=len(classes),
            target_index=target_index,
            temporal_aggregation=temporal_aggregation,
            temporal_model=temporal_model,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            temporal_feature_dim=temporal_feature_dim,
            temporal_feature_hidden_dim=temporal_feature_hidden_dim,
        ).to(device)
    else:
        model = SingleStreamEmbeddingTemporalTransformer(
            feature_dim=feature_dim,
            stream=stream,
            num_classes=len(classes),
            target_index=target_index,
            temporal_aggregation=temporal_aggregation,
            temporal_model=temporal_model,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            temporal_feature_dim=temporal_feature_dim,
            temporal_feature_hidden_dim=temporal_feature_hidden_dim,
        ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Infer phenology with a cached/tiled embedding temporal model.")
    parser.add_argument("--checkpoint", required=True, help="Path to fold/best_model.pt or final best_model.pt.")
    parser.add_argument("--image-path", required=True, help="Target-day 10X image path.")
    parser.add_argument("--image-dir", default=None, help="Optional folder containing previous days. Dates are parsed from filenames.")
    parser.add_argument("--target-date", default=None, help="Target date YYYY-MM-DD. Defaults to date parsed from --image-path.")
    parser.add_argument("--planting-date", default=None, help="Planting/1-Ekim date YYYY-MM-DD. Required for non-zero days-since-planting metadata.")
    parser.add_argument("--station-code", default=None, help="Station code such as 02.02. Defaults to the code parsed from the image filename when possible.")
    parser.add_argument("--weather-cache", default=None, help="Meteostat weather cache CSV used when the checkpoint was trained with weather metadata.")
    parser.add_argument("--embedding-cache", default=None, help="Optional vit_embeddings.pt used to read tiling/backbone settings.")
    parser.add_argument("--stream", choices=["micro", "macro", "both"], default=None)
    parser.add_argument("--window-days", type=int, default=None)
    parser.add_argument("--image-backbone", default=DINO_DEFAULT_BACKBONE)
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained DINOv2 if no embedding cache is supplied.")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--tile-pooling", choices=["attention", "mean"], default="attention")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tile-stride", type=int, default=224)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--tile-batch-size", type=int, default=64)
    parser.add_argument("--repeat-missing", action="store_true", help="Repeat target image for missing previous days instead of masking them out.")
    parser.add_argument("--debug-window", action="store_true", help="Print parsed window dates and paths before encoding.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not checkpoint.get("uses_embedding_cache"):
        raise RuntimeError("This script is for checkpoints trained with --embedding-cache.")

    embedding_cache = torch.load(args.embedding_cache, map_location="cpu") if args.embedding_cache else None
    classes = checkpoint.get("classes", BASE_CLASSES)
    stream = args.stream or checkpoint.get("stream", "micro")
    window_days = args.window_days or int(checkpoint.get("window_days", 31))
    target_date = infer_date(args)
    tiling = load_tiling_config(checkpoint, embedding_cache, args)
    feature_dim = int(embedding_cache["feature_dim"]) if embedding_cache is not None else 768
    use_days_since_planting = bool(
        checkpoint.get("use_days_since_planting", int(checkpoint.get("temporal_feature_dim", 0)) > 0)
    )
    temporal_feature_columns = temporal_feature_columns_from_checkpoint(checkpoint)
    station_code = args.station_code or parse_station_code(args.image_path)
    backbone = embedding_cache.get("image_backbone", args.image_backbone) if embedding_cache is not None else args.image_backbone
    pretrained = bool(embedding_cache.get("pretrained", args.pretrained)) if embedding_cache is not None else args.pretrained

    encoder = ViTBackboneFeatureExtractor(backbone, pretrained=pretrained).to(device)
    encoder.eval()
    preprocess = embedding_cache.get("preprocess", {}) if embedding_cache is not None else {}
    preprocess_mean = preprocess.get("mean", getattr(encoder, "preprocess_mean", [0.485, 0.456, 0.406]))
    preprocess_std = preprocess.get("std", getattr(encoder, "preprocess_std", [0.229, 0.224, 0.225]))

    window_paths = build_window_paths(args.image_path, args.image_dir, target_date, window_days, stream)
    if args.debug_window:
        start_date = target_date - pd.Timedelta(days=window_days - 1)
        print("Resolved inference window:", flush=True)
        for offset, path in enumerate(window_paths):
            date = start_date + pd.Timedelta(days=offset)
            status = "FOUND" if path else "MISSING"
            print(f"  {date.date()} {status} {path or ''}", flush=True)
    target_feature = None
    encoded_by_path = {}
    for path in sorted({path_key(p) for p in window_paths if p is not None}):
        encoded_by_path[path] = encode_one_image(
            encoder,
            path,
            device,
            tile_size=int(tiling.get("tile_size", 224)),
            tile_stride=int(tiling.get("tile_stride", 224)),
            max_tiles=int(tiling.get("max_tiles", 0)),
            image_size=int(tiling.get("vit_image_size", 224)),
            tile_pooling=tiling.get("tile_pooling", "attention"),
            tile_batch_size=args.tile_batch_size,
            mean=preprocess_mean,
            std=preprocess_std,
        )
        if path_key(args.image_path) == path:
            target_feature = encoded_by_path[path]

    if target_feature is None:
        raise RuntimeError("Target image was not encoded. Check --image-path.")

    features, mask = [], []
    for path in window_paths:
        if path is None:
            if args.repeat_missing:
                features.append(target_feature)
                mask.append(True)
            else:
                zero_shape = target_feature.shape
                features.append(torch.zeros(zero_shape, dtype=torch.float32))
                mask.append(False)
            continue
        features.append(encoded_by_path[path_key(path)])
        mask.append(True)

    model = build_model(checkpoint, feature_dim, classes, stream, device)
    mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
    temporal_features = build_temporal_features(
        target_date,
        args.planting_date,
        window_days,
        temporal_feature_columns,
        weather_cache=args.weather_cache,
        station_code=station_code,
        gdd_base_temp=float(checkpoint.get("gdd_base_temp", 0.0)),
    ).unsqueeze(0).to(device)

    if tiling.get("tile_pooling") == "attention":
        feature_tensor, tile_mask = pad_tile_features(features, feature_dim)
        feature_tensor = feature_tensor.unsqueeze(0).to(device)
        tile_mask = tile_mask.unsqueeze(0).to(device)
        macro = feature_tensor if stream in {"macro", "both"} else torch.zeros_like(feature_tensor)
        micro = feature_tensor if stream in {"micro", "both"} else torch.zeros_like(feature_tensor)
        macro_tile_mask = tile_mask if stream in {"macro", "both"} else torch.zeros_like(tile_mask)
        micro_tile_mask = tile_mask if stream in {"micro", "both"} else torch.zeros_like(tile_mask)
        with torch.no_grad():
            logits = model(macro, micro, mask_tensor, macro_tile_mask, micro_tile_mask, temporal_features)
    else:
        feature_tensor = torch.stack([feat.float() if feat.ndim == 1 else feat.float().mean(dim=0) for feat in features])
        feature_tensor = feature_tensor.unsqueeze(0).to(device)
        macro = feature_tensor if stream in {"macro", "both"} else torch.zeros_like(feature_tensor)
        micro = feature_tensor if stream in {"micro", "both"} else torch.zeros_like(feature_tensor)
        with torch.no_grad():
            logits = model(macro, micro, mask_tensor, temporal_features=temporal_features)

    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
    top_k = min(args.top_k, len(classes))
    values, indices = torch.topk(probs, k=top_k)
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "image_path": os.path.abspath(args.image_path),
        "image_dir": os.path.abspath(args.image_dir) if args.image_dir else None,
        "target_date": str(target_date.date()),
        "window_days": window_days,
        "available_window_images": int(sum(mask)),
        "missing_window_images": int(window_days - sum(mask)),
        "stream": stream,
        "temporal_model": checkpoint.get("temporal_model", "transformer"),
        "tile_pooling": tiling.get("tile_pooling", "attention"),
        "planting_date": args.planting_date,
        "use_days_since_planting": use_days_since_planting,
        "station_code": station_code,
        "temporal_feature_columns": temporal_feature_columns,
        "weather_cache": os.path.abspath(args.weather_cache) if args.weather_cache else None,
        "prediction": classes[int(indices[0])],
        "confidence": float(values[0]),
        "top_k": [
            {"label": classes[int(idx)], "probability": float(prob)}
            for prob, idx in zip(values, indices)
        ],
        "used_dates": [
            {
                "date": str((target_date - pd.Timedelta(days=window_days - 1 - i)).date()),
                "path": os.path.abspath(path) if path else None,
                "available": bool(path is not None or args.repeat_missing),
            }
            for i, path in enumerate(window_paths)
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
