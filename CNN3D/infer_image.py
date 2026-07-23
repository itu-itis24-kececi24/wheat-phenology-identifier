import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None

from multiscale_phenology import (
    BASE_CLASSES,
    DEFAULT_TEMPORAL_FEATURE_COLUMNS,
    IMAGE_EXTS,
    MultiScale3DCNN,
    SingleStream3DCNN,
    _extract_date,
    _temporal_features_for_date,
)


CURRENT_DATA_RE = re.compile(
    r"(?P<station>\d{2}[_.,]\d{2})-(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
)
DATE_RE = re.compile(r"(?P<year>\d{4})[_-](?P<month>\d{2})[_-](?P<day>\d{2})")


def path_key(path: str) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


def parse_image_date(path_or_name: str) -> Optional[pd.Timestamp]:
    name = os.path.basename(path_or_name)
    match = CURRENT_DATA_RE.search(name) or DATE_RE.search(name)
    if match is None:
        date = _extract_date(path_or_name)
        return None if date is None else date.normalize()
    try:
        return pd.Timestamp(
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
        ).normalize()
    except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
        return None


def image_matches_stream(path: str, stream: str) -> bool:
    lowered = os.path.normpath(path).lower()
    filename = os.path.basename(lowered)
    parts = lowered.split(os.sep)
    if stream == "micro":
        return "10x" in parts or "-10x" in filename or "_10x" in filename
    if stream == "macro":
        return "1x" in parts or "-1x" in filename or "_1x" in filename
    return True


def find_images_by_date(image_dir: Optional[str], stream: str) -> Dict[pd.Timestamp, str]:
    if not image_dir:
        return {}
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
            key = date.normalize()
            if key not in by_date or path_key(path) < path_key(by_date[key]):
                by_date[key] = path
    return by_date


def infer_target_date(args, *paths: Optional[str]) -> pd.Timestamp:
    if args.target_date:
        return pd.Timestamp(args.target_date).normalize()
    for path in paths:
        if path:
            date = parse_image_date(path)
            if date is not None:
                return date.normalize()
    raise ValueError("Provide --target-date, or use image filenames that contain a date.")


def load_image(path: str, image_size: int = 224) -> torch.Tensor:
    if T is None:
        raise ImportError("torchvision is required for image transforms")
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(path) as image:
        image = image.convert("RGB")
        return transform(image)


def window_dates(target_date: pd.Timestamp, window_days: int) -> List[pd.Timestamp]:
    start = target_date - pd.Timedelta(days=window_days - 1)
    return [start + pd.Timedelta(days=offset) for offset in range(window_days)]


def resolve_window_paths(
    target_path: Optional[str],
    image_dir: Optional[str],
    target_date: pd.Timestamp,
    window_days: int,
    stream: str,
) -> List[Optional[str]]:
    by_date = find_images_by_date(image_dir, stream)
    if target_path:
        by_date[target_date] = path_key(target_path)
    return [by_date.get(date) for date in window_dates(target_date, window_days)]


def build_sequence(
    paths: List[Optional[str]],
    target_path: Optional[str],
    image_size: int,
    repeat_missing: bool,
    repeat_all: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_path is None and (repeat_missing or repeat_all):
        raise ValueError("Repeating missing images requires a target image path.")
    target_image = load_image(target_path, image_size) if target_path else None
    zero = torch.zeros(3, image_size, image_size)

    frames, mask = [], []
    for path in paths:
        if repeat_all:
            frames.append(target_image)
            mask.append(True)
        elif path:
            frames.append(load_image(path, image_size))
            mask.append(True)
        elif repeat_missing:
            frames.append(target_image)
            mask.append(True)
        else:
            frames.append(zero)
            mask.append(False)
    return torch.stack(frames), torch.tensor(mask, dtype=torch.bool)


def empty_sequence(window_days: int, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(window_days, 3, image_size, image_size),
        torch.zeros(window_days, dtype=torch.bool),
    )


def build_temporal_features(
    target_date: pd.Timestamp,
    planting_date: Optional[str],
    window_days: int,
    feature_columns: List[str],
) -> torch.Tensor:
    if not feature_columns:
        return torch.zeros(window_days, 0)
    if planting_date is None:
        print("Warning: checkpoint expects days-since-planting metadata but --planting-date was not supplied; metadata features are zero.", flush=True)
        return torch.zeros(window_days, len(feature_columns))
    dates = window_dates(target_date, window_days)
    planting = pd.Timestamp(planting_date).normalize()
    return torch.stack(
        [
            _temporal_features_for_date(
                date,
                planting,
                feature_columns=feature_columns,
            )
            for date in dates
        ]
    )


def build_model(checkpoint: Dict, classes: List[str], stream: str, device: torch.device):
    architecture = checkpoint.get("architecture", "cnn3d")
    if architecture != "cnn3d":
        raise RuntimeError(f"This CNN3D inference script only supports cnn3d checkpoints, got: {architecture}")
    target_index = checkpoint.get("target_index")
    temporal_feature_dim = int(checkpoint.get("temporal_feature_dim", 0))
    temporal_feature_hidden_dim = int(checkpoint.get("temporal_feature_hidden_dim", 32))
    if stream == "both":
        model = MultiScale3DCNN(
            num_classes=len(classes),
            base_channels=int(checkpoint.get("cnn3d_base_channels", 24)),
            feature_dim=int(checkpoint.get("cnn3d_feature_dim", 256)),
            dropout=float(checkpoint.get("cnn3d_dropout", 0.25)),
            temporal_feature_dim=temporal_feature_dim,
            temporal_feature_hidden_dim=temporal_feature_hidden_dim,
            target_index=target_index,
        ).to(device)
    else:
        model = SingleStream3DCNN(
            stream=stream,
            num_classes=len(classes),
            base_channels=int(checkpoint.get("cnn3d_base_channels", 24)),
            feature_dim=int(checkpoint.get("cnn3d_feature_dim", 256)),
            dropout=float(checkpoint.get("cnn3d_dropout", 0.25)),
            temporal_feature_dim=temporal_feature_dim,
            temporal_feature_hidden_dim=temporal_feature_hidden_dim,
            target_index=target_index,
        ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, architecture


def main():
    parser = argparse.ArgumentParser(description="Infer phenology from one image or a previous-days image folder.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--image-path", default=None, help="Target-day image path for the selected stream.")
    parser.add_argument("--image-dir", default=None, help="Folder containing dated previous-day images for the selected stream.")
    parser.add_argument("--macro-path", default=None, help="Target-day 1X/canopy image path for --stream both or macro.")
    parser.add_argument("--micro-path", default=None, help="Target-day 10X/close-up image path for --stream both or micro.")
    parser.add_argument("--macro-dir", default=None, help="Folder containing dated 1X/canopy images.")
    parser.add_argument("--micro-dir", default=None, help="Folder containing dated 10X/close-up images.")
    parser.add_argument("--target-date", default=None, help="Target date YYYY-MM-DD. Defaults to date parsed from target image filename.")
    parser.add_argument("--planting-date", default=None, help="Planting/1-Ekim date YYYY-MM-DD. Needed for checkpoints trained with days-since-planting metadata.")
    parser.add_argument("--stream", choices=["macro", "micro", "both"], default=None, help="Override checkpoint stream.")
    parser.add_argument("--window-days", type=int, default=None, help="Override checkpoint window length.")
    parser.add_argument("--image-size", type=int, default=None, help="Override checkpoint image size. Defaults to the training image size saved in the checkpoint.")
    parser.add_argument("--repeat-missing", action="store_true", help="Repeat target image for missing dates instead of masking them out.")
    parser.add_argument("--repeat-target", action="store_true", help="Ignore folders and repeat target image across the whole window.")
    parser.add_argument("--debug-window", action="store_true", help="Print resolved window dates and paths.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("uses_embedding_cache"):
        raise RuntimeError("This script expects a full-image checkpoint, not a cached-embedding checkpoint.")

    classes = checkpoint.get("classes", BASE_CLASSES)
    stream = args.stream or checkpoint.get("stream", "micro")
    window_days = args.window_days or int(checkpoint.get("window_days", 31))
    image_size = args.image_size or int(checkpoint.get("image_size", 224))
    device = torch.device(args.device)

    if args.image_path and (args.macro_path or args.micro_path):
        raise ValueError("Use either --image-path or explicit --macro-path/--micro-path, not both.")

    if stream == "micro":
        micro_target = args.micro_path or args.image_path
        macro_target = None
    elif stream == "macro":
        macro_target = args.macro_path or args.image_path
        micro_target = None
    else:
        macro_target = args.macro_path or args.image_path
        micro_target = args.micro_path or args.image_path

    target_date = infer_target_date(args, micro_target, macro_target, args.image_path)
    temporal_feature_columns = checkpoint.get("temporal_feature_columns")
    if temporal_feature_columns is None:
        temporal_feature_columns = list(DEFAULT_TEMPORAL_FEATURE_COLUMNS) if checkpoint.get("use_days_since_planting", False) else []
    temporal_features = build_temporal_features(
        target_date,
        args.planting_date,
        window_days,
        list(temporal_feature_columns),
    )
    repeat_all = args.repeat_target or not any([args.image_dir, args.macro_dir, args.micro_dir])

    macro_paths = resolve_window_paths(
        macro_target,
        args.macro_dir or args.image_dir,
        target_date,
        window_days,
        "macro",
    )
    micro_paths = resolve_window_paths(
        micro_target,
        args.micro_dir or args.image_dir,
        target_date,
        window_days,
        "micro",
    )

    if args.debug_window:
        print("Resolved inference window:", flush=True)
        for idx, date in enumerate(window_dates(target_date, window_days)):
            macro_status = macro_paths[idx] if macro_paths[idx] else ""
            micro_status = micro_paths[idx] if micro_paths[idx] else ""
            print(f"  {date.date()} macro={macro_status} micro={micro_status}", flush=True)

    if stream in {"macro", "both"}:
        macro, macro_mask = build_sequence(
            macro_paths,
            macro_target,
            image_size,
            repeat_missing=args.repeat_missing,
            repeat_all=repeat_all and macro_target is not None,
        )
    else:
        macro, macro_mask = empty_sequence(window_days, image_size)

    if stream in {"micro", "both"}:
        micro, micro_mask = build_sequence(
            micro_paths,
            micro_target,
            image_size,
            repeat_missing=args.repeat_missing,
            repeat_all=repeat_all and micro_target is not None,
        )
    else:
        micro, micro_mask = empty_sequence(window_days, image_size)

    if stream == "micro":
        mask = micro_mask
    elif stream == "macro":
        mask = macro_mask
    else:
        mask = macro_mask & micro_mask

    model, architecture = build_model(checkpoint, classes, stream, device)
    with torch.no_grad():
        macro_batch = macro.unsqueeze(0).to(device)
        micro_batch = micro.unsqueeze(0).to(device)
        mask_batch = mask.unsqueeze(0).to(device)
        if int(checkpoint.get("temporal_feature_dim", 0)) > 0:
            logits = model(
                macro_batch,
                micro_batch,
                mask_batch,
                temporal_features.unsqueeze(0).to(device),
            )
        else:
            logits = model(macro_batch, micro_batch, mask_batch)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

    top_k = min(args.top_k, len(classes))
    values, indices = torch.topk(probs, k=top_k)
    dates = window_dates(target_date, window_days)
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "target_date": str(target_date.date()),
        "window_days": window_days,
        "image_size": image_size,
        "stream": stream,
        "architecture": architecture,
        "mode": "repeat_target" if repeat_all else "folder_window",
        "available_window_images": int(mask.sum().item()),
        "missing_window_images": int(window_days - mask.sum().item()),
        "prediction": classes[int(indices[0])],
        "confidence": float(values[0]),
        "top_k": [
            {"label": classes[int(idx)], "probability": float(prob)}
            for prob, idx in zip(values, indices)
        ],
        "used_dates": [
            {
                "date": str(dates[idx].date()),
                "macro_path": os.path.abspath(macro_paths[idx]) if macro_paths[idx] else None,
                "micro_path": os.path.abspath(micro_paths[idx]) if micro_paths[idx] else None,
                "available": bool(mask[idx].item()),
            }
            for idx in range(window_days)
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
