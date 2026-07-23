import argparse
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from multiscale_phenology import (
    BASE_CLASSES,
    CNNBackboneFeatureExtractor,
    DEFAULT_IMAGE_BACKBONE,
    build_multiscale_daily_dataframe,
)

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None


def path_key(path: str) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], image_size: int = 224):
        if T is None:
            raise ImportError("torchvision is required for image transforms")
        self.paths = [path_key(p) for p in paths]
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return path, self.transform(image)


class TileTransform:
    def __init__(self, image_size: int = 224):
        if T is None:
            raise ImportError("torchvision is required for image transforms")
        steps = []
        if image_size is not None:
            steps.append(T.Resize((image_size, image_size)))
        steps.extend(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform = T.Compose(steps)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class TiledImagePathDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        tile_size: int = 224,
        tile_stride: int = 224,
        max_tiles: int = 0,
        image_size: int = 224,
    ):
        self.paths = [path_key(p) for p in paths]
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.max_tiles = max_tiles
        self.transform = TileTransform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        key = self.paths[idx]
        try:
            with Image.open(key) as image:
                image = image.convert("RGB")
                boxes = select_tiles(
                    tile_boxes(image.width, image.height, self.tile_size, self.tile_stride),
                    self.max_tiles,
                )
                tiles = torch.stack([self.transform(image.crop(box)) for box in boxes])
            return {
                "path": key,
                "tiles": tiles,
                "tile_count": len(boxes),
                "error": None,
            }
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
            return {
                "path": key,
                "tiles": None,
                "tile_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }


def single_item_collate(batch: List[Dict]) -> Dict:
    return batch[0]


def tile_boxes(width: int, height: int, tile_size: int, tile_stride: int) -> List[Tuple[int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if tile_stride <= 0:
        raise ValueError("tile_stride must be positive")

    def starts(length: int) -> List[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, length - tile_size + 1, tile_stride))
        last = length - tile_size
        if values[-1] != last:
            values.append(last)
        return values

    xs = starts(width)
    ys = starts(height)
    return [(x, y, x + tile_size, y + tile_size) for y in ys for x in xs]


def select_tiles(boxes: List[Tuple[int, int, int, int]], max_tiles: int) -> List[Tuple[int, int, int, int]]:
    if max_tiles is None or max_tiles <= 0 or len(boxes) <= max_tiles:
        return boxes
    if max_tiles == 1:
        return [boxes[len(boxes) // 2]]
    step = (len(boxes) - 1) / (max_tiles - 1)
    indices = sorted({round(i * step) for i in range(max_tiles)})
    return [boxes[i] for i in indices]


@torch.no_grad()
def encode_paths(
    model,
    paths: List[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_dtype: torch.dtype = torch.float16,
) -> dict:
    dataset = ImagePathDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    encoded = {}
    model.eval()
    for batch_paths, images in loader:
        images = images.to(device, non_blocking=True)
        features = model(images).detach().cpu()
        for path, feature in zip(batch_paths, features):
            encoded[path_key(path)] = feature.to(output_dtype)
    return encoded


@torch.no_grad()
def encode_tiled_paths(
    model,
    paths: List[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    tile_size: int = 224,
    tile_stride: int = 224,
    max_tiles: int = 0,
    image_size: int = 224,
    tile_pooling: str = "attention",
    output_dtype: torch.dtype = torch.float16,
) -> Tuple[dict, dict, dict]:
    dataset = TiledImagePathDataset(
        paths,
        tile_size=tile_size,
        tile_stride=tile_stride,
        max_tiles=max_tiles,
        image_size=image_size,
    )
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": single_item_collate,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    encoded = {}
    tile_counts = {}
    failures = {}
    model.eval()

    for idx, item in enumerate(loader, 1):
        key = item["path"]
        if item["error"] is not None:
            failures[key] = item["error"]
            print(f"Skipping corrupt/unreadable tiled image {idx}/{len(paths)}: {key} ({item['error']})")
            continue

        tiles = item["tiles"]
        tile_counts[key] = int(item["tile_count"])
        features = []
        for start in range(0, tiles.shape[0], batch_size):
            batch = tiles[start:start + batch_size]
            batch = batch.to(device, non_blocking=True)
            features.append(model(batch).detach().cpu())

        tile_features = torch.cat(features, dim=0)
        encoded[key] = (
            tile_features.to(output_dtype)
            if tile_pooling == "attention"
            else tile_features.mean(dim=0).to(output_dtype)
        )
        if idx == 1 or idx % 25 == 0 or idx == len(paths):
            print(f"Encoded tiled image {idx}/{len(paths)} ({tile_counts[key]} tiles/image for current image)")

    if failures:
        print(f"Skipped {len(failures)} corrupt/unreadable tiled images")
    return encoded, tile_counts, failures


def unique_existing_paths(series: pd.Series) -> List[str]:
    paths = []
    for value in series.dropna().tolist():
        key = path_key(value)
        if os.path.isfile(key):
            paths.append(key)
    return sorted(set(paths))


def embedding_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported embedding dtype: {name}")


def main():
    parser = argparse.ArgumentParser(description="Precompute frozen CNN/EfficientNet image embeddings for temporal phenology training.")
    parser.add_argument(
        "--excel-path",
        "--label-path",
        dest="excel_path",
        default="labeling.xlsx",
        help="Path to the phenology label table. Supports .xlsx/.xls and .csv exports with the same columns.",
    )
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_multiscale")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--date-tolerance-days", type=int, default=7, help="Days outside a predicted stage window that still receive partial metric credit.")
    parser.add_argument("--preplant-days", type=int, default=30, help="Number of days before seeding to keep as OffSeason.")
    parser.add_argument("--postharvest-days", type=int, default=30, help="Number of days after harvest to keep as OffSeason.")
    parser.add_argument("--camera", default="AUTO", help="Camera folder to use: AUTO prefers K1 and falls back to K2/available camera; use K1, K2, or ALL to override.")
    parser.add_argument("--stream", choices=["micro", "macro", "both"], default="micro", help="Which image stream to cache. micro means 10X only.")
    parser.add_argument("--ignore-status-csv", action="store_true")
    parser.add_argument("--image-backbone", default=DEFAULT_IMAGE_BACKBONE)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-streams", choices=["none", "micro", "macro", "both"], default="micro", help="Which streams use native-resolution tiling before CNN encoding.")
    parser.add_argument("--tile-size", type=int, default=224, help="Crop size in original pixels for tiled encoding.")
    parser.add_argument("--tile-stride", type=int, default=224, help="Stride in original pixels for tiled encoding.")
    parser.add_argument("--max-tiles", type=int, default=0, help="Maximum tiles per image. 0 means use all tiles.")
    parser.add_argument("--vit-image-size", type=int, default=224, help="CNN input size after each tile is cropped. Kept as --vit-image-size for compatibility.")
    parser.add_argument("--tile-pooling", choices=["attention", "mean"], default="attention", help="attention stores all tile features for learned pooling; mean averages tiles during precompute.")
    parser.add_argument("--embedding-dtype", choices=["float16", "float32"], default="float16", help="Storage dtype for cached embeddings. float16 roughly halves cache size and RAM use.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = args.cache_path or os.path.join(args.out_dir, "cnn_embeddings.pt")
    metadata_path = os.path.join(args.out_dir, "multiscale_daily_metadata.csv")
    device = torch.device(args.device)
    output_dtype = embedding_dtype(args.embedding_dtype)

    daily_df = build_multiscale_daily_dataframe(
        args.excel_path,
        args.data_path,
        include_preplant_days=args.preplant_days,
        include_postharvest_days=args.postharvest_days,
        transition_days=args.transition_days,
        date_tolerance_days=args.date_tolerance_days,
        classes=BASE_CLASSES,
        preferred_camera=None if args.camera.upper() == "ALL" else args.camera,
        use_status_csv=not args.ignore_status_csv,
    )
    if daily_df.empty:
        raise RuntimeError("No paired daily rows were created. Check data_path, folder names, and filename dates.")

    daily_df.to_csv(metadata_path, index=False)
    macro_paths = unique_existing_paths(daily_df["macro_path"]) if args.stream in {"macro", "both"} else []
    micro_paths = unique_existing_paths(daily_df["micro_path"]) if args.stream in {"micro", "both"} else []
    print(f"Saved metadata: {metadata_path}")
    print(f"Macro images: {len(macro_paths)}")
    print(f"Micro images: {len(micro_paths)}")

    model = CNNBackboneFeatureExtractor(args.image_backbone, pretrained=args.pretrained).to(device)
    tile_streams = {args.tile_streams} if args.tile_streams in {"macro", "micro"} else {"macro", "micro"} if args.tile_streams == "both" else set()

    macro_tile_counts = {}
    micro_tile_counts = {}
    if macro_paths and "macro" in tile_streams:
        print("Encoding macro images with tiling")
        macro_embeddings, macro_tile_counts, macro_failures = encode_tiled_paths(
            model,
            macro_paths,
            args.batch_size,
            args.num_workers,
            device,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            max_tiles=args.max_tiles,
            image_size=args.vit_image_size,
            tile_pooling=args.tile_pooling,
            output_dtype=output_dtype,
        )
    else:
        macro_embeddings = encode_paths(model, macro_paths, args.batch_size, args.num_workers, device, output_dtype=output_dtype) if macro_paths else {}
        macro_failures = {}

    if micro_paths and "micro" in tile_streams:
        print("Encoding micro images with tiling")
        micro_embeddings, micro_tile_counts, micro_failures = encode_tiled_paths(
            model,
            micro_paths,
            args.batch_size,
            args.num_workers,
            device,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            max_tiles=args.max_tiles,
            image_size=args.vit_image_size,
            tile_pooling=args.tile_pooling,
            output_dtype=output_dtype,
        )
    else:
        micro_embeddings = encode_paths(model, micro_paths, args.batch_size, args.num_workers, device, output_dtype=output_dtype) if micro_paths else {}
        micro_failures = {}

    torch.save(
        {
            "feature_dim": model.out_dim,
            "image_backbone": args.image_backbone,
            "pretrained": args.pretrained,
            "stream": args.stream,
            "embedding_dtype": args.embedding_dtype,
            "tiling": {
                "tile_streams": args.tile_streams,
                "tile_size": args.tile_size,
                "tile_stride": args.tile_stride,
                "max_tiles": args.max_tiles,
                "image_size": args.vit_image_size,
                "vit_image_size": args.vit_image_size,
                "tile_pooling": args.tile_pooling,
                "macro_tile_counts": macro_tile_counts,
                "micro_tile_counts": micro_tile_counts,
                "macro_failures": macro_failures,
                "micro_failures": micro_failures,
            },
            "macro": macro_embeddings,
            "micro": micro_embeddings,
        },
        cache_path,
    )
    print(f"Saved embedding cache: {cache_path}")


if __name__ == "__main__":
    main()
