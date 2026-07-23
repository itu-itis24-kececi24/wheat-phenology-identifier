import argparse
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from multiscale_phenology import (
    BASE_CLASSES,
    DINO_DEFAULT_BACKBONE,
    ViTBackboneFeatureExtractor,
    build_multiscale_daily_dataframe,
    print_station_image_edges,
)

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None


def path_key(path: str) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


def build_image_transform(
    image_size: int = 224,
    augment: bool = False,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.15,
    hue: float = 0.03,
    rotation: float = 5.0,
    blur_prob: float = 0.15,
):
    steps = []
    if augment:
        steps.extend(
            [
                T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=rotation),
                T.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=blur_prob),
            ]
        )
    else:
        steps.append(T.Resize((image_size, image_size)))
    steps.extend(
        [
            T.ToTensor(),
            T.Normalize(mean or [0.485, 0.456, 0.406], std or [0.229, 0.224, 0.225]),
        ]
    )
    return T.Compose(steps)


class ImagePathDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        image_size: int = 224,
        augment: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.15,
        hue: float = 0.03,
        rotation: float = 5.0,
        blur_prob: float = 0.15,
    ):
        if T is None:
            raise ImportError("torchvision is required for image transforms")
        self.paths = [path_key(p) for p in paths]
        self.transform = build_image_transform(
            image_size=image_size,
            augment=augment,
            mean=mean,
            std=std,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            rotation=rotation,
            blur_prob=blur_prob,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.paths[idx]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
            return {"path": path, "image": tensor, "error": None}
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
            return {"path": path, "image": None, "error": f"{type(exc).__name__}: {exc}"}


class TileTransform:
    def __init__(
        self,
        image_size: int = 224,
        augment: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.15,
        hue: float = 0.03,
        rotation: float = 5.0,
        blur_prob: float = 0.15,
    ):
        if T is None:
            raise ImportError("torchvision is required for image transforms")
        self.transform = build_image_transform(
            image_size=image_size,
            augment=augment,
            mean=mean,
            std=std,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            rotation=rotation,
            blur_prob=blur_prob,
        )

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
        augment: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.15,
        hue: float = 0.03,
        rotation: float = 5.0,
        blur_prob: float = 0.15,
    ):
        self.paths = [path_key(p) for p in paths]
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.max_tiles = max_tiles
        self.transform = TileTransform(
            image_size=image_size,
            augment=augment,
            mean=mean,
            std=std,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            rotation=rotation,
            blur_prob=blur_prob,
        )

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


def image_item_collate(batch: List[Dict]) -> List[Dict]:
    return batch


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def resolve_amp_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    if name == "none" or device.type != "cuda":
        return None
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def enable_fast_cuda_runtime(device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


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
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    augment: bool = False,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.15,
    hue: float = 0.03,
    rotation: float = 5.0,
    blur_prob: float = 0.15,
    amp_dtype: Optional[torch.dtype] = None,
) -> dict:
    dataset = ImagePathDataset(
        paths,
        image_size=image_size,
        augment=augment,
        mean=mean,
        std=std,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        rotation=rotation,
        blur_prob=blur_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=image_item_collate,
    )
    encoded = {}
    model.eval()
    for items in loader:
        valid_items = [item for item in items if item["error"] is None]
        for item in items:
            if item["error"] is not None:
                print(f"Skipping corrupt/unreadable image: {item['path']} ({item['error']})")
        if not valid_items:
            continue
        batch_paths = [item["path"] for item in valid_items]
        images = torch.stack([item["image"] for item in valid_items])
        images = images.to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            features = model(images)
        features = features.detach().to("cpu", dtype=output_dtype)
        for path, feature in zip(batch_paths, features):
            encoded[path_key(path)] = feature
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
    augment: bool = False,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.15,
    hue: float = 0.03,
    rotation: float = 5.0,
    blur_prob: float = 0.15,
    amp_dtype: Optional[torch.dtype] = None,
    dense_features: bool = True,
    dense_grid_size: int = 2,
    dense_include_cls: bool = True,
) -> Tuple[dict, dict, dict]:
    dataset = TiledImagePathDataset(
        paths,
        tile_size=tile_size,
        tile_stride=tile_stride,
        max_tiles=max_tiles,
        image_size=image_size,
        augment=augment,
        mean=mean,
        std=std,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        rotation=rotation,
        blur_prob=blur_prob,
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
            with autocast_context(device, amp_dtype):
                batch_features = (
                    model.forward_dense(
                        batch,
                        grid_size=dense_grid_size,
                        include_cls=dense_include_cls,
                    )
                    if dense_features
                    else model(batch)
                )
                features.append(batch_features.detach().to("cpu", dtype=output_dtype))

        tile_features = torch.cat(features, dim=0)
        encoded[key] = (
            tile_features
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
    parser = argparse.ArgumentParser(description="Precompute frozen DINOv3 dense image embeddings for BBCH phenology training.")
    parser.add_argument(
        "--excel-path",
        "--label-path",
        dest="excel_path",
        default="labeling_bbch_iso_dates.csv",
        help="Path to the BBCH label table. Supports the revised .xlsx or .csv files with the same columns.",
    )
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_dinov3_bbch_cache")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--date-tolerance-days", type=int, default=7, help="Days outside a predicted stage window that still receive partial metric credit.")
    parser.add_argument("--preplant-days", type=int, default=30, help="Number of days before seeding to keep as OffSeason.")
    parser.add_argument("--postharvest-days", type=int, default=30, help="Number of days after harvest to keep as OffSeason.")
    parser.add_argument("--camera", default="AUTO", help="Camera folder to use. AUTO uses the label table kamera/Camera column when present, otherwise prefers K1 and falls back to the available camera. Use K1, K2, or ALL to override.")
    parser.add_argument("--stream", choices=["micro", "macro", "both"], default="micro", help="Which image stream to cache. micro means 10X only.")
    parser.add_argument("--ignore-status-csv", action="store_true")
    parser.add_argument("--image-backbone", default=DINO_DEFAULT_BACKBONE, help="DINOv3 Hugging Face backbone model ID.")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=["auto", "float16", "bfloat16", "none"], default="auto", help="Mixed precision used for frozen DINOv3 extraction.")
    parser.add_argument("--tile-streams", choices=["none", "micro", "macro", "both"], default="micro", help="Which streams use native-resolution tiling before ViT encoding.")
    parser.add_argument("--tile-size", type=int, default=224, help="Crop size in original pixels for tiled encoding.")
    parser.add_argument("--tile-stride", type=int, default=224, help="Stride in original pixels for tiled encoding.")
    parser.add_argument("--max-tiles", type=int, default=0, help="Maximum tiles per image. 0 means use all tiles.")
    parser.add_argument("--vit-image-size", type=int, default=224, help="ViT input size after each tile is cropped.")
    parser.add_argument("--tile-pooling", choices=["attention", "mean"], default="attention", help="attention stores all tile features for learned pooling; mean averages tiles during precompute.")
    parser.add_argument("--dense-features", dest="dense_features", action="store_true", default=True, help="Cache compact DINOv3 patch descriptors per tile (recommended).")
    parser.add_argument("--no-dense-features", dest="dense_features", action="store_false", help="Cache only one global descriptor per tile for an ablation.")
    parser.add_argument("--dense-grid-size", type=int, default=2, help="Adaptive patch-token grid per tile. 2 stores a 2x2 grid; 4 is more detailed but much larger.")
    parser.add_argument("--dense-include-cls", dest="dense_include_cls", action="store_true", default=True, help="Keep the DINOv3 CLS descriptor beside the dense patch grid.")
    parser.add_argument("--no-dense-include-cls", dest="dense_include_cls", action="store_false")
    parser.add_argument("--embedding-dtype", choices=["float16", "float32"], default="float16", help="Storage dtype for cached embeddings. float16 roughly halves cache size and RAM use.")
    parser.add_argument("--augment-views", type=int, default=0, help="Number of augmented embedding variants to store per image. 0 disables augmentation.")
    parser.add_argument("--augment-streams", choices=["auto", "none", "micro", "macro", "both"], default="auto", help="Which streams receive augmented embedding variants. auto follows --stream.")
    parser.add_argument("--augment-brightness", type=float, default=0.2)
    parser.add_argument("--augment-contrast", type=float, default=0.2)
    parser.add_argument("--augment-saturation", type=float, default=0.15)
    parser.add_argument("--augment-hue", type=float, default=0.03)
    parser.add_argument("--augment-rotation", type=float, default=5.0)
    parser.add_argument("--augment-blur-prob", type=float, default=0.15)
    args = parser.parse_args()

    if args.dense_grid_size < 1:
        parser.error("--dense-grid-size must be at least 1")
    if args.dense_features and args.tile_pooling != "attention":
        parser.error("--dense-features requires --tile-pooling attention so patch/tile hierarchy is preserved")

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = args.cache_path or os.path.join(args.out_dir, "vit_embeddings.pt")
    metadata_path = os.path.join(args.out_dir, "multiscale_daily_metadata.csv")
    device = torch.device(args.device)
    enable_fast_cuda_runtime(device)
    amp_dtype = resolve_amp_dtype(args.amp, device)
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
    print_station_image_edges(
        daily_df,
        stream=args.stream,
        base_dir=args.data_path,
        title="First/last resolved images used for DINOv3 precompute:",
    )

    model = ViTBackboneFeatureExtractor(args.image_backbone, pretrained=args.pretrained).to(device)
    max_dense_grid = args.vit_image_size // int(getattr(model, "patch_size", 16))
    if args.dense_features and args.dense_grid_size > max_dense_grid:
        raise ValueError(
            f"--dense-grid-size {args.dense_grid_size} exceeds the {max_dense_grid}x{max_dense_grid} "
            f"DINOv3 patch grid for --vit-image-size {args.vit_image_size}"
        )
    preprocess_mean = list(getattr(model, "preprocess_mean", [0.485, 0.456, 0.406]))
    preprocess_std = list(getattr(model, "preprocess_std", [0.229, 0.224, 0.225]))
    preprocess_image_size = int(getattr(model, "preprocess_image_size", args.vit_image_size))
    if args.vit_image_size != preprocess_image_size:
        print(
            f"Using --vit-image-size {args.vit_image_size}; backbone default is {preprocess_image_size}.",
            flush=True,
        )
    print(f"Backbone: {getattr(model, 'backbone_name', args.image_backbone)} ({getattr(model, 'backbone_source', 'unknown')})")
    print(f"Feature dimension: {model.out_dim}")
    dense_tokens_per_tile = args.dense_grid_size ** 2 + int(args.dense_include_cls)
    print(
        "Dense feature cache: "
        + (
            f"enabled ({args.dense_grid_size}x{args.dense_grid_size} patch grid + "
            f"{'CLS' if args.dense_include_cls else 'no CLS'} = {dense_tokens_per_tile} descriptors/tile)"
            if args.dense_features
            else "disabled"
        )
    )
    print(f"Embedding extraction AMP: {str(amp_dtype).replace('torch.', '') if amp_dtype is not None else 'disabled'}")
    tile_streams = {args.tile_streams} if args.tile_streams in {"macro", "micro"} else {"macro", "micro"} if args.tile_streams == "both" else set()
    if args.augment_streams == "auto":
        augment_streams = {args.stream} if args.stream in {"macro", "micro"} else {"macro", "micro"} if args.stream == "both" else set()
    elif args.augment_streams == "both":
        augment_streams = {"macro", "micro"}
    elif args.augment_streams == "none":
        augment_streams = set()
    else:
        augment_streams = {args.augment_streams}
    if args.augment_views <= 0:
        augment_streams = set()
    print(f"Augmented embedding views: {max(args.augment_views, 0)}")
    print(f"Augmented streams: {sorted(augment_streams) if augment_streams else 'none'}")

    def encode_stream(paths: List[str], stream_name: str, augment: bool = False):
        if not paths:
            return {}, {}, {}
        if stream_name in tile_streams:
            print(f"Encoding {stream_name} images with tiling" + (" and augmentation" if augment else ""))
            return encode_tiled_paths(
                model,
                paths,
                args.batch_size,
                args.num_workers,
                device,
                tile_size=args.tile_size,
                tile_stride=args.tile_stride,
                max_tiles=args.max_tiles,
                image_size=args.vit_image_size,
                tile_pooling=args.tile_pooling,
                output_dtype=output_dtype,
                augment=augment,
                mean=preprocess_mean,
                std=preprocess_std,
                brightness=args.augment_brightness,
                contrast=args.augment_contrast,
                saturation=args.augment_saturation,
                hue=args.augment_hue,
                rotation=args.augment_rotation,
                blur_prob=args.augment_blur_prob,
                amp_dtype=amp_dtype,
                dense_features=args.dense_features,
                dense_grid_size=args.dense_grid_size,
                dense_include_cls=args.dense_include_cls,
            )
        print(f"Encoding {stream_name} images" + (" with augmentation" if augment else ""))
        embeddings = encode_paths(
            model,
            paths,
            args.batch_size,
            args.num_workers,
            device,
            output_dtype=output_dtype,
            image_size=args.vit_image_size,
            mean=preprocess_mean,
            std=preprocess_std,
            augment=augment,
            brightness=args.augment_brightness,
            contrast=args.augment_contrast,
            saturation=args.augment_saturation,
            hue=args.augment_hue,
            rotation=args.augment_rotation,
            blur_prob=args.augment_blur_prob,
            amp_dtype=amp_dtype,
        )
        return embeddings, {}, {}

    def encode_augmented_views(paths: List[str], stream_name: str, clean_embeddings: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        if args.augment_views <= 0 or stream_name not in augment_streams or not paths:
            return {}
        augmented = {key: [] for key in clean_embeddings.keys()}
        for view_idx in range(args.augment_views):
            print(f"Encoding augmented {stream_name} view {view_idx + 1}/{args.augment_views}")
            view_embeddings, _, view_failures = encode_stream(paths, stream_name, augment=True)
            for key, value in view_embeddings.items():
                if key in augmented:
                    augmented[key].append(value)
            if view_failures:
                print(f"Augmented {stream_name} view {view_idx + 1}: skipped {len(view_failures)} images")
        return {key: values for key, values in augmented.items() if values}

    macro_tile_counts = {}
    micro_tile_counts = {}
    if macro_paths and "macro" in tile_streams:
        macro_embeddings, macro_tile_counts, macro_failures = encode_stream(macro_paths, "macro", augment=False)
    else:
        macro_embeddings, macro_tile_counts, macro_failures = encode_stream(macro_paths, "macro", augment=False) if macro_paths else ({}, {}, {})

    if micro_paths and "micro" in tile_streams:
        micro_embeddings, micro_tile_counts, micro_failures = encode_stream(micro_paths, "micro", augment=False)
    else:
        micro_embeddings, micro_tile_counts, micro_failures = encode_stream(micro_paths, "micro", augment=False) if micro_paths else ({}, {}, {})

    macro_aug_embeddings = encode_augmented_views(macro_paths, "macro", macro_embeddings)
    micro_aug_embeddings = encode_augmented_views(micro_paths, "micro", micro_embeddings)

    torch.save(
        {
            "cache_format": "dinov3_dense_v1" if args.dense_features else "dinov3_global_v1",
            "feature_dim": model.out_dim,
            "image_backbone": getattr(model, "backbone_name", args.image_backbone),
            "backbone_source": getattr(model, "backbone_source", "torchvision"),
            "pretrained": args.pretrained,
            "preprocess": {
                "image_size": args.vit_image_size,
                "backbone_default_image_size": preprocess_image_size,
                "mean": preprocess_mean,
                "std": preprocess_std,
            },
            "stream": args.stream,
            "embedding_dtype": args.embedding_dtype,
            "dense_features": {
                "enabled": bool(args.dense_features),
                "streams": sorted(tile_streams) if args.dense_features else [],
                "grid_size": int(args.dense_grid_size),
                "include_cls": bool(args.dense_include_cls),
                "tokens_per_tile": int(dense_tokens_per_tile if args.dense_features else 1),
                "patch_size": int(getattr(model, "patch_size", 16)),
                "num_register_tokens": int(getattr(model, "num_register_tokens", 0)),
                "pooling": "adaptive_avg",
            },
            "tiling": {
                "tile_streams": args.tile_streams,
                "tile_size": args.tile_size,
                "tile_stride": args.tile_stride,
                "max_tiles": args.max_tiles,
                "vit_image_size": args.vit_image_size,
                "tile_pooling": args.tile_pooling,
                "macro_tile_counts": macro_tile_counts,
                "micro_tile_counts": micro_tile_counts,
                "macro_failures": macro_failures,
                "micro_failures": micro_failures,
            },
            "augmentation": {
                "views": max(args.augment_views, 0),
                "streams": sorted(augment_streams),
                "brightness": args.augment_brightness,
                "contrast": args.augment_contrast,
                "saturation": args.augment_saturation,
                "hue": args.augment_hue,
                "rotation": args.augment_rotation,
                "blur_prob": args.augment_blur_prob,
            },
            "macro": macro_embeddings,
            "micro": micro_embeddings,
            "macro_aug": macro_aug_embeddings,
            "micro_aug": micro_aug_embeddings,
        },
        cache_path,
    )
    print(f"Saved embedding cache: {cache_path}")


if __name__ == "__main__":
    main()
