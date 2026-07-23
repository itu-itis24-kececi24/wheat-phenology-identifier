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

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return path, self.transform(image)


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


def tiled_item_collate(batch: List[Dict]) -> List[Dict]:
    """Keep variable-length tile tensors together without padding them."""
    return batch


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def resolve_amp_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    if name == "none" or device.type != "cuda":
        return None
    if name in {"auto", "float16"}:
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def enable_fast_cuda_runtime(device: torch.device) -> None:
    """Enable safe CUDA math paths for frozen embedding extraction."""
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


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
    prefetch_factor: int = 4,
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
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(prefetch_factor, 1)
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    encoded = {}
    model.eval()
    for batch_paths, images in loader:
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
    prefetch_factor: int = 4,
    tile_loader_batch_size: int = 4,
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
        "batch_size": max(tile_loader_batch_size, 1),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": tiled_item_collate,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(prefetch_factor, 1)
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    encoded = {}
    tile_counts = {}
    failures = {}
    model.eval()

    processed = 0
    for items in loader:
        valid_items = []
        for item in items:
            processed += 1
            key = item["path"]
            if item["error"] is not None:
                failures[key] = item["error"]
                print(f"Skipping corrupt/unreadable tiled image {processed}/{len(paths)}: {key} ({item['error']})")
                continue
            tile_counts[key] = int(item["tile_count"])
            valid_items.append((processed, item))

        if not valid_items:
            continue

        # Flatten tiles from several decoded images, then keep GPU inference at
        # the configured tile batch size. This reduces loader/GPU hand-off gaps.
        all_tiles = torch.cat([item["tiles"] for _, item in valid_items], dim=0)
        features = []
        for start in range(0, all_tiles.shape[0], batch_size):
            batch = all_tiles[start:start + batch_size].to(device, non_blocking=True)
            with autocast_context(device, amp_dtype):
                # torch.compile may reuse its CUDA-graph output buffer on the
                # next invocation, so retain an independent tensor per chunk.
                features.append(model(batch).clone())
        all_features = torch.cat(features, dim=0).detach().to("cpu", dtype=output_dtype)

        offset = 0
        for item_index, item in valid_items:
            key = item["path"]
            count = int(item["tile_count"])
            tile_features = all_features[offset:offset + count]
            offset += count
            encoded[key] = (
                tile_features.clone()
                if tile_pooling == "attention"
                else tile_features.float().mean(dim=0).to(output_dtype)
            )
            if item_index == 1 or item_index % 25 == 0 or item_index == len(paths):
                print(f"Encoded tiled image {item_index}/{len(paths)} ({tile_counts[key]} tiles/image for current image)")

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
    parser = argparse.ArgumentParser(description="Precompute frozen DINOv2 image embeddings for multi-scale phenology training.")
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
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per worker. Higher values keep a fast GPU fed at the cost of host RAM.")
    parser.add_argument("--tile-loader-batch-size", type=int, default=4, help="Number of full-resolution tiled images decoded together before their tiles are flattened into --batch-size GPU batches.")
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--date-tolerance-days", type=int, default=7, help="Days outside a predicted stage window that still receive partial metric credit.")
    parser.add_argument("--preplant-days", type=int, default=30, help="Number of days before seeding to keep as OffSeason.")
    parser.add_argument("--postharvest-days", type=int, default=30, help="Number of days after harvest to keep as OffSeason.")
    parser.add_argument("--camera", default="AUTO", help="Camera folder to use. AUTO uses the label table kamera/Camera column when present, otherwise prefers K1 and falls back to the available camera. Use K1, K2, or ALL to override.")
    parser.add_argument("--stream", choices=["micro", "macro", "both"], default="both", help="Which image stream to cache. both caches 1X and 10X for gated fusion.")
    parser.add_argument("--ignore-status-csv", action="store_true")
    parser.add_argument("--image-backbone", default=DINO_DEFAULT_BACKBONE, help="DINOv2/Hugging Face backbone, e.g. facebook/dinov2-base.")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=["auto", "float16", "bfloat16", "none"], default="auto", help="CUDA mixed precision used while extracting frozen features. auto uses float16 on CUDA.")
    parser.add_argument("--compile", dest="compile", action="store_true", default=True, help="Use torch.compile for faster long embedding jobs when supported.")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="Disable torch.compile.")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="max-autotune", help="torch.compile optimization mode.")
    parser.add_argument("--tile-streams", choices=["none", "micro", "macro", "both"], default="both", help="Which streams use native-resolution tiling before ViT encoding.")
    parser.add_argument("--tile-size", type=int, default=224, help="Crop size in original pixels for tiled encoding.")
    parser.add_argument("--tile-stride", type=int, default=224, help="Stride in original pixels for tiled encoding.")
    parser.add_argument("--max-tiles", type=int, default=0, help="Maximum tiles per image. 0 means use all tiles.")
    parser.add_argument("--vit-image-size", type=int, default=224, help="ViT input size after each tile is cropped.")
    parser.add_argument("--tile-pooling", choices=["attention", "mean"], default="attention", help="attention stores all tile features for learned pooling; mean averages tiles during precompute.")
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

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = args.cache_path or os.path.join(args.out_dir, "vit_embeddings.pt")
    metadata_path = os.path.join(args.out_dir, "multiscale_daily_metadata.csv")
    device = torch.device(args.device)
    output_dtype = embedding_dtype(args.embedding_dtype)
    amp_dtype = resolve_amp_dtype(args.amp, device)
    enable_fast_cuda_runtime(device)

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
        title="First/last resolved images used for DINOv2 precompute:",
    )

    model = ViTBackboneFeatureExtractor(args.image_backbone, pretrained=args.pretrained).to(device)
    feature_dim = model.out_dim
    backbone_name = getattr(model, "backbone_name", args.image_backbone)
    backbone_source = getattr(model, "backbone_source", "torchvision")
    preprocess_mean = list(getattr(model, "preprocess_mean", [0.485, 0.456, 0.406]))
    preprocess_std = list(getattr(model, "preprocess_std", [0.229, 0.224, 0.225]))
    preprocess_image_size = int(getattr(model, "preprocess_image_size", args.vit_image_size))
    if args.vit_image_size != preprocess_image_size:
        print(
            f"Using --vit-image-size {args.vit_image_size}; backbone default is {preprocess_image_size}.",
            flush=True,
        )
    print(f"Backbone: {backbone_name} ({backbone_source})")
    print(f"Feature dimension: {feature_dim}")
    if device.type == "cuda":
        amp_name = str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "disabled"
        print(f"CUDA fast paths: TF32 enabled | AMP: {amp_name}")
    if args.compile and device.type == "cuda":
        if hasattr(torch, "compile"):
            print(f"Compiling backbone with torch.compile ({args.compile_mode}, dynamic shapes enabled)")
            model = torch.compile(model, mode=args.compile_mode, dynamic=True)
        else:
            print("torch.compile is unavailable in this PyTorch version; continuing without compilation")
    elif args.compile:
        print("torch.compile is enabled but skipped because embedding extraction is not using CUDA")
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
                prefetch_factor=args.prefetch_factor,
                tile_loader_batch_size=args.tile_loader_batch_size,
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
            prefetch_factor=args.prefetch_factor,
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
            "feature_dim": feature_dim,
            "image_backbone": backbone_name,
            "backbone_source": backbone_source,
            "pretrained": args.pretrained,
            "preprocess": {
                "image_size": args.vit_image_size,
                "backbone_default_image_size": preprocess_image_size,
                "mean": preprocess_mean,
                "std": preprocess_std,
            },
            "stream": args.stream,
            "embedding_dtype": args.embedding_dtype,
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
