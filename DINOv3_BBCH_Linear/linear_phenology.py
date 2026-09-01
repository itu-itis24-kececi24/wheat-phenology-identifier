"""Single-day DINOv3 BBCH model with direct metadata concatenation."""

from __future__ import annotations

import ast
import inspect
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DINOV3_DIR = PROJECT_ROOT / "DINOv3_BBCH"
if str(DINOV3_DIR) not in sys.path:
    sys.path.insert(0, str(DINOV3_DIR))

from multiscale_phenology import (  # noqa: E402
    DINO_DEFAULT_BACKBONE,
    HierarchicalDenseTilePooler,
    LOCATION_FEATURE_COLUMNS,
    ViTBackboneFeatureExtractor,
)
from precompute_multiscale_embeddings import (  # noqa: E402
    build_image_transform,
    select_tiles,
    tile_boxes,
)


LINEAR_METADATA_COLUMNS = (
    "weather_gdd_cum_norm",
    *LOCATION_FEATURE_COLUMNS,
)


def create_backbone_extractor(
    backbone_name: str,
    pretrained: bool = True,
    backbone_config: Optional[Dict] = None,
    extractor_cls=None,
) -> nn.Module:
    """Construct the shared extractor while tolerating its pre-local_config API."""
    extractor_cls = extractor_cls or ViTBackboneFeatureExtractor
    kwargs = {"pretrained": pretrained}
    if backbone_config is not None:
        parameters = inspect.signature(extractor_cls.__init__).parameters
        if "local_config" not in parameters:
            raise RuntimeError(
                "This checkpoint contains an offline backbone configuration, but the installed "
                "DINOv3_BBCH/ViTBackboneFeatureExtractor is an older version without "
                "'local_config' support. Sync DINOv3_BBCH/multiscale_phenology.py from the "
                "same repository version as DINOv3_BBCH_Linear."
            )
        kwargs["local_config"] = backbone_config
    return extractor_cls(str(backbone_name), **kwargs)


def path_key(path: object) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


def parse_target(value: object) -> List[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    if isinstance(value, str):
        return [float(item) for item in ast.literal_eval(value)]
    raise TypeError(f"Unsupported target value: {type(value).__name__}")


def prepare_image_rows(
    frame: pd.DataFrame,
    stream: str,
    active_classes: Sequence[str],
    base_classes: Sequence[str],
    metadata_columns: Sequence[str] = LINEAR_METADATA_COLUMNS,
) -> pd.DataFrame:
    path_column = "micro_path" if stream == "micro" else "macro_path"
    rows = frame.loc[frame["label"].isin(active_classes) & frame[path_column].notna()].copy()
    rows["image_path"] = rows[path_column].map(path_key)
    rows = rows.loc[rows["image_path"].map(os.path.isfile)].copy()
    class_indices = [base_classes.index(name) for name in active_classes]

    def active_target(value: object) -> List[float]:
        full = parse_target(value)
        target = np.asarray([full[index] for index in class_indices], dtype=np.float32)
        total = float(target.sum())
        if total <= 0:
            raise ValueError("Image target has no probability mass in active classes")
        return (target / total).tolist()

    rows["active_target"] = rows["target"].map(active_target)
    for column in metadata_columns:
        if column not in rows:
            raise ValueError(f"Required metadata column is missing: {column}")
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0).astype(float)
    return (
        rows.sort_values(["station_year", "date"])
        .drop_duplicates("image_path", keep="first")
        .reset_index(drop=True)
    )


class LinearTiledImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
        tile_size: int,
        tile_stride: int,
        max_tiles: int,
        train: bool,
        mean: Sequence[float],
        std: Sequence[float],
        metadata_columns: Sequence[str] = LINEAR_METADATA_COLUMNS,
    ):
        if max_tiles < 1:
            raise ValueError("--max-tiles must be at least 1 during fine-tuning")
        self.rows = list(frame.itertuples(index=False))
        self.tile_size = int(tile_size)
        self.tile_stride = int(tile_stride)
        self.max_tiles = int(max_tiles)
        self.train = bool(train)
        self.metadata_columns = tuple(metadata_columns)
        self.transform = build_image_transform(
            image_size=image_size,
            augment=train,
            mean=list(mean),
            std=list(std),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _choose_boxes(self, boxes: List[Tuple[int, int, int, int]]):
        if len(boxes) <= self.max_tiles:
            return boxes
        if self.train:
            indices = sorted(random.sample(range(len(boxes)), self.max_tiles))
            return [boxes[index] for index in indices]
        return select_tiles(boxes, self.max_tiles)

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        try:
            with Image.open(row.image_path) as image:
                image = image.convert("RGB")
                boxes = self._choose_boxes(
                    tile_boxes(image.width, image.height, self.tile_size, self.tile_stride)
                )
                tiles = torch.stack([self.transform(image.crop(box)) for box in boxes])
            metadata = torch.tensor(
                [float(getattr(row, column)) for column in self.metadata_columns],
                dtype=torch.float32,
            )
            return {
                "tiles": tiles,
                "metadata": metadata,
                "target": torch.tensor(row.active_target, dtype=torch.float32),
                "path": row.image_path,
                "station_year": str(row.station_year),
                "station_code": str(row.station_code),
                "date": str(row.date),
                "error": None,
            }
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
            return {
                "tiles": None,
                "metadata": None,
                "target": None,
                "path": row.image_path,
                "station_year": str(row.station_year),
                "station_code": str(row.station_code),
                "date": str(row.date),
                "error": f"{type(exc).__name__}: {exc}",
            }


def collate_tiled_images(items: List[Dict]) -> Optional[Dict]:
    valid = [item for item in items if item["error"] is None]
    for item in items:
        if item["error"] is not None:
            print(f"Skipping unreadable image {item['path']}: {item['error']}", flush=True)
    if not valid:
        return None
    max_tiles = max(item["tiles"].shape[0] for item in valid)
    sample = valid[0]["tiles"]
    tiles = sample.new_zeros((len(valid), max_tiles, *sample.shape[1:]))
    mask = torch.zeros((len(valid), max_tiles), dtype=torch.bool)
    for index, item in enumerate(valid):
        count = item["tiles"].shape[0]
        tiles[index, :count] = item["tiles"]
        mask[index, :count] = True
    return {
        "tiles": tiles,
        "tile_mask": mask,
        "metadata": torch.stack([item["metadata"] for item in valid]),
        "target": torch.stack([item["target"] for item in valid]),
        "path": [item["path"] for item in valid],
        "station_year": [item["station_year"] for item in valid],
        "station_code": [item["station_code"] for item in valid],
        "date": [item["date"] for item in valid],
    }


def module_by_path(module: nn.Module, path: str) -> Optional[nn.Module]:
    current = module
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def find_transformer_blocks(backbone: nn.Module) -> Tuple[str, nn.ModuleList]:
    preferred = ("layer", "encoder.layer", "encoder.layers", "blocks", "transformer.layer")
    for name in preferred:
        candidate = module_by_path(backbone, name)
        if isinstance(candidate, nn.ModuleList) and len(candidate):
            return name, candidate
    candidates = [
        (name, module)
        for name, module in backbone.named_modules()
        if isinstance(module, nn.ModuleList) and len(module)
    ]
    if not candidates:
        raise ValueError("Could not locate the backbone Transformer blocks")
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    return candidates[0]


def configure_partial_finetuning(
    extractor: nn.Module,
    unfreeze_last_blocks: int,
    unfreeze_final_norm: bool,
) -> Dict:
    for parameter in extractor.parameters():
        parameter.requires_grad = False
    block_path, blocks = find_transformer_blocks(extractor.backbone)
    if not 1 <= unfreeze_last_blocks <= len(blocks):
        raise ValueError(
            f"--unfreeze-last-blocks must be in [1, {len(blocks)}], got {unfreeze_last_blocks}"
        )
    for block in blocks[-unfreeze_last_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    norm_names = []
    if unfreeze_final_norm:
        block_prefix = f"{block_path}."
        for name, module in extractor.backbone.named_modules():
            if name.startswith(block_prefix):
                continue
            if isinstance(module, nn.LayerNorm) or name.lower().endswith(("norm", "layernorm")):
                parameters = list(module.parameters(recurse=False))
                if parameters:
                    for parameter in parameters:
                        parameter.requires_grad = True
                    norm_names.append(name)
    return {
        "block_path": block_path,
        "total_blocks": len(blocks),
        "unfrozen_blocks": list(range(len(blocks) - unfreeze_last_blocks, len(blocks))),
        "unfrozen_norms": norm_names,
    }


class DINOv3LinearClassifier(nn.Module):
    """Dense DINOv3 image representation + metadata + one linear BBCH head."""

    def __init__(
        self,
        backbone_name: str = DINO_DEFAULT_BACKBONE,
        num_classes: int = 7,
        dense_grid_size: int = 2,
        dense_include_cls: bool = True,
        metadata_columns: Sequence[str] = LINEAR_METADATA_COLUMNS,
        dropout: float = 0.2,
        pretrained: bool = True,
        backbone_config: Optional[Dict] = None,
        extractor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.dense_grid_size = int(dense_grid_size)
        self.dense_include_cls = bool(dense_include_cls)
        self.metadata_columns = tuple(metadata_columns)
        self.extractor = extractor or create_backbone_extractor(
            self.backbone_name,
            pretrained=pretrained,
            backbone_config=backbone_config,
        )
        self.tokens_per_tile = self.dense_grid_size**2 + int(self.dense_include_cls)
        self.pooler = HierarchicalDenseTilePooler(
            self.extractor.out_dim,
            self.tokens_per_tile,
        )
        fused_dim = self.extractor.out_dim + len(self.metadata_columns)
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(
        self,
        tiles: torch.Tensor,
        tile_mask: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        if tiles.ndim != 5:
            raise ValueError(f"tiles must be [batch, tiles, C, H, W], got {tuple(tiles.shape)}")
        batch, tile_count = tiles.shape[:2]
        if tile_mask.shape != (batch, tile_count):
            raise ValueError(f"tile_mask must be {(batch, tile_count)}, got {tuple(tile_mask.shape)}")
        if metadata.shape != (batch, len(self.metadata_columns)):
            raise ValueError(
                f"metadata must be [batch, {len(self.metadata_columns)}], got {tuple(metadata.shape)}"
            )
        dense = self.extractor.forward_dense(
            tiles.flatten(0, 1),
            grid_size=self.dense_grid_size,
            include_cls=self.dense_include_cls,
        )
        dense = dense.reshape(batch, tile_count, self.tokens_per_tile, -1).unsqueeze(1)
        visual = self.pooler(dense, tile_mask.unsqueeze(1)).squeeze(1)
        fused = torch.cat([visual, metadata.to(device=visual.device, dtype=visual.dtype)], dim=-1)
        return self.classifier(self.dropout(self.fusion_norm(fused)))

    def checkpoint_config(self) -> Dict:
        backbone_config = getattr(getattr(self.extractor, "backbone", None), "config", None)
        return {
            "backbone_name": self.backbone_name,
            "num_classes": self.classifier.out_features,
            "dense_grid_size": self.dense_grid_size,
            "dense_include_cls": self.dense_include_cls,
            "metadata_columns": list(self.metadata_columns),
            "dropout": float(self.dropout.p),
            "backbone_config": backbone_config.to_dict() if backbone_config is not None else None,
            "preprocess": {
                "image_size": int(getattr(self.extractor, "preprocess_image_size", 224)),
                "mean": list(getattr(self.extractor, "preprocess_mean", [0.485, 0.456, 0.406])),
                "std": list(getattr(self.extractor, "preprocess_std", [0.229, 0.224, 0.225])),
                "patch_size": int(getattr(self.extractor, "patch_size", 16)),
                "num_register_tokens": int(getattr(self.extractor, "num_register_tokens", 0)),
            },
        }
