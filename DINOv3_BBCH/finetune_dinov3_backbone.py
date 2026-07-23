import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from multiscale_phenology import (
    BASE_CLASSES,
    DINO_DEFAULT_BACKBONE,
    HierarchicalDenseTilePooler,
    HybridOrdinalLoss,
    ViTBackboneFeatureExtractor,
    build_multiscale_daily_dataframe,
    generate_loso_train_val_test_folds,
)
from precompute_multiscale_embeddings import build_image_transform, select_tiles, tile_boxes


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def path_key(path: object) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


def parse_target(value: object) -> List[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    if isinstance(value, str):
        import ast

        return [float(item) for item in ast.literal_eval(value)]
    raise TypeError(f"Unsupported target value: {type(value).__name__}")


def prepare_image_rows(
    frame: pd.DataFrame,
    stream: str,
    classes: Sequence[str],
) -> pd.DataFrame:
    path_column = "micro_path" if stream == "micro" else "macro_path"
    rows = frame.loc[frame["label"].isin(classes) & frame[path_column].notna()].copy()
    rows["image_path"] = rows[path_column].map(path_key)
    rows = rows.loc[rows["image_path"].map(os.path.isfile)].copy()
    class_indices = [BASE_CLASSES.index(name) for name in classes]

    def active_target(value: object) -> List[float]:
        full = parse_target(value)
        target = np.asarray([full[index] for index in class_indices], dtype=np.float32)
        total = float(target.sum())
        if total <= 0:
            raise ValueError("Image target has no probability mass in active classes")
        return (target / total).tolist()

    rows["active_target"] = rows["target"].map(active_target)
    # One camera image should contribute only once even if malformed metadata duplicated a date.
    rows = rows.sort_values(["station_year", "date"]).drop_duplicates("image_path", keep="first")
    return rows.reset_index(drop=True)


class LabeledTiledImageDataset(Dataset):
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
    ):
        if max_tiles < 1:
            raise ValueError("Fine-tuning requires --max-tiles >= 1 to bound GPU memory")
        self.rows = list(frame.itertuples(index=False))
        self.tile_size = int(tile_size)
        self.tile_stride = int(tile_stride)
        self.max_tiles = int(max_tiles)
        self.train = bool(train)
        self.transform = build_image_transform(
            image_size=image_size,
            augment=train,
            mean=list(mean),
            std=list(std),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _choose_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        if len(boxes) <= self.max_tiles:
            return boxes
        if self.train:
            return [boxes[index] for index in sorted(random.sample(range(len(boxes)), self.max_tiles))]
        return select_tiles(boxes, self.max_tiles)

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        try:
            with Image.open(row.image_path) as image:
                image = image.convert("RGB")
                boxes = tile_boxes(image.width, image.height, self.tile_size, self.tile_stride)
                boxes = self._choose_boxes(boxes)
                tiles = torch.stack([self.transform(image.crop(box)) for box in boxes])
            return {
                "tiles": tiles,
                "target": torch.tensor(row.active_target, dtype=torch.float32),
                "path": row.image_path,
                "station_year": str(row.station_year),
                "date": str(row.date),
                "error": None,
            }
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
            return {
                "tiles": None,
                "target": None,
                "path": row.image_path,
                "station_year": str(row.station_year),
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
        "target": torch.stack([item["target"] for item in valid]),
        "path": [item["path"] for item in valid],
        "station_year": [item["station_year"] for item in valid],
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
        if isinstance(module, nn.ModuleList) and len(module) and any(p.numel() for p in module.parameters())
    ]
    if not candidates:
        raise ValueError("Could not locate the Hugging Face backbone Transformer block list")
    candidates.sort(key=lambda item: (len(item[1]), "layer" in item[0] or "block" in item[0]), reverse=True)
    return candidates[0]


def configure_partial_finetuning(
    extractor: ViTBackboneFeatureExtractor,
    unfreeze_last_blocks: int,
    unfreeze_final_norm: bool,
) -> Dict:
    for parameter in extractor.parameters():
        parameter.requires_grad = False
    block_path, blocks = find_transformer_blocks(extractor.backbone)
    if not 1 <= unfreeze_last_blocks <= len(blocks):
        raise ValueError(
            f"--unfreeze-last-blocks must be in [1, {len(blocks)}] for {block_path}, "
            f"got {unfreeze_last_blocks}"
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


class DINOv3TiledClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dense_grid_size: int,
        dense_include_cls: bool,
        dropout: float,
    ):
        super().__init__()
        self.extractor = ViTBackboneFeatureExtractor(backbone_name, pretrained=True)
        self.dense_grid_size = int(dense_grid_size)
        self.dense_include_cls = bool(dense_include_cls)
        self.tokens_per_tile = self.dense_grid_size ** 2 + int(self.dense_include_cls)
        self.pooler = HierarchicalDenseTilePooler(self.extractor.out_dim, self.tokens_per_tile)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.extractor.out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.extractor.out_dim, num_classes),
        )

    def forward(self, tiles: torch.Tensor, tile_mask: torch.Tensor) -> torch.Tensor:
        batch, tile_count = tiles.shape[:2]
        flat = tiles.flatten(0, 1)
        features = self.extractor.forward_dense(
            flat,
            grid_size=self.dense_grid_size,
            include_cls=self.dense_include_cls,
        )
        features = features.reshape(batch, tile_count, self.tokens_per_tile, -1).unsqueeze(1)
        pooled = self.pooler(features, tile_mask.unsqueeze(1)).squeeze(1)
        return self.classifier(pooled)


def macro_f1(true_indices: List[int], pred_indices: List[int], num_classes: int) -> float:
    scores = []
    true = np.asarray(true_indices)
    pred = np.asarray(pred_indices)
    for index in range(num_classes):
        tp = int(((true == index) & (pred == index)).sum())
        fp = int(((true != index) & (pred == index)).sum())
        fn = int(((true == index) & (pred != index)).sum())
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    accumulation_steps: int,
    scheduler=None,
) -> Tuple[Dict, List[Dict]]:
    train = optimizer is not None
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total = 0
    true_indices: List[int] = []
    pred_indices: List[int] = []
    predictions: List[Dict] = []
    valid_steps = 0
    pending_steps = 0

    def optimizer_step(pending_count: int) -> None:
        # Losses are divided by the requested accumulation count. Rescale a
        # short final accumulation group so it remains an average, not a
        # smaller update merely because the epoch length is not divisible.
        if pending_count < accumulation_steps:
            correction = accumulation_steps / pending_count
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
    for step, batch in enumerate(loader, 1):
        if batch is None:
            continue
        valid_steps += 1
        tiles = batch["tiles"].to(device, non_blocking=True)
        tile_mask = batch["tile_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train), amp_context(device, scaler is not None and scaler.is_enabled()):
            logits = model(tiles, tile_mask)
            raw_loss = criterion(logits, targets)
            loss = raw_loss / accumulation_steps if train else raw_loss
        if train:
            scaler.scale(loss).backward()
            pending_steps += 1
            if pending_steps == accumulation_steps:
                optimizer_step(pending_steps)
                pending_steps = 0
        predicted = logits.detach().argmax(dim=-1).cpu()
        actual = targets.detach().argmax(dim=-1).cpu()
        batch_size = len(actual)
        total_loss += float(raw_loss.detach().cpu()) * batch_size
        total += batch_size
        true_indices.extend(actual.tolist())
        pred_indices.extend(predicted.tolist())
        for idx in range(batch_size):
            predictions.append(
                {
                    "path": batch["path"][idx],
                    "station_year": batch["station_year"][idx],
                    "date": batch["date"][idx],
                    "true_idx": int(actual[idx]),
                    "pred_idx": int(predicted[idx]),
                }
            )
    if train and pending_steps:
        optimizer_step(pending_steps)
    if not total:
        raise RuntimeError("No readable images were produced by the DataLoader")
    metrics = {
        "loss": total_loss / total,
        "accuracy": float(np.mean(np.asarray(true_indices) == np.asarray(pred_indices))),
        "macro_f1": macro_f1(true_indices, pred_indices, len(targets[0])),
        "samples": total,
    }
    return metrics, predictions


def cosine_warmup_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def build_loader(dataset: Dataset, args, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed + (0 if shuffle else 10_000))
    kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "collate_fn": collate_tiled_images,
        "generator": generator,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 1
    return DataLoader(**kwargs)


def save_backbone(model: DINOv3TiledClassifier, output_dir: Path, metadata: Dict) -> None:
    backbone_dir = output_dir / "backbone"
    backbone_dir.mkdir(parents=True, exist_ok=True)
    model.extractor.backbone.save_pretrained(backbone_dir, safe_serialization=True)
    processor = getattr(model.extractor, "processor", None)
    if processor is not None:
        processor.save_pretrained(backbone_dir)
    with open(backbone_dir / "wheat_finetune_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold-safe partial fine-tuning of DINOv3 on tiled BBCH crop images.")
    parser.add_argument("--label-path", default="labeling_bbch_iso_dates.csv")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_dinov3_backbone_finetune")
    parser.add_argument("--image-backbone", default=DINO_DEFAULT_BACKBONE)
    parser.add_argument("--stream", choices=["micro", "macro"], default="micro")
    parser.add_argument("--camera", default="AUTO")
    parser.add_argument("--fold-id", type=int, required=True, help="One-based LOSO fold to fine-tune. Run separately for each fold.")
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--validation-groups", type=int, default=2)
    parser.add_argument("--expected-stations", type=int, default=None)
    parser.add_argument("--exclude-offseason", action="store_true", default=True)
    parser.add_argument("--include-offseason", dest="exclude_offseason", action="store_false")
    parser.add_argument("--preplant-days", type=int, default=30)
    parser.add_argument("--postharvest-days", type=int, default=30)
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--date-tolerance-days", type=int, default=5)
    parser.add_argument("--ignore-status-csv", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of source images per batch; each image contains max_tiles crops.")
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument("--unfreeze-final-norm", action="store_true", default=True)
    parser.add_argument("--freeze-final-norm", dest="unfreeze_final_norm", action="store_false")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tile-stride", type=int, default=224)
    parser.add_argument("--max-tiles", type=int, default=16)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--dense-grid-size", type=int, default=2)
    parser.add_argument("--dense-include-cls", action="store_true", default=True)
    parser.add_argument("--no-dense-cls", dest="dense_include_cls", action="store_false")
    parser.add_argument("--ordinal-power", type=int, choices=[1, 2], default=2)
    parser.add_argument("--ordinal-ce-weight", type=float, default=0.5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.fold_id < 1:
        raise ValueError("--fold-id is one-based and must be at least 1")
    if args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError("Batch size and accumulation steps must be positive")
    seed_everything(args.seed + args.fold_id)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir) / f"fold_{args.fold_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_df = build_multiscale_daily_dataframe(
        args.label_path,
        args.data_path,
        include_preplant_days=args.preplant_days,
        include_postharvest_days=args.postharvest_days,
        transition_days=args.transition_days,
        date_tolerance_days=args.date_tolerance_days,
        classes=BASE_CLASSES,
        preferred_camera=None if args.camera.upper() == "ALL" else args.camera,
        use_status_csv=not args.ignore_status_csv,
    ).reset_index(drop=True)
    station_count = int(daily_df["station_code"].nunique())
    if args.expected_stations is not None and station_count != args.expected_stations:
        stations = sorted(daily_df["station_code"].dropna().astype(str).unique())
        raise ValueError(f"Expected {args.expected_stations} stations, found {station_count}: {stations}")
    folds = generate_loso_train_val_test_folds(
        daily_df,
        group_col="station_code",
        n_val=args.validation_groups,
        random_state=args.fold_seed,
    )
    if args.fold_id > len(folds):
        raise ValueError(f"--fold-id {args.fold_id} exceeds the {len(folds)} generated LOSO folds")
    train_idx, val_idx, test_idx = folds[args.fold_id - 1]
    active_classes = [name for name in BASE_CLASSES if name != "OffSeason"] if args.exclude_offseason else list(BASE_CLASSES)
    train_rows = prepare_image_rows(daily_df.iloc[train_idx], args.stream, active_classes)
    val_rows = prepare_image_rows(daily_df.iloc[val_idx], args.stream, active_classes)
    test_rows = prepare_image_rows(daily_df.iloc[test_idx], args.stream, active_classes)
    split_info = {
        "fold": args.fold_id,
        "train_stations": sorted(daily_df.iloc[train_idx]["station_code"].astype(str).unique()),
        "val_stations": sorted(daily_df.iloc[val_idx]["station_code"].astype(str).unique()),
        "test_stations": sorted(daily_df.iloc[test_idx]["station_code"].astype(str).unique()),
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "test_images": len(test_rows),
        "classes": active_classes,
    }
    print(json.dumps(split_info, indent=2), flush=True)
    with open(out_dir / "split_info.json", "w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2)

    model = DINOv3TiledClassifier(
        args.image_backbone,
        len(active_classes),
        args.dense_grid_size,
        args.dense_include_cls,
        args.dropout,
    ).to(device)
    max_dense_grid = args.vit_image_size // int(model.extractor.patch_size)
    if not 1 <= args.dense_grid_size <= max_dense_grid:
        raise ValueError(
            f"--dense-grid-size must be in [1, {max_dense_grid}] for "
            f"image_size={args.vit_image_size} and patch_size={model.extractor.patch_size}"
        )
    tune_info = configure_partial_finetuning(model.extractor, args.unfreeze_last_blocks, args.unfreeze_final_norm)
    print(f"Partial fine-tuning: {json.dumps(tune_info)}", flush=True)
    mean = model.extractor.preprocess_mean
    std = model.extractor.preprocess_std
    datasets = {
        "train": LabeledTiledImageDataset(train_rows, args.vit_image_size, args.tile_size, args.tile_stride, args.max_tiles, True, mean, std),
        "val": LabeledTiledImageDataset(val_rows, args.vit_image_size, args.tile_size, args.tile_stride, args.max_tiles, False, mean, std),
        "test": LabeledTiledImageDataset(test_rows, args.vit_image_size, args.tile_size, args.tile_stride, args.max_tiles, False, mean, std),
    }
    loaders = {
        name: build_loader(dataset, args, shuffle=name == "train")
        for name, dataset in datasets.items()
    }
    backbone_params = [p for p in model.extractor.parameters() if p.requires_grad]
    head_params = [p for module in (model.pooler, model.classifier) for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    optimizer_steps = max(1, math.ceil(len(loaders["train"]) / args.accumulation_steps) * args.epochs)
    scheduler = cosine_warmup_scheduler(optimizer, optimizer_steps, args.warmup_ratio)
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    criterion = HybridOrdinalLoss(args.ordinal_power, args.ordinal_ce_weight)
    best_f1 = -math.inf
    history = []
    best_path = out_dir / "best_finetune.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(
            model, loaders["train"], criterion, device, optimizer, scaler,
            args.accumulation_steps, scheduler,
        )
        val_metrics, val_predictions = run_epoch(
            model, loaders["val"], criterion, device, None, None, 1,
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        pd.DataFrame(val_predictions).to_csv(out_dir / "val_predictions.csv", index=False)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "classes": active_classes,
                    "tune_info": tune_info,
                    "args": vars(args),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, test_predictions = run_epoch(model, loaders["test"], criterion, device, None, None, 1)
    pd.DataFrame(test_predictions).to_csv(out_dir / "test_predictions.csv", index=False)
    result = {
        "best_epoch": checkpoint["epoch"],
        "best_val_macro_f1": checkpoint["val_macro_f1"],
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    metadata = {
        **split_info,
        **tune_info,
        "source_backbone": args.image_backbone,
        "best_epoch": checkpoint["epoch"],
        "best_val_macro_f1": checkpoint["val_macro_f1"],
        "test_metrics": test_metrics,
        "dense_grid_size": args.dense_grid_size,
        "dense_include_cls": args.dense_include_cls,
    }
    save_backbone(model, out_dir, metadata)
    print(f"Saved fine-tuned backbone: {out_dir / 'backbone'}", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
