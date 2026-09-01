"""Fine-tune DINOv3 for single-day BBCH classification with a linear metadata head."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from torch.amp import GradScaler
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DINOV3_DIR = PROJECT_ROOT / "DINOv3_BBCH"
for path in (str(DINOV3_DIR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from linear_phenology import (  # noqa: E402
    DINOv3LinearClassifier,
    LINEAR_METADATA_COLUMNS,
    LinearTiledImageDataset,
    collate_tiled_images,
    configure_partial_finetuning,
    prepare_image_rows,
)
from multiscale_phenology import (  # noqa: E402
    BASE_CLASSES,
    DINO_DEFAULT_BACKBONE,
    HybridOrdinalLoss,
    WEATHER_MISSING_FEATURE_COLUMNS,
    add_location_metadata,
    add_weather_metadata,
    build_multiscale_daily_dataframe,
    generate_loso_train_val_test_folds,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_metrics(y_true, y_pred, classes: List[str]) -> Dict:
    labels = np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "plus_minus_one_accuracy": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1)),
        "mean_absolute_stage_error": float(mean_absolute_error(y_true, y_pred)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(classes)
        },
    }


def amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    classes,
    optimizer=None,
    scaler=None,
    accumulation_steps=1,
    scheduler=None,
):
    train = optimizer is not None
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    true_indices, pred_indices, rows = [], [], []
    total = 0
    pending = 0

    def optimizer_step(pending_count):
        if pending_count < accumulation_steps:
            correction = accumulation_steps / pending_count
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    for batch in loader:
        if batch is None:
            continue
        tiles = batch["tiles"].to(device, non_blocking=True)
        tile_mask = batch["tile_mask"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        use_amp = scaler is not None and scaler.is_enabled()
        with torch.set_grad_enabled(train), amp_context(device, use_amp):
            logits = model(tiles, tile_mask, metadata)
            raw_loss = criterion(logits, targets)
            loss = raw_loss / accumulation_steps if train else raw_loss
        if train:
            scaler.scale(loss).backward()
            pending += 1
            if pending == accumulation_steps:
                optimizer_step(pending)
                pending = 0
        probs = torch.softmax(logits.detach(), dim=-1).cpu()
        predicted = probs.argmax(dim=-1)
        actual = targets.detach().argmax(dim=-1).cpu()
        batch_size = len(actual)
        total_loss += float(raw_loss.detach().cpu()) * batch_size
        total += batch_size
        true_indices.extend(actual.tolist())
        pred_indices.extend(predicted.tolist())
        for index in range(batch_size):
            row = {
                "path": batch["path"][index],
                "station_year": batch["station_year"][index],
                "station_code": batch["station_code"][index],
                "date": batch["date"][index],
                "true_idx": int(actual[index]),
                "true_label": classes[int(actual[index])],
                "pred_idx": int(predicted[index]),
                "pred_label": classes[int(predicted[index])],
            }
            for class_index, class_name in enumerate(classes):
                row[f"prob_{class_name}"] = float(probs[index, class_index])
            rows.append(row)
    if train and pending:
        optimizer_step(pending)
    if not total:
        raise RuntimeError("No readable images were produced by the DataLoader")
    metrics = classification_metrics(true_indices, pred_indices, classes)
    metrics["loss"] = total_loss / total
    metrics["samples"] = total
    return metrics, rows


def cosine_warmup_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def multiplier(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def build_loader(dataset, args, shuffle):
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
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def save_metric_tables(fold_dir: Path, metrics: Dict, classes: List[str]) -> None:
    pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes).to_csv(
        fold_dir / "confusion_matrix.csv"
    )
    pd.DataFrame.from_dict(metrics["per_class"], orient="index").to_csv(
        fold_dir / "per_class_metrics.csv"
    )
    with open(fold_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def train_fold(args, daily_df, folds, fold_id, active_classes, device):
    seed_everything(args.seed + fold_id)
    fold_dir = Path(args.out_dir) / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_idx, val_idx, test_idx = folds[fold_id - 1]
    train_rows = prepare_image_rows(
        daily_df.iloc[train_idx], args.stream, active_classes, BASE_CLASSES
    )
    val_rows = prepare_image_rows(
        daily_df.iloc[val_idx], args.stream, active_classes, BASE_CLASSES
    )
    test_rows = prepare_image_rows(
        daily_df.iloc[test_idx], args.stream, active_classes, BASE_CLASSES
    )
    split_info = {
        "fold": fold_id,
        "train_stations": sorted(daily_df.iloc[train_idx]["station_code"].astype(str).unique()),
        "val_stations": sorted(daily_df.iloc[val_idx]["station_code"].astype(str).unique()),
        "test_stations": sorted(daily_df.iloc[test_idx]["station_code"].astype(str).unique()),
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "test_images": len(test_rows),
        "classes": active_classes,
    }
    with open(fold_dir / "split_info.json", "w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2)

    model = DINOv3LinearClassifier(
        backbone_name=args.image_backbone,
        num_classes=len(active_classes),
        dense_grid_size=args.dense_grid_size,
        dense_include_cls=args.dense_include_cls,
        metadata_columns=LINEAR_METADATA_COLUMNS,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)
    max_grid = args.vit_image_size // int(model.extractor.patch_size)
    if not 1 <= args.dense_grid_size <= max_grid:
        raise ValueError(f"--dense-grid-size must be in [1, {max_grid}]")
    tune_info = configure_partial_finetuning(
        model.extractor,
        args.unfreeze_last_blocks,
        args.unfreeze_final_norm,
    )
    mean, std = model.extractor.preprocess_mean, model.extractor.preprocess_std
    datasets = {
        "train": LinearTiledImageDataset(
            train_rows, args.vit_image_size, args.tile_size, args.tile_stride,
            args.max_tiles, True, mean, std,
        ),
        "val": LinearTiledImageDataset(
            val_rows, args.vit_image_size, args.tile_size, args.tile_stride,
            args.max_tiles, False, mean, std,
        ),
        "test": LinearTiledImageDataset(
            test_rows, args.vit_image_size, args.tile_size, args.tile_stride,
            args.max_tiles, False, mean, std,
        ),
    }
    loaders = {
        name: build_loader(dataset, args, shuffle=name == "train")
        for name, dataset in datasets.items()
    }
    backbone_params = [p for p in model.extractor.parameters() if p.requires_grad]
    head_params = [
        p
        for module in (model.pooler, model.fusion_norm, model.classifier)
        for p in module.parameters()
        if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    optimizer_steps = max(
        1,
        math.ceil(len(loaders["train"]) / args.accumulation_steps) * args.epochs,
    )
    scheduler = cosine_warmup_scheduler(optimizer, optimizer_steps, args.warmup_ratio)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    criterion = HybridOrdinalLoss(args.ordinal_power, args.ordinal_ce_weight)
    best_path = fold_dir / "best_model.pt"
    best_f1 = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(
            model, loaders["train"], criterion, device, active_classes,
            optimizer, scaler, args.accumulation_steps, scheduler,
        )
        val_metrics, val_predictions = run_epoch(
            model, loaders["val"], criterion, device, active_classes,
        )
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items() if not isinstance(v, (dict, list))},
            **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, (dict, list))},
        })
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        pd.DataFrame(val_predictions).to_csv(fold_dir / "val_predictions.csv", index=False)
        print(
            f"fold={fold_id} epoch={epoch}/{args.epochs} "
            f"train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save({
                "model": model.state_dict(),
                "model_config": model.checkpoint_config(),
                "tiling": {
                    "tile_size": args.tile_size,
                    "tile_stride": args.tile_stride,
                    "max_tiles": args.max_tiles,
                    "vit_image_size": args.vit_image_size,
                },
                "metadata_columns": list(LINEAR_METADATA_COLUMNS),
                "classes": active_classes,
                "epoch": epoch,
                "val_macro_f1": best_f1,
                "tune_info": tune_info,
                "split_info": split_info,
                "gdd_base_temp": args.gdd_base_temp,
                "args": vars(args),
            }, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, test_predictions = run_epoch(
        model, loaders["test"], criterion, device, active_classes,
    )
    pd.DataFrame(test_predictions).to_csv(fold_dir / "test_predictions.csv", index=False)
    save_metric_tables(fold_dir, test_metrics, active_classes)
    if args.export_backbone:
        backbone_dir = fold_dir / "backbone"
        backbone_dir.mkdir(exist_ok=True)
        model.extractor.backbone.save_pretrained(backbone_dir, safe_serialization=True)
        processor = getattr(model.extractor, "processor", None)
        if processor is not None:
            processor.save_pretrained(backbone_dir)
    summary = {
        "fold": fold_id,
        "best_epoch": checkpoint["epoch"],
        "best_val_macro_f1": checkpoint["val_macro_f1"],
        "test_stations": split_info["test_stations"],
        **{key: value for key, value in test_metrics.items() if key not in {"per_class", "confusion_matrix"}},
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary, test_predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-path", default="labeling_bbch_iso_dates.csv")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_dinov3_bbch_linear")
    parser.add_argument("--image-backbone", default=DINO_DEFAULT_BACKBONE)
    parser.add_argument("--stream", choices=["micro", "macro"], default="micro")
    parser.add_argument("--camera", default="AUTO")
    parser.add_argument("--weather-cache", default=None)
    parser.add_argument("--weather-force-refresh", action="store_true")
    parser.add_argument("--allow-missing-weather", action="store_true")
    parser.add_argument("--gdd-base-temp", type=float, default=0.0)
    parser.add_argument("--fold-id", type=int, default=0, help="One-based fold; 0 trains all LOSO folds.")
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--validation-groups", type=int, default=2)
    parser.add_argument("--expected-stations", type=int, default=None)
    parser.add_argument("--exclude-offseason", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preplant-days", type=int, default=30)
    parser.add_argument("--postharvest-days", type=int, default=30)
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--date-tolerance-days", type=int, default=5)
    parser.add_argument("--ignore-status-csv", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument("--unfreeze-final-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tile-stride", type=int, default=224)
    parser.add_argument("--max-tiles", type=int, default=16)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--dense-grid-size", type=int, default=2)
    parser.add_argument("--dense-include-cls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ordinal-power", type=int, choices=[1, 2], default=2)
    parser.add_argument("--ordinal-ce-weight", type=float, default=0.5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--export-backbone", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fold_id < 0:
        raise ValueError("--fold-id must be zero or a positive one-based fold")
    if min(args.batch_size, args.accumulation_steps, args.max_tiles) < 1:
        raise ValueError("Batch size, accumulation steps, and max tiles must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir)
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
    if daily_df.empty:
        raise RuntimeError("No daily image rows were created")
    station_count = int(daily_df["station_code"].nunique())
    if args.expected_stations is not None and station_count != args.expected_stations:
        stations = sorted(daily_df["station_code"].dropna().astype(str).unique())
        raise ValueError(f"Expected {args.expected_stations} stations, found {station_count}: {stations}")
    weather_cache = args.weather_cache or str(out_dir / "meteostat_weather_cache.csv")
    daily_df = add_weather_metadata(
        daily_df,
        weather_cache,
        force_refresh=args.weather_force_refresh,
        gdd_base_temp=args.gdd_base_temp,
    )
    if (
        not args.allow_missing_weather
        and daily_df[list(WEATHER_MISSING_FEATURE_COLUMNS)].to_numpy(dtype=float).mean() >= 0.999
    ):
        raise RuntimeError(
            "Weather is unavailable for every row. Provide --weather-cache, allow Meteostat "
            "access, or explicitly use --allow-missing-weather for a smoke test."
        )
    daily_df = add_location_metadata(daily_df, strict=True)
    daily_df.to_csv(out_dir / "daily_metadata.csv", index=False)
    folds = generate_loso_train_val_test_folds(
        daily_df,
        group_col="station_code",
        n_val=args.validation_groups,
        random_state=args.fold_seed,
    )
    if args.fold_id > len(folds):
        raise ValueError(f"--fold-id {args.fold_id} exceeds {len(folds)} LOSO folds")
    assignments = []
    for fold_id, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
        for role, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
            for station in sorted(daily_df.iloc[indices]["station_code"].astype(str).unique()):
                assignments.append({"fold": fold_id, "role": role, "station_code": station})
    pd.DataFrame(assignments).to_csv(out_dir / "fold_assignments.csv", index=False)

    active_classes = (
        [name for name in BASE_CLASSES if name != "OffSeason"]
        if args.exclude_offseason
        else list(BASE_CLASSES)
    )
    fold_ids = [args.fold_id] if args.fold_id else list(range(1, len(folds) + 1))
    summaries, predictions = [], []
    for fold_id in fold_ids:
        summary, fold_predictions = train_fold(
            args, daily_df, folds, fold_id, active_classes, device
        )
        summaries.append(summary)
        predictions.extend(fold_predictions)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined = pd.DataFrame(predictions)
    combined.to_csv(out_dir / "combined_test_predictions.csv", index=False)
    aggregate = classification_metrics(
        combined["true_idx"].astype(int).tolist(),
        combined["pred_idx"].astype(int).tolist(),
        active_classes,
    )
    aggregate["folds"] = summaries
    with open(out_dir / "aggregate_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    pd.DataFrame(aggregate["confusion_matrix"], index=active_classes, columns=active_classes).to_csv(
        out_dir / "aggregate_confusion_matrix.csv"
    )
    pd.DataFrame.from_dict(aggregate["per_class"], orient="index").to_csv(
        out_dir / "aggregate_per_class_metrics.csv"
    )
    print(
        f"aggregate accuracy={aggregate['accuracy']:.4f} macro_f1={aggregate['macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
