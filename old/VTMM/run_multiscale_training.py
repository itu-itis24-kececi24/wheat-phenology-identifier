import argparse
import json
import os
import time
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from framework import generate_group_folds
except ModuleNotFoundError:
    from .framework import generate_group_folds
from multiscale_phenology import (
    BASE_CLASSES,
    MultiScaleEmbeddingTemporalTransformer,
    MultiScaleEmbeddingWindowDataset,
    MultiScaleTemporalTransformer,
    MultiScaleWindowDataset,
    SoftTargetCrossEntropy,
    WindowConfig,
    build_multiscale_daily_dataframe,
)


def log(message: str):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def stage_distance_accuracy(pred: torch.Tensor, label: torch.Tensor, tolerance: int = 1) -> float:
    return (torch.abs(pred - label) <= tolerance).float().mean().item()


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    accumulation_steps: int = 1,
    log_interval: int = 25,
    phase: str = "train",
    fold_id: int = 0,
    epoch: int = 0,
):
    model.train(train)
    total_loss, total_correct, total_neighbor, total = 0.0, 0.0, 0.0, 0
    accumulation_steps = max(1, accumulation_steps)
    epoch_start = time.time()

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, 1):
        batch_start = time.time()
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            logits = model(batch["macro"], batch["micro"], batch["mask"])
            loss = criterion(logits, batch["target"])

        if train:
            (loss / accumulation_steps).backward()
            if step % accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        pred = torch.argmax(logits, dim=1)
        label = batch["label"]
        batch_size = label.numel()
        total += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (pred == label).float().sum().item()
        total_neighbor += stage_distance_accuracy(pred, label, tolerance=1) * batch_size

        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == len(loader)):
            elapsed = time.time() - epoch_start
            batch_elapsed = time.time() - batch_start
            current_loss = total_loss / max(total, 1)
            current_acc = total_correct / max(total, 1)
            msg = (
                f"fold={fold_id} epoch={epoch} {phase} "
                f"batch={step}/{len(loader)} samples={total} "
                f"loss={current_loss:.4f} acc={current_acc:.4f} "
                f"elapsed={elapsed:.1f}s batch_time={batch_elapsed:.2f}s"
            )
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                msg += f" max_vram={mem:.2f}GB"
            log(msg)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "plus_minus_1_accuracy": total_neighbor / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the multi-scale temporal wheat phenology model.")
    parser.add_argument("--excel-path", default="labeling.xlsx")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_multiscale")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2, help="Physical DataLoader batch size. Keep this small for ViT windows.")
    parser.add_argument("--accumulation-steps", type=int, default=16, help="Gradient accumulation steps. Effective batch size = batch_size * accumulation_steps.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window-days", type=int, default=31)
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--preplant-days", type=int, default=30, help="Number of days before seeding to keep as OffSeason.")
    parser.add_argument("--postharvest-days", type=int, default=30, help="Number of days after harvest to keep as OffSeason.")
    parser.add_argument("--camera", default="K1", help="Camera folder to use, e.g. K1 or K2. Use ALL for both.")
    parser.add_argument("--ignore-status-csv", action="store_true", help="Scan image folders directly instead of using day_image_status CSV files.")
    parser.add_argument("--embedding-cache", default=None, help="Optional .pt cache from precompute_multiscale_embeddings.py.")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=25, help="Print progress every N batches. Use 0 to disable batch progress logs.")
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    effective_batch = args.batch_size * max(1, args.accumulation_steps)
    log("Starting multi-scale phenology training")
    log(f"Working directory: {os.getcwd()}")
    log(f"Arguments: {json.dumps(vars(args), default=str)}")
    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"CUDA available: {torch.cuda.is_available()}")
        log(f"CUDA device: {torch.cuda.get_device_name(device)}")
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        log(f"CUDA total VRAM: {total_vram:.2f}GB")
    log(f"Physical batch size: {args.batch_size}")
    log(f"Gradient accumulation steps: {args.accumulation_steps}")
    log(f"Effective training batch size: {effective_batch}")

    log("Building daily metadata dataframe")
    daily_df = build_multiscale_daily_dataframe(
        args.excel_path,
        args.data_path,
        include_preplant_days=args.preplant_days,
        include_postharvest_days=args.postharvest_days,
        transition_days=args.transition_days,
        classes=BASE_CLASSES,
        preferred_camera=None if args.camera.upper() == "ALL" else args.camera,
        use_status_csv=not args.ignore_status_csv,
    )
    if daily_df.empty:
        raise RuntimeError("No paired daily rows were created. Check data_path, folder names, and filename dates.")

    meta_path = os.path.join(args.out_dir, "multiscale_daily_metadata.csv")
    daily_df.to_csv(meta_path, index=False)
    log(f"Saved metadata: {meta_path}")
    log(f"Metadata rows: {len(daily_df)}")
    log(f"Station-years: {daily_df['station_year'].nunique()} | Groups: {daily_df['group_id'].nunique()}")
    log(f"Date range: {daily_df['date'].min()} -> {daily_df['date'].max()}")
    log("Label counts:\n" + daily_df["label"].value_counts().sort_index().to_string())
    log("Path availability:\n" + daily_df[["macro_path", "micro_path"]].notna().mean().to_string())

    log("Generating group folds")
    folds = generate_group_folds(
        daily_df,
        group_col="group_id",
        n_train=8,
        n_test=2,
        num_folds=args.folds,
        random_state=42,
    )

    cfg = WindowConfig(window_days=args.window_days, classes=tuple(BASE_CLASSES))
    all_history = []
    if args.embedding_cache:
        log(f"Loading embedding cache: {args.embedding_cache}")
        embedding_cache = torch.load(args.embedding_cache, map_location="cpu")
        log(
            "Embedding cache loaded: "
            f"feature_dim={embedding_cache['feature_dim']} "
            f"macro={len(embedding_cache.get('macro', {}))} "
            f"micro={len(embedding_cache.get('micro', {}))}"
        )
    else:
        embedding_cache = None
        log("No embedding cache supplied; training will load images and run ViT each epoch")

    for fold_id, (train_idx, test_idx) in enumerate(folds, 1):
        log(f"Preparing fold {fold_id}/{len(folds)}")
        fold_dir = os.path.join(args.out_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = daily_df.iloc[train_idx].reset_index(drop=True)
        test_df = daily_df.iloc[test_idx].reset_index(drop=True)
        log(
            f"Fold {fold_id}: train rows={len(train_df)} test rows={len(test_df)} "
            f"train groups={sorted(train_df['group_id'].unique().tolist())} "
            f"test groups={sorted(test_df['group_id'].unique().tolist())}"
        )
        if embedding_cache is None:
            log(f"Fold {fold_id}: creating image window datasets")
            train_ds = MultiScaleWindowDataset(train_df, cfg)
            test_ds = MultiScaleWindowDataset(test_df, cfg)
        else:
            log(f"Fold {fold_id}: creating embedding window datasets")
            train_ds = MultiScaleEmbeddingWindowDataset(train_df, cfg, embedding_cache)
            test_ds = MultiScaleEmbeddingWindowDataset(test_df, cfg, embedding_cache)
        log(f"Fold {fold_id}: train samples={len(train_ds)} test samples={len(test_ds)}")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        log(
            f"Fold {fold_id}: loaders ready "
            f"train_batches={len(train_loader)} test_batches={len(test_loader)} "
            f"num_workers={args.num_workers}"
        )

        if embedding_cache is None:
            log(f"Fold {fold_id}: initializing full image model")
            model = MultiScaleTemporalTransformer(
                num_classes=len(BASE_CLASSES),
                pretrained=args.pretrained,
            ).to(device)
        else:
            log(f"Fold {fold_id}: initializing cached-embedding temporal model")
            model = MultiScaleEmbeddingTemporalTransformer(
                feature_dim=int(embedding_cache["feature_dim"]),
                num_classes=len(BASE_CLASSES),
            ).to(device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        log(f"Fold {fold_id}: model params total={total_params:,} trainable={trainable_params:,}")
        criterion = SoftTargetCrossEntropy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        best_acc = -1.0
        history = []
        for epoch in range(1, args.epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            log(f"Fold {fold_id} epoch {epoch}/{args.epochs}: training started")
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train=True,
                accumulation_steps=args.accumulation_steps,
                log_interval=args.log_interval,
                phase="train",
                fold_id=fold_id,
                epoch=epoch,
            )
            log(f"Fold {fold_id} epoch {epoch}/{args.epochs}: validation started")
            test_metrics = run_epoch(
                model,
                test_loader,
                criterion,
                optimizer,
                device,
                train=False,
                log_interval=args.log_interval,
                phase="val",
                fold_id=fold_id,
                epoch=epoch,
            )
            row = {
                "fold": fold_id,
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            history.append(row)
            all_history.append(row)
            log("Epoch metrics: " + json.dumps(row, indent=None))

            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                checkpoint_path = os.path.join(fold_dir, "best_model.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "classes": BASE_CLASSES,
                        "window_days": args.window_days,
                        "uses_embedding_cache": embedding_cache is not None,
                    },
                    checkpoint_path,
                )
                log(f"Fold {fold_id}: saved new best checkpoint to {checkpoint_path} acc={best_acc:.4f}")

        history_path = os.path.join(fold_dir, "history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False)
        log(f"Fold {fold_id}: saved history to {history_path}")

    all_history_path = os.path.join(args.out_dir, "all_history.csv")
    pd.DataFrame(all_history).to_csv(all_history_path, index=False)
    log(f"Saved combined history to {all_history_path}")
    log(f"Done in {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
