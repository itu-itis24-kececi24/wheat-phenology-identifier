import argparse
import json
import math
import os
import time
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from multiscale_phenology import (
    BASE_CLASSES,
    DEFAULT_TEMPORAL_FEATURE_COLUMNS,
    MultiScale3DCNN,
    MultiScaleWindowDataset,
    SingleStream3DCNN,
    SoftTargetCrossEntropy,
    WindowConfig,
    build_image_transform,
    build_multiscale_daily_dataframe,
    generate_group_train_val_test_folds,
)


# Edit these values to change the fold split for CNN3D experiments.
# Training groups are derived automatically:
# train_groups = total_groups - VALIDATION_FOLD_STATIONS - TEST_FOLD_STATIONS
VALIDATION_FOLD_STATIONS = 2
TEST_FOLD_STATIONS = 2


def log(message: str):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def maybe_plot_training_outputs(history_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str):
    if history_df.empty:
        log("No history rows available; skipping plots")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib is not installed; skipping PNG plots. Install with: python -m pip install matplotlib")
        return

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if col not in {"fold", "epoch"}:
                converted = pd.to_numeric(out[col], errors="coerce")
                if converted.notna().any() or out[col].isna().all():
                    out[col] = converted
        return out

    history_df = numeric_frame(history_df)

    def plot_epoch_metric(metric: str, ylabel: str, filename: str):
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        if train_col not in history_df.columns and val_col not in history_df.columns:
            return

        fig, ax = plt.subplots(figsize=(9, 5))
        for col, label, color in [
            (train_col, "train", "#2563eb"),
            (val_col, "validation", "#dc2626"),
        ]:
            if col not in history_df.columns:
                continue
            grouped = history_df.groupby("epoch")[col].agg(["mean", "std"]).reset_index()
            grouped["std"] = grouped["std"].fillna(0.0)
            epochs = grouped["epoch"].astype(float).to_numpy()
            mean = grouped["mean"].astype(float).to_numpy()
            std = grouped["std"].astype(float).to_numpy()
            ax.plot(epochs, mean, label=label, color=color, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(ylabel)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if "accuracy" in metric:
            ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(plot_dir, filename)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        log(f"Saved plot: {path}")

    plot_epoch_metric("accuracy", "Accuracy", "epoch_accuracy.png")
    plot_epoch_metric("date_window_accuracy", "Date-window accuracy", "epoch_date_window_accuracy.png")
    plot_epoch_metric("plus_minus_1_accuracy", "Plus/minus 1 stage accuracy", "epoch_plus_minus_1_accuracy.png")
    plot_epoch_metric("loss", "Loss", "epoch_loss.png")

    if test_df.empty:
        log("No test metric rows available; skipping final test plots")
        return

    test_df = numeric_frame(test_df)
    test_metrics = [
        ("test_accuracy", "Test accuracy"),
        ("test_date_window_accuracy", "Test date-window accuracy"),
        ("test_plus_minus_1_accuracy", "Test plus/minus 1 accuracy"),
    ]
    available = [(col, label) for col, label in test_metrics if col in test_df.columns]
    if not available:
        return

    folds = test_df["fold"].astype(str).tolist()
    x = range(len(folds))
    width = 0.8 / max(1, len(available))
    fig, ax = plt.subplots(figsize=(max(9, len(folds) * 0.9), 5))
    for idx, (col, label) in enumerate(available):
        values = pd.to_numeric(test_df[col], errors="coerce").fillna(0.0).to_numpy()
        offsets = [pos + (idx - (len(available) - 1) / 2) * width for pos in x]
        ax.bar(offsets, values, width=width, label=label)

    ax.set_title("Final held-out test metrics by fold")
    ax.set_xlabel("fold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(folds)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(plot_dir, "final_test_metrics_by_fold.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log(f"Saved plot: {path}")

    summary_rows = []
    for col, label in available:
        values = pd.to_numeric(test_df[col], errors="coerce").dropna()
        summary_rows.append({"metric": label, "mean": values.mean(), "std": values.std(ddof=0)})
    summary_df = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary_df["metric"], summary_df["mean"], yerr=summary_df["std"], capsize=4, color="#16a34a")
    ax.set_title("Final held-out test mean +/- std")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(plot_dir, "final_test_mean_std.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log(f"Saved plot: {path}")


def log_tb_metrics(writer, prefix: str, metrics: Dict[str, float], step: int):
    if writer is None:
        return
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{name}", value, step)


def log_tb_text(writer, tag: str, value: object, step: int = 0):
    if writer is None:
        return
    text = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
    writer.add_text(tag, f"```json\n{text}\n```", step)


def save_resume_checkpoint(path: str, state: Dict):
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def model_state_dict(model):
    module = getattr(model, "_orig_mod", model)
    return module.state_dict()


def load_model_state(model, state_dict):
    module = getattr(model, "_orig_mod", model)
    try:
        module.load_state_dict(state_dict)
        return
    except RuntimeError:
        if all(str(key).startswith("_orig_mod.") for key in state_dict):
            stripped = {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}
            module.load_state_dict(stripped)
            return
        raise


def maybe_compile_model(model, args, device):
    if not args.compile_model:
        return model, False
    if device.type != "cuda":
        log("torch.compile requested but disabled because device is not CUDA")
        return model, False
    if not hasattr(torch, "compile"):
        log("torch.compile requested but this PyTorch version does not provide torch.compile")
        return model, False
    try:
        log(f"Compiling model with torch.compile(mode={args.compile_mode})")
        return torch.compile(model, mode=args.compile_mode), True
    except Exception as exc:
        if args.compile_strict:
            raise
        log(f"torch.compile failed during setup; continuing without compile: {type(exc).__name__}: {exc}")
        return model, False


def build_optimizer(model, args, device):
    use_fused = args.fused_optimizer and device.type == "cuda"
    if use_fused:
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
            log("Using fused AdamW optimizer")
            return optimizer, True
        except (TypeError, RuntimeError) as exc:
            log(f"Fused AdamW unavailable; falling back to standard AdamW: {type(exc).__name__}: {exc}")
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4), False


def build_scheduler(optimizer, args, steps_per_epoch: int):
    if args.lr_scheduler == "none":
        return None, 0, 0

    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    min_factor = args.eta_min / args.lr if args.lr > 0 else 0.0

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_factor, float(step + 1) / float(warmup_steps))
        cosine_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(cosine_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, total_steps, warmup_steps


def build_loader(dataset, args, device, shuffle: bool, drop_last: bool = False):
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
        "drop_last": drop_last,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def stage_distance_accuracy(pred: torch.Tensor, label: torch.Tensor, tolerance: int = 1) -> float:
    return (torch.abs(pred - label) <= tolerance).float().mean().item()


def forward_model(model, batch: Dict):
    temporal_features = batch.get("temporal_features")
    if temporal_features is not None and int(getattr(model, "temporal_feature_dim", 0)) > 0:
        return model(batch["macro"], batch["micro"], batch["mask"], temporal_features)
    return model(batch["macro"], batch["micro"], batch["mask"])


def build_model(args, cfg, device):
    if args.stream == "both":
        log("Initializing two-stream 3D CNN model")
        return MultiScale3DCNN(
            num_classes=len(BASE_CLASSES),
            base_channels=args.cnn3d_base_channels,
            feature_dim=args.cnn3d_feature_dim,
            dropout=args.cnn3d_dropout,
            temporal_feature_dim=cfg.temporal_feature_dim,
            temporal_feature_hidden_dim=args.temporal_feature_hidden_dim,
            target_index=cfg.center,
        ).to(device)
    log(f"Initializing {args.stream} single-stream 3D CNN model")
    return SingleStream3DCNN(
        stream=args.stream,
        num_classes=len(BASE_CLASSES),
        base_channels=args.cnn3d_base_channels,
        feature_dim=args.cnn3d_feature_dim,
        dropout=args.cnn3d_dropout,
        temporal_feature_dim=cfg.temporal_feature_dim,
        temporal_feature_hidden_dim=args.temporal_feature_hidden_dim,
        target_index=cfg.center,
    ).to(device)


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
    use_amp: bool = False,
    scaler: GradScaler = None,
    scheduler=None,
):
    model.train(train)
    total_loss, total_correct, total_neighbor, total_date_score, total = 0.0, 0.0, 0.0, 0.0, 0
    accumulation_steps = max(1, accumulation_steps)
    epoch_start = time.time()

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, 1):
        batch_start = time.time()
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            with autocast(device_type=device.type, enabled=use_amp):
                logits = forward_model(model, batch)
                loss = criterion(logits, batch["target"])

        if train:
            scaled_loss = loss / accumulation_steps
            if scaler is not None and use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if step % accumulation_steps == 0 or step == len(loader):
                if scaler is not None and use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        pred = torch.argmax(logits, dim=1)
        label = batch["label"]
        date_score = batch.get("date_score")
        batch_size = label.numel()
        total += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (pred == label).float().sum().item()
        total_neighbor += stage_distance_accuracy(pred, label, tolerance=1) * batch_size
        if date_score is not None:
            total_date_score += date_score.gather(1, pred.unsqueeze(1)).sum().item()

        if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == len(loader)):
            elapsed = time.time() - epoch_start
            batch_elapsed = time.time() - batch_start
            current_loss = total_loss / max(total, 1)
            current_acc = total_correct / max(total, 1)
            current_date_acc = total_date_score / max(total, 1)
            msg = (
                f"fold={fold_id} epoch={epoch} {phase} "
                f"batch={step}/{len(loader)} samples={total} "
                f"loss={current_loss:.4f} acc={current_acc:.4f} date_acc={current_date_acc:.4f} "
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
        "date_window_accuracy": total_date_score / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the multi-scale temporal wheat phenology model.")
    parser.add_argument(
        "--excel-path",
        "--label-path",
        dest="excel_path",
        default="labeling.xlsx",
        help="Path to the phenology label table. Supports .xlsx/.xls and .csv exports with the same columns.",
    )
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_multiscale")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2, help="Physical DataLoader batch size. Keep this small for image-window training.")
    parser.add_argument("--accumulation-steps", type=int, default=16, help="Gradient accumulation steps. Effective batch size = batch_size * accumulation_steps.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=["cosine", "none"], default="cosine", help="Learning rate schedule.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Fraction of optimizer steps used for linear LR warmup.")
    parser.add_argument("--eta-min", type=float, default=1e-6, help="Minimum LR for cosine decay.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True, help="Use pinned CPU memory for faster CUDA transfers.")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Disable pinned CPU memory.")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=True, help="Keep DataLoader workers alive between epochs.")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false", help="Restart DataLoader workers each epoch.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Number of batches prefetched per DataLoader worker.")
    parser.add_argument("--drop-last", action="store_true", help="Drop the last incomplete training batch.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window-days", type=int, default=31, help="Temporal window length. In causal mode, 31 means previous 30 days + target day.")
    parser.add_argument("--window-mode", choices=["causal", "center"], default="causal", help="causal predicts the last day; center predicts the middle day.")
    parser.add_argument("--stream", choices=["micro", "macro", "both"], default="micro", help="Image stream to train on. micro means 10X only.")
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--date-tolerance-days", type=int, default=7, help="Days outside a predicted stage window that still receive partial metric credit.")
    parser.add_argument("--preplant-days", type=int, default=30, help="Number of days before seeding to keep as OffSeason.")
    parser.add_argument("--postharvest-days", type=int, default=30, help="Number of days after harvest to keep as OffSeason.")
    parser.add_argument("--camera", default="AUTO", help="Camera folder to use. AUTO uses the label table kamera/Camera column when present, otherwise prefers K1 and falls back to the available camera. Use K1, K2, or ALL to override.")
    parser.add_argument("--ignore-status-csv", action="store_true", help="Scan image folders directly instead of using day_image_status CSV files.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size fed into CNN3D.")
    parser.add_argument("--augment", action="store_true", help="Enable training-only image augmentation.")
    parser.add_argument("--augment-crop-scale-min", type=float, default=0.75, help="Minimum random resized crop scale when --augment is enabled.")
    parser.add_argument("--augment-hflip-prob", type=float, default=0.5, help="Horizontal flip probability when --augment is enabled.")
    parser.add_argument("--augment-rotation-degrees", type=float, default=5.0, help="Small random rotation range in degrees when --augment is enabled.")
    parser.add_argument("--augment-color-jitter", type=float, default=0.2, help="Brightness/contrast/saturation jitter strength when --augment is enabled.")
    parser.add_argument("--augment-blur-prob", type=float, default=0.1, help="Gaussian blur probability when --augment is enabled.")
    parser.add_argument("--augment-erasing-prob", type=float, default=0.1, help="Random erasing probability after normalization when --augment is enabled.")
    parser.add_argument("--cnn3d-base-channels", type=int, default=24, help="Width of the compact 3D CNN. Reduce to 16 if VRAM is tight.")
    parser.add_argument("--cnn3d-feature-dim", type=int, default=256, help="Feature size after 3D CNN pooling.")
    parser.add_argument("--cnn3d-dropout", type=float, default=0.25, help="Dropout inside the 3D CNN classifier head.")
    parser.add_argument("--use-days-since-planting", action="store_true", help="Add normalized days-since-planting metadata to the CNN3D classifier head.")
    parser.add_argument("--temporal-feature-hidden-dim", type=int, default=32, help="Hidden size for the small CNN3D temporal metadata MLP.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="Use CUDA automatic mixed precision.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision.")
    parser.add_argument("--compile-model", action="store_true", help="Use torch.compile for the model. Best tested on CUDA/Linux; optional on Windows.")
    parser.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode.")
    parser.add_argument("--compile-strict", action="store_true", help="Fail if torch.compile setup fails instead of falling back.")
    parser.add_argument("--fused-optimizer", dest="fused_optimizer", action="store_true", default=True, help="Try fused AdamW on CUDA.")
    parser.add_argument("--no-fused-optimizer", dest="fused_optimizer", action="store_false", help="Disable fused AdamW.")
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true", default=True, help="Write TensorBoard event logs.")
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false", help="Disable TensorBoard logging.")
    parser.add_argument("--tensorboard-dir", default=None, help="TensorBoard log directory. Defaults to <out-dir>/tensorboard.")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True, help="Disable PNG training/test visualizations.")
    parser.add_argument("--resume-checkpoint", default=None, help="Resume training from a last_checkpoint.pt file.")
    parser.add_argument("--save-last-checkpoint", dest="save_last_checkpoint", action="store_true", default=True, help="Save resumable last_checkpoint.pt after every epoch.")
    parser.add_argument("--no-save-last-checkpoint", dest="save_last_checkpoint", action="store_false", help="Disable per-epoch resumable checkpoint saves.")
    parser.add_argument("--log-interval", type=int, default=25, help="Print progress every N batches. Use 0 to disable batch progress logs.")
    parser.add_argument("--checkpoint-metric", default="date_window_accuracy", help="Validation metric used for best_model.pt.")
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
    log("Architecture: cnn3d")
    log(f"Days-since-planting metadata: {'enabled' if args.use_days_since_planting else 'disabled'}")
    log(f"Temporal metadata hidden dim: {args.temporal_feature_hidden_dim}")
    log(f"Image size: {args.image_size}")
    log(
        "Training augmentation: "
        + (
            "enabled "
            f"(crop_min={args.augment_crop_scale_min}, hflip={args.augment_hflip_prob}, "
            f"rotation={args.augment_rotation_degrees}, color_jitter={args.augment_color_jitter}, "
            f"blur={args.augment_blur_prob}, erasing={args.augment_erasing_prob})"
            if args.augment
            else "disabled"
        )
    )
    use_amp = args.amp and device.type == "cuda"
    log(f"Automatic mixed precision: {'enabled' if use_amp else 'disabled'}")
    resume_state = None
    resume_fold_id = None
    resume_completed_epoch = 0
    if args.resume_checkpoint:
        if not os.path.isfile(args.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        log(f"Loading resume checkpoint: {args.resume_checkpoint}")
        resume_state = torch.load(args.resume_checkpoint, map_location=device)
        resume_fold_id = int(resume_state["fold_id"])
        resume_completed_epoch = int(resume_state["epoch"])
        log(f"Resume target: fold={resume_fold_id} completed_epoch={resume_completed_epoch}")
    tb_writer = None
    tb_log_dir = args.tensorboard_dir or os.path.join(args.out_dir, "tensorboard")
    if args.tensorboard:
        if SummaryWriter is None:
            log("TensorBoard logging requested, but tensorboard is not installed. Run: python -m pip install tensorboard")
        else:
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            log(f"TensorBoard logs: {tb_log_dir}")
            log_tb_text(tb_writer, "config/args", vars(args))
            tb_writer.add_scalar("config/effective_batch_size", effective_batch, 0)
            tb_writer.add_scalar("config/use_amp", int(use_amp), 0)
            tb_writer.add_scalar("config/compile_requested", int(args.compile_model), 0)
            tb_writer.add_scalar("config/fused_optimizer_requested", int(args.fused_optimizer), 0)

    log("Building daily metadata dataframe")
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

    meta_path = os.path.join(args.out_dir, "multiscale_daily_metadata.csv")
    daily_df.to_csv(meta_path, index=False)
    log(f"Saved metadata: {meta_path}")
    log(f"Metadata rows: {len(daily_df)}")
    log(f"Station-years: {daily_df['station_year'].nunique()} | Groups: {daily_df['group_id'].nunique()}")
    log(f"Date range: {daily_df['date'].min()} -> {daily_df['date'].max()}")
    log("Label counts:\n" + daily_df["label"].value_counts().sort_index().to_string())
    log("Path availability:\n" + daily_df[["macro_path", "micro_path"]].notna().mean().to_string())
    if tb_writer is not None:
        tb_writer.add_scalar("metadata/rows", len(daily_df), 0)
        tb_writer.add_scalar("metadata/station_years", daily_df["station_year"].nunique(), 0)
        tb_writer.add_scalar("metadata/groups", daily_df["group_id"].nunique(), 0)
        for label_name, count in daily_df["label"].value_counts().sort_index().items():
            tb_writer.add_scalar(f"metadata/label_count/{label_name}", int(count), 0)
        for path_col, availability in daily_df[["macro_path", "micro_path"]].notna().mean().items():
            tb_writer.add_scalar(f"metadata/path_availability/{path_col}", float(availability), 0)

    log("Generating strict train/validation/test group folds")
    total_groups = int(daily_df["group_id"].nunique())
    n_val_groups = int(VALIDATION_FOLD_STATIONS)
    n_test_groups = int(TEST_FOLD_STATIONS)
    n_train_groups = total_groups - n_val_groups - n_test_groups
    if n_val_groups < 1:
        raise ValueError("VALIDATION_FOLD_STATIONS must be at least 1")
    if n_test_groups < 1:
        raise ValueError("TEST_FOLD_STATIONS must be at least 1")
    if n_train_groups < 1:
        raise ValueError(
            "Invalid CNN3D split constants: "
            f"total_groups={total_groups}, "
            f"VALIDATION_FOLD_STATIONS={VALIDATION_FOLD_STATIONS}, "
            f"TEST_FOLD_STATIONS={TEST_FOLD_STATIONS}. "
            "At least one training group must remain."
        )
    log(
        "CNN3D fold station counts from file constants: "
        f"train={n_train_groups} val={n_val_groups} test={n_test_groups}"
    )
    if tb_writer is not None:
        tb_writer.add_scalar("config/train_fold_stations", n_train_groups, 0)
        tb_writer.add_scalar("config/validation_fold_stations", n_val_groups, 0)
        tb_writer.add_scalar("config/test_fold_stations", n_test_groups, 0)
    folds = generate_group_train_val_test_folds(
        daily_df,
        group_col="group_id",
        n_train=n_train_groups,
        n_val=n_val_groups,
        n_test=n_test_groups,
        num_folds=args.folds,
        random_state=42,
    )
    if resume_fold_id is not None and not (1 <= resume_fold_id <= len(folds)):
        raise ValueError(f"Resume checkpoint fold_id={resume_fold_id} is outside generated fold range 1..{len(folds)}")

    temporal_feature_columns = DEFAULT_TEMPORAL_FEATURE_COLUMNS if args.use_days_since_planting else ()
    target_offset = args.window_days - 1 if args.window_mode == "causal" else None
    cfg = WindowConfig(
        window_days=args.window_days,
        center_offset=target_offset,
        classes=tuple(BASE_CLASSES),
        stream=args.stream,
        temporal_feature_columns=tuple(temporal_feature_columns),
    )
    log(
        f"Window mode: {args.window_mode} | window_days={args.window_days} "
        f"| target_index={cfg.center} | stream={args.stream}"
    )
    log(
        "Temporal metadata columns: "
        + (", ".join(temporal_feature_columns) if temporal_feature_columns else "none")
    )
    train_transform = build_image_transform(
        image_size=args.image_size,
        train=True,
        augment=args.augment,
        crop_scale_min=args.augment_crop_scale_min,
        hflip_prob=args.augment_hflip_prob,
        rotation_degrees=args.augment_rotation_degrees,
        color_jitter=args.augment_color_jitter,
        blur_prob=args.augment_blur_prob,
        erasing_prob=args.augment_erasing_prob,
    )
    eval_transform = build_image_transform(image_size=args.image_size, train=False, augment=False)
    all_history = []
    all_test_metrics = []
    all_history_loaded_from_disk = False
    if resume_state is not None:
        existing_history_path = os.path.join(args.out_dir, "all_history.csv")
        if os.path.isfile(existing_history_path):
            all_history = pd.read_csv(existing_history_path).to_dict("records")
            all_history_loaded_from_disk = True
            log(f"Loaded existing combined history for resume: {existing_history_path}")
        existing_test_metrics_path = os.path.join(args.out_dir, "all_test_metrics.csv")
        if os.path.isfile(existing_test_metrics_path):
            all_test_metrics = pd.read_csv(existing_test_metrics_path).to_dict("records")
            log(f"Loaded existing held-out test metrics for resume: {existing_test_metrics_path}")
    for fold_id, (train_idx, val_idx, test_idx) in enumerate(folds, 1):
        if resume_state is not None and fold_id < resume_fold_id:
            log(f"Skipping fold {fold_id}/{len(folds)} because resume starts at fold {resume_fold_id}")
            continue

        log(f"Preparing fold {fold_id}/{len(folds)}")
        fold_dir = os.path.join(args.out_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        is_resume_fold = resume_state is not None and fold_id == resume_fold_id

        train_df = daily_df.iloc[train_idx].reset_index(drop=True)
        val_df = daily_df.iloc[val_idx].reset_index(drop=True)
        test_df = daily_df.iloc[test_idx].reset_index(drop=True)
        log(
            f"Fold {fold_id}: train rows={len(train_df)} val rows={len(val_df)} test rows={len(test_df)} "
            f"train groups={sorted(train_df['group_id'].unique().tolist())} "
            f"val groups={sorted(val_df['group_id'].unique().tolist())} "
            f"test groups={sorted(test_df['group_id'].unique().tolist())}"
        )
        split_info = {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_groups": sorted(train_df["group_id"].unique().tolist()),
            "val_groups": sorted(val_df["group_id"].unique().tolist()),
            "test_groups": sorted(test_df["group_id"].unique().tolist()),
        }
        log_tb_text(tb_writer, f"fold_{fold_id}/split", split_info)
        if tb_writer is not None:
            tb_writer.add_scalar(f"fold_{fold_id}/split/train_rows", len(train_df), 0)
            tb_writer.add_scalar(f"fold_{fold_id}/split/val_rows", len(val_df), 0)
            tb_writer.add_scalar(f"fold_{fold_id}/split/test_rows", len(test_df), 0)
        log(f"Fold {fold_id}: creating raw-image 3D CNN window datasets")
        train_ds = MultiScaleWindowDataset(
            train_df,
            cfg,
            transform=train_transform,
            image_size=args.image_size,
            synchronized_transform=args.augment,
        )
        val_ds = MultiScaleWindowDataset(
            val_df,
            cfg,
            transform=eval_transform,
            image_size=args.image_size,
            synchronized_transform=False,
        )
        test_ds = MultiScaleWindowDataset(
            test_df,
            cfg,
            transform=eval_transform,
            image_size=args.image_size,
            synchronized_transform=False,
        )
        log(f"Fold {fold_id}: train samples={len(train_ds)} val samples={len(val_ds)} test samples={len(test_ds)}")
        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            raise RuntimeError(
                f"Fold {fold_id} produced an empty dataset split "
                f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})."
            )

        train_loader = build_loader(train_ds, args, device, shuffle=True, drop_last=args.drop_last)
        val_loader = build_loader(val_ds, args, device, shuffle=False)
        test_loader = build_loader(test_ds, args, device, shuffle=False)
        log(
            f"Fold {fold_id}: loaders ready "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)} "
            f"num_workers={args.num_workers} pin_memory={args.pin_memory and device.type == 'cuda'} "
            f"persistent_workers={args.persistent_workers and args.num_workers > 0} drop_last={args.drop_last}"
        )
        if len(train_loader) == 0:
            raise RuntimeError(
                f"Fold {fold_id} train loader has zero batches. "
                "Disable --drop-last or reduce --batch-size."
            )

        model = build_model(args, cfg, device)
        if is_resume_fold:
            log(f"Fold {fold_id}: restoring model state from resume checkpoint")
            load_model_state(model, resume_state["model"])
        model, compiled_model = maybe_compile_model(model, args, device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        log(f"Fold {fold_id}: model params total={total_params:,} trainable={trainable_params:,}")
        log(f"Fold {fold_id}: torch.compile active={compiled_model}")
        if tb_writer is not None:
            tb_writer.add_scalar(f"fold_{fold_id}/model/total_params", total_params, 0)
            tb_writer.add_scalar(f"fold_{fold_id}/model/trainable_params", trainable_params, 0)
            tb_writer.add_scalar(f"fold_{fold_id}/model/compiled", int(compiled_model), 0)
        criterion = SoftTargetCrossEntropy()
        optimizer, fused_optimizer = build_optimizer(model, args, device)
        if tb_writer is not None:
            tb_writer.add_scalar(f"fold_{fold_id}/optimizer/fused", int(fused_optimizer), 0)
        optimizer_steps_per_epoch = math.ceil(len(train_loader) / max(1, args.accumulation_steps))
        scheduler, total_scheduler_steps, warmup_steps = build_scheduler(optimizer, args, optimizer_steps_per_epoch)
        log(
            f"Fold {fold_id}: lr_scheduler={args.lr_scheduler} "
            f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} "
            f"total_steps={total_scheduler_steps} warmup_steps={warmup_steps}"
        )
        if tb_writer is not None:
            tb_writer.add_scalar(f"fold_{fold_id}/optimizer/steps_per_epoch", optimizer_steps_per_epoch, 0)
            tb_writer.add_scalar(f"fold_{fold_id}/optimizer/warmup_steps", warmup_steps, 0)
        scaler = GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

        best_score = -1.0
        best_epoch = None
        checkpoint_path = os.path.join(fold_dir, "best_model.pt")
        history = []
        start_epoch = 1
        if is_resume_fold:
            log(f"Fold {fold_id}: restoring optimizer/scaler state from resume checkpoint")
            optimizer.load_state_dict(resume_state["optimizer"])
            if scheduler is not None and resume_state.get("scheduler") is not None:
                scheduler.load_state_dict(resume_state["scheduler"])
            if scaler is not None and resume_state.get("scaler") is not None:
                scaler.load_state_dict(resume_state["scaler"])
            best_score = float(resume_state.get("best_score", -1.0))
            best_epoch = resume_state.get("best_epoch")
            history = list(resume_state.get("history", []))
            start_epoch = resume_completed_epoch + 1
            if not all_history_loaded_from_disk:
                all_history.extend(history)
            log(
                f"Fold {fold_id}: resume restored {len(history)} history rows; "
                f"next_epoch={start_epoch}; best_epoch={best_epoch}; best_score={best_score:.4f}"
            )
            if start_epoch > args.epochs:
                log(f"Fold {fold_id}: resume checkpoint already completed requested epochs; moving to final test")

        for epoch in range(start_epoch, args.epochs + 1):
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
                use_amp=use_amp,
                scaler=scaler,
                scheduler=scheduler,
            )
            log(f"Fold {fold_id} epoch {epoch}/{args.epochs}: validation started")
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                train=False,
                log_interval=args.log_interval,
                phase="val",
                fold_id=fold_id,
                epoch=epoch,
                use_amp=use_amp,
            )
            row = {
                "fold": fold_id,
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            all_history.append(row)
            log("Epoch metrics: " + json.dumps(row, indent=None))
            log_tb_metrics(tb_writer, f"fold_{fold_id}/train", train_metrics, epoch)
            log_tb_metrics(tb_writer, f"fold_{fold_id}/val", val_metrics, epoch)
            for metric_name, metric_value in train_metrics.items():
                if isinstance(metric_value, (int, float)):
                    tb_writer and tb_writer.add_scalar(f"cv/train/{metric_name}", metric_value, (fold_id - 1) * args.epochs + epoch)
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)):
                    tb_writer and tb_writer.add_scalar(f"cv/val/{metric_name}", metric_value, (fold_id - 1) * args.epochs + epoch)
            if tb_writer is not None:
                tb_writer.add_scalar(f"fold_{fold_id}/optimizer/lr", optimizer.param_groups[0]["lr"], epoch)
                if device.type == "cuda":
                    max_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    tb_writer.add_scalar(f"fold_{fold_id}/system/max_vram_gb", max_vram, epoch)

            checkpoint_score = val_metrics.get(args.checkpoint_metric)
            if checkpoint_score is None:
                raise KeyError(f"Unknown checkpoint metric: {args.checkpoint_metric}")
            if checkpoint_score > best_score:
                best_score = checkpoint_score
                best_epoch = epoch
                torch.save(
                    {
                        "model": model_state_dict(model),
                        "classes": BASE_CLASSES,
                        "window_days": args.window_days,
                        "window_mode": args.window_mode,
                        "target_index": cfg.center,
                        "stream": args.stream,
                        "architecture": "cnn3d",
                        "image_size": args.image_size,
                        "augment": args.augment,
                        "augment_crop_scale_min": args.augment_crop_scale_min,
                        "augment_hflip_prob": args.augment_hflip_prob,
                        "augment_rotation_degrees": args.augment_rotation_degrees,
                        "augment_color_jitter": args.augment_color_jitter,
                        "augment_blur_prob": args.augment_blur_prob,
                        "augment_erasing_prob": args.augment_erasing_prob,
                        "augment_synchronized_per_window": args.augment,
                        "cnn3d_base_channels": args.cnn3d_base_channels,
                        "cnn3d_feature_dim": args.cnn3d_feature_dim,
                        "cnn3d_dropout": args.cnn3d_dropout,
                        "use_days_since_planting": args.use_days_since_planting,
                        "temporal_feature_dim": cfg.temporal_feature_dim,
                        "temporal_feature_columns": list(cfg.temporal_feature_columns),
                        "temporal_feature_hidden_dim": args.temporal_feature_hidden_dim,
                        "date_tolerance_days": args.date_tolerance_days,
                        "checkpoint_metric": args.checkpoint_metric,
                        "validation_score": best_score,
                        "best_epoch": best_epoch,
                        "split_train_group_count": n_train_groups,
                        "split_val_group_count": n_val_groups,
                        "split_test_group_count": n_test_groups,
                        "train_groups": sorted(train_df["group_id"].unique().tolist()),
                        "val_groups": sorted(val_df["group_id"].unique().tolist()),
                        "test_groups": sorted(test_df["group_id"].unique().tolist()),
                        "uses_embedding_cache": False,
                        "compiled_model": compiled_model,
                        "fused_optimizer": fused_optimizer,
                        "lr_scheduler": args.lr_scheduler,
                        "warmup_steps": warmup_steps,
                        "total_scheduler_steps": total_scheduler_steps,
                    },
                    checkpoint_path,
                )
                log(f"Fold {fold_id}: saved new best checkpoint to {checkpoint_path} val_{args.checkpoint_metric}={best_score:.4f}")

            if args.save_last_checkpoint:
                last_state = {
                    "checkpoint_type": "last",
                    "fold_id": fold_id,
                    "epoch": epoch,
                    "model": model_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "history": history,
                    "classes": BASE_CLASSES,
                    "window_days": args.window_days,
                    "window_mode": args.window_mode,
                    "target_index": cfg.center,
                    "stream": args.stream,
                    "architecture": "cnn3d",
                    "image_size": args.image_size,
                    "augment": args.augment,
                    "augment_crop_scale_min": args.augment_crop_scale_min,
                    "augment_hflip_prob": args.augment_hflip_prob,
                    "augment_rotation_degrees": args.augment_rotation_degrees,
                    "augment_color_jitter": args.augment_color_jitter,
                    "augment_blur_prob": args.augment_blur_prob,
                    "augment_erasing_prob": args.augment_erasing_prob,
                    "augment_synchronized_per_window": args.augment,
                    "cnn3d_base_channels": args.cnn3d_base_channels,
                    "cnn3d_feature_dim": args.cnn3d_feature_dim,
                    "cnn3d_dropout": args.cnn3d_dropout,
                    "use_days_since_planting": args.use_days_since_planting,
                    "temporal_feature_dim": cfg.temporal_feature_dim,
                    "temporal_feature_columns": list(cfg.temporal_feature_columns),
                    "temporal_feature_hidden_dim": args.temporal_feature_hidden_dim,
                    "date_tolerance_days": args.date_tolerance_days,
                    "checkpoint_metric": args.checkpoint_metric,
                    "split_train_group_count": n_train_groups,
                    "split_val_group_count": n_val_groups,
                    "split_test_group_count": n_test_groups,
                    "train_groups": sorted(train_df["group_id"].unique().tolist()),
                    "val_groups": sorted(val_df["group_id"].unique().tolist()),
                    "test_groups": sorted(test_df["group_id"].unique().tolist()),
                    "uses_embedding_cache": False,
                    "compiled_model": compiled_model,
                    "fused_optimizer": fused_optimizer,
                    "lr_scheduler": args.lr_scheduler,
                    "warmup_steps": warmup_steps,
                    "total_scheduler_steps": total_scheduler_steps,
                    "args": vars(args),
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                fold_last_path = os.path.join(fold_dir, "last_checkpoint.pt")
                run_last_path = os.path.join(args.out_dir, "last_checkpoint.pt")
                save_resume_checkpoint(fold_last_path, last_state)
                save_resume_checkpoint(run_last_path, last_state)
                log(f"Fold {fold_id}: saved resumable checkpoint at epoch {epoch} to {fold_last_path}")

        log(f"Fold {fold_id}: loading best checkpoint from epoch {best_epoch} for final test evaluation")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_model_state(model, checkpoint["model"])
        test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer,
            device,
            train=False,
            log_interval=args.log_interval,
            phase="test",
            fold_id=fold_id,
            epoch=best_epoch or args.epochs,
            use_amp=use_amp,
        )
        test_row = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            f"best_val_{args.checkpoint_metric}": best_score,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        all_test_metrics.append(test_row)
        log_tb_metrics(tb_writer, f"fold_{fold_id}/test", test_metrics, best_epoch or args.epochs)
        if tb_writer is not None:
            tb_writer.add_scalar(f"fold_{fold_id}/best_epoch", best_epoch or 0, 0)
            tb_writer.add_scalar(f"fold_{fold_id}/best_val/{args.checkpoint_metric}", best_score, 0)
            for metric_name, metric_value in test_metrics.items():
                if isinstance(metric_value, (int, float)):
                    tb_writer.add_scalar(f"cv/test/{metric_name}", metric_value, fold_id)
        test_metrics_path = os.path.join(fold_dir, "test_metrics.json")
        with open(test_metrics_path, "w", encoding="utf-8") as f:
            json.dump(test_row, f, indent=2)
        log(f"Fold {fold_id}: final held-out test metrics saved to {test_metrics_path}")
        log("Final test metrics: " + json.dumps(test_row, indent=None))

        history_path = os.path.join(fold_dir, "history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False)
        log(f"Fold {fold_id}: saved history to {history_path}")

    all_history_path = os.path.join(args.out_dir, "all_history.csv")
    history_df = pd.DataFrame(all_history)
    history_df.to_csv(all_history_path, index=False)
    log(f"Saved combined history to {all_history_path}")
    all_test_metrics_path = os.path.join(args.out_dir, "all_test_metrics.csv")
    test_metrics_df = pd.DataFrame(all_test_metrics)
    test_metrics_df.to_csv(all_test_metrics_path, index=False)
    log(f"Saved combined held-out test metrics to {all_test_metrics_path}")
    if args.plots:
        maybe_plot_training_outputs(history_df, test_metrics_df, args.out_dir)
    else:
        log("PNG plot generation disabled by --no-plots")
    if tb_writer is not None:
        if all_test_metrics:
            test_df = test_metrics_df
            numeric_cols = test_df.select_dtypes(include="number").columns
            for col in numeric_cols:
                if col == "fold":
                    continue
                tb_writer.add_scalar(f"summary/{col}_mean", float(test_df[col].mean()), 0)
                tb_writer.add_scalar(f"summary/{col}_std", float(test_df[col].std(ddof=0)), 0)
            hparams = {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "accumulation_steps": args.accumulation_steps,
                "effective_batch_size": effective_batch,
                "lr": args.lr,
                "folds": args.folds,
                "split_train_group_count": n_train_groups,
                "split_val_group_count": n_val_groups,
                "split_test_group_count": n_test_groups,
                "window_days": args.window_days,
                "window_mode": args.window_mode,
                "stream": args.stream,
                "architecture": "cnn3d",
                "image_size": args.image_size,
                "augment": args.augment,
                "augment_crop_scale_min": args.augment_crop_scale_min,
                "augment_hflip_prob": args.augment_hflip_prob,
                "augment_rotation_degrees": args.augment_rotation_degrees,
                "augment_color_jitter": args.augment_color_jitter,
                "augment_blur_prob": args.augment_blur_prob,
                "augment_erasing_prob": args.augment_erasing_prob,
                "augment_synchronized_per_window": args.augment,
                "amp": use_amp,
                "embedding_cache": False,
                "compile_model": args.compile_model,
                "compile_mode": args.compile_mode,
                "fused_optimizer": args.fused_optimizer,
                "lr_scheduler": args.lr_scheduler,
                "warmup_ratio": args.warmup_ratio,
                "eta_min": args.eta_min,
                "pin_memory": args.pin_memory,
                "persistent_workers": args.persistent_workers,
                "drop_last": args.drop_last,
            }
            hparam_metrics = {
                col: float(test_df[col].mean())
                for col in numeric_cols
                if col.startswith("test_")
            }
            if hparam_metrics:
                tb_writer.add_hparams(hparams, hparam_metrics)
        tb_writer.flush()
        tb_writer.close()
        log(f"Closed TensorBoard writer. View with: tensorboard --logdir {tb_log_dir}")
    log(f"Done in {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
