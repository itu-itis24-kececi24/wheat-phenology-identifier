import argparse
import ast
import json
import math
import os
import random
import time
from typing import Dict, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from multiscale_phenology import (
    BASE_CLASSES,
    DINO_DEFAULT_BACKBONE,
    DEFAULT_TEMPORAL_FEATURE_COLUMNS,
    HybridOrdinalLoss,
    MultiScaleEmbeddingTemporalTransformer,
    MultiScaleEmbeddingWindowDataset,
    MultiScaleTemporalTransformer,
    MultiScaleWindowDataset,
    OrdinalRegressionLoss,
    SMOTEEmbeddingWindowDataset,
    SingleStreamEmbeddingTemporalTransformer,
    SingleStreamTemporalTransformer,
    SoftTargetCrossEntropy,
    WEATHER_MISSING_FEATURE_COLUMNS,
    WEATHER_TEMPORAL_FEATURE_COLUMNS,
    WindowConfig,
    add_weather_metadata,
    build_multiscale_daily_dataframe,
    generate_group_train_val_test_folds,
    print_station_image_edges,
)


# Edit these values to change the fold split for DINOv2 experiments.
# Training groups are derived automatically:
# train_groups = total_groups - VALIDATION_FOLD_STATIONS - TEST_FOLD_STATIONS
VALIDATION_FOLD_STATIONS = 2
TEST_FOLD_STATIONS = 2
REQUIRE_DIVERSE_VALIDATION_STATIONS = True
REQUIRE_DIVERSE_TEST_STATIONS = True


def log(message: str):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def split_group_target_eligibility(
    daily_df: pd.DataFrame,
    group_col: str,
    stream: str,
    active_classes,
    embedding_cache: Optional[Dict] = None,
) -> pd.DataFrame:
    """Count target days each split group can actually supply to a dataset."""

    def cache_contains(path, table: Dict) -> bool:
        if path is None or pd.isna(path):
            return False
        key = os.path.abspath(os.path.normpath(str(path)))
        return key in table

    active = daily_df["label"].isin(active_classes)
    if embedding_cache is None:
        macro_available = daily_df["macro_path"].notna()
        micro_available = daily_df["micro_path"].notna()
    else:
        macro_table = embedding_cache.get("macro", {})
        micro_table = embedding_cache.get("micro", {})
        macro_available = daily_df["macro_path"].map(lambda path: cache_contains(path, macro_table))
        micro_available = daily_df["micro_path"].map(lambda path: cache_contains(path, micro_table))

    if stream == "macro":
        usable = active & macro_available
    elif stream == "micro":
        usable = active & micro_available
    else:
        usable = active & (macro_available | micro_available)

    audit = daily_df[[group_col]].copy()
    audit["active_stage_row"] = active.astype(int)
    audit["macro_target_available"] = (active & macro_available).astype(int)
    audit["micro_target_available"] = (active & micro_available).astype(int)
    audit["eligible_target"] = usable.astype(int)
    return (
        audit.groupby(group_col, dropna=False)
        .agg(
            metadata_rows=(group_col, "size"),
            active_stage_rows=("active_stage_row", "sum"),
            macro_target_days=("macro_target_available", "sum"),
            micro_target_days=("micro_target_available", "sum"),
            eligible_target_days=("eligible_target", "sum"),
        )
        .reset_index()
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_runtime(device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def capture_rng_state() -> Dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Optional[Dict]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


BRIEF_HISTORY_COLUMNS = [
    "fold",
    "epoch",
    "train_loss",
    "train_accuracy",
    "train_plus_minus_1_accuracy",
    "train_date_window_accuracy",
    "val_loss",
    "val_accuracy",
    "val_plus_minus_1_accuracy",
    "val_date_window_accuracy",
]


def brief_history_frame(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for col in BRIEF_HISTORY_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    return frame[BRIEF_HISTORY_COLUMNS]


def compact_epoch_metrics(row: Dict) -> Dict:
    return {key: row.get(key) for key in BRIEF_HISTORY_COLUMNS if key in row}


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


def save_confusion_matrix(prediction_rows, out_dir: str, name: str, classes):
    if not prediction_rows:
        log(f"No prediction rows available; skipping confusion matrix for {name}")
        return

    os.makedirs(out_dir, exist_ok=True)
    matrix = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    for row in prediction_rows:
        true_label = row["true_label"]
        pred_label = row["pred_label"]
        if true_label in matrix.index and pred_label in matrix.columns:
            matrix.loc[true_label, pred_label] += 1

    csv_path = os.path.join(out_dir, f"{name}_confusion_matrix.csv")
    matrix.to_csv(csv_path)
    log(f"Saved confusion matrix CSV: {csv_path}")

    normalized = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    normalized_path = os.path.join(out_dir, f"{name}_confusion_matrix_normalized.csv")
    normalized.to_csv(normalized_path)
    log(f"Saved normalized confusion matrix CSV: {normalized_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib is not installed; skipping confusion matrix PNG")
        return

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(normalized.to_numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(f"{name.replace('_', ' ').title()} Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            count = int(matrix.loc[true_label, pred_label])
            value = float(normalized.loc[true_label, pred_label])
            if count:
                ax.text(j, i, f"{count}\n{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="row-normalized score")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{name}_confusion_matrix.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    log(f"Saved confusion matrix PNG: {png_path}")


def _list_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return value


def filter_offseason_rows(daily_df: pd.DataFrame, keep_as_context: bool = False) -> pd.DataFrame:
    if "OffSeason" not in BASE_CLASSES:
        return daily_df
    off_idx = BASE_CLASSES.index("OffSeason")
    if keep_as_context:
        filtered = daily_df.copy().reset_index(drop=True)
    else:
        filtered = daily_df.loc[daily_df["label"] != "OffSeason"].copy().reset_index(drop=True)
    for col in ("target", "date_score"):
        filtered[col] = filtered[col].apply(
            lambda value: [x for idx, x in enumerate(_list_value(value)) if idx != off_idx]
        )
    return filtered


def build_criterion(args):
    if args.loss == "soft_ce":
        return SoftTargetCrossEntropy()
    if args.loss == "hybrid":
        return HybridOrdinalLoss(
            power=args.ordinal_power,
            cross_entropy_weight=args.ordinal_ce_weight,
        )
    return OrdinalRegressionLoss(power=args.ordinal_power)


def checkpoint_mode(metric_name: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    return "min" if metric_name == "loss" or metric_name.endswith("_error") else "max"


def checkpoint_improved(score: float, best_score: float, mode: str) -> bool:
    return score < best_score if mode == "min" else score > best_score


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
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
            log("Using fused AdamW optimizer")
            return optimizer, True
        except (TypeError, RuntimeError) as exc:
            log(f"Fused AdamW unavailable; falling back to standard AdamW: {type(exc).__name__}: {exc}")
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay), False


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


def build_loader(dataset, args, device, shuffle: bool, drop_last: bool = False, seed: Optional[int] = None):
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
        kwargs["worker_init_fn"] = seed_worker
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


def empty_metrics() -> Dict[str, Optional[float]]:
    return {
        "loss": None,
        "accuracy": None,
        "plus_minus_1_accuracy": None,
        "date_window_accuracy": None,
        "macro_f1": None,
        "quadratic_weighted_kappa": None,
        "mean_absolute_stage_error": None,
    }


def move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def stage_distance_accuracy(pred: torch.Tensor, label: torch.Tensor, tolerance: int = 1) -> float:
    return (torch.abs(pred - label) <= tolerance).float().mean().item()


def _metric_class_name(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")


def classification_metrics_from_indices(true_indices, pred_indices, classes) -> Dict[str, float]:
    class_names = list(classes or BASE_CLASSES)
    num_classes = len(class_names)
    metrics: Dict[str, float] = {}
    if num_classes == 0 or len(true_indices) == 0:
        metrics.update(
            {
                "macro_f1": 0.0,
                "quadratic_weighted_kappa": 0.0,
                "mean_absolute_stage_error": 0.0,
            }
        )
        return metrics

    true = torch.tensor(true_indices, dtype=torch.long)
    pred = torch.tensor(pred_indices, dtype=torch.long)
    valid = (true >= 0) & (true < num_classes) & (pred >= 0) & (pred < num_classes)
    true = true[valid]
    pred = pred[valid]
    if true.numel() == 0:
        metrics.update(
            {
                "macro_f1": 0.0,
                "quadratic_weighted_kappa": 0.0,
                "mean_absolute_stage_error": 0.0,
            }
        )
        return metrics

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    for true_idx, pred_idx in zip(true.tolist(), pred.tolist()):
        confusion[int(true_idx), int(pred_idx)] += 1.0

    per_class_f1 = []
    for idx, class_name in enumerate(class_names):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        support = confusion[idx, :].sum()
        predicted = confusion[:, idx].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, dtype=torch.float64)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, dtype=torch.float64)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else torch.tensor(0.0, dtype=torch.float64)
        key = _metric_class_name(class_name)
        metrics[f"per_class_{key}_precision"] = float(precision)
        metrics[f"per_class_{key}_recall"] = float(recall)
        metrics[f"per_class_{key}_f1"] = float(f1)
        metrics[f"per_class_{key}_support"] = float(support)
        if support > 0 or predicted > 0:
            per_class_f1.append(float(f1))

    metrics["macro_f1"] = float(sum(per_class_f1) / max(len(per_class_f1), 1))
    metrics["mean_absolute_stage_error"] = float(torch.abs(pred - true).double().mean())

    if num_classes <= 1:
        metrics["quadratic_weighted_kappa"] = 1.0
        return metrics
    observed = confusion
    true_hist = observed.sum(dim=1)
    pred_hist = observed.sum(dim=0)
    expected = torch.outer(true_hist, pred_hist) / observed.sum().clamp_min(1.0)
    indices = torch.arange(num_classes, dtype=torch.float64)
    weights = (indices[:, None] - indices[None, :]).pow(2) / float((num_classes - 1) ** 2)
    observed_weighted = (weights * observed).sum()
    expected_weighted = (weights * expected).sum()
    if expected_weighted <= 1e-12:
        qwk = 1.0 if observed_weighted <= 1e-12 else 0.0
    else:
        qwk = 1.0 - float(observed_weighted / expected_weighted)
    metrics["quadratic_weighted_kappa"] = float(qwk)
    return metrics


def classification_metrics_from_rows(prediction_rows, classes) -> Dict[str, float]:
    true_indices = [int(row["true_idx"]) for row in prediction_rows]
    pred_indices = [int(row["pred_idx"]) for row in prediction_rows]
    return classification_metrics_from_indices(true_indices, pred_indices, classes)


def monotonic_viterbi_decode(probabilities, advance_penalty: float = 0.0, max_advances=None):
    """Decode one ordered sequence using only stay or next-stage transitions."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"Expected [time, classes] probabilities, got shape {probs.shape}")
    if probs.shape[0] == 0:
        return []
    if probs.shape[1] == 0:
        raise ValueError("Cannot decode probabilities with zero classes")

    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    scores = np.full_like(log_probs, -np.inf)
    backpointers = np.full(log_probs.shape, -1, dtype=np.int64)
    scores[0] = log_probs[0]
    if max_advances is None:
        max_advances = [1] * max(0, probs.shape[0] - 1)
    if len(max_advances) != max(0, probs.shape[0] - 1):
        raise ValueError("max_advances must contain one value for every transition between observations")

    for time_idx in range(1, probs.shape[0]):
        max_advance = max(1, int(max_advances[time_idx - 1]))
        for current_stage in range(probs.shape[1]):
            candidates = [
                (
                    scores[time_idx - 1, previous_stage]
                    - float(advance_penalty) * (current_stage - previous_stage),
                    previous_stage,
                )
                for previous_stage in range(max(0, current_stage - max_advance), current_stage + 1)
            ]
            best_score, best_previous = max(candidates, key=lambda item: item[0])
            scores[time_idx, current_stage] = best_score + log_probs[time_idx, current_stage]
            backpointers[time_idx, current_stage] = best_previous

    decoded = [int(np.argmax(scores[-1]))]
    for time_idx in range(probs.shape[0] - 1, 0, -1):
        decoded.append(int(backpointers[time_idx, decoded[-1]]))
    return list(reversed(decoded))


def apply_monotonic_decoder(prediction_rows, classes, advance_penalty: float = 0.0):
    """Add decoded fields without overwriting the raw neural predictions."""
    if not prediction_rows:
        return prediction_rows

    grouped = {}
    for row_index, row in enumerate(prediction_rows):
        key = (row.get("fold"), str(row.get("station_year", "")))
        grouped.setdefault(key, []).append((row_index, row))

    for sequence in grouped.values():
        sequence.sort(key=lambda item: pd.Timestamp(item[1].get("date")))
        sequence_dates = [pd.Timestamp(item[1].get("date")) for item in sequence]
        max_advances = [
            max(1, int((current - previous).days))
            for previous, current in zip(sequence_dates, sequence_dates[1:])
        ]
        probabilities = []
        for _, row in sequence:
            values = _list_value(row.get("class_probabilities"))
            if values is None or len(values) != len(classes):
                raise ValueError("Monotonic decoding requires one class-probability vector per prediction row")
            probabilities.append(values)
        decoded = monotonic_viterbi_decode(
            probabilities,
            advance_penalty=advance_penalty,
            max_advances=max_advances,
        )
        for (_, row), decoded_idx in zip(sequence, decoded):
            row["raw_pred_idx"] = int(row["pred_idx"])
            row["raw_pred_label"] = str(row["pred_label"])
            row["decoded_pred_idx"] = int(decoded_idx)
            row["decoded_pred_label"] = str(classes[decoded_idx])
            row["decoded_changed"] = int(decoded_idx != int(row["pred_idx"]))
            date_scores = _list_value(row.get("date_window_scores"))
            row["decoded_date_window_score"] = (
                float(date_scores[decoded_idx])
                if date_scores is not None and len(date_scores) > decoded_idx
                else None
            )
    return prediction_rows


def decoded_prediction_view(prediction_rows):
    decoded_rows = []
    for row in prediction_rows:
        if "decoded_pred_idx" not in row:
            continue
        decoded_row = dict(row)
        decoded_row["pred_idx"] = int(row["decoded_pred_idx"])
        decoded_row["pred_label"] = str(row["decoded_pred_label"])
        decoded_row["date_window_score"] = row.get("decoded_date_window_score")
        decoded_rows.append(decoded_row)
    return decoded_rows


def prediction_summary_from_rows(prediction_rows, classes) -> Dict[str, float]:
    if not prediction_rows:
        return {
            "accuracy": 0.0,
            "plus_minus_1_accuracy": 0.0,
            "date_window_accuracy": 0.0,
            **classification_metrics_from_indices([], [], classes),
        }
    true_indices = np.asarray([int(row["true_idx"]) for row in prediction_rows])
    pred_indices = np.asarray([int(row["pred_idx"]) for row in prediction_rows])
    distance = np.abs(true_indices - pred_indices)
    date_scores = [row.get("date_window_score") for row in prediction_rows]
    valid_date_scores = [float(value) for value in date_scores if value is not None and not pd.isna(value)]
    return {
        "accuracy": float(np.mean(distance == 0)),
        "plus_minus_1_accuracy": float(np.mean(distance <= 1)),
        "date_window_accuracy": float(np.mean(valid_date_scores)) if valid_date_scores else 0.0,
        **classification_metrics_from_indices(true_indices.tolist(), pred_indices.tolist(), classes),
    }


def save_classification_metrics(prediction_rows, out_dir: str, name: str, classes):
    if not prediction_rows:
        log(f"No prediction rows available; skipping classification metrics for {name}")
        return
    os.makedirs(out_dir, exist_ok=True)
    metrics = classification_metrics_from_rows(prediction_rows, classes)
    json_path = os.path.join(out_dir, f"{name}_classification_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved classification metrics JSON: {json_path}")

    rows = []
    for class_name in classes:
        key = _metric_class_name(class_name)
        rows.append(
            {
                "class": class_name,
                "precision": metrics.get(f"per_class_{key}_precision", 0.0),
                "recall": metrics.get(f"per_class_{key}_recall", 0.0),
                "f1": metrics.get(f"per_class_{key}_f1", 0.0),
                "support": metrics.get(f"per_class_{key}_support", 0.0),
            }
        )
    csv_path = os.path.join(out_dir, f"{name}_per_class_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log(f"Saved per-class metrics CSV: {csv_path}")


def save_coverage_stratified_metrics(
    prediction_rows,
    out_dir: str,
    name: str,
    classes,
    window_days: int,
    min_stage_support_days: int = 0,
    min_window_coverage_days: int = 0,
):
    if not prediction_rows:
        return
    frame = pd.DataFrame(prediction_rows)
    required = {"true_idx", "pred_idx", "window_coverage_days", "stage_support_days"}
    if not required.issubset(frame.columns):
        return

    def window_coverage_bin(value):
        value = int(value)
        if value < 10:
            return "0-9"
        if value < 20:
            return "10-19"
        if value < 26:
            return "20-25"
        return f"26-{int(window_days)}"

    frame["window_coverage_bin"] = frame["window_coverage_days"].map(window_coverage_bin)
    frame["stage_support_bin"] = pd.cut(
        frame["stage_support_days"],
        bins=[-1, 9, 19, 39, float("inf")],
        labels=["0-9", "10-19", "20-39", "40+"],
    ).astype("string").fillna("outside")

    strata = []
    for column in ("window_coverage_bin", "stage_support_bin"):
        for value, subset in frame.groupby(column, dropna=False):
            strata.append((column, str(value), subset))
    if min_window_coverage_days > 0:
        below = frame["window_coverage_days"] < min_window_coverage_days
        strata.extend(
            [
                ("window_threshold", "below", frame[below]),
                ("window_threshold", "meets", frame[~below]),
            ]
        )
    if min_stage_support_days > 0:
        below = frame["stage_support_days"] < min_stage_support_days
        strata.extend(
            [
                ("stage_support_threshold", "below", frame[below]),
                ("stage_support_threshold", "meets", frame[~below]),
            ]
        )

    rows = []
    for stratification, stratum, subset in strata:
        if subset.empty:
            continue
        metrics = classification_metrics_from_indices(
            subset["true_idx"].astype(int).tolist(),
            subset["pred_idx"].astype(int).tolist(),
            classes,
        )
        distance = (subset["true_idx"].astype(int) - subset["pred_idx"].astype(int)).abs()
        rows.append(
            {
                "stratification": stratification,
                "stratum": stratum,
                "samples": int(len(subset)),
                "accuracy": float((distance == 0).mean()),
                "plus_minus_1_accuracy": float((distance <= 1).mean()),
                "date_window_accuracy": float(subset["date_window_score"].mean())
                if "date_window_score" in subset
                else None,
                "macro_f1": metrics["macro_f1"],
                "mean_absolute_stage_error": metrics["mean_absolute_stage_error"],
                "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
            }
        )
    path = os.path.join(out_dir, f"{name}_coverage_metrics.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    log(f"Saved coverage-stratified metrics CSV: {path}")


def forward_model(model, batch: Dict):
    temporal_features = batch.get("temporal_features")
    module = getattr(model, "_orig_mod", model)
    modality_kwargs = {}
    if getattr(module, "supports_modality_masks", False):
        modality_kwargs = {
            "macro_valid": batch.get("macro_valid"),
            "micro_valid": batch.get("micro_valid"),
        }
    if "macro_tile_mask" in batch:
        return model(
            batch["macro"],
            batch["micro"],
            batch["mask"],
            batch["macro_tile_mask"],
            batch["micro_tile_mask"],
            temporal_features,
            **modality_kwargs,
        )
    return model(
        batch["macro"],
        batch["micro"],
        batch["mask"],
        temporal_features,
        **modality_kwargs,
    )


def build_model(args, cfg, embedding_cache, device, classes):
    temporal_feature_dim = cfg.temporal_feature_dim
    if embedding_cache is None:
        if args.stream == "both":
            log("Initializing full two-stream gated-fusion image model")
            return MultiScaleTemporalTransformer(
                num_classes=len(classes),
                image_backbone=args.image_backbone,
                pretrained=args.pretrained,
                target_index=cfg.center,
                temporal_aggregation=args.temporal_aggregation,
                temporal_model=args.temporal_model,
                temporal_layers=args.temporal_layers,
                temporal_heads=args.temporal_heads,
                dropout=args.dropout,
                temporal_feature_dim=temporal_feature_dim,
                temporal_feature_hidden_dim=args.temporal_feature_hidden_dim,
                gate_hidden_dim=args.gate_hidden_dim,
                modality_dropout=args.modality_dropout,
                temporal_norm_first=args.temporal_norm_first,
                temporal_ffn_multiplier=args.temporal_ffn_multiplier,
            ).to(device)
        log(f"Initializing {args.stream} single-stream image model")
        return SingleStreamTemporalTransformer(
            stream=args.stream,
            num_classes=len(classes),
            image_backbone=args.image_backbone,
            pretrained=args.pretrained,
            target_index=cfg.center,
            temporal_aggregation=args.temporal_aggregation,
            temporal_model=args.temporal_model,
            temporal_layers=args.temporal_layers,
            temporal_heads=args.temporal_heads,
            dropout=args.dropout,
            temporal_feature_dim=temporal_feature_dim,
            temporal_feature_hidden_dim=args.temporal_feature_hidden_dim,
            temporal_norm_first=args.temporal_norm_first,
            temporal_ffn_multiplier=args.temporal_ffn_multiplier,
        ).to(device)

    if args.stream == "both":
        log("Initializing cached two-stream gated-fusion embedding model")
        return MultiScaleEmbeddingTemporalTransformer(
            feature_dim=int(embedding_cache["feature_dim"]),
            num_classes=len(classes),
            target_index=cfg.center,
            temporal_aggregation=args.temporal_aggregation,
            temporal_model=args.temporal_model,
            temporal_layers=args.temporal_layers,
            temporal_heads=args.temporal_heads,
            dropout=args.dropout,
            temporal_feature_dim=temporal_feature_dim,
            temporal_feature_hidden_dim=args.temporal_feature_hidden_dim,
            gate_hidden_dim=args.gate_hidden_dim,
            modality_dropout=args.modality_dropout,
            temporal_norm_first=args.temporal_norm_first,
            temporal_ffn_multiplier=args.temporal_ffn_multiplier,
        ).to(device)
    log(f"Initializing cached {args.stream} single-stream embedding model")
    return SingleStreamEmbeddingTemporalTransformer(
        feature_dim=int(embedding_cache["feature_dim"]),
        stream=args.stream,
        num_classes=len(classes),
        target_index=cfg.center,
        temporal_aggregation=args.temporal_aggregation,
        temporal_model=args.temporal_model,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        dropout=args.dropout,
        temporal_feature_dim=temporal_feature_dim,
        temporal_feature_hidden_dim=args.temporal_feature_hidden_dim,
        temporal_norm_first=args.temporal_norm_first,
        temporal_ffn_multiplier=args.temporal_ffn_multiplier,
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
    prediction_rows=None,
    classes=None,
):
    model.train(train)
    total_loss, total_correct, total_neighbor, total_date_score, total = 0.0, 0.0, 0.0, 0.0, 0
    all_true_indices = []
    all_pred_indices = []
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
            group_start = ((step - 1) // accumulation_steps) * accumulation_steps + 1
            group_size = min(accumulation_steps, len(loader) - group_start + 1)
            scaled_loss = loss / group_size
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
        all_true_indices.extend(label.detach().cpu().tolist())
        all_pred_indices.extend(pred.detach().cpu().tolist())
        if date_score is not None:
            total_date_score += date_score.gather(1, pred.unsqueeze(1)).sum().item()
        if prediction_rows is not None:
            class_names = classes or BASE_CLASSES
            pred_cpu = pred.detach().cpu().tolist()
            label_cpu = label.detach().cpu().tolist()
            probability_cpu = torch.softmax(logits.float(), dim=1).detach().cpu().tolist()
            date_score_cpu = date_score.detach().cpu().tolist() if date_score is not None else [None] * batch_size
            dates = batch.get("date", [""] * batch_size)
            station_years = batch.get("station_year", [""] * batch_size)
            window_coverage = batch.get("window_coverage_days")
            stage_support = batch.get("stage_support_days")
            window_coverage = (
                window_coverage.detach().cpu().tolist()
                if torch.is_tensor(window_coverage)
                else [None] * batch_size
            )
            stage_support = (
                stage_support.detach().cpu().tolist()
                if torch.is_tensor(stage_support)
                else [None] * batch_size
            )
            predicted_date_scores = (
                date_score.gather(1, pred.unsqueeze(1)).squeeze(1).detach().cpu().tolist()
                if date_score is not None
                else [None] * batch_size
            )
            for sample_idx, (true_idx, pred_idx) in enumerate(zip(label_cpu, pred_cpu)):
                prediction_rows.append(
                    {
                        "station_year": station_years[sample_idx],
                        "date": dates[sample_idx],
                        "true_idx": int(true_idx),
                        "pred_idx": int(pred_idx),
                        "true_label": class_names[int(true_idx)],
                        "pred_label": class_names[int(pred_idx)],
                        "window_coverage_days": window_coverage[sample_idx],
                        "stage_support_days": stage_support[sample_idx],
                        "date_window_score": predicted_date_scores[sample_idx],
                        "class_probabilities": json.dumps(probability_cpu[sample_idx]),
                        "date_window_scores": json.dumps(date_score_cpu[sample_idx])
                        if date_score_cpu[sample_idx] is not None
                        else None,
                    }
                )

        if log_interval > 0 and step % log_interval == 0:
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

    metrics = {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "plus_minus_1_accuracy": total_neighbor / max(total, 1),
        "date_window_accuracy": total_date_score / max(total, 1),
    }
    metrics.update(classification_metrics_from_indices(all_true_indices, all_pred_indices, classes or BASE_CLASSES))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the multi-scale temporal wheat phenology model.")
    parser.add_argument(
        "--excel-path",
        "--label-path",
        dest="excel_path",
        default="labeling_bbch_iso_dates.csv",
        help="Path to the BBCH label table. Supports the revised .xlsx or .csv files with the same columns.",
    )
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_dinov2_bbch_gated")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2, help="Physical DataLoader batch size. Keep this small for ViT windows.")
    parser.add_argument("--accumulation-steps", type=int, default=16, help="Gradient accumulation steps. Effective batch size = batch_size * accumulation_steps.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in the temporal model/fusion layers.")
    parser.add_argument("--gate-hidden-dim", type=int, default=128, help="Hidden size of the per-day macro/micro gating MLP.")
    parser.add_argument("--modality-dropout", type=float, default=0.2, help="During gated two-stream training, hide one available camera on this fraction of day tokens.")
    parser.add_argument("--loss", choices=["ordinal", "soft_ce", "hybrid"], default="ordinal", help="Training loss. hybrid combines exact soft-label cross entropy with ordinal CDF distance.")
    parser.add_argument("--ordinal-power", type=int, choices=[1, 2], default=2, help="Ordinal CDF loss power: 1=L1 earth-mover style, 2=squared CDF distance.")
    parser.add_argument("--ordinal-ce-weight", type=float, default=0.5, help="Cross-entropy weight used by --loss hybrid; the remainder weights ordinal CDF loss.")
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
    parser.add_argument("--validation-groups", type=int, default=None, help="Override VALIDATION_FOLD_STATIONS for this run.")
    parser.add_argument("--test-groups", type=int, default=None, help="Override TEST_FOLD_STATIONS for this run; use 0 for validation-only sweeps.")
    parser.add_argument("--fold-seed", type=int, default=42, help="Random seed used when sampling train/validation/test fold combinations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model initialization, shuffling, augmentation selection, and workers.")
    parser.add_argument("--fold-group-by", choices=["station", "station_year"], default="station", help="station keeps every year of a physical station in one split; station_year reproduces the older ID-based split.")
    parser.add_argument("--window-days", type=int, default=31, help="Temporal window length. In causal mode, 31 means previous 30 days + target day.")
    parser.add_argument("--window-mode", choices=["causal", "center"], default="causal", help="causal predicts the last day; center predicts the middle day.")
    parser.add_argument("--min-train-stage-days", type=int, default=0, help="Training only: exclude a station-stage when it has fewer usable target-image days than this value. 0 disables the filter.")
    parser.add_argument("--min-train-window-coverage-days", type=int, default=0, help="Training only: exclude a sample when its temporal window contains fewer exact usable image days than this value. 0 disables the filter.")
    parser.add_argument("--temporal-aggregation", choices=["target", "mean", "cls"], default="cls", help="How the temporal Transformer aggregates the window before classification.")
    parser.add_argument("--temporal-model", choices=["transformer", "lstm", "gru"], default="transformer", help="Temporal sequence backend. lstm/gru are PhenoNet-style recurrent alternatives.")
    parser.add_argument("--temporal-layers", type=int, default=4, help="Number of temporal Transformer/LSTM/GRU layers.")
    parser.add_argument("--temporal-heads", type=int, default=8, help="Number of attention heads for the Transformer backend. Ignored by LSTM/GRU.")
    parser.add_argument("--temporal-norm-first", dest="temporal_norm_first", action="store_true", default=True, help="Use pre-norm temporal Transformer blocks for more stable optimization.")
    parser.add_argument("--temporal-post-norm", dest="temporal_norm_first", action="store_false", help="Use the legacy post-norm temporal Transformer behavior.")
    parser.add_argument("--temporal-ffn-multiplier", type=float, default=4.0, help="Temporal Transformer feed-forward width as a multiple of the 512-dimensional model width.")
    parser.add_argument("--use-days-since-planting", dest="use_days_since_planting", action="store_true", default=True, help="Append normalized days since 1-Sowing as a per-day temporal feature.")
    parser.add_argument("--no-days-since-planting", dest="use_days_since_planting", action="store_false", help="Disable days-since-planting temporal metadata feature.")
    parser.add_argument("--temporal-feature-hidden-dim", type=int, default=32, help="Hidden size for the temporal metadata fusion MLP. Use 0 for the old single Linear fusion.")
    parser.add_argument("--use-weather-metadata", action="store_true", help="Fetch/cache Meteostat weather and append normalized temperature, precipitation, and GDD features.")
    parser.add_argument("--weather-cache", default=None, help="CSV cache for Meteostat daily weather. Defaults to <out-dir>/meteostat_weather_cache.csv when weather metadata is enabled.")
    parser.add_argument("--weather-force-refresh", action="store_true", help="Ignore an existing weather cache and fetch Meteostat data again.")
    parser.add_argument("--gdd-base-temp", type=float, default=0.0, help="Base temperature in Celsius for growing degree day metadata.")
    parser.add_argument("--stream", choices=["micro", "macro", "both"], default="both", help="Image stream to train on. both enables gated 1X + 10X fusion.")
    parser.add_argument("--exclude-offseason", action="store_true", help="Exclude OffSeason as a target class while retaining its images as causal context for BBCH windows.")
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--date-tolerance-days", type=int, default=7, help="Days outside a predicted stage window that still receive partial metric credit.")
    parser.add_argument("--monotonic-decoding", choices=["none", "viterbi"], default="none", help="Post-process each station-year sequence with non-regressing Viterbi decoding. Requires --exclude-offseason.")
    parser.add_argument("--monotonic-advance-penalty", type=float, default=0.0, help="Non-negative log-score penalty for advancing one stage during monotonic Viterbi decoding.")
    parser.add_argument("--preplant-days", type=int, default=30, help="Number of days before seeding to keep as OffSeason.")
    parser.add_argument("--postharvest-days", type=int, default=30, help="Number of days after harvest to keep as OffSeason.")
    parser.add_argument("--camera", default="AUTO", help="Camera folder to use. AUTO uses the label table kamera/Camera column when present, otherwise prefers K1 and falls back to the available camera. Use K1, K2, or ALL to override.")
    parser.add_argument("--ignore-status-csv", action="store_true", help="Scan image folders directly instead of using day_image_status CSV files.")
    parser.add_argument("--embedding-cache", default=None, help="Optional .pt cache from precompute_multiscale_embeddings.py.")
    parser.add_argument("--use-augmented-embeddings", dest="use_augmented_embeddings", action="store_true", default=True, help="Use augmented embedding variants from the cache for training folds.")
    parser.add_argument("--no-augmented-embeddings", dest="use_augmented_embeddings", action="store_false", help="Ignore augmented embedding variants even if present in the cache.")
    parser.add_argument("--embedding-augmentation-multiplier", type=int, default=None, help="Training sample multiplier when augmented embeddings exist. Defaults to clean + all available augmented views.")
    parser.add_argument("--embedding-smote", action="store_true", help="Use leakage-safe SMOTE on cached embedding windows in training folds only. Requires --embedding-cache.")
    parser.add_argument("--smote-target-ratio", type=float, default=1.0, help="SMOTE target count for each class as a fraction of the largest training class. 1.0 balances minority classes to the largest class.")
    parser.add_argument("--smote-k-neighbors", type=int, default=5, help="Number of same-stage nearest cached windows considered by SMOTE.")
    parser.add_argument("--smote-max-synthetic-samples", type=int, default=0, help="Maximum synthetic SMOTE windows per fold; 0 means no cap.")
    parser.add_argument("--smote-seed", type=int, default=42, help="Random seed for reproducible within-fold SMOTE interpolation.")
    parser.add_argument("--image-backbone", default=DINO_DEFAULT_BACKBONE, help="Backbone used only for full-image training without --embedding-cache.")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
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
    parser.add_argument("--resume-checkpoint", default=None, help="Resume training from a last_checkpoint.pt file.")
    parser.add_argument("--save-last-checkpoint", dest="save_last_checkpoint", action="store_true", default=True, help="Save resumable last_checkpoint.pt after every epoch.")
    parser.add_argument("--no-save-last-checkpoint", dest="save_last_checkpoint", action="store_false", help="Disable per-epoch resumable checkpoint saves.")
    parser.add_argument("--log-interval", type=int, default=25, help="Print progress every N batches. Use 0 to disable batch progress logs.")
    parser.add_argument("--checkpoint-metric", default="date_window_accuracy", help="Validation metric used for best_model.pt.")
    parser.add_argument("--checkpoint-mode", choices=["auto", "max", "min"], default="auto", help="Whether a higher or lower checkpoint metric is better. auto minimizes loss/errors and maximizes other metrics.")
    args = parser.parse_args()

    if not 0.0 <= args.ordinal_ce_weight <= 1.0:
        parser.error("--ordinal-ce-weight must be in [0, 1]")
    if args.temporal_ffn_multiplier <= 0:
        parser.error("--temporal-ffn-multiplier must be positive")
    if args.gate_hidden_dim <= 0:
        parser.error("--gate-hidden-dim must be positive")
    if not 0.0 <= args.modality_dropout < 1.0:
        parser.error("--modality-dropout must be in [0, 1)")
    if args.monotonic_advance_penalty < 0:
        parser.error("--monotonic-advance-penalty must be non-negative")
    if args.monotonic_decoding != "none" and not args.exclude_offseason:
        parser.error("--monotonic-decoding requires --exclude-offseason because OffSeason occurs both before sowing and after harvest")

    start_time = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    configure_runtime(device)
    seed_everything(args.seed)
    effective_batch = args.batch_size * max(1, args.accumulation_steps)
    log("Starting gated multi-scale BBCH phenology training")
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
    log(f"Temporal model: {args.temporal_model} layers={args.temporal_layers} heads={args.temporal_heads}")
    log(f"Dropout: {args.dropout}")
    log(
        f"Fusion: {'gated macro/micro' if args.stream == 'both' else 'single stream'} "
        f"gate_hidden_dim={args.gate_hidden_dim} modality_dropout={args.modality_dropout}"
    )
    log(f"Temporal metadata hidden dim: {args.temporal_feature_hidden_dim}")
    log(f"AdamW weight decay: {args.weight_decay}")
    log(f"Loss: {args.loss} ordinal_power={args.ordinal_power}")
    log(f"Days-since-planting feature: {'enabled' if args.use_days_since_planting else 'disabled'}")
    log(f"Weather metadata: {'enabled' if args.use_weather_metadata else 'disabled'}")
    log(f"Exclude OffSeason: {args.exclude_offseason}")
    log(
        "Monotonic sequence decoding: "
        + (
            f"viterbi (stay/advance, advance_penalty={args.monotonic_advance_penalty:g})"
            if args.monotonic_decoding == "viterbi"
            else "disabled"
        )
    )
    log(
        "Training-only coverage filters: "
        f"min_stage_days={args.min_train_stage_days} "
        f"min_window_days={args.min_train_window_coverage_days}; "
        "validation/test remain unfiltered"
    )
    log(
        "Embedding SMOTE: "
        + (
            f"enabled target_ratio={args.smote_target_ratio} k_neighbors={args.smote_k_neighbors} "
            f"max_synthetic={args.smote_max_synthetic_samples} seed={args.smote_seed}"
            if args.embedding_smote
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
        resume_state = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        resume_fold_id = int(resume_state["fold_id"])
        resume_completed_epoch = int(resume_state["epoch"])
        if "fold_seed" in resume_state and int(resume_state["fold_seed"]) != int(args.fold_seed):
            raise ValueError(
                f"Resume checkpoint was created with fold_seed={resume_state['fold_seed']}, "
                f"but current --fold-seed is {args.fold_seed}. Use the same seed to resume safely."
            )
        resume_grouping = resume_state.get("fold_group_by", "station_year")
        if resume_grouping != args.fold_group_by:
            raise ValueError(
                f"Resume checkpoint uses fold_group_by={resume_grouping}, but current value is "
                f"{args.fold_group_by}. Use --fold-group-by {resume_grouping}."
            )
        resume_norm_first = bool(resume_state.get("temporal_norm_first", False))
        if resume_norm_first != args.temporal_norm_first:
            required_flag = "--temporal-norm-first" if resume_norm_first else "--temporal-post-norm"
            raise ValueError(f"Resume checkpoint temporal normalization differs. Resume with {required_flag}.")
        for checkpoint_key, current_value, cli_flag in (
            ("min_train_stage_days", args.min_train_stage_days, "--min-train-stage-days"),
            (
                "min_train_window_coverage_days",
                args.min_train_window_coverage_days,
                "--min-train-window-coverage-days",
            ),
        ):
            checkpoint_value = int(resume_state.get(checkpoint_key, 0))
            if checkpoint_value != int(current_value):
                raise ValueError(
                    f"Resume checkpoint uses {checkpoint_key}={checkpoint_value}, but current value is "
                    f"{current_value}. Resume with {cli_flag} {checkpoint_value}."
                )
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
            tb_writer.add_scalar("config/temporal_layers", args.temporal_layers, 0)
            tb_writer.add_scalar("config/temporal_heads", args.temporal_heads, 0)
            tb_writer.add_scalar("config/dropout", args.dropout, 0)
            tb_writer.add_scalar("config/gate_hidden_dim", args.gate_hidden_dim, 0)
            tb_writer.add_scalar("config/modality_dropout", args.modality_dropout, 0)
            tb_writer.add_scalar("config/temporal_feature_hidden_dim", args.temporal_feature_hidden_dim, 0)
            tb_writer.add_scalar("config/weight_decay", args.weight_decay, 0)
            tb_writer.add_scalar("config/ordinal_power", args.ordinal_power, 0)
            tb_writer.add_scalar("config/use_days_since_planting", int(args.use_days_since_planting), 0)
            tb_writer.add_scalar("config/use_weather_metadata", int(args.use_weather_metadata), 0)
            tb_writer.add_scalar("config/monotonic_decoding", int(args.monotonic_decoding == "viterbi"), 0)
            tb_writer.add_scalar("config/monotonic_advance_penalty", args.monotonic_advance_penalty, 0)
            tb_writer.add_scalar("config/exclude_offseason", int(args.exclude_offseason), 0)
            tb_writer.add_scalar("config/min_train_stage_days", args.min_train_stage_days, 0)
            tb_writer.add_scalar("config/min_train_window_coverage_days", args.min_train_window_coverage_days, 0)
            tb_writer.add_scalar("config/embedding_smote", int(args.embedding_smote), 0)
            if args.embedding_smote:
                tb_writer.add_scalar("config/smote_target_ratio", args.smote_target_ratio, 0)
                tb_writer.add_scalar("config/smote_k_neighbors", args.smote_k_neighbors, 0)
                tb_writer.add_scalar("config/smote_max_synthetic_samples", args.smote_max_synthetic_samples, 0)
            log_tb_text(tb_writer, "config/loss", {"loss": args.loss, "ordinal_power": args.ordinal_power})

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

    active_classes = list(BASE_CLASSES)
    if args.exclude_offseason:
        # Keep OffSeason rows as historical context for causal BBCH0 windows,
        # but WindowConfig.classes prevents them from becoming target samples.
        daily_df = filter_offseason_rows(daily_df, keep_as_context=True)
        active_classes = [name for name in BASE_CLASSES if name != "OffSeason"]
        if not daily_df["label"].isin(active_classes).any():
            raise RuntimeError("No BBCH stage rows remain after --exclude-offseason filtering.")
        log(
            "Excluded OffSeason as a prediction target while retaining its rows as causal context; "
            "training classes are: " + ", ".join(active_classes)
        )

    temporal_feature_columns = []
    if args.use_days_since_planting:
        temporal_feature_columns.extend(DEFAULT_TEMPORAL_FEATURE_COLUMNS)
    if args.use_weather_metadata:
        weather_cache = args.weather_cache or os.path.join(args.out_dir, "meteostat_weather_cache.csv")
        log(f"Adding Meteostat weather metadata using cache: {weather_cache}")
        daily_df = add_weather_metadata(
            daily_df,
            weather_cache,
            force_refresh=args.weather_force_refresh,
            gdd_base_temp=args.gdd_base_temp,
        )
        if "weather_gdd_cum_raw" in daily_df.columns:
            gdd_stats = daily_df["weather_gdd_cum_raw"].describe()[["min", "mean", "max"]]
            log(
                "Cumulative GDD since planting: "
                f"min={gdd_stats['min']:.1f} mean={gdd_stats['mean']:.1f} max={gdd_stats['max']:.1f} "
                f"(base_temp={args.gdd_base_temp:g}C, normalized by 2500)"
            )
        missing_rates = daily_df[list(WEATHER_MISSING_FEATURE_COLUMNS)].mean()
        log(
            "Weather missingness rates before local interpolation: "
            + ", ".join(
                f"{column.replace('weather_', '').replace('_missing', '')}={float(rate):.1%}"
                for column, rate in missing_rates.items()
            )
        )
        temporal_feature_columns.extend(WEATHER_TEMPORAL_FEATURE_COLUMNS)
    temporal_feature_columns = tuple(temporal_feature_columns)
    log(
        "Temporal metadata columns: "
        + (", ".join(temporal_feature_columns) if temporal_feature_columns else "none")
    )
    if tb_writer is not None:
        tb_writer.add_scalar("config/temporal_feature_dim", len(temporal_feature_columns), 0)
        log_tb_text(tb_writer, "config/temporal_feature_columns", {"columns": list(temporal_feature_columns)})

    meta_path = os.path.join(args.out_dir, "multiscale_daily_metadata.csv")
    daily_df.to_csv(meta_path, index=False)
    log(f"Saved metadata: {meta_path}")
    log(f"Metadata rows: {len(daily_df)}")
    log(f"Station-years: {daily_df['station_year'].nunique()} | Groups: {daily_df['group_id'].nunique()}")
    log(f"Date range: {daily_df['date'].min()} -> {daily_df['date'].max()}")
    log("Label counts:\n" + daily_df["label"].value_counts().sort_index().to_string())
    log("Path availability:\n" + daily_df[["macro_path", "micro_path"]].notna().mean().to_string())
    print_station_image_edges(
        daily_df,
        stream=args.stream,
        base_dir=args.data_path,
        printer=log,
        title="First/last resolved images used for DINOv2 training:",
    )
    if tb_writer is not None:
        tb_writer.add_scalar("metadata/rows", len(daily_df), 0)
        tb_writer.add_scalar("metadata/station_years", daily_df["station_year"].nunique(), 0)
        tb_writer.add_scalar("metadata/groups", daily_df["group_id"].nunique(), 0)
        for label_name, count in daily_df["label"].value_counts().sort_index().items():
            tb_writer.add_scalar(f"metadata/label_count/{label_name}", int(count), 0)
        for path_col, availability in daily_df[["macro_path", "micro_path"]].notna().mean().items():
            tb_writer.add_scalar(f"metadata/path_availability/{path_col}", float(availability), 0)

    if args.embedding_cache:
        log(f"Loading embedding cache: {args.embedding_cache}")
        embedding_cache = torch.load(args.embedding_cache, map_location="cpu", weights_only=False)
        log(
            "Embedding cache loaded: "
            f"feature_dim={embedding_cache['feature_dim']} "
            f"macro={len(embedding_cache.get('macro', {}))} "
            f"micro={len(embedding_cache.get('micro', {}))} "
            f"macro_aug={len(embedding_cache.get('macro_aug', {}))} "
            f"micro_aug={len(embedding_cache.get('micro_aug', {}))} "
            f"aug_views={embedding_cache.get('augmentation', {}).get('views', 0)}"
        )
        if args.stream == "both" and (
            not embedding_cache.get("macro") or not embedding_cache.get("micro")
        ):
            raise ValueError(
                "--stream both requires a cache containing macro and micro embeddings. "
                "Re-run precompute with --stream both --tile-streams both."
            )
    else:
        embedding_cache = None
        log("No embedding cache supplied; training will load images and run ViT each epoch")

    log("Generating strict train/validation/test group folds")
    split_group_col = "station_code" if args.fold_group_by == "station" else "group_id"
    eligibility = split_group_target_eligibility(
        daily_df,
        split_group_col,
        args.stream,
        active_classes,
        embedding_cache=embedding_cache,
    )
    eligibility_path = os.path.join(args.out_dir, "split_group_target_eligibility.csv")
    eligibility.to_csv(eligibility_path, index=False)
    eligible_groups = eligibility.loc[
        eligibility["eligible_target_days"] > 0,
        split_group_col,
    ].tolist()
    ignored_groups = eligibility.loc[
        eligibility["eligible_target_days"] == 0,
        split_group_col,
    ].tolist()
    if ignored_groups:
        log(
            "Ignoring split groups with zero eligible target days: "
            f"{ignored_groups}. Audit: {eligibility_path}"
        )
    else:
        log(f"All split groups have eligible targets. Audit: {eligibility_path}")
    fold_df = daily_df[daily_df[split_group_col].isin(eligible_groups)]
    total_groups = int(fold_df[split_group_col].nunique())
    n_val_groups = int(VALIDATION_FOLD_STATIONS if args.validation_groups is None else args.validation_groups)
    n_test_groups = int(TEST_FOLD_STATIONS if args.test_groups is None else args.test_groups)
    n_train_groups = total_groups - n_val_groups - n_test_groups
    if n_val_groups < 1:
        raise ValueError("VALIDATION_FOLD_STATIONS must be at least 1")
    if n_test_groups < 0:
        raise ValueError("TEST_FOLD_STATIONS must be at least 0")
    if n_train_groups < 1:
        raise ValueError(
            "Invalid split after target-eligibility filtering: "
            f"eligible_groups={total_groups}, validation_groups={n_val_groups}, "
            f"test_groups={n_test_groups}. "
            "At least one training group must remain."
        )
    log(
        "DINOv2 eligible fold group counts: "
        f"train={n_train_groups} val={n_val_groups} test={n_test_groups}"
    )
    log(
        "DINOv2 fold station diversity requirements: "
        f"validation={REQUIRE_DIVERSE_VALIDATION_STATIONS} "
        f"test={REQUIRE_DIVERSE_TEST_STATIONS}"
    )
    log(f"Fold sampling seed: {args.fold_seed}")
    log(f"Training seed: {args.seed}")
    if tb_writer is not None:
        tb_writer.add_scalar("config/train_fold_stations", n_train_groups, 0)
        tb_writer.add_scalar("config/validation_fold_stations", n_val_groups, 0)
        tb_writer.add_scalar("config/test_fold_stations", n_test_groups, 0)
        tb_writer.add_scalar("config/require_diverse_validation_stations", int(REQUIRE_DIVERSE_VALIDATION_STATIONS), 0)
        tb_writer.add_scalar("config/require_diverse_test_stations", int(REQUIRE_DIVERSE_TEST_STATIONS), 0)
        tb_writer.add_scalar("config/fold_seed", args.fold_seed, 0)
    log(f"Fold grouping: {args.fold_group_by} ({split_group_col})")
    folds = generate_group_train_val_test_folds(
        fold_df,
        group_col=split_group_col,
        n_train=n_train_groups,
        n_val=n_val_groups,
        n_test=n_test_groups,
        num_folds=args.folds,
        random_state=args.fold_seed,
        station_col="station_code",
        require_diverse_val_stations=REQUIRE_DIVERSE_VALIDATION_STATIONS,
        require_diverse_test_stations=REQUIRE_DIVERSE_TEST_STATIONS,
    )
    fold_assignment_rows = []
    for fold_id, (train_idx, val_idx, test_idx) in enumerate(folds, 1):
        for role, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            role_df = daily_df.iloc[idx]
            for group_id, group_df in role_df.groupby(split_group_col):
                fold_assignment_rows.append(
                    {
                        "fold": fold_id,
                        "role": role,
                        "split_group_column": split_group_col,
                        "group_id": group_id,
                        "station_years": "|".join(sorted(group_df["station_year"].dropna().astype(str).unique())),
                        "station_codes": "|".join(sorted(group_df["station_code"].dropna().astype(str).unique()))
                        if "station_code" in group_df.columns
                        else "",
                        "rows": len(group_df),
                    }
                )
    fold_assignment_path = os.path.join(args.out_dir, "fold_assignments.csv")
    pd.DataFrame(fold_assignment_rows).to_csv(fold_assignment_path, index=False)
    log(f"Saved fold assignments: {fold_assignment_path}")
    if resume_fold_id is not None and not (1 <= resume_fold_id <= len(folds)):
        raise ValueError(f"Resume checkpoint fold_id={resume_fold_id} is outside generated fold range 1..{len(folds)}")

    target_offset = args.window_days - 1 if args.window_mode == "causal" else None
    cfg = WindowConfig(
        window_days=args.window_days,
        center_offset=target_offset,
        classes=tuple(active_classes),
        stream=args.stream,
        temporal_feature_columns=temporal_feature_columns,
    )
    log(
        f"Window mode: {args.window_mode} | window_days={args.window_days} "
        f"| target_index={cfg.center} | stream={args.stream} "
        f"| temporal_feature_dim={cfg.temporal_feature_dim}"
    )
    all_history = []
    all_test_metrics = []
    all_val_metrics = []
    all_val_predictions = []
    all_test_predictions = []
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
        existing_test_predictions_path = os.path.join(args.out_dir, "all_test_predictions.csv")
        if os.path.isfile(existing_test_predictions_path):
            all_test_predictions = pd.read_csv(existing_test_predictions_path).to_dict("records")
            log(f"Loaded existing held-out test predictions for resume: {existing_test_predictions_path}")
        existing_val_metrics_path = os.path.join(args.out_dir, "all_val_metrics.csv")
        if os.path.isfile(existing_val_metrics_path):
            all_val_metrics = pd.read_csv(existing_val_metrics_path).to_dict("records")
            log(f"Loaded existing validation metrics for resume: {existing_val_metrics_path}")
        existing_val_predictions_path = os.path.join(args.out_dir, "all_val_predictions.csv")
        if os.path.isfile(existing_val_predictions_path):
            all_val_predictions = pd.read_csv(existing_val_predictions_path).to_dict("records")
            log(f"Loaded existing validation predictions for resume: {existing_val_predictions_path}")
    if args.embedding_smote and embedding_cache is None:
        raise ValueError("--embedding-smote requires --embedding-cache because SMOTE is applied to frozen embedding windows, not raw images.")

    for fold_id, (train_idx, val_idx, test_idx) in enumerate(folds, 1):
        if resume_state is not None and fold_id < resume_fold_id:
            log(f"Skipping fold {fold_id}/{len(folds)} because resume starts at fold {resume_fold_id}")
            continue

        log(f"Preparing fold {fold_id}/{len(folds)}")
        fold_seed = int(args.seed) + fold_id
        seed_everything(fold_seed)
        fold_dir = os.path.join(args.out_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        is_resume_fold = resume_state is not None and fold_id == resume_fold_id

        train_df = daily_df.iloc[train_idx].reset_index(drop=True)
        val_df = daily_df.iloc[val_idx].reset_index(drop=True)
        test_df = daily_df.iloc[test_idx].reset_index(drop=True)
        log(
            f"Fold {fold_id}: train rows={len(train_df)} val rows={len(val_df)} test rows={len(test_df)} "
            f"train groups={sorted(train_df[split_group_col].unique().tolist())} "
            f"val groups={sorted(val_df[split_group_col].unique().tolist())} "
            f"test groups={sorted(test_df[split_group_col].unique().tolist())}"
        )
        split_info = {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "fold_group_by": args.fold_group_by,
            "train_groups": sorted(train_df[split_group_col].unique().tolist()),
            "val_groups": sorted(val_df[split_group_col].unique().tolist()),
            "test_groups": sorted(test_df[split_group_col].unique().tolist()),
        }
        split_info_path = os.path.join(fold_dir, "split_info.json")
        with open(split_info_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)
        log_tb_text(tb_writer, f"fold_{fold_id}/split", split_info)
        if tb_writer is not None:
            tb_writer.add_scalar(f"fold_{fold_id}/split/train_rows", len(train_df), 0)
            tb_writer.add_scalar(f"fold_{fold_id}/split/val_rows", len(val_df), 0)
            tb_writer.add_scalar(f"fold_{fold_id}/split/test_rows", len(test_df), 0)
        if embedding_cache is None:
            log(f"Fold {fold_id}: creating image window datasets")
            train_ds = MultiScaleWindowDataset(
                train_df,
                cfg,
                min_stage_support_days=args.min_train_stage_days,
                min_window_coverage_days=args.min_train_window_coverage_days,
            )
            val_ds = MultiScaleWindowDataset(val_df, cfg)
            test_ds = MultiScaleWindowDataset(test_df, cfg) if len(test_df) else None
        else:
            log(f"Fold {fold_id}: creating embedding window datasets")
            train_ds = MultiScaleEmbeddingWindowDataset(
                train_df,
                cfg,
                embedding_cache,
                use_augmentation=args.use_augmented_embeddings,
                augmentation_multiplier=args.embedding_augmentation_multiplier,
                min_stage_support_days=args.min_train_stage_days,
                min_window_coverage_days=args.min_train_window_coverage_days,
            )
            val_ds = MultiScaleEmbeddingWindowDataset(val_df, cfg, embedding_cache)
            test_ds = MultiScaleEmbeddingWindowDataset(test_df, cfg, embedding_cache) if len(test_df) else None
            if getattr(train_ds, "sample_multiplier", 1) > 1:
                log(
                    f"Fold {fold_id}: augmented embedding training multiplier="
                    f"{train_ds.sample_multiplier} base_samples={len(train_ds.samples)}"
                )
            if args.embedding_smote:
                log(f"Fold {fold_id}: building SMOTE neighbours from training embeddings only")
                train_ds = SMOTEEmbeddingWindowDataset(
                    train_ds,
                    target_ratio=args.smote_target_ratio,
                    k_neighbors=args.smote_k_neighbors,
                    max_synthetic_samples=args.smote_max_synthetic_samples,
                    seed=args.smote_seed + fold_id,
                )
                smote_summary_path = os.path.join(fold_dir, "smote_summary.json")
                with open(smote_summary_path, "w", encoding="utf-8") as f:
                    json.dump(train_ds.summary, f, indent=2)
                log(
                    f"Fold {fold_id}: SMOTE added {train_ds.summary['synthetic_samples']} synthetic windows "
                    f"({train_ds.summary['original_samples']} real -> {train_ds.summary['total_samples']} total); "
                    f"summary: {smote_summary_path}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar(
                        f"fold_{fold_id}/smote/synthetic_samples",
                        train_ds.summary["synthetic_samples"],
                        0,
                    )
                    tb_writer.add_scalar(
                        f"fold_{fold_id}/smote/total_train_samples",
                        train_ds.summary["total_samples"],
                        0,
                    )
                    log_tb_text(tb_writer, f"fold_{fold_id}/smote/summary", train_ds.summary)
        coverage_filter_summary = getattr(
            train_ds.base_dataset if isinstance(train_ds, SMOTEEmbeddingWindowDataset) else train_ds,
            "filter_summary",
            {},
        )
        coverage_filter_path = os.path.join(fold_dir, "training_coverage_filter.json")
        with open(coverage_filter_path, "w", encoding="utf-8") as f:
            json.dump(coverage_filter_summary, f, indent=2)
        log(
            f"Fold {fold_id}: training coverage filter kept="
            f"{coverage_filter_summary.get('kept_samples', len(train_ds))} "
            f"excluded_stage={coverage_filter_summary.get('excluded_stage_support_samples', 0)} "
            f"excluded_window={coverage_filter_summary.get('excluded_window_coverage_samples', 0)}; "
            f"summary: {coverage_filter_path}"
        )
        log_tb_text(tb_writer, f"fold_{fold_id}/coverage_filter", coverage_filter_summary)
        test_samples = 0 if test_ds is None else len(test_ds)
        log(f"Fold {fold_id}: train samples={len(train_ds)} val samples={len(val_ds)} test samples={test_samples}")
        if len(train_ds) == 0 or len(val_ds) == 0 or (test_ds is not None and len(test_ds) == 0):
            raise RuntimeError(
                f"Fold {fold_id} produced an empty dataset split "
                f"(train={len(train_ds)}, val={len(val_ds)}, test={test_samples})."
            )

        train_loader = build_loader(train_ds, args, device, shuffle=True, drop_last=args.drop_last, seed=fold_seed)
        val_loader = build_loader(val_ds, args, device, shuffle=False, seed=fold_seed + 10_000)
        test_loader = build_loader(test_ds, args, device, shuffle=False, seed=fold_seed + 20_000) if test_ds is not None else None
        log(
            f"Fold {fold_id}: loaders ready "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
            f"test_batches={0 if test_loader is None else len(test_loader)} "
            f"loader_batch_size={train_loader.batch_size} requested_batch_size={args.batch_size} "
            f"log_interval={args.log_interval} "
            f"num_workers={args.num_workers} pin_memory={args.pin_memory and device.type == 'cuda'} "
            f"persistent_workers={args.persistent_workers and args.num_workers > 0} drop_last={args.drop_last}"
        )
        if len(train_loader) == 0:
            raise RuntimeError(
                f"Fold {fold_id} train loader has zero batches. "
                "Disable --drop-last or reduce --batch-size."
            )

        model = build_model(args, cfg, embedding_cache, device, active_classes)
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
        criterion = build_criterion(args)
        log(f"Fold {fold_id}: criterion={criterion.__class__.__name__}")
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

        best_mode = checkpoint_mode(args.checkpoint_metric, args.checkpoint_mode)
        best_score = float("inf") if best_mode == "min" else -float("inf")
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
            if train_loader.generator is not None and resume_state.get("train_loader_generator_state") is not None:
                train_loader.generator.set_state(resume_state["train_loader_generator_state"])
            restore_rng_state(resume_state.get("rng_state"))
            best_score = float(resume_state.get("best_score", best_score))
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
                classes=active_classes,
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
                classes=active_classes,
            )
            row = {
                "fold": fold_id,
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            all_history.append(row)
            log("Epoch accuracies: " + json.dumps(compact_epoch_metrics(row), indent=None))
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
            if checkpoint_improved(float(checkpoint_score), best_score, best_mode):
                best_score = checkpoint_score
                best_epoch = epoch
                torch.save(
                    {
                        "model": model_state_dict(model),
                        "classes": active_classes,
                        "window_days": args.window_days,
                        "window_mode": args.window_mode,
                        "target_index": cfg.center,
                        "stream": args.stream,
                        "image_backbone": args.image_backbone,
                        "pretrained": args.pretrained,
                        "temporal_aggregation": args.temporal_aggregation,
                        "temporal_model": args.temporal_model,
                        "temporal_layers": args.temporal_layers,
                        "temporal_heads": args.temporal_heads,
                        "temporal_norm_first": args.temporal_norm_first,
                        "temporal_ffn_multiplier": args.temporal_ffn_multiplier,
                        "dropout": args.dropout,
                        "seed": args.seed,
                        "fold_group_by": args.fold_group_by,
                        "ordinal_ce_weight": args.ordinal_ce_weight,
                        "use_days_since_planting": args.use_days_since_planting,
                        "use_weather_metadata": args.use_weather_metadata,
                        "temporal_feature_dim": cfg.temporal_feature_dim,
                        "temporal_feature_columns": list(cfg.temporal_feature_columns),
                        "temporal_feature_hidden_dim": args.temporal_feature_hidden_dim,
                        "gate_hidden_dim": args.gate_hidden_dim,
                        "modality_dropout": args.modality_dropout,
                        "fusion": "gated" if args.stream == "both" else "single_stream",
                        "gdd_base_temp": args.gdd_base_temp,
                        "exclude_offseason": args.exclude_offseason,
                        "date_tolerance_days": args.date_tolerance_days,
                        "monotonic_decoding": args.monotonic_decoding,
                        "monotonic_advance_penalty": args.monotonic_advance_penalty,
                        "min_train_stage_days": args.min_train_stage_days,
                        "min_train_window_coverage_days": args.min_train_window_coverage_days,
                        "fold_seed": args.fold_seed,
                        "checkpoint_metric": args.checkpoint_metric,
                        "checkpoint_mode": best_mode,
                        "loss": args.loss,
                        "ordinal_power": args.ordinal_power,
                        "validation_score": best_score,
                        "best_epoch": best_epoch,
                        "train_groups": sorted(train_df[split_group_col].unique().tolist()),
                        "val_groups": sorted(val_df[split_group_col].unique().tolist()),
                        "test_groups": sorted(test_df[split_group_col].unique().tolist()),
                        "uses_embedding_cache": embedding_cache is not None,
                        "use_augmented_embeddings": args.use_augmented_embeddings,
                        "embedding_augmentation_multiplier": getattr(train_ds, "sample_multiplier", 1),
                        "embedding_smote": args.embedding_smote,
                        "smote_target_ratio": args.smote_target_ratio,
                        "smote_k_neighbors": args.smote_k_neighbors,
                        "smote_max_synthetic_samples": args.smote_max_synthetic_samples,
                        "smote_seed": args.smote_seed,
                        "compiled_model": compiled_model,
                        "fused_optimizer": fused_optimizer,
                        "weight_decay": args.weight_decay,
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
                    "rng_state": capture_rng_state(),
                    "train_loader_generator_state": train_loader.generator.get_state() if train_loader.generator is not None else None,
                    "classes": active_classes,
                    "window_days": args.window_days,
                    "window_mode": args.window_mode,
                    "target_index": cfg.center,
                    "stream": args.stream,
                    "temporal_aggregation": args.temporal_aggregation,
                    "temporal_model": args.temporal_model,
                    "temporal_layers": args.temporal_layers,
                    "temporal_heads": args.temporal_heads,
                    "temporal_norm_first": args.temporal_norm_first,
                    "temporal_ffn_multiplier": args.temporal_ffn_multiplier,
                    "dropout": args.dropout,
                    "seed": args.seed,
                    "fold_group_by": args.fold_group_by,
                    "ordinal_ce_weight": args.ordinal_ce_weight,
                    "use_days_since_planting": args.use_days_since_planting,
                    "use_weather_metadata": args.use_weather_metadata,
                    "temporal_feature_dim": cfg.temporal_feature_dim,
                    "temporal_feature_columns": list(cfg.temporal_feature_columns),
                    "temporal_feature_hidden_dim": args.temporal_feature_hidden_dim,
                    "gate_hidden_dim": args.gate_hidden_dim,
                    "modality_dropout": args.modality_dropout,
                    "fusion": "gated" if args.stream == "both" else "single_stream",
                    "gdd_base_temp": args.gdd_base_temp,
                    "exclude_offseason": args.exclude_offseason,
                    "date_tolerance_days": args.date_tolerance_days,
                    "monotonic_decoding": args.monotonic_decoding,
                    "monotonic_advance_penalty": args.monotonic_advance_penalty,
                    "min_train_stage_days": args.min_train_stage_days,
                    "min_train_window_coverage_days": args.min_train_window_coverage_days,
                    "fold_seed": args.fold_seed,
                    "checkpoint_metric": args.checkpoint_metric,
                    "checkpoint_mode": best_mode,
                    "loss": args.loss,
                    "ordinal_power": args.ordinal_power,
                    "train_groups": sorted(train_df[split_group_col].unique().tolist()),
                    "val_groups": sorted(val_df[split_group_col].unique().tolist()),
                    "test_groups": sorted(test_df[split_group_col].unique().tolist()),
                    "uses_embedding_cache": embedding_cache is not None,
                    "use_augmented_embeddings": args.use_augmented_embeddings,
                    "embedding_augmentation_multiplier": getattr(train_ds, "sample_multiplier", 1),
                    "embedding_smote": args.embedding_smote,
                    "smote_target_ratio": args.smote_target_ratio,
                    "smote_k_neighbors": args.smote_k_neighbors,
                    "smote_max_synthetic_samples": args.smote_max_synthetic_samples,
                    "smote_seed": args.smote_seed,
                    "compiled_model": compiled_model,
                    "fused_optimizer": fused_optimizer,
                    "weight_decay": args.weight_decay,
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

        best_val_row = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            "checkpoint_metric": args.checkpoint_metric,
            f"best_val_{args.checkpoint_metric}": best_score,
            "train_groups": "|".join(map(str, sorted(train_df[split_group_col].unique().tolist()))),
            "val_groups": "|".join(map(str, sorted(val_df[split_group_col].unique().tolist()))),
            "test_groups": "|".join(map(str, sorted(test_df[split_group_col].unique().tolist()))),
        }
        if history and best_epoch is not None:
            best_history = next((row for row in history if int(row["epoch"]) == int(best_epoch)), None)
            if best_history is not None:
                best_val_row.update(
                    {
                        key: value
                        for key, value in best_history.items()
                        if key.startswith("train_") or key.startswith("val_")
                    }
                )
        all_val_metrics.append(best_val_row)
        val_metrics_path = os.path.join(fold_dir, "val_metrics.json")
        with open(val_metrics_path, "w", encoding="utf-8") as f:
            json.dump(best_val_row, f, indent=2)

        log(f"Fold {fold_id}: loading best checkpoint from epoch {best_epoch} for coverage evaluation")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        load_model_state(model, checkpoint["model"])
        fold_val_predictions = []
        run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            log_interval=args.log_interval,
            phase="val_coverage",
            fold_id=fold_id,
            epoch=best_epoch or args.epochs,
            use_amp=use_amp,
            prediction_rows=fold_val_predictions,
            classes=active_classes,
        )
        for prediction_row in fold_val_predictions:
            prediction_row["fold"] = fold_id
        if args.monotonic_decoding == "viterbi":
            apply_monotonic_decoder(
                fold_val_predictions,
                active_classes,
                advance_penalty=args.monotonic_advance_penalty,
            )
            decoded_val_predictions = decoded_prediction_view(fold_val_predictions)
            decoded_val_metrics = prediction_summary_from_rows(decoded_val_predictions, active_classes)
            best_val_row.update({f"monotonic_val_{key}": value for key, value in decoded_val_metrics.items()})
            with open(val_metrics_path, "w", encoding="utf-8") as f:
                json.dump(best_val_row, f, indent=2)
            save_confusion_matrix(decoded_val_predictions, fold_dir, "val_monotonic", active_classes)
            save_classification_metrics(decoded_val_predictions, fold_dir, "val_monotonic", active_classes)
            save_coverage_stratified_metrics(
                decoded_val_predictions,
                fold_dir,
                "val_monotonic",
                active_classes,
                window_days=args.window_days,
                min_stage_support_days=args.min_train_stage_days,
                min_window_coverage_days=args.min_train_window_coverage_days,
            )
            log_tb_metrics(tb_writer, f"fold_{fold_id}/val_monotonic", decoded_val_metrics, best_epoch or args.epochs)
        all_val_predictions.extend(fold_val_predictions)
        val_predictions_path = os.path.join(fold_dir, "val_predictions.csv")
        pd.DataFrame(fold_val_predictions).to_csv(val_predictions_path, index=False)
        save_coverage_stratified_metrics(
            fold_val_predictions,
            fold_dir,
            "val",
            active_classes,
            window_days=args.window_days,
            min_stage_support_days=args.min_train_stage_days,
            min_window_coverage_days=args.min_train_window_coverage_days,
        )

        if test_loader is None:
            log(f"Fold {fold_id}: no test groups requested; skipping final test evaluation")
            test_row = {
                "fold": fold_id,
                "best_epoch": best_epoch,
                f"best_val_{args.checkpoint_metric}": best_score,
                **{f"test_{k}": v for k, v in empty_metrics().items()},
            }
            all_test_metrics.append(test_row)
            history_path = os.path.join(fold_dir, "history.csv")
            pd.DataFrame(history).to_csv(history_path, index=False)
            log(f"Fold {fold_id}: saved history to {history_path}")
            brief_history_path = os.path.join(fold_dir, "brief_history.csv")
            brief_history_frame(history).to_csv(brief_history_path, index=False)
            log(f"Fold {fold_id}: saved brief history to {brief_history_path}")
            continue

        log(f"Fold {fold_id}: evaluating the best checkpoint on the final held-out test split")
        fold_test_predictions = []
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
            prediction_rows=fold_test_predictions,
            classes=active_classes,
        )
        for prediction_row in fold_test_predictions:
            prediction_row["fold"] = fold_id
        decoded_test_metrics = None
        if args.monotonic_decoding == "viterbi":
            apply_monotonic_decoder(
                fold_test_predictions,
                active_classes,
                advance_penalty=args.monotonic_advance_penalty,
            )
            decoded_test_predictions = decoded_prediction_view(fold_test_predictions)
            decoded_test_metrics = prediction_summary_from_rows(decoded_test_predictions, active_classes)
        all_test_predictions.extend(fold_test_predictions)
        test_row = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            f"best_val_{args.checkpoint_metric}": best_score,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        if decoded_test_metrics is not None:
            test_row.update({f"test_monotonic_{key}": value for key, value in decoded_test_metrics.items()})
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
        test_predictions_path = os.path.join(fold_dir, "test_predictions.csv")
        pd.DataFrame(fold_test_predictions).to_csv(test_predictions_path, index=False)
        log(f"Fold {fold_id}: final held-out test predictions saved to {test_predictions_path}")
        save_confusion_matrix(fold_test_predictions, fold_dir, "test", active_classes)
        save_classification_metrics(fold_test_predictions, fold_dir, "test", active_classes)
        save_coverage_stratified_metrics(
            fold_test_predictions,
            fold_dir,
            "test",
            active_classes,
            window_days=args.window_days,
            min_stage_support_days=args.min_train_stage_days,
            min_window_coverage_days=args.min_train_window_coverage_days,
        )
        if decoded_test_metrics is not None:
            save_confusion_matrix(decoded_test_predictions, fold_dir, "test_monotonic", active_classes)
            save_classification_metrics(decoded_test_predictions, fold_dir, "test_monotonic", active_classes)
            save_coverage_stratified_metrics(
                decoded_test_predictions,
                fold_dir,
                "test_monotonic",
                active_classes,
                window_days=args.window_days,
                min_stage_support_days=args.min_train_stage_days,
                min_window_coverage_days=args.min_train_window_coverage_days,
            )
            log_tb_metrics(tb_writer, f"fold_{fold_id}/test_monotonic", decoded_test_metrics, best_epoch or args.epochs)

        history_path = os.path.join(fold_dir, "history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False)
        log(f"Fold {fold_id}: saved history to {history_path}")
        brief_history_path = os.path.join(fold_dir, "brief_history.csv")
        brief_history_frame(history).to_csv(brief_history_path, index=False)
        log(f"Fold {fold_id}: saved brief history to {brief_history_path}")

    all_history_path = os.path.join(args.out_dir, "all_history.csv")
    pd.DataFrame(all_history).to_csv(all_history_path, index=False)
    log(f"Saved combined history to {all_history_path}")
    brief_history_path = os.path.join(args.out_dir, "brief_history.csv")
    brief_history_frame(all_history).to_csv(brief_history_path, index=False)
    log(f"Saved brief combined history to {brief_history_path}")
    all_val_metrics_path = os.path.join(args.out_dir, "all_val_metrics.csv")
    pd.DataFrame(all_val_metrics).to_csv(all_val_metrics_path, index=False)
    log(f"Saved combined validation metrics to {all_val_metrics_path}")
    all_val_predictions_path = os.path.join(args.out_dir, "all_val_predictions.csv")
    pd.DataFrame(all_val_predictions).to_csv(all_val_predictions_path, index=False)
    log(f"Saved combined validation predictions to {all_val_predictions_path}")
    save_coverage_stratified_metrics(
        all_val_predictions,
        args.out_dir,
        "all_val",
        active_classes,
        window_days=args.window_days,
        min_stage_support_days=args.min_train_stage_days,
        min_window_coverage_days=args.min_train_window_coverage_days,
    )
    if args.monotonic_decoding == "viterbi":
        all_val_decoded = decoded_prediction_view(all_val_predictions)
        save_confusion_matrix(all_val_decoded, args.out_dir, "all_val_monotonic", active_classes)
        save_classification_metrics(all_val_decoded, args.out_dir, "all_val_monotonic", active_classes)
        save_coverage_stratified_metrics(
            all_val_decoded,
            args.out_dir,
            "all_val_monotonic",
            active_classes,
            window_days=args.window_days,
            min_stage_support_days=args.min_train_stage_days,
            min_window_coverage_days=args.min_train_window_coverage_days,
        )
    all_test_metrics_path = os.path.join(args.out_dir, "all_test_metrics.csv")
    pd.DataFrame(all_test_metrics).to_csv(all_test_metrics_path, index=False)
    log(f"Saved combined held-out test metrics to {all_test_metrics_path}")
    all_test_predictions_path = os.path.join(args.out_dir, "all_test_predictions.csv")
    pd.DataFrame(all_test_predictions).to_csv(all_test_predictions_path, index=False)
    log(f"Saved combined held-out test predictions to {all_test_predictions_path}")
    save_confusion_matrix(all_test_predictions, args.out_dir, "all_test", active_classes)
    save_classification_metrics(all_test_predictions, args.out_dir, "all_test", active_classes)
    save_coverage_stratified_metrics(
        all_test_predictions,
        args.out_dir,
        "all_test",
        active_classes,
        window_days=args.window_days,
        min_stage_support_days=args.min_train_stage_days,
        min_window_coverage_days=args.min_train_window_coverage_days,
    )
    if args.monotonic_decoding == "viterbi":
        all_test_decoded = decoded_prediction_view(all_test_predictions)
        save_confusion_matrix(all_test_decoded, args.out_dir, "all_test_monotonic", active_classes)
        save_classification_metrics(all_test_decoded, args.out_dir, "all_test_monotonic", active_classes)
        save_coverage_stratified_metrics(
            all_test_decoded,
            args.out_dir,
            "all_test_monotonic",
            active_classes,
            window_days=args.window_days,
            min_stage_support_days=args.min_train_stage_days,
            min_window_coverage_days=args.min_train_window_coverage_days,
        )
    if tb_writer is not None:
        if all_test_metrics:
            test_df = pd.DataFrame(all_test_metrics)
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
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "seed": args.seed,
                "fold_group_by": args.fold_group_by,
                "ordinal_ce_weight": args.ordinal_ce_weight,
                "loss": args.loss,
                "ordinal_power": args.ordinal_power,
                "folds": args.folds,
                "fold_seed": args.fold_seed,
                "window_days": args.window_days,
                "window_mode": args.window_mode,
                "temporal_aggregation": args.temporal_aggregation,
                "temporal_model": args.temporal_model,
                "temporal_layers": args.temporal_layers,
                "temporal_heads": args.temporal_heads,
                "temporal_norm_first": args.temporal_norm_first,
                "temporal_ffn_multiplier": args.temporal_ffn_multiplier,
                "use_days_since_planting": args.use_days_since_planting,
                "use_weather_metadata": args.use_weather_metadata,
                "temporal_feature_dim": cfg.temporal_feature_dim,
                "temporal_feature_hidden_dim": args.temporal_feature_hidden_dim,
                "gate_hidden_dim": args.gate_hidden_dim,
                "modality_dropout": args.modality_dropout,
                "fusion": "gated" if args.stream == "both" else "single_stream",
                "gdd_base_temp": args.gdd_base_temp,
                "exclude_offseason": args.exclude_offseason,
                "min_train_stage_days": args.min_train_stage_days,
                "min_train_window_coverage_days": args.min_train_window_coverage_days,
                "stream": args.stream,
                "amp": use_amp,
                "embedding_cache": bool(args.embedding_cache),
                "use_augmented_embeddings": args.use_augmented_embeddings,
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
