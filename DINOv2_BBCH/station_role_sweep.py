import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from multiscale_phenology import BASE_CLASSES, build_multiscale_daily_dataframe


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PRECOMPUTE_SCRIPT = SCRIPT_DIR / "precompute_multiscale_embeddings.py"
TRAIN_SCRIPT = SCRIPT_DIR / "run_multiscale_training.py"


def log(message: str):
    print(message, flush=True)


def script_arg(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def run_command(command: List[str], dry_run: bool = False):
    log("Command:")
    log(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in command))
    if dry_run:
        return
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def parse_group_list(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part for part in text.split("|") if part != ""]


def group_station_labels(daily_df: pd.DataFrame, group_col: str) -> Dict[str, str]:
    mapping = {}
    for group_id, group in daily_df.groupby(group_col):
        station_years = sorted(str(x) for x in group["station_year"].dropna().unique())
        mapping[str(group_id)] = ", ".join(station_years)
    return mapping


def summarize_station_roles(
    val_metrics_path: Path,
    daily_df: pd.DataFrame,
    out_dir: Path,
    group_col: str,
    metric_cols: Iterable[str] = ("val_accuracy", "val_date_window_accuracy", "val_plus_minus_1_accuracy"),
) -> pd.DataFrame:
    if not val_metrics_path.is_file():
        raise FileNotFoundError(f"Validation metrics not found: {val_metrics_path}")

    results = pd.read_csv(val_metrics_path)
    station_labels = group_station_labels(daily_df, group_col)
    all_groups = sorted(station_labels, key=lambda x: int(float(x)) if str(x).replace(".", "", 1).isdigit() else str(x))
    rows = []

    for group_id in all_groups:
        for role in ("train", "val"):
            role_mask = results[f"{role}_groups"].apply(lambda value: group_id in parse_group_list(value))
            subset = results[role_mask]
            row = {
                "group_id": group_id,
                "station_years": station_labels.get(group_id, ""),
                "role": role,
                "count": int(len(subset)),
            }
            for metric in metric_cols:
                if metric in subset.columns:
                    values = pd.to_numeric(subset[metric], errors="coerce").dropna()
                    row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
                    row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else None
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "station_role_summary.csv"
    summary.to_csv(summary_path, index=False)
    log(f"Saved station role summary: {summary_path}")
    return summary


def plot_station_role_summary(summary: pd.DataFrame, out_dir: Path, metric: str):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary.columns:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib is not installed; skipping plots.")
        return

    pivot_mean = summary.pivot(index="group_id", columns="role", values=mean_col)
    pivot_std = summary.pivot(index="group_id", columns="role", values=std_col)
    x = list(range(len(pivot_mean.index)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.75), 5))
    train_mean = pivot_mean.get("train")
    val_mean = pivot_mean.get("val")
    train_std = pivot_std.get("train")
    val_std = pivot_std.get("val")

    if train_mean is not None:
        ax.bar([i - width / 2 for i in x], train_mean, width, yerr=train_std, label="station in train", capsize=3)
    if val_mean is not None:
        ax.bar([i + width / 2 for i in x], val_mean, width, yerr=val_std, label="station in validation", capsize=3)

    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xlabel("group_id")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_mean.index.astype(str), rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / f"station_role_{metric}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log(f"Saved plot: {path}")


def build_precompute_command(args, cache_path: Path) -> List[str]:
    return [
        args.python,
        script_arg(PRECOMPUTE_SCRIPT),
        "--excel-path",
        args.excel_path,
        "--data-path",
        args.data_path,
        "--out-dir",
        str(cache_path.parent),
        "--cache-path",
        str(cache_path),
        "--camera",
        args.camera,
        "--stream",
        args.stream,
        "--tile-streams",
        args.tile_streams,
        "--tile-pooling",
        args.tile_pooling,
        "--tile-size",
        str(args.tile_size),
        "--tile-stride",
        str(args.tile_stride),
        "--max-tiles",
        str(args.max_tiles),
        "--preplant-days",
        str(args.preplant_days),
        "--postharvest-days",
        str(args.postharvest_days),
        "--date-tolerance-days",
        str(args.date_tolerance_days),
        "--batch-size",
        str(args.precompute_batch_size),
        "--num-workers",
        str(args.precompute_workers),
        "--embedding-dtype",
        args.embedding_dtype,
        "--image-backbone",
        args.image_backbone,
        "--device",
        args.device,
    ] + (["--pretrained"] if args.pretrained else [])


def build_training_command(args, cache_path: Path, folds: int, extra_train_args: List[str]) -> List[str]:
    command = [
        args.python,
        script_arg(TRAIN_SCRIPT),
        "--excel-path",
        args.excel_path,
        "--data-path",
        args.data_path,
        "--out-dir",
        args.out_dir,
        "--embedding-cache",
        str(cache_path),
        "--epochs",
        str(args.epochs),
        "--folds",
        str(folds),
        "--validation-groups",
        str(args.val_groups),
        "--test-groups",
        "0",
        "--fold-group-by",
        args.fold_group_by,
        "--window-days",
        str(args.window_days),
        "--window-mode",
        args.window_mode,
        "--temporal-aggregation",
        args.temporal_aggregation,
        "--stream",
        args.stream,
        "--preplant-days",
        str(args.preplant_days),
        "--postharvest-days",
        str(args.postharvest_days),
        "--date-tolerance-days",
        str(args.date_tolerance_days),
        "--batch-size",
        str(args.train_batch_size),
        "--accumulation-steps",
        str(args.accumulation_steps),
        "--num-workers",
        str(args.train_workers),
        "--camera",
        args.camera,
        "--device",
        args.device,
        "--log-interval",
        str(args.log_interval),
        "--checkpoint-metric",
        args.checkpoint_metric,
    ]
    if not args.tensorboard:
        command.append("--no-tensorboard")
    if args.no_amp:
        command.append("--no-amp")
    command.extend(extra_train_args)
    return command


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute DINOv2 embeddings once, train every 12-train/3-val/no-test station combination, "
            "and summarize how each station behaves when used for training versus validation."
        )
    )
    parser.add_argument("--excel-path", default="labeling_bbch_iso_dates.csv")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_dinov2_bbch_station_role_sweep")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera", default="AUTO")
    parser.add_argument("--stream", choices=["micro", "macro", "both"], default="micro")
    parser.add_argument("--train-groups", type=int, default=12)
    parser.add_argument("--val-groups", type=int, default=3)
    parser.add_argument("--fold-group-by", choices=["station", "station_year"], default="station")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--window-days", type=int, default=31)
    parser.add_argument("--window-mode", choices=["causal", "center"], default="causal")
    parser.add_argument("--temporal-aggregation", choices=["target", "mean", "cls"], default="cls")
    parser.add_argument("--preplant-days", type=int, default=30)
    parser.add_argument("--postharvest-days", type=int, default=30)
    parser.add_argument("--date-tolerance-days", type=int, default=7)
    parser.add_argument("--checkpoint-metric", default="date_window_accuracy")
    parser.add_argument("--precompute-batch-size", type=int, default=64)
    parser.add_argument("--precompute-workers", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--train-workers", type=int, default=0)
    parser.add_argument("--tile-streams", choices=["none", "micro", "macro", "both"], default="micro")
    parser.add_argument("--tile-pooling", choices=["attention", "mean"], default="attention")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tile-stride", type=int, default=224)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--embedding-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--image-backbone", default="facebook/dinov2-base")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--force-precompute", action="store_true")
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None, help="Run only a random subset of validation combinations for a pilot.")
    parser.add_argument("--tensorboard", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args, extra_train_args = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_path) if args.cache_path else out_dir / "precompute" / "vit_embeddings.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    daily_df = build_multiscale_daily_dataframe(
        args.excel_path,
        args.data_path,
        include_preplant_days=args.preplant_days,
        include_postharvest_days=args.postharvest_days,
        date_tolerance_days=args.date_tolerance_days,
        classes=BASE_CLASSES,
        preferred_camera=None if args.camera.upper() == "ALL" else args.camera,
    )
    group_col = "station_code" if args.fold_group_by == "station" else "group_id"
    groups = sorted(daily_df[group_col].dropna().unique().tolist())
    total_groups = len(groups)
    expected_total = math.comb(total_groups, args.val_groups)
    folds_to_run = min(args.max_runs, expected_total) if args.max_runs else expected_total
    if args.train_groups + args.val_groups != total_groups:
        raise ValueError(
            f"train_groups + val_groups must equal available groups for no-test sweep. "
            f"Got {args.train_groups} + {args.val_groups} but found {total_groups} groups: {groups}"
        )

    plan = {
        "groups": [str(x) for x in groups],
        "total_groups": total_groups,
        "train_groups": args.train_groups,
        "val_groups": args.val_groups,
        "all_validation_combinations": expected_total,
        "folds_to_run": folds_to_run,
        "cache_path": str(cache_path),
        "extra_train_args": extra_train_args,
        "fold_group_by": args.fold_group_by,
    }
    plan_path = out_dir / "station_role_sweep_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log(f"Saved sweep plan: {plan_path}")
    log(f"Validation combinations: {expected_total}; folds_to_run={folds_to_run}")

    if not args.analyze_only:
        if not args.skip_precompute and (args.force_precompute or not cache_path.is_file()):
            run_command(build_precompute_command(args, cache_path), dry_run=args.dry_run)
        elif args.skip_precompute:
            log("Skipping precompute by request.")
        else:
            log(f"Using existing embedding cache: {cache_path}")

        if not args.skip_training:
            run_command(build_training_command(args, cache_path, folds_to_run, extra_train_args), dry_run=args.dry_run)
        else:
            log("Skipping training by request.")

    if args.dry_run:
        return

    val_metrics_path = out_dir / "all_val_metrics.csv"
    summary = summarize_station_roles(val_metrics_path, daily_df, out_dir, group_col=group_col)
    for metric in ("val_accuracy", "val_date_window_accuracy", "val_plus_minus_1_accuracy"):
        plot_station_role_summary(summary, out_dir, metric)


if __name__ == "__main__":
    main()
