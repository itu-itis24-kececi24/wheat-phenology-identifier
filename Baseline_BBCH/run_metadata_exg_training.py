"""Leakage-safe BBCH baseline using metadata, weather/GDD, and ExG features."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

# Avoid joblib's Windows physical-core probe in every spawned fold worker.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from threadpoolctl import threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DINOV3_DIR = PROJECT_ROOT / "DINOv3_BBCH"
sys.path.insert(0, str(DINOV3_DIR))

from multiscale_phenology import (  # noqa: E402
    BASE_CLASSES,
    LOCATION_FEATURE_COLUMNS,
    WEATHER_TEMPORAL_FEATURE_COLUMNS,
    add_location_metadata,
    add_weather_metadata,
    build_multiscale_daily_dataframe,
    generate_loso_train_val_test_folds,
)


EXG_VERSION = "exg-v1"
EXG_COLUMNS = (
    "exg_mean",
    "exg_std",
    "exg_p10",
    "exg_p50",
    "exg_p90",
    "exg_positive_fraction",
    "green_dominance_fraction",
    "brightness_mean",
    "exg_missing",
)
ROLLING_SOURCE_COLUMNS = (
    "exg_mean",
    "exg_std",
    "exg_positive_fraction",
    "green_dominance_fraction",
    "weather_tavg_norm",
    "weather_prcp_norm",
    "weather_gdd_norm",
    "weather_gdd_cum_norm",
)


def image_signature(path: str) -> str:
    stat = os.stat(path)
    text = f"{EXG_VERSION}|{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def extract_exg_features(path: str, max_side: int = 1024) -> dict[str, float]:
    """Compute illumination-tolerant ExG summary statistics from one RGB image."""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError, ValueError):
        return {column: (1.0 if column == "exg_missing" else 0.0) for column in EXG_COLUMNS}

    total = rgb.sum(axis=-1)
    normalized = rgb / np.maximum(total[..., None], 1e-6)
    red, green, blue = (normalized[..., i] for i in range(3))
    exg = 2.0 * green - red - blue
    raw_green = rgb[..., 1]
    raw_red = rgb[..., 0]
    raw_blue = rgb[..., 2]
    return {
        "exg_mean": float(np.mean(exg)),
        "exg_std": float(np.std(exg)),
        "exg_p10": float(np.percentile(exg, 10)),
        "exg_p50": float(np.percentile(exg, 50)),
        "exg_p90": float(np.percentile(exg, 90)),
        "exg_positive_fraction": float(np.mean(exg > 0.05)),
        "green_dominance_fraction": float(
            np.mean((raw_green > raw_red) & (raw_green > raw_blue))
        ),
        "brightness_mean": float(np.mean(rgb)),
        "exg_missing": 0.0,
    }


def _extract_exg_task(task):
    signature, path, max_side = task
    return signature, path, extract_exg_features(path, max_side=max_side)


def resolve_worker_count(requested: int, task_count: int | None = None) -> int:
    if requested < 0:
        raise ValueError("Worker counts must be zero (automatic) or positive.")
    workers = requested or (os.cpu_count() or 1)
    return max(1, min(workers, task_count)) if task_count is not None else max(1, workers)


def add_exg_features(
    daily_df: pd.DataFrame,
    cache_path: str,
    stream: str = "micro",
    max_side: int = 1024,
    workers: int = 0,
) -> pd.DataFrame:
    path_column = "micro_path" if stream == "micro" else "macro_path"
    cache_file = Path(cache_path)
    if cache_file.is_file():
        cache = pd.read_csv(cache_file)
    else:
        cache = pd.DataFrame(columns=["signature", "path", *EXG_COLUMNS])
    cached = {
        str(row.signature): {column: float(getattr(row, column)) for column in EXG_COLUMNS}
        for row in cache.itertuples(index=False)
    }

    missing_features = {
        column: (1.0 if column == "exg_missing" else 0.0) for column in EXG_COLUMNS
    }
    row_keys = []
    pending = {}
    for value in daily_df[path_column]:
        if value is None or pd.isna(value) or not os.path.isfile(str(value)):
            row_keys.append(None)
            continue
        path = str(value)
        signature = image_signature(path)
        row_keys.append(signature)
        if signature not in cached:
            pending[signature] = os.path.abspath(path)

    new_rows = []
    if pending:
        tasks = [(signature, path, max_side) for signature, path in pending.items()]
        worker_count = resolve_worker_count(workers, len(tasks))
        print(
            f"Extracting ExG from {len(tasks)} uncached images with {worker_count} processes",
            flush=True,
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            for signature, path, features in executor.map(
                _extract_exg_task,
                tasks,
                chunksize=max(1, len(tasks) // max(1, worker_count * 8)),
            ):
                cached[signature] = features
                new_rows.append({"signature": signature, "path": path, **features})

    row_features = [
        missing_features if signature is None else cached[signature]
        for signature in row_keys
    ]

    if new_rows:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        updated = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        updated.drop_duplicates("signature", keep="last").to_csv(cache_file, index=False)

    out = daily_df.copy()
    feature_df = pd.DataFrame(row_features, index=out.index)
    for column in EXG_COLUMNS:
        out[column] = feature_df[column].astype(float)
    return out


def add_causal_features(daily_df: pd.DataFrame, window_days: int) -> tuple[pd.DataFrame, list[str]]:
    out = daily_df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["planting_date"] = pd.to_datetime(out["planting_date"]).dt.normalize()
    out["days_since_planting"] = (
        (out["date"] - out["planting_date"]).dt.days.astype(float) / 365.0
    )

    rolling_columns = []
    parts = []
    for _, group in out.groupby("station_year", sort=False):
        group = group.sort_values("date").copy()
        dates = pd.DatetimeIndex(group["date"])
        for column in ROLLING_SOURCE_COLUMNS:
            values = pd.Series(
                pd.to_numeric(group[column], errors="coerce").fillna(0.0).to_numpy(),
                index=dates,
            )
            rolling = values.rolling(f"{window_days}D", min_periods=1)
            for suffix, result in (
                ("mean", rolling.mean()),
                ("std", rolling.std().fillna(0.0)),
                ("min", rolling.min()),
                ("max", rolling.max()),
                ("delta", rolling.apply(lambda x: float(x[-1] - x[0]), raw=True)),
            ):
                name = f"{column}_causal_{window_days}d_{suffix}"
                group[name] = result.to_numpy(dtype=np.float32)
                if name not in rolling_columns:
                    rolling_columns.append(name)
        parts.append(group)
    out = pd.concat(parts).sort_index()

    feature_columns = [
        "days_since_planting",
        *LOCATION_FEATURE_COLUMNS,
        *WEATHER_TEMPORAL_FEATURE_COLUMNS,
        *EXG_COLUMNS,
        *rolling_columns,
    ]
    for column in feature_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype(np.float32)
    return out, feature_columns


def classification_metrics(y_true, y_pred, classes: list[str]) -> dict:
    labels = np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "plus_minus_one_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mean_absolute_stage_error": float(mean_absolute_error(y_true, y_pred)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(classes)
        },
    }


def _fit_fold_task(task):
    fold_id, train_idx, val_idx, test_idx, x, y, classes, config = task
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    best = None
    # Folds are parallelized at the process level. Limiting native libraries to
    # one thread per process avoids exponential OpenMP/BLAS oversubscription.
    with threadpool_limits(limits=1):
        for max_iter in config["max_iter_candidates"]:
            model = HistGradientBoostingClassifier(
                learning_rate=config["learning_rate"],
                max_iter=max_iter,
                max_leaf_nodes=config["max_leaf_nodes"],
                l2_regularization=config["l2_regularization"],
                class_weight="balanced",
                early_stopping=False,
                random_state=config["seed"],
            )
            model.fit(x_train, y_train)
            val_pred = model.predict(x_val)
            score = f1_score(
                y_val,
                val_pred,
                labels=np.arange(len(classes)),
                average="macro",
                zero_division=0,
            )
            if best is None or score > best[0]:
                best = (float(score), max_iter, model)
        val_macro_f1, selected_iter, model = best
        test_pred = model.predict(x[test_idx]).astype(int)
    metrics = classification_metrics(y[test_idx], test_pred, classes)
    return {
        "fold_id": fold_id,
        "test_idx": test_idx,
        "model": model,
        "test_pred": test_pred,
        "metrics": metrics,
        "validation_macro_f1": val_macro_f1,
        "selected_iter": selected_iter,
    }


def train(args) -> None:
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [name for name in BASE_CLASSES if not (args.exclude_offseason and name == "OffSeason")]
    daily = build_multiscale_daily_dataframe(
        args.label_path,
        args.data_path,
        include_preplant_days=args.preplant_days,
        include_postharvest_days=args.postharvest_days,
        transition_days=args.transition_days,
        date_tolerance_days=args.date_tolerance_days,
        classes=BASE_CLASSES,
        preferred_camera=None if args.camera.upper() == "ALL" else args.camera,
        use_status_csv=not args.ignore_status_csv,
    )
    if daily.empty:
        raise RuntimeError("No daily image rows were found.")

    weather_cache = args.weather_cache or str(out_dir / "meteostat_weather_cache.csv")
    daily = add_weather_metadata(
        daily,
        weather_cache,
        force_refresh=args.weather_force_refresh,
        gdd_base_temp=args.gdd_base_temp,
    )
    missing_columns = [column for column in WEATHER_TEMPORAL_FEATURE_COLUMNS if column.endswith("_missing")]
    if (
        not args.allow_missing_weather
        and missing_columns
        and daily[missing_columns].to_numpy(dtype=float).mean() >= 0.999
    ):
        raise RuntimeError(
            "Weather is unavailable for every row. Supply a populated --weather-cache, "
            "allow Meteostat access, or explicitly use --allow-missing-weather."
        )
    daily = add_location_metadata(daily, strict=True)
    daily = add_exg_features(
        daily,
        str(out_dir / "exg_features.csv"),
        stream=args.stream,
        max_side=args.exg_max_side,
        workers=args.exg_workers,
    )
    daily, feature_columns = add_causal_features(daily, args.window_days)
    daily.to_csv(out_dir / "metadata_exg_daily.csv", index=False)

    targets = daily[daily["label"].isin(classes)].copy().reset_index(drop=True)
    if args.expected_stations:
        discovered = targets["station_code"].astype(str).nunique()
        if discovered != args.expected_stations:
            raise ValueError(
                f"Expected {args.expected_stations} stations, discovered {discovered}."
            )

    folds = generate_loso_train_val_test_folds(
        targets,
        group_col="station_code",
        n_val=args.validation_groups,
        random_state=args.fold_seed,
    )
    class_to_idx = {name: i for i, name in enumerate(classes)}
    targets["target_idx"] = targets["label"].map(class_to_idx).astype(int)
    predictions = []
    fold_summaries = []
    x_all = targets[feature_columns].to_numpy(np.float32)
    y_all = targets["target_idx"].to_numpy()
    fold_config = {
        "max_iter_candidates": args.max_iter_candidates,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "l2_regularization": args.l2_regularization,
        "seed": args.seed,
    }
    fold_tasks = [
        (fold_id, train_idx, val_idx, test_idx, x_all, y_all, classes, fold_config)
        for fold_id, (train_idx, val_idx, test_idx) in enumerate(folds, start=1)
        if not args.only_fold or fold_id == args.only_fold
    ]
    if not fold_tasks:
        raise RuntimeError("No folds matched --only-fold.")
    worker_count = resolve_worker_count(args.fold_workers, len(fold_tasks))
    print(
        f"Training {len(fold_tasks)} LOSO folds with {worker_count} parallel processes",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        fold_results = executor.map(_fit_fold_task, fold_tasks)

        for result in fold_results:
            fold_id = result["fold_id"]
            test_idx = result["test_idx"]
            model = result["model"]
            test_pred = result["test_pred"]
            metrics = result["metrics"]
            val_macro_f1 = result["validation_macro_f1"]
            selected_iter = result["selected_iter"]
            test_df = targets.iloc[test_idx]

            fold_dir = out_dir / f"fold_{fold_id}"
            fold_dir.mkdir(exist_ok=True)
            with open(fold_dir / "best_model.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "model": model,
                        "classes": classes,
                        "feature_columns": feature_columns,
                        "window_days": args.window_days,
                        "stream": args.stream,
                        "selected_max_iter": selected_iter,
                        "validation_macro_f1": val_macro_f1,
                    },
                    handle,
                )
            with open(fold_dir / "metrics.json", "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)

            fold_predictions = test_df[
                ["station_year", "station_code", "date", "label"]
            ].copy()
            fold_predictions["fold"] = fold_id
            fold_predictions["pred_idx"] = test_pred
            fold_predictions["pred_label"] = [classes[index] for index in test_pred]
            fold_predictions.to_csv(fold_dir / "test_predictions.csv", index=False)
            predictions.append(fold_predictions)
            fold_summaries.append(
                {
                    "fold": fold_id,
                    "test_station": sorted(test_df["station_code"].astype(str).unique().tolist()),
                    "validation_macro_f1": val_macro_f1,
                    "selected_max_iter": selected_iter,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"confusion_matrix", "per_class"}
                    },
                }
            )
            print(
                f"fold={fold_id} test={fold_summaries[-1]['test_station']} "
                f"val_macro_f1={val_macro_f1:.4f} test_macro_f1={metrics['macro_f1']:.4f}",
                flush=True,
            )

    if not predictions:
        raise RuntimeError("No folds were trained.")
    combined = pd.concat(predictions, ignore_index=True)
    combined.to_csv(out_dir / "loso_test_predictions.csv", index=False)
    y_true = combined["label"].map(class_to_idx).to_numpy()
    y_pred = combined["pred_idx"].to_numpy()
    aggregate = classification_metrics(y_true, y_pred, classes)
    aggregate["folds"] = fold_summaries
    aggregate["feature_columns"] = feature_columns
    aggregate["configuration"] = vars(args)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(
        f"aggregate accuracy={aggregate['accuracy']:.4f} "
        f"macro_f1={aggregate['macro_f1']:.4f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-path", default="labeling_bbch_iso_dates.csv")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--out-dir", default="results_metadata_exg_bbch")
    parser.add_argument("--weather-cache", default=None)
    parser.add_argument("--weather-force-refresh", action="store_true")
    parser.add_argument("--allow-missing-weather", action="store_true")
    parser.add_argument("--gdd-base-temp", type=float, default=0.0)
    parser.add_argument("--stream", choices=["micro", "macro"], default="micro")
    parser.add_argument("--camera", default="AUTO")
    parser.add_argument("--ignore-status-csv", action="store_true")
    parser.add_argument("--exclude-offseason", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preplant-days", type=int, default=30)
    parser.add_argument("--postharvest-days", type=int, default=30)
    parser.add_argument("--transition-days", type=int, default=2)
    parser.add_argument("--date-tolerance-days", type=int, default=5)
    parser.add_argument("--window-days", type=int, default=21)
    parser.add_argument("--exg-max-side", type=int, default=1024)
    parser.add_argument("--exg-workers", type=int, default=0, help="ExG worker processes; 0 uses every CPU core.")
    parser.add_argument("--fold-workers", type=int, default=0, help="Parallel LOSO folds; 0 uses every CPU core.")
    parser.add_argument("--validation-groups", type=int, default=2)
    parser.add_argument("--expected-stations", type=int, default=0)
    parser.add_argument("--only-fold", type=int, default=0)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter-candidates", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
