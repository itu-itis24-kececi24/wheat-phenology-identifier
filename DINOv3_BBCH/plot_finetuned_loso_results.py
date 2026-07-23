"""Create publication-ready diagnostic plots for completed LOSO folds.

The Taylor diagram treats BBCH class indices as an ordinal sequence. It compares
the predicted and true stage trajectories within each held-out station. This is
a diagnostic complement to classification metrics, not a replacement for
macro-F1, exact accuracy, or the confusion matrix.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_ORDER = ["BBCH0", "BBCH1", "BBCH2", "BBCH3", "BBCH5", "BBCH6_7", "BBCH8"]
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Taylor and box-and-whisker diagnostics for LOSO results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Experiment root containing temporal/run_fold_* or run_fold_*.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <results-dir>/figures).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution.",
    )
    return parser.parse_args()


def temporal_root(results_dir: Path) -> Path:
    candidate = results_dir / "temporal"
    return candidate if candidate.is_dir() else results_dir


def fold_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"fold_(\d+)", part)
        if match:
            return int(match.group(1))
        match = re.fullmatch(r"run_fold_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer fold number from {path}")


def physical_station(station_year: object) -> str:
    text = str(station_year)
    return text.rsplit("_", 1)[0] if "_" in text else text


def load_fold_outputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = temporal_root(results_dir)
    prediction_paths = sorted(root.glob("run_fold_*/fold_*/test_predictions.csv"))
    metric_paths = sorted(root.glob("run_fold_*/fold_*/test_metrics.json"))

    if not prediction_paths:
        aggregate = root / "aggregate"
        prediction_path = aggregate / "all_test_predictions.csv"
        metric_path = aggregate / "all_test_metrics.csv"
        if not prediction_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(
                "No per-fold outputs or complete aggregate CSV files were found under "
                f"{root}"
            )
        predictions = pd.read_csv(prediction_path)
        metrics = pd.read_csv(metric_path)
    else:
        prediction_frames: list[pd.DataFrame] = []
        for path in prediction_paths:
            fold = fold_from_path(path)
            frame = pd.read_csv(path)
            frame["fold"] = fold
            prediction_frames.append(frame)
        predictions = pd.concat(prediction_frames, ignore_index=True)

        metric_rows: list[dict[str, object]] = []
        for path in metric_paths:
            row = json.loads(path.read_text(encoding="utf-8"))
            row.setdefault("fold", fold_from_path(path))
            metric_rows.append(row)
        if not metric_rows:
            raise FileNotFoundError(f"No test_metrics.json files found under {root}")
        metrics = pd.DataFrame(metric_rows)

    required_prediction_columns = {
        "fold",
        "station_year",
        "true_idx",
        "pred_idx",
        "date_window_score",
    }
    missing = required_prediction_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction files are missing columns: {sorted(missing)}")

    predictions = predictions.copy()
    predictions["fold"] = pd.to_numeric(predictions["fold"], errors="raise").astype(int)
    predictions["station"] = predictions["station_year"].map(physical_station)
    dedup_columns = ["fold", "station_year"]
    if "date" in predictions.columns:
        dedup_columns.append("date")
    predictions = predictions.drop_duplicates(dedup_columns, keep="last")

    metrics = metrics.copy()
    metrics["fold"] = pd.to_numeric(metrics["fold"], errors="raise").astype(int)
    metrics = metrics.drop_duplicates("fold", keep="last").sort_values("fold")
    sort_columns = ["fold", "station_year"]
    if "date" in predictions.columns:
        sort_columns.append("date")
    return predictions.sort_values(sort_columns), metrics


def station_taylor_statistics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (fold, station), frame in predictions.groupby(["fold", "station"], sort=True):
        true = frame["true_idx"].to_numpy(dtype=np.float64)
        pred = frame["pred_idx"].to_numpy(dtype=np.float64)
        true_std = float(np.std(true, ddof=0))
        pred_std = float(np.std(pred, ddof=0))
        if len(frame) < 2 or true_std <= 0.0 or pred_std <= 0.0:
            continue
        correlation = float(np.corrcoef(true, pred)[0, 1])
        correlation = float(np.clip(correlation, -1.0, 1.0))
        centered_error = (pred - pred.mean()) - (true - true.mean())
        normalized_crmse = float(np.sqrt(np.mean(centered_error**2)) / true_std)
        rows.append(
            {
                "fold": int(fold),
                "station": str(station),
                "samples": int(len(frame)),
                "correlation": correlation,
                "predicted_to_observed_std_ratio": pred_std / true_std,
                "normalized_centered_rmse": normalized_crmse,
                "accuracy": float(np.mean(pred == true)),
                "mean_absolute_stage_error": float(np.mean(np.abs(pred - true))),
                "date_window_accuracy": float(frame["date_window_score"].mean()),
            }
        )
    if not rows:
        raise ValueError("No station had enough class variation for a Taylor diagram")
    return pd.DataFrame(rows).sort_values("fold")


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi)
    plt.close(fig)


def plot_taylor(stats: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    correlations_observed = np.clip(stats["correlation"].to_numpy(), 0.0, 1.0)
    theta = np.arccos(correlations_observed)
    radius = stats["predicted_to_observed_std_ratio"].to_numpy()
    accuracy = stats["accuracy"].to_numpy()
    color_min = max(0.7, float(accuracy.min()) - 0.02)
    color_max = 1.0

    fig = plt.figure(figsize=(9.0, 4.9))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    ax = fig.add_subplot(grid[0, 0], projection="polar")
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim(0.0, 1.5)
    correlations = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_xticks(np.arccos(correlations))
    ax.set_xticklabels([f"{value:g}" for value in correlations])
    ax.grid(color="#B0BEC5", alpha=0.45, linewidth=0.7)

    contour_theta = np.linspace(0.0, np.pi / 2.0, 180)
    contour_radius = np.linspace(0.0, 1.5, 180)
    theta_grid, radius_grid = np.meshgrid(contour_theta, contour_radius)
    crmse = np.sqrt(
        np.maximum(
            0.0,
            1.0 + radius_grid**2 - 2.0 * radius_grid * np.cos(theta_grid),
        )
    )
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    contours = ax.contour(
        theta_grid,
        radius_grid,
        crmse,
        levels=levels,
        colors="#7B8794",
        linewidths=0.65,
        linestyles="dashed",
    )
    ax.clabel(contours, inline=True, fontsize=7, fmt="nCRMSD %.1f")
    ax.scatter([0.0], [1.0], marker="*", s=145, color="#D55E00", label="Reference", zorder=5)
    ax.scatter(
        theta,
        radius,
        c=accuracy,
        cmap="cividis",
        vmin=color_min,
        vmax=color_max,
        s=95,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
    )
    for point_theta, point_radius, fold in zip(theta, radius, stats["fold"]):
        ax.text(
            point_theta,
            point_radius,
            str(int(fold)),
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="white",
            zorder=5,
        )
    ax.text(
        0.04,
        0.93,
        "(a)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    ax.text(0.53, 1.04, "Correlation", transform=ax.transAxes, ha="center", fontsize=9)
    ax.text(
        -0.13,
        0.5,
        "Standard deviation ratio",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        fontsize=9,
    )

    detail_ax = fig.add_subplot(grid[0, 1])
    correlation_min = max(0.0, float(correlations_observed.min()) - 0.003)
    ratio_min = max(0.0, float(radius.min()) - 0.025)
    ratio_max = float(radius.max()) + 0.025
    detail_correlation = np.linspace(correlation_min, 1.0, 220)
    detail_ratio = np.linspace(ratio_min, ratio_max, 220)
    corr_grid, ratio_grid = np.meshgrid(detail_correlation, detail_ratio)
    detail_crmse = np.sqrt(
        np.maximum(0.0, 1.0 + ratio_grid**2 - 2.0 * ratio_grid * corr_grid)
    )
    detail_contours = detail_ax.contour(
        corr_grid,
        ratio_grid,
        detail_crmse,
        levels=[0.10, 0.15, 0.20, 0.25],
        colors="#7B8794",
        linewidths=0.7,
        linestyles="dashed",
    )
    detail_ax.clabel(detail_contours, inline=True, fontsize=7, fmt="nCRMSD %.2f")
    detail_points = detail_ax.scatter(
        correlations_observed,
        radius,
        c=accuracy,
        cmap="cividis",
        vmin=color_min,
        vmax=color_max,
        s=125,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
    )
    detail_ax.scatter([1.0], [1.0], marker="*", s=145, color="#D55E00", zorder=5)
    for point_correlation, point_radius, fold in zip(
        correlations_observed, radius, stats["fold"]
    ):
        detail_ax.text(
            point_correlation,
            point_radius,
            str(int(fold)),
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="white",
            zorder=5,
        )
    detail_ax.set_xlim(correlation_min, 1.001)
    detail_ax.set_ylim(ratio_min, ratio_max)
    detail_ax.set_xlabel("Correlation")
    detail_ax.set_ylabel("Predicted / observed standard deviation")
    detail_ax.set_title("(b) High-skill detail", pad=8)
    detail_ax.grid(color="#B0BEC5", alpha=0.35)

    colorbar_ax = fig.add_axes([0.905, 0.31, 0.018, 0.45])
    colorbar = fig.colorbar(detail_points, cax=colorbar_ax)
    colorbar.set_label("Exact accuracy")

    station_entries = [
        f"{int(row.fold)}: {row.station}" for row in stats.itertuples(index=False)
    ]
    station_key = "   ".join(station_entries[:7]) + "\n" + "   ".join(station_entries[7:])
    fig.text(
        0.47,
        0.055,
        station_key,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#263238",
        wrap=True,
    )
    fig.text(
        0.47,
        0.018,
        "Ordinal-stage diagnostic; each point is one held-out station.",
        ha="center",
        fontsize=8,
        color="#455A64",
    )
    fig.suptitle(
        "Fine-tuned DINOv3 LOSO: normalized Taylor diagram",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.06, right=0.875, top=0.84, bottom=0.18, wspace=0.32)
    save_figure(fig, out_dir, "finetuned_loso_taylor_diagram", dpi)


def metric_column(metrics: pd.DataFrame, name: str) -> pd.Series:
    if name not in metrics.columns:
        raise ValueError(f"Metric column {name!r} is missing from test_metrics.json")
    return pd.to_numeric(metrics[name], errors="raise")


def plot_box_whisker(
    metrics: pd.DataFrame,
    stats: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    metric_specs = [
        ("Exact accuracy", "test_accuracy"),
        ("Macro-F1", "test_macro_f1"),
        ("Date-window", "test_date_window_accuracy"),
        ("Quadratic kappa", "test_quadratic_weighted_kappa"),
    ]
    values = [metric_column(metrics, column).to_numpy() for _, column in metric_specs]
    labels = [label for label, _ in metric_specs]
    station_by_fold = stats.set_index("fold")["station"].to_dict()

    fig, (ax, error_ax) = plt.subplots(
        1,
        2,
        figsize=(8.2, 3.8),
        gridspec_kw={"width_ratios": [3.5, 1.0]},
        constrained_layout=True,
    )
    box = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "#D55E00", "markeredgecolor": "white"},
        medianprops={"color": "#263238", "linewidth": 1.5},
        whiskerprops={"color": "#546E7A"},
        capprops={"color": "#546E7A"},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#999999"},
    )
    for patch, color in zip(box["boxes"], OKABE_ITO):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    rng = np.random.default_rng(42)
    folds = metrics["fold"].to_numpy(dtype=int)
    for index, metric_values in enumerate(values, start=1):
        jitter = rng.uniform(-0.09, 0.09, size=len(metric_values))
        ax.scatter(
            index + jitter,
            metric_values,
            s=16,
            color="#263238",
            alpha=0.65,
            linewidth=0,
            zorder=3,
        )
        median = float(np.median(metric_values))
        ax.text(index, median + 0.012, f"{median:.3f}", ha="center", fontsize=7.5)
    ax.set_ylim(0.55, 1.015)
    ax.set_ylabel("Held-out fold score")
    ax.set_title("Across-station score distributions")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", color="#B0BEC5", alpha=0.35)

    mae = metric_column(metrics, "test_mean_absolute_stage_error").to_numpy()
    error_box = error_ax.boxplot(
        [mae],
        tick_labels=["Stage MAE"],
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "#0072B2", "markeredgecolor": "white"},
        medianprops={"color": "#263238", "linewidth": 1.5},
    )
    error_box["boxes"][0].set_facecolor("#E69F00")
    error_box["boxes"][0].set_alpha(0.75)
    jitter = rng.uniform(-0.07, 0.07, size=len(mae))
    error_ax.scatter(1 + jitter, mae, s=19, color="#263238", alpha=0.7, linewidth=0)
    for x, y, fold in zip(1 + jitter, mae, folds):
        station = station_by_fold.get(int(fold), f"fold {fold}")
        if y >= np.quantile(mae, 0.85):
            error_ax.annotate(
                station,
                (x, y),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=7,
            )
    error_ax.set_ylim(0.0, max(0.25, float(mae.max()) + 0.025))
    error_ax.set_ylabel("Ordinal stages (lower is better)")
    error_ax.set_title("Error distribution")
    error_ax.grid(axis="y", color="#B0BEC5", alpha=0.35)

    fig.suptitle("Fine-tuned DINOv3 LOSO fold variability", fontsize=12, fontweight="bold")
    save_figure(fig, out_dir, "finetuned_loso_box_whisker", dpi)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    out_dir = (args.out_dir or results_dir / "figures").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions, metrics = load_fold_outputs(results_dir)
    stats = station_taylor_statistics(predictions)
    stats.to_csv(out_dir / "finetuned_loso_taylor_statistics.csv", index=False)
    metrics.to_csv(out_dir / "finetuned_loso_fold_metrics.csv", index=False)

    set_publication_style()
    plot_taylor(stats, out_dir, args.dpi)
    plot_box_whisker(metrics, stats, out_dir, args.dpi)
    print(f"Loaded {len(predictions)} predictions from {predictions.fold.nunique()} folds")
    print(f"Saved plots and source tables to {out_dir}")


if __name__ == "__main__":
    main()
