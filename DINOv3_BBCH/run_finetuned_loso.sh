#!/usr/bin/env bash
set -Eeuo pipefail

# Fold-safe DINOv3 fine-tuning -> dense cache -> temporal LOSO training.
# Environment overrides:
#   FOLD_START=3 FOLD_END=5 KEEP_CACHES=1 bash DINOv3_BBCH/run_finetuned_loso.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python3}"
LABEL_PATH="${LABEL_PATH:-labeling_truncated3.csv}"
DATA_PATH="${DATA_PATH:-data}"
RESULT_ROOT="${RESULT_ROOT:-results_dinov3_finetuned_loso}"
BACKBONE_ROOT="${BACKBONE_ROOT:-${RESULT_ROOT}/backbones}"
CACHE_ROOT="${CACHE_ROOT:-${RESULT_ROOT}/caches}"
TEMPORAL_ROOT="${TEMPORAL_ROOT:-${RESULT_ROOT}/temporal}"
EXPECTED_STATIONS="${EXPECTED_STATIONS:-15}"
FOLD_START="${FOLD_START:-1}"
FOLD_END="${FOLD_END:-15}"
FOLD_SEED="${FOLD_SEED:-42}"
SEED="${SEED:-42}"
IMAGE_BACKBONE="${IMAGE_BACKBONE:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
# Dense caches are very large. By default they are removed only after the
# corresponding temporal fold has produced final test metrics. Set to 1 to keep.
KEEP_CACHES="${KEEP_CACHES:-0}"

if [[ ! -f "${LABEL_PATH}" ]]; then
  echo "Label file not found: ${LABEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "Data directory not found: ${DATA_PATH}" >&2
  exit 1
fi
if (( FOLD_START < 1 || FOLD_END < FOLD_START || FOLD_END > EXPECTED_STATIONS )); then
  echo "Invalid fold range ${FOLD_START}..${FOLD_END} for ${EXPECTED_STATIONS} stations" >&2
  exit 1
fi
if [[ "${KEEP_CACHES}" != "0" && "${KEEP_CACHES}" != "1" ]]; then
  echo "KEEP_CACHES must be 0 or 1" >&2
  exit 1
fi

mkdir -p "${BACKBONE_ROOT}" "${CACHE_ROOT}" "${TEMPORAL_ROOT}"

echo "Checking Hugging Face backbone access..."
"${PYTHON}" - "${IMAGE_BACKBONE}" <<'PY'
import os
import sys
from pathlib import Path

model_id = sys.argv[1]
if Path(model_id).is_dir():
    config = Path(model_id) / "config.json"
    if not config.is_file():
        raise SystemExit(f"Local IMAGE_BACKBONE has no config.json: {model_id}")
    print(f"Using local backbone: {model_id}")
else:
    from huggingface_hub import HfApi, get_token, whoami

    token = get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token is visible to this Python process. Run `hf auth login` "
            "with the same Linux user and Python environment, or export HF_TOKEN."
        )
    try:
        account = whoami(token=token)["name"]
        HfApi().model_info(model_id, token=token)
    except Exception as exc:
        raise SystemExit(
            f"The active Hugging Face token cannot access {model_id}: "
            f"{type(exc).__name__}: {exc}. Verify gated-model approval and check whether "
            "a stale HF_TOKEN environment variable overrides your saved login."
        ) from exc
    print(f"Authenticated Hugging Face access confirmed for {model_id} as {account}.")
PY

on_error() {
  local exit_code=$?
  echo "Pipeline failed near line ${BASH_LINENO[0]} with exit code ${exit_code}." >&2
  echo "Completed stage markers were preserved; rerun the same command to continue." >&2
  exit "${exit_code}"
}
trap on_error ERR

for fold in $(seq "${FOLD_START}" "${FOLD_END}"); do
  echo
  echo "================================================================"
  echo "LOSO fold ${fold}/${EXPECTED_STATIONS}"
  echo "================================================================"

  backbone_fold_dir="${BACKBONE_ROOT}/fold_${fold}"
  backbone_dir="${backbone_fold_dir}/backbone"
  cache_dir="${CACHE_ROOT}/fold_${fold}"
  cache_path="${cache_dir}/vit_embeddings.pt"
  temporal_run_dir="${TEMPORAL_ROOT}/run_fold_${fold}"
  temporal_fold_dir="${temporal_run_dir}/fold_${fold}"
  temporal_done="${temporal_fold_dir}/test_metrics.json"

  if [[ -f "${backbone_dir}/config.json" && -f "${backbone_dir}/wheat_finetune_metadata.json" ]]; then
    echo "[fold ${fold}] Fine-tuned backbone already exists; skipping."
  else
    echo "[fold ${fold}] Fine-tuning DINOv3 backbone..."
    "${PYTHON}" DINOv3_BBCH/finetune_dinov3_backbone.py \
      --label-path "${LABEL_PATH}" \
      --data-path "${DATA_PATH}" \
      --out-dir "${BACKBONE_ROOT}" \
      --image-backbone "${IMAGE_BACKBONE}" \
      --stream micro \
      --camera AUTO \
      --fold-id "${fold}" \
      --fold-seed "${FOLD_SEED}" \
      --validation-groups 2 \
      --expected-stations "${EXPECTED_STATIONS}" \
      --epochs 8 \
      --batch-size 1 \
      --accumulation-steps 8 \
      --unfreeze-last-blocks 1 \
      --backbone-lr 2e-6 \
      --head-lr 1e-4 \
      --weight-decay 1e-2 \
      --dropout 0.2 \
      --tile-size 224 \
      --tile-stride 224 \
      --max-tiles 16 \
      --vit-image-size 224 \
      --dense-grid-size 2 \
      --dense-include-cls \
      --ordinal-ce-weight 0.5 \
      --exclude-offseason \
      --num-workers 4 \
      --seed "${SEED}" \
      --device cuda
  fi

  if [[ -f "${temporal_done}" ]]; then
    echo "[fold ${fold}] Temporal test metrics already exist; fold is complete."
  else
    if [[ -f "${cache_path}" ]]; then
      echo "[fold ${fold}] Dense embedding cache already exists; skipping precompute."
    else
      echo "[fold ${fold}] Precomputing fold-specific dense embeddings..."
      mkdir -p "${cache_dir}"
      "${PYTHON}" DINOv3_BBCH/precompute_multiscale_embeddings.py \
        --label-path "${LABEL_PATH}" \
        --data-path "${DATA_PATH}" \
        --out-dir "${cache_dir}" \
        --image-backbone "${backbone_dir}" \
        --camera AUTO \
        --stream micro \
        --tile-streams micro \
        --tile-pooling attention \
        --tile-size 224 \
        --tile-stride 112 \
        --max-tiles 0 \
        --vit-image-size 224 \
        --dense-features \
        --dense-grid-size 2 \
        --dense-include-cls \
        --embedding-dtype float16 \
        --batch-size 256 \
        --num-workers 8 \
        --device cuda
    fi

    echo "[fold ${fold}] Training temporal model on generated LOSO fold ${fold}..."
    resume_args=()
    if [[ -f "${temporal_fold_dir}/last_checkpoint.pt" ]]; then
      echo "[fold ${fold}] Resuming temporal checkpoint: ${temporal_fold_dir}/last_checkpoint.pt"
      resume_args=(--resume-checkpoint "${temporal_fold_dir}/last_checkpoint.pt")
    fi
    "${PYTHON}" DINOv3_BBCH/run_multiscale_training.py \
      --label-path "${LABEL_PATH}" \
      --data-path "${DATA_PATH}" \
      --out-dir "${temporal_run_dir}" \
      --embedding-cache "${cache_path}" \
      --epochs 30 \
      --folds "${EXPECTED_STATIONS}" \
      --fold-strategy loso \
      --fold-group-by station \
      --only-fold "${fold}" \
      --expected-stations "${EXPECTED_STATIONS}" \
      --validation-groups 2 \
      --test-groups 1 \
      --fold-seed "${FOLD_SEED}" \
      --seed "${SEED}" \
      --window-days 21 \
      --window-mode causal \
      --date-tolerance-days 5 \
      --min-train-stage-days 8 \
      --min-train-window-coverage-days 12 \
      --stream micro \
      --camera AUTO \
      --batch-size 32 \
      --accumulation-steps 1 \
      --num-workers 2 \
      --temporal-model transformer \
      --temporal-aggregation cls \
      --temporal-layers 2 \
      --temporal-heads 8 \
      --temporal-norm-first \
      --temporal-ffn-multiplier 2 \
      --loss hybrid \
      --ordinal-ce-weight 0.5 \
      --checkpoint-metric macro_f1 \
      --dropout 0.2 \
      --weight-decay 3e-4 \
      --lr 1e-4 \
      --lr-scheduler cosine \
      --warmup-ratio 0.05 \
      --use-days-since-planting \
      --temporal-feature-fusion gated \
      --temporal-feature-hidden-dim 32 \
      --temporal-feature-gate-init 0.1 \
      --use-location-metadata \
      --location-feature-hidden-dim 16 \
      --location-gate-init 0.1 \
      --exclude-offseason \
      --monotonic-decoding none \
      --device cuda \
      --log-interval 50 \
      "${resume_args[@]}"
  fi

  if [[ ! -f "${temporal_done}" ]]; then
    echo "Fold ${fold} did not produce ${temporal_done}; refusing cache cleanup." >&2
    exit 1
  fi
  if [[ "${KEEP_CACHES}" == "0" && -f "${cache_path}" ]]; then
    echo "[fold ${fold}] Removing completed fold cache to reclaim disk: ${cache_path}"
    rm -f -- "${cache_path}"
  fi
done

echo
echo "Aggregating completed per-fold outputs..."
"${PYTHON}" - "${TEMPORAL_ROOT}" "${FOLD_START}" "${FOLD_END}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
fold_start = int(sys.argv[2])
fold_end = int(sys.argv[3])
aggregate_dir = root / "aggregate"
aggregate_dir.mkdir(parents=True, exist_ok=True)

metric_rows = []
prediction_frames = []
validation_frames = []
history_frames = []
missing = []
for fold in range(fold_start, fold_end + 1):
    run_dir = root / f"run_fold_{fold}"
    fold_dir = run_dir / f"fold_{fold}"
    metric_path = fold_dir / "test_metrics.json"
    prediction_path = fold_dir / "test_predictions.csv"
    if not metric_path.is_file() or not prediction_path.is_file():
        missing.append(fold)
        continue
    metric_rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
    predictions = pd.read_csv(prediction_path)
    predictions["fold"] = fold
    prediction_frames.append(predictions)
    val_path = run_dir / "all_val_metrics.csv"
    if val_path.is_file():
        validation_frames.append(pd.read_csv(val_path))
    history_path = fold_dir / "history.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        history["fold"] = fold
        history_frames.append(history)

if missing:
    raise SystemExit(f"Cannot aggregate; missing completed folds: {missing}")

metrics = pd.DataFrame(metric_rows).sort_values("fold")
predictions = pd.concat(prediction_frames, ignore_index=True)
metrics.to_csv(aggregate_dir / "all_test_metrics.csv", index=False)
predictions.to_csv(aggregate_dir / "all_test_predictions.csv", index=False)
if validation_frames:
    pd.concat(validation_frames, ignore_index=True).sort_values("fold").to_csv(
        aggregate_dir / "all_val_metrics.csv", index=False
    )
if history_frames:
    pd.concat(history_frames, ignore_index=True).sort_values(["fold", "epoch"]).to_csv(
        aggregate_dir / "all_history.csv", index=False
    )

classes = ["BBCH0", "BBCH1", "BBCH2", "BBCH3", "BBCH5", "BBCH6_7", "BBCH8"]


def evaluate_prediction_columns(frame, pred_idx_col, pred_label_col, date_score_col):
    true_indices = pd.to_numeric(frame["true_idx"], errors="raise").astype(int)
    pred_indices = pd.to_numeric(frame[pred_idx_col], errors="raise").astype(int)
    true_labels = frame["true_label"].astype(str)
    pred_labels = frame[pred_label_col].astype(str)

    confusion = pd.crosstab(true_labels, pred_labels)
    confusion = confusion.reindex(index=classes, columns=classes, fill_value=0)

    per_class = []
    for label in classes:
        true_is_label = true_labels == label
        pred_is_label = pred_labels == label
        tp = int((true_is_label & pred_is_label).sum())
        fp = int((~true_is_label & pred_is_label).sum())
        fn = int((true_is_label & ~pred_is_label).sum())
        support = int(true_is_label.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    per_class_frame = pd.DataFrame(per_class)
    distance = (true_indices - pred_indices).abs()
    date_scores = pd.to_numeric(frame[date_score_col], errors="coerce")
    summary = {
        "folds": len(metrics),
        "samples": len(frame),
        "accuracy": float((true_indices == pred_indices).mean()),
        "plus_minus_1_accuracy": float((distance <= 1).mean()),
        "mean_absolute_stage_error": float(distance.mean()),
        "macro_f1": float(per_class_frame.f1.mean()),
        "date_window_accuracy": float(date_scores.mean()),
    }
    return confusion, per_class_frame, summary


confusion, per_class_frame, summary = evaluate_prediction_columns(
    predictions,
    pred_idx_col="pred_idx",
    pred_label_col="pred_label",
    date_score_col="date_window_score",
)
confusion.to_csv(aggregate_dir / "all_test_confusion_matrix.csv")
per_class_frame.to_csv(aggregate_dir / "all_test_per_class_metrics.csv", index=False)
summary.update(
    {
        "fold_accuracy_mean": float(metrics.test_accuracy.mean()),
        "fold_accuracy_std": float(metrics.test_accuracy.std(ddof=0)),
        "fold_macro_f1_mean": float(metrics.test_macro_f1.mean()),
        "fold_macro_f1_std": float(metrics.test_macro_f1.std(ddof=0)),
    }
)
(aggregate_dir / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("Raw aggregate:")
print(json.dumps(summary, indent=2))

decoded_columns = {
    "decoded_pred_idx",
    "decoded_pred_label",
    "decoded_date_window_score",
}
present_decoded_columns = decoded_columns.intersection(predictions.columns)
if present_decoded_columns and present_decoded_columns != decoded_columns:
    missing_decoded_columns = sorted(decoded_columns - present_decoded_columns)
    raise SystemExit(
        "Cannot aggregate monotonic predictions; missing decoded columns: "
        f"{missing_decoded_columns}"
    )

if decoded_columns.issubset(predictions.columns):
    monotonic_predictions = predictions.copy()
    monotonic_predictions["raw_pred_idx"] = monotonic_predictions["pred_idx"]
    monotonic_predictions["raw_pred_label"] = monotonic_predictions["pred_label"]
    monotonic_predictions["raw_date_window_score"] = monotonic_predictions[
        "date_window_score"
    ]
    monotonic_predictions["pred_idx"] = monotonic_predictions["decoded_pred_idx"]
    monotonic_predictions["pred_label"] = monotonic_predictions["decoded_pred_label"]
    monotonic_predictions["date_window_score"] = monotonic_predictions[
        "decoded_date_window_score"
    ]
    monotonic_predictions.to_csv(
        aggregate_dir / "all_test_monotonic_predictions.csv",
        index=False,
    )

    monotonic_confusion, monotonic_per_class, monotonic_summary = (
        evaluate_prediction_columns(
            monotonic_predictions,
            pred_idx_col="pred_idx",
            pred_label_col="pred_label",
            date_score_col="date_window_score",
        )
    )
    monotonic_confusion.to_csv(
        aggregate_dir / "all_test_monotonic_confusion_matrix.csv"
    )
    monotonic_per_class.to_csv(
        aggregate_dir / "all_test_monotonic_per_class_metrics.csv",
        index=False,
    )

    if "test_monotonic_accuracy" in metrics:
        monotonic_summary["fold_accuracy_mean"] = float(
            metrics.test_monotonic_accuracy.mean()
        )
        monotonic_summary["fold_accuracy_std"] = float(
            metrics.test_monotonic_accuracy.std(ddof=0)
        )
    if "test_monotonic_macro_f1" in metrics:
        monotonic_summary["fold_macro_f1_mean"] = float(
            metrics.test_monotonic_macro_f1.mean()
        )
        monotonic_summary["fold_macro_f1_std"] = float(
            metrics.test_monotonic_macro_f1.std(ddof=0)
        )
    monotonic_summary["accuracy_delta_vs_raw"] = (
        monotonic_summary["accuracy"] - summary["accuracy"]
    )
    monotonic_summary["macro_f1_delta_vs_raw"] = (
        monotonic_summary["macro_f1"] - summary["macro_f1"]
    )
    monotonic_summary["date_window_accuracy_delta_vs_raw"] = (
        monotonic_summary["date_window_accuracy"]
        - summary["date_window_accuracy"]
    )
    (aggregate_dir / "monotonic_summary.json").write_text(
        json.dumps(monotonic_summary, indent=2),
        encoding="utf-8",
    )
    print("Monotonic aggregate:")
    print(json.dumps(monotonic_summary, indent=2))
else:
    print(
        "No decoded prediction columns found; monotonic aggregate was not generated."
    )

print(f"Combined outputs: {aggregate_dir}")
PY

echo "Fine-tuned LOSO pipeline completed successfully."
