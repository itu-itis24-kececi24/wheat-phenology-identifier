# DINOv2 BBCH Gated

This package combines the revised BBCH interval labels with learned 1X/10X
fusion. It is based on `DINOv2_BBCH`, but a daily sample may contain either or
both cameras and the model learns how much to trust each available view.

## Architecture

For every day in the causal window:

1. Native-resolution 1X and 10X images are split into tiles.
2. Frozen DINOv2 encodes every tile once during precomputation.
3. Separate macro and micro tile-attention modules pool each camera's tiles
   into one daily feature.
4. `GatedMultiviewFusion` projects the two camera features and predicts two
   per-day fusion weights.
5. Missing cameras are hard-masked before softmax. A missing view therefore
   receives exactly zero weight.
6. Days-since-planting and optional weather features are available to both the
   gate and temporal model.
7. A temporal Transformer processes the previous 30 days plus the target day.
8. The classifier predicts the ordered BBCH interval.

The gate starts as a neutral 50/50 mixture when both views exist. During
training, modality dropout occasionally hides one of two available cameras so
the model remains useful when deployment provides only a 10X image. It never
drops the only available camera.

Single-stream experiments remain supported with `--stream micro` or
`--stream macro`; those use the ordinary single-stream temporal model.

## Labels

The default label file is `labeling_bbch_iso_dates.csv`. The expected ordered
classes are:

```text
OffSeason, BBCH0, BBCH1, BBCH2, BBCH3, BBCH5, BBCH6_7, BBCH8
```

CSV and XLSX versions with the same columns are supported. `--exclude-offseason`
removes OffSeason as a target while retaining those days as causal context.

## Sanity Check

Run this before a long cache or training job:

```powershell
python DINOv2_BBCH_Gated/metadata_sanity_check.py `
  --label-path labeling_bbch_iso_dates.csv `
  --data-path data `
  --camera AUTO `
  --stream both
```

`AUTO` follows the `kamera`/`Camera` value in each label row, falling back to
an available camera only when needed.

## Precompute Both Cameras

```powershell
python DINOv2_BBCH_Gated/precompute_multiscale_embeddings.py `
  --excel-path labeling_bbch_iso_dates.csv `
  --data-path data `
  --out-dir results_dinov2_bbch_gated_cache `
  --camera AUTO `
  --stream both `
  --tile-streams both `
  --tile-pooling attention `
  --tile-size 224 `
  --tile-stride 224 `
  --max-tiles 0 `
  --batch-size 64 `
  --num-workers 4 `
  --embedding-dtype float16 `
  --pretrained `
  --device cuda
```

The cache stores macro and micro embeddings separately. Reusing a micro-only
cache with `--stream both` is not equivalent: the macro branch would always be
missing.

Overlapping tiles (`--tile-stride 112`) preserve more boundary detail but make
the cache and each training batch substantially larger. Establish a 224-stride
baseline first.

## Promising Training Run

```powershell
python DINOv2_BBCH_Gated/run_multiscale_training.py `
  --excel-path labeling_bbch_iso_dates.csv `
  --data-path data `
  --out-dir results_dinov2_bbch_gated `
  --embedding-cache results_dinov2_bbch_gated_cache/vit_embeddings.pt `
  --epochs 40 `
  --folds 8 `
  --fold-seed 42 `
  --window-days 31 `
  --window-mode causal `
  --stream both `
  --temporal-model transformer `
  --temporal-aggregation cls `
  --temporal-layers 4 `
  --temporal-heads 8 `
  --batch-size 8 `
  --accumulation-steps 4 `
  --dropout 0.2 `
  --weight-decay 0.0003 `
  --gate-hidden-dim 128 `
  --modality-dropout 0.2 `
  --loss hybrid `
  --ordinal-ce-weight 0.5 `
  --use-days-since-planting `
  --exclude-offseason `
  --monotonic-decoding none `
  --min-train-stage-days 20 `
  --min-train-window-coverage-days 20 `
  --num-workers 4 `
  --device cuda
```

The physical batch size controls peak VRAM. `8 x 4` gives an effective batch
size of 32. Increase the physical batch only when profiling shows that the GPU
is underused; do not change the effective batch during an A/B comparison.

`--min-train-stage-days` and `--min-train-window-coverage-days` affect training
only. Validation and test retain difficult low-coverage station-years so the
reported result still measures deployment behavior.

Before sampling folds, the script writes `split_group_target_eligibility.csv`.
A station group with no active-stage target backed by the selected embedding
cache cannot form a validation or test dataset and is excluded from fold
sampling with an explicit warning. Low-coverage groups that still have at least
one usable target remain eligible for validation and test.

Weather can be added as a separate ablation:

```powershell
  --use-weather-metadata `
  --weather-cache results_dinov2_bbch_gated_cache/meteostat_weather_cache.csv
```

Missing weather values have explicit missingness indicators. Do not mix a
weather-enabled checkpoint with inference that silently omits the weather
cache unless that missing-weather condition was included in validation.

## Cached Inference

With both camera histories:

```powershell
python DINOv2_BBCH_Gated/infer_cached_window.py `
  --checkpoint results_dinov2_bbch_gated/fold_1/best_model.pt `
  --embedding-cache results_dinov2_bbch_gated_cache/vit_embeddings.pt `
  --macro-path path/to/target/1X/image.jpeg `
  --macro-dir path/to/1X `
  --micro-path path/to/target/10X/image.jpeg `
  --micro-dir path/to/10X `
  --planting-date 2016-11-12 `
  --device cuda `
  --debug-window
```

With only the deployment 10X camera:

```powershell
python DINOv2_BBCH_Gated/infer_cached_window.py `
  --checkpoint results_dinov2_bbch_gated/fold_1/best_model.pt `
  --embedding-cache results_dinov2_bbch_gated_cache/vit_embeddings.pt `
  --micro-path path/to/target/10X/image.jpeg `
  --micro-dir path/to/10X `
  --planting-date 2016-11-12 `
  --device cuda
```

The second command does not duplicate 10X embeddings into the 1X branch. The
macro view is marked unavailable and receives zero gate weight.

`--repeat-missing` repeats the target image for absent dates within a supplied
camera stream. The default masked gaps are more faithful to real deployment.

## Resume and Outputs

Every epoch updates `last_checkpoint.pt` with model, optimizer, scaler,
scheduler, epoch, and fold state. Resume with:

```powershell
python DINOv2_BBCH_Gated/run_multiscale_training.py `
  --excel-path labeling_bbch_iso_dates.csv `
  --data-path data `
  --out-dir results_dinov2_bbch_gated `
  --embedding-cache results_dinov2_bbch_gated_cache/vit_embeddings.pt `
  --resume-checkpoint results_dinov2_bbch_gated/last_checkpoint.pt `
  --stream both `
  --device cuda
```

Important outputs include `test_metrics.csv`, per-fold prediction CSV files,
confusion matrices, coverage-stratified metrics, `best_model.pt`, and TensorBoard
logs under `<out-dir>/tensorboard`.

```powershell
tensorboard --logdir results_dinov2_bbch_gated/tensorboard
```

Compare the gated model against a micro-only BBCH run using identical folds,
seed, labels, temporal settings, and effective batch size. The useful result is
not merely whether gated fusion wins overall, but whether it improves macro-F1
and weak-station performance without reducing the 10X-only deployment test.
