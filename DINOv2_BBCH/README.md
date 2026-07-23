# DINOv2 BBCH Wheat Phenology

This folder is the BBCH-label variant of the reviewed `DINOv2` pipeline. It uses
`labeling_bbch_iso_dates.csv` or `labeling_bbch_iso_dates.xlsx` and predicts seven
ordered wheat-development intervals from causal image windows.

## BBCH Label Mapping

The model uses the milestone dates as interval boundaries:

| Model class | Date interval |
| --- | --- |
| `BBCH0` | `1-Sowing` to `2 - Emergence` |
| `BBCH1` | `2 - Emergence` to `3 - Tillering` |
| `BBCH2` | `3 - Tillering` to `4 - Stem Elongation` |
| `BBCH3` | `4 - Stem Elongation` to `5 - Heading` |
| `BBCH5` | `5 - Heading` to `6 - Flowering` |
| `BBCH6_7` | `6 - Flowering` to `7 - Maturity` |
| `BBCH8` | `7 - Maturity` to `8 - Harvest` |

`OffSeason` is optional and covers the configured period before sowing and after
harvest. The numeric duration columns (`BBCH 0`, `BBCH 1`, and so on) are useful
for auditing the table but are not used as independent model inputs; durations
are derived from the milestone dates.

The classes remain in the table order above for ordinal loss, mean absolute stage
error, plus/minus-one accuracy, and quadratic weighted kappa.

## Architecture

1. Each 10X image is split into overlapping 224-pixel tiles.
2. Frozen `facebook/dinov2-base` converts every tile into an embedding.
3. Learned tile attention pools informative plant details within each day.
4. A causal 31-day window contains the previous 30 days plus the target day.
5. Separate residual MLPs project days-since-sowing and cumulative GDD into the
   daily visual embedding. Learnable gates keep both metadata sources weak at
   initialization without compressing the DINOv2 representation.
6. A temporal Transformer aggregates the resulting causal daily sequence.
7. Optional latitude, longitude, and elevation pass through a small MLP and a
   learnable gated residual connection after temporal aggregation.
8. The hybrid loss combines soft cross entropy with an ordinal CDF loss, so distant
   BBCH errors are penalized more strongly than neighboring errors.

Precomputing caches frozen DINOv2 embeddings. It changes training speed and storage,
not the label mapping or temporal model.

## Files

- `multiscale_phenology.py`: BBCH parsing, metadata, datasets, models, and losses.
- `precompute_multiscale_embeddings.py`: tiled DINOv2 embedding cache creation.
- `run_multiscale_training.py`: grouped cross-validation training and evaluation.
- `metadata_sanity_check.py`: class, camera, path, and station-year checks.
- `infer_cached_window.py`: deployment-style inference from up to 31 causal images.
- `infer_single_image.py`: repeated-single-image diagnostic inference.
- `station_role_sweep.py`: station-role sensitivity experiments.

## Requirements

Use the repository virtual environment and install the root requirements:

```bash
python -m pip install -r requirements.txt
```

The DINOv2 backbone requires `torch`, `torchvision`, and `transformers`. XLSX input
also requires `openpyxl`.

## Metadata Check

From the repository root:

```bash
python DINOv2_BBCH/metadata_sanity_check.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --camera AUTO \
  --stream micro
```

The XLSX version is accepted as well:

```bash
python DINOv2_BBCH/metadata_sanity_check.py \
  --label-path labeling_bbch_iso_dates.xlsx \
  --data-path data
```

Check that all seven BBCH classes appear, camera choices match the `kamera` column,
and the expected station-years have image paths. The revised table still contains
the row marked `fotograflar iyi degil`; remove or explicitly exclude that row if it
should not participate in an experiment.

## Recommended Cached Run

Create detailed overlapping-tile embeddings once:

```bash
python DINOv2_BBCH/precompute_multiscale_embeddings.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --out-dir results_dinov2_bbch_cache \
  --image-backbone facebook/dinov2-base \
  --camera AUTO \
  --stream micro \
  --tile-streams micro \
  --tile-pooling attention \
  --tile-size 224 \
  --tile-stride 112 \
  --max-tiles 0 \
  --batch-size 64 \
  --num-workers 4 \
  --embedding-dtype float16 \
  --augment-views 1 \
  --augment-streams micro \
  --amp auto \
  --pretrained \
  --device cuda
```

Train the temporal classifier:

```bash
python DINOv2_BBCH/run_multiscale_training.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --out-dir results_dinov2_bbch \
  --embedding-cache results_dinov2_bbch_cache/vit_embeddings.pt \
  --epochs 40 \
  --folds 8 \
  --fold-group-by station \
  --validation-groups 2 \
  --test-groups 2 \
  --fold-seed 42 \
  --seed 42 \
  --window-days 31 \
  --window-mode causal \
  --min-train-stage-days 8 \
  --min-train-window-coverage-days 20 \
  --stream micro \
  --camera AUTO \
  --batch-size 32 \
  --accumulation-steps 1 \
  --num-workers 0 \
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
  --use-weather-metadata \
  --weather-feature-set cumulative \
  --weather-feature-gate-init 0.1 \
  --weather-cache results_dinov2_bbch_cache/meteostat_weather_cache.csv \
  --exclude-offseason \
  --monotonic-decoding viterbi \
  --monotonic-advance-penalty 0 \
  --device cuda \
  --log-interval 50
```

Location metadata is static for a station window and is fused only after temporal
aggregation. Latitude, longitude, and elevation are normalized with fixed physical
reference values before entering the MLP. The gate starts at `0.1`, so the visual
and temporal representation dominates initially, while training can increase or
decrease the geographic contribution. The learned gate is written to each fold's
history, validation metrics, test metrics, and TensorBoard logs.

Existing DINOv2 embedding caches can be reused because location is fused by the
temporal classifier rather than the frozen image backbone. A location-enabled run
does require retraining that classifier. For a clean ablation, run the exact same
fold assignments and seeds once without `--use-location-metadata` and once with it.
You can also test location without weather by omitting `--use-weather-metadata` and
`--weather-cache`; this is useful when weather coverage is poor.

`--weather-feature-set cumulative` supplies only normalized cumulative GDD since
planting to the weather MLP. `daily_cumulative` supplies daily and cumulative GDD,
while `full` retains the previous ten-value temperature/precipitation/GDD vector
for ablation experiments. Weather is calculated and audited in the metadata CSV
for every mode, but only the selected columns enter the model.

The gated fusion projects metadata independently and adds it residually to each
daily image embedding. It therefore avoids the old `image + metadata -> 32 -> 512`
bottleneck. `temporal_feature_gate` and `weather_feature_gate` are recorded in fold
history, metrics, and TensorBoard. Old checkpoints remain loadable by inference:
checkpoints without a fusion-version field automatically reconstruct the legacy
concatenation module.

`--monotonic-decoding viterbi` is an evaluation/deployment post-processor, not a
training loss. It decodes each station-year chronologically and permits only staying
in the current stage or advancing. Consecutive calendar days can advance by one
stage; larger image gaps permit a correspondingly larger jump but never a regression.
Raw predictions remain in
`pred_idx`/`pred_label`; constrained predictions are saved in
`decoded_pred_idx`/`decoded_pred_label`. It requires `--exclude-offseason`, because
OffSeason appears both before sowing and after harvest and is therefore not monotonic
in the current class order. `--monotonic-advance-penalty` can discourage overly fast
advancement; start at `0` and tune it using validation only.

The current randomized fold generator writes `fold_assignments.csv`. Always inspect
that file: repeated random test combinations are not the same as exhaustive grouped
cross-validation, even when `--folds 8` is used.

## Training Coverage Filters

Coverage filtering is deliberately training-only:

- `--min-train-stage-days N` removes target samples from a station-year/BBCH class
  when fewer than `N` dates have a usable target image or cached embedding.
- `--min-train-window-coverage-days N` removes a target sample when fewer than `N`
  exact dates in its causal window have the requested image stream. Nearest-date
  fallbacks do not count.
- Validation and test remain unfiltered. Removing sparse cases there would inflate
  reported performance and hide the deployment problem.
- When `--exclude-offseason` is active, OffSeason is excluded as a target but retained
  as historical context for early BBCH0 windows.

Do not begin with a 20-day stage threshold blindly. Some valid wheat stages are
naturally shorter than 20 days. On the current metadata, stage `20` removes 694
otherwise usable targets. A safer first comparison is:

```text
baseline:        stage=0, window=0
coverage only:   stage=0, window=20
combined:        stage=8, window=20
```

Each fold writes `training_coverage_filter.json` with exclusions by station-stage.
The best checkpoint is also evaluated on the complete validation and test datasets,
producing `val_coverage_metrics.csv` and `test_coverage_metrics.csv`. These stratify
accuracy, macro-F1, MAE, and QWK by window coverage and stage support.

## Lightweight Smoke Test

```bash
python DINOv2_BBCH/run_multiscale_training.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --out-dir results_dinov2_bbch_smoke \
  --epochs 1 \
  --folds 1 \
  --validation-groups 2 \
  --test-groups 2 \
  --window-days 3 \
  --window-mode causal \
  --stream micro \
  --batch-size 1 \
  --accumulation-steps 1 \
  --num-workers 0 \
  --camera AUTO \
  --device cuda \
  --log-interval 1
```

## Outputs

Each fold saves its validation-selected `best_model.pt`, resumable
`last_checkpoint.pt`, history, test predictions, per-class metrics, and confusion
matrix. Aggregate files include:

- `all_test_metrics.csv`
- `all_test_per_class_metrics.csv`
- `all_test_predictions.csv`
- `all_test_confusion_matrix.png`
- `all_test_monotonic_confusion_matrix.png` when monotonic decoding is enabled
- `all_test_monotonic_per_class_metrics.csv` when monotonic decoding is enabled
- `all_val_coverage_metrics.csv`
- `all_test_coverage_metrics.csv`
- `fold_assignments.csv`
- `tensorboard/`

Select experiments primarily by validation macro-F1. Report exact accuracy,
date-window accuracy, mean absolute stage error, and quadratic weighted kappa as
complementary metrics. Do not select checkpoints using test results.

## Resume

```bash
python DINOv2_BBCH/run_multiscale_training.py \
  --label-path labeling_bbch_iso_dates.csv \
  --data-path data \
  --out-dir results_dinov2_bbch \
  --embedding-cache results_dinov2_bbch_cache/vit_embeddings.pt \
  --resume-checkpoint results_dinov2_bbch/last_checkpoint.pt \
  --epochs 40 \
  --device cuda
```

Resume with the same architecture, class configuration, fold grouping, and cache
settings used to create the checkpoint.

For inference with a location-enabled checkpoint, provide the station code (for
example `--station-code 02.06`). The code maps its city-family prefix to the
configured coordinate and elevation values.
