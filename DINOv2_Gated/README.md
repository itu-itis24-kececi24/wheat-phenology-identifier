# DINOv2 Gated Fusion

This folder is a copy of the DINOv2 temporal pipeline with one important change:

See `../DINOV2_ARCHITECTURE_REVIEW.md` for the focused code review, corrected evaluation protocol, and ranked accuracy experiments.

```text
1X macro/canopy feature
10X micro/close-up feature
weather + days-since-planting metadata
        -> gated macro/micro fusion
        -> temporal Transformer/LSTM/GRU
        -> phenology stage
```

This folder defaults to the gated two-stream setup:

```text
--stream both
--tile-streams both
```

Single-stream runs still work if you explicitly pass `--stream micro` or `--stream macro`, but they behave like the normal DINOv2 pipeline.

## Why This Exists

The goal is to let the model learn which view matters at each stage. Early stages may rely more on close-up crop texture, later stages may need spike/organ detail, and harvest/off-season context may benefit from canopy-level 1X information.

The fusion block computes a softmax gate per day. It starts at a neutral 50/50 split and masks unavailable streams before softmax:

```text
macro_weight + micro_weight = 1
```

Then it combines the two streams:

```text
fused = macro_weight * macro_feature + micro_weight * micro_feature
```

The gate can also see temporal metadata, so weather and days-since-planting can help decide whether macro or micro evidence should dominate.

Missing 1X or 10X images no longer invalidate the entire day. If one stream is available, its gate weight becomes 1.0. Optional modality dropout can make the model less dependent on either camera:

```text
--modality-dropout 0.1
```

Use physical-station-held-out evaluation by leaving `--fold-group-by station` enabled. Use `--fold-group-by station_year` only for comparison with older runs.

## Precompute Both Streams

From the project root:

```bash
python3 DINOv2_Gated/precompute_multiscale_embeddings.py \
  --label-path labeling3.csv \
  --data-path data \
  --out-dir results_dinov2_gated \
  --image-backbone facebook/dinov2-base \
  --camera AUTO \
  --stream both \
  --tile-streams both \
  --tile-pooling attention \
  --tile-size 224 \
  --tile-stride 112 \
  --max-tiles 0 \
  --preplant-days 30 \
  --postharvest-days 30 \
  --date-tolerance-days 2 \
  --batch-size 64 \
  --num-workers 4 \
  --embedding-dtype float16 \
  --pretrained \
  --amp auto \
  --device cuda
```

## Train With Weather Metadata

Weather metadata includes normalized average/min/max temperature, precipitation,
daily GDD, and cumulative GDD since planting. The metadata CSV also saves
`weather_gdd_cum_raw` so you can inspect the actual accumulated heat units.

```bash
python3 DINOv2_Gated/run_multiscale_training.py \
  --label-path labeling3.csv \
  --data-path data \
  --out-dir results_dinov2_gated \
  --embedding-cache results_dinov2_gated/vit_embeddings.pt \
  --epochs 30 \
  --folds 7 \
  --window-days 31 \
  --window-mode causal \
  --stream both \
  --preplant-days 30 \
  --postharvest-days 30 \
  --date-tolerance-days 2 \
  --batch-size 16 \
  --accumulation-steps 2 \
  --num-workers 0 \
  --camera AUTO \
  --device cuda \
  --temporal-model transformer \
  --temporal-aggregation cls \
  --temporal-norm-first \
  --loss hybrid \
  --ordinal-ce-weight 0.5 \
  --use-days-since-planting \
  --use-weather-metadata \
  --weather-cache results_dinov2_gated/meteostat_weather_cache.csv \
  --dropout 0.3 \
  --weight-decay 3e-4 \
  --log-interval 50
```

For a quick smoke test, change:

```bash
--epochs 2 --folds 1 --batch-size 4 --accumulation-steps 1
```

## Cached Two-Stream Inference

Gated cached inference accepts separate 1X and 10X histories. Do not pass one image as both camera views:

```bash
python3 DINOv2_Gated/infer_cached_window.py \
  --checkpoint results_dinov2_gated/fold_1/best_model.pt \
  --embedding-cache results_dinov2_gated/vit_embeddings.pt \
  --macro-path path/to/target-1X.jpeg \
  --micro-path path/to/target-10X.jpeg \
  --macro-dir path/to/1X \
  --micro-dir path/to/10X \
  --planting-date 2016-11-12 \
  --stream both \
  --device cuda
```

## Notes

- This model needs both 1X and 10X embeddings. Use `--stream both` in both precompute and training.
- `--tile-streams both` keeps detail in both views but increases cache time and size.
- A target day is usable when at least one requested stream exists; the gate masks the missing stream.
- Compare this against the 10X-only DINOv2 baseline using the same folds and label file.

## Metrics

Training and test outputs include accuracy, date-window accuracy, macro F1,
quadratic weighted kappa, mean absolute stage error, and per-class
precision/recall/F1. Per-fold and combined test reports are saved as:

```text
test_classification_metrics.json
test_per_class_metrics.csv
all_test_classification_metrics.json
all_test_per_class_metrics.csv
```
