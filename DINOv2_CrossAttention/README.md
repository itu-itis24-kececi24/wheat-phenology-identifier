# DINOv2 Tile Cross-Attention

This folder is a DINOv2 temporal pipeline variant for testing 1X + 10X fusion with cross-attention before tile pooling.

```text
1X macro/canopy tile tokens
10X micro/close-up tile tokens
weather + days-since-planting metadata
        -> bidirectional macro/micro tile cross-attention
        -> attention pooling into one macro vector and one micro vector per day
        -> gated macro/micro fusion
        -> temporal Transformer/LSTM/GRU
        -> phenology stage
```

The cross-attention path is used for cached tiled embeddings with:

```text
--stream both
--tile-streams both
--tile-pooling attention
```

Single-stream runs still work, but they behave like the normal DINOv2 single-stream pipeline. Full-image non-cache runs use vector-level gated fusion because they do not have tile tokens.

## Why This Exists

The gated model fuses one pooled 1X vector with one pooled 10X vector. This model lets the 1X and 10X tiles interact first. For example, canopy context can attend to crop-organ detail, and close-up evidence can attend back to field-level state before the model compresses tiles into daily vectors.

This is more expensive than plain gated fusion, but it is the more direct test of whether 1X context improves 10X phenology classification.

## Precompute Both Tiled Streams

From the project root:

```bash
python3 DINOv2_CrossAttention/precompute_multiscale_embeddings.py \
  --label-path labeling3.csv \
  --data-path data \
  --out-dir results_dinov2_crossattn \
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
  --device cuda
```

Use `--tile-stride 224` for a faster, lighter cache. Use `--tile-stride 112` when you want more spatial detail and can afford the larger cache.

## Train

```bash
python3 DINOv2_CrossAttention/run_multiscale_training.py \
  --label-path labeling3.csv \
  --data-path data \
  --out-dir results_dinov2_crossattn \
  --embedding-cache results_dinov2_crossattn/vit_embeddings.pt \
  --epochs 30 \
  --folds 7 \
  --window-days 31 \
  --window-mode causal \
  --stream both \
  --exclude-offseason \
  --transition-days 5 \
  --date-tolerance-days 2 \
  --batch-size 32 \
  --accumulation-steps 1 \
  --num-workers 0 \
  --camera AUTO \
  --device cuda \
  --temporal-model transformer \
  --temporal-aggregation cls \
  --use-days-since-planting \
  --use-weather-metadata \
  --weather-cache results_dinov2_crossattn/meteostat_weather_cache.csv \
  --dropout 0.2 \
  --weight-decay 3e-4 \
  --log-interval 50
```

For a quick smoke test, change:

```bash
--epochs 2 --folds 1 --batch-size 4 --accumulation-steps 1
```

## Outputs

Checkpoints from cached two-stream runs include:

```text
fusion = tile_cross_attention
```

Training and test outputs include accuracy, date-window accuracy, macro F1, quadratic weighted kappa, mean absolute stage error, per-class precision/recall/F1, and confusion matrices.

Important output files:

```text
brief_history.csv
history.csv
test_metrics.csv
all_test_classification_metrics.json
all_test_per_class_metrics.csv
all_test_confusion_matrix.png
tensorboard/
```

## Notes

- This experiment needs both 1X and 10X tiled embeddings to test cross-view tile interaction.
- If one stream is missing for a date, that timestep is masked when possible. Target samples still require the selected target-day stream to exist.
- Compare against `DINOv2_Gated` using the same label file, fold seed, window settings, weather metadata, and augmentation settings.
