# DINOv3 BBCH Linear

This pipeline is the single-day counterpart to `DINOv3_BBCH`.

```text
image tiles
  -> fine-tuned DINOv3 dense descriptors
  -> patch attention
  -> tile attention
  -> concatenate [cumulative GDD, latitude, longitude, elevation]
  -> LayerNorm + Dropout + one Linear BBCH classifier
```

It contains no temporal window, temporal Transformer, RNN, or causal image
sequence. “Linear” refers to the post-fusion classifier; the selected dense
patch/tile attention and fine-tuned ViT remain learned nonlinear components.

## Train one LOSO fold

```powershell
.\.venv\Scripts\python.exe DINOv3_BBCH_Linear\run_linear_training.py `
  --label-path labeling_bbch_iso_dates.csv `
  --data-path data `
  --out-dir results_dinov3_bbch_linear `
  --weather-cache results_dinov3_bbch_linear\meteostat_weather_cache.csv `
  --fold-id 1 `
  --expected-stations 15 `
  --epochs 8 `
  --stream micro `
  --device cuda
```

Use `--fold-id 0` (the default) to train every LOSO fold sequentially. Each fold
starts from the selected pretrained DINOv3 model, fine-tunes the final block and
final norm, selects `best_model.pt` using validation macro-F1, and evaluates its
test station once.

Default metadata is deliberately fixed to normalized cumulative GDD plus
normalized latitude, longitude, and elevation. Unknown station locations fail
strictly. A run also fails if weather is unavailable for every row; use
`--allow-missing-weather` only for pipeline smoke testing.

## Single-image inference

```powershell
.\.venv\Scripts\python.exe DINOv3_BBCH_Linear\infer_single_image.py `
  --checkpoint results_dinov3_bbch_linear\fold_1\best_model.pt `
  --image-path data\02.02\2015\K1\10X\example.jpeg `
  --target-date 2015-03-04 `
  --planting-date 2014-10-20 `
  --station-code 02.02 `
  --weather-cache results_dinov3_bbch_linear\meteostat_weather_cache.csv `
  --device cuda
```

Checkpoints contain the full model, DINOv3 architecture config, preprocessing,
tiling, metadata order, class order, fold split, and fine-tuning policy. Inference
therefore reconstructs the model without downloading the original tensor weights.

## Outputs

Each fold saves training history, validation/test predictions, split information,
the validation-selected full checkpoint, test metrics, confusion matrix, and
per-class metrics. Complete or partial runs also save combined predictions and
aggregate metrics. Add `--export-backbone` to save the selected Hugging Face
backbone separately.

## Version compatibility

Training remains compatible with the older shared
`DINOv3_BBCH/ViTBackboneFeatureExtractor` constructor: when no saved
architecture config is being restored, the linear pipeline does not pass the
new `local_config` argument.

Offline inference from a full checkpoint does use `local_config` to reconstruct
the backbone without downloading its original weights. For that path,
`DINOv3_BBCH/multiscale_phenology.py` and `DINOv3_BBCH_Linear` must come from
the same repository version. If they do not, inference now reports an explicit
version-mismatch error instead of Python's unexpected-keyword exception.
