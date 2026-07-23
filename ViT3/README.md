# Multi-Scale Wheat Phenology

This folder contains the deployment-oriented temporal implementation for wheat phenology classification.

The default setup in `ViT3` is:

```text
10X close-up images only
previous 30 days + target day
predict the target day
```

This matches a realistic deployment setting where future images are unavailable and users may only provide close-up crop photos.

## Files

- `framework.py`: deprecated compatibility shim. It re-exports the active helpers from `multiscale_phenology.py` for old notebooks.
- `multiscale_phenology.py`: metadata builder, Excel/CSV date normalization, station-folder matching, group folds, datasets, ViT feature encoders, temporal Transformer models, and ordinal/soft-label losses.
- `run_multiscale_training.py`: main training script.
- `precompute_multiscale_embeddings.py`: optional frozen-backbone embedding cache builder.
- `infer_single_image.py`: quick single-image inference helper for full-image checkpoints.
- `wrapper.py`: deprecated preset launcher kept only for compatibility. Prefer the direct script commands below.
- `metadata_sanity_check.py`: small script for checking label counts and 1X/10X path availability.

The active ViT3 pipeline is self-contained in `multiscale_phenology.py` for metadata creation, label mapping, station lookup, image pairing, `OffSeason` windows, soft labels, temporal windows, and model definitions.

## Architecture Overview

The project is a temporal classifier. By default in `ViT3`, it predicts the last day in a causal rolling window.

Input for one sample:

```text
station-year daily sequence
        |
        |-- 1X image for each day  -> macro/canopy stream
        |-- 10X image for each day -> micro/leaf-spikelet stream
```

For example, with `--window-days 31 --window-mode causal --stream micro`, one training sample contains:

```text
31 micro/10X images
```

The label is the stage of the last day:

```text
previous 30 days + target day
```

### Full Image Model

In full-image training, every image is passed through a Vision Transformer during training:

```text
1X image sequence
  -> macro ViT encoder
  -> macro daily feature vectors

10X image sequence
  -> micro ViT encoder
  -> micro daily feature vectors

micro features per day
  -> temporal backend
  -> target-day classifier
  -> OffSeason / PS1 / ... / PS7
```

The default temporal backend is a Transformer, which attends across the previous days in the window so it can use recent growth history without relying on future photos. You can also test PhenoNet-style recurrent backends:

```powershell
--temporal-model transformer
--temporal-model lstm
--temporal-model gru
```

For LSTM/GRU experiments, a good starting point is:

```powershell
--temporal-model lstm --temporal-aggregation target --temporal-layers 2
```

By default, the temporal model prepends a learnable `CLS` token and classifies from that aggregated sequence representation. You can switch this with:

```powershell
--temporal-aggregation cls
--temporal-aggregation mean
--temporal-aggregation target
```

`cls` is the default and is usually the best starting point because it lets the model summarize the whole causal window instead of forcing the final day vector to be the only classifier input.

`cls` is Transformer-specific. LSTM/GRU runs automatically fall back to target-day aggregation if `cls` is left enabled.

### Cached Embedding Model

The cached workflow freezes the image ViT. It first computes image features once:

```text
all 1X/10X JPEGs -> frozen ViT -> saved feature vectors
```

Training then loads feature vectors instead of JPEGs:

```text
cached micro feature sequence
  + days since planting for each date
  -> temporal Transformer
  -> target-day classifier
```

This is much faster, but the image backbone does not learn during training. Use it for quick experiments and use full-image training when you want end-to-end fine-tuning.

### Missing Days

If a date is missing, the dataset can fall back to the nearest available day and marks that timestep in the mask. The temporal Transformer receives the mask so it can distinguish real dates from filled gaps.

### Planting-Date Metadata

By default, each timestep also receives:

```text
days_since_planting / 365
```

The planting date comes from the label table's `1-Ekim` column. Disable this feature for an image-only ablation with:

```powershell
--no-days-since-planting
```

The metadata is fused into the visual sequence through a small MLP. To give the time metadata more capacity, tune:

```powershell
--temporal-feature-hidden-dim 32
```

Use `--temporal-feature-hidden-dim 0` to reproduce the older single-layer metadata fusion.

### Weather Metadata

If `meteostat` is installed, ViT3 can append daily weather features to each timestep:

```text
tavg, tmin, tmax, precipitation, daily GDD, cumulative GDD since planting
```

The values are normalized before entering the temporal Transformer. Enable this with:

```powershell
--use-weather-metadata
```

The first run builds a cache, and later runs reuse it:

```powershell
--weather-cache results_ViT3_weather\meteostat_weather_cache.csv
```

The station coordinates are read from `STATION_COORDINATES` in [multiscale_phenology.py](multiscale_phenology.py). The keys are city plate-code prefixes such as `02`, `06`, `11`, `26`, and `27`.

## Data Layout

The current dataset is expected at the project root:

```text
data/
  02.02/
    2014/
      day_image_status_02.02_2014.csv
      K1/
        1X/
        10X/
      K2/
        1X/
        10X/
```

Supported filename pattern:

```text
02_02-2014_01_01-10_09-K1-10X.jpeg
```

The metadata builder maps:

```text
1X  -> macro/canopy stream
10X -> micro/leaf-spikelet stream
```

By default it uses `--camera AUTO` and trains on the `10X` stream. `AUTO` prefers `K1` for each station-year, but falls back to `K2` or the next available camera when `K1` does not exist. Use `--camera K1`, `--camera K2`, or `--camera ALL` to override this behavior.

## Labels

The metadata builder creates:

```text
OffSeason, PS1, PS2, PS3, PS4, PS5, PS6, PS7
```

It reads phenology dates from `labeling.xlsx`, then:

- keeps 30 days before seeding and the `1-Ekim -> 2-Çıkış` pre-emergence interval as `OffSeason`,
- keeps 30 days after harvest and labels them as `OffSeason`,
- labels `2-Çıkış -> 3-Çimlenme` as `PS1`,
- shifts the remaining stage intervals right by one,
- labels `8-Olgunlaşma -> 9-Hasat` as `PS7`,
- uses soft labels near transition boundaries via `--transition-days`.
- stores date-aware metric scores via `--date-tolerance-days`.

You can change the off-season window with:

```powershell
--preplant-days 30 --postharvest-days 30
```

Date-aware metric credit is controlled with:

```powershell
--date-tolerance-days 7
```

If the model predicts a stage whose real calendar window is close to the image date, it receives partial credit. Inside the predicted stage window gets `1.0`; outside the window decays linearly to `0.0` across the tolerance.

## Environment

Use the same Python environment as your notebooks.

Install minimum dependencies:

```powershell
python -m pip install pandas numpy pillow torch torchvision scikit-learn openpyxl meteostat
```

Check CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

## Metadata Sanity Check

From the project root:

```powershell
cd C:\PATH_TO_DATA_FOLDER
```

Option 1: run the helper script from `ViT3`:

```powershell
cd ViT3
python metadata_sanity_check.py
```

Option 2: run this in Python or a notebook from the project root:

```python
from ViT3.multiscale_phenology import build_multiscale_daily_dataframe

df = build_multiscale_daily_dataframe(
    "labeling.xlsx",
    "data",
    preferred_camera="AUTO",
    include_preplant_days=30,
    include_postharvest_days=30,
)
print(df.head())
print(df["label"].value_counts().sort_index())
print(df[["macro_path", "micro_path"]].notna().mean())
```

A healthy result should include all `PS1` to `PS7`, plus `OffSeason`. The `macro_path` and `micro_path` availability should be high, ideally close to `1.0`.

The label table can be Excel or a CSV export with the same columns:

```powershell
python ViT3\run_multiscale_training.py --excel-path labeling.xlsx ...
python ViT3\run_multiscale_training.py --label-path labeling_iso_dates.csv ...
```

CSV loading accepts comma or semicolon separators and common Turkish/Windows encodings. ISO dates like `2014-12-02` are parsed as `YYYY-MM-DD`.

Expected labels:

```text
OffSeason
PS1
PS2
PS3
PS4
PS5
PS6
PS7
```

## Lightweight Smoke Test

This checks that the training loop works. It is not meant to produce good accuracy.

```powershell
python ViT3\run_multiscale_training.py --excel-path labeling.xlsx --data-path data --out-dir results_smoke_ViT3 --epochs 1 --folds 1 --window-days 3 --window-mode causal --stream micro --preplant-days 30 --postharvest-days 30 --date-tolerance-days 7 --batch-size 1 --accumulation-steps 1 --num-workers 0 --camera AUTO --device cuda --log-interval 1
```

If CUDA is not available, use:

```powershell
--device cpu
```

CPU training will be much slower.

## Fold Split

For `ViT3`, validation/test station counts are edited in [run_multiscale_training.py](run_multiscale_training.py), not passed as command-line arguments:

```python
VALIDATION_FOLD_STATIONS = 2
TEST_FOLD_STATIONS = 2
REQUIRE_DIVERSE_VALIDATION_STATIONS = True
REQUIRE_DIVERSE_TEST_STATIONS = True
```

Training station count is derived automatically:

```text
train = total station groups - validation - test
```

When diversity is enabled and a split has more than one group, the selected groups must include at least two station-code prefixes. For example, `02.02 + 06.01` is allowed, while `02.02 + 02.03` is rejected because both belong to prefix `02`.

Fold combinations are sampled randomly but reproducibly. Change the seed to get a different fold arrangement:

```powershell
--fold-seed 42
--fold-seed 123
--fold-seed 2026
```

The selected split is written to:

```text
<out-dir>/fold_assignments.csv
```

## Recommended Fast Workflow

For practical experiments, freeze the image ViT and precompute image embeddings once.

For `10X` close-up images, use tiled embeddings so the full `2288x1712` image is not squeezed into one `224x224` view. The script crops native-resolution tiles first and runs ViT on each tile. With `--tile-pooling attention`, it stores all tile features and the training model learns which tiles matter.

The tile attention pooler uses lightweight row/column positional embeddings, so tile order carries spatial information instead of treating all patches as locationless crop fragments.

Recommended detail-preserving setup:

```text
original 2288x1712 image
  -> 224x224 native crops
  -> ViT per crop
  -> learned tile attention pooling
  -> one daily 10X feature inside the temporal model
```

Step 1: build cached embeddings:

```powershell
python ViT3\precompute_multiscale_embeddings.py --excel-path labeling.xlsx --data-path data --out-dir results_ViT3_cache --camera AUTO --stream micro --tile-streams micro --tile-pooling attention --tile-size 224 --tile-stride 224 --max-tiles 0 --preplant-days 30 --postharvest-days 30 --date-tolerance-days 7 --batch-size 64 --num-workers 4 --embedding-dtype float16 --pretrained --device cuda
```

Step 2: train from cached vectors:

```powershell
python ViT3\run_multiscale_training.py --excel-path labeling.xlsx --data-path data --out-dir results_ViT3_cache --embedding-cache results_ViT3_cache\vit_embeddings.pt --epochs 5 --folds 1 --window-days 31 --window-mode causal --stream micro --preplant-days 30 --postharvest-days 30 --date-tolerance-days 7 --batch-size 32 --accumulation-steps 1 --num-workers 0 --camera AUTO --device cuda --log-interval 10
```

This is much faster because training no longer decodes JPEGs or runs ViT for every frame during every epoch.

Use this workflow when the image backbone should stay frozen.

Tiling options:

- `--tile-streams micro`: tile only the 10X stream.
- `--tile-size 224`: crop 224x224 regions from the original image.
- `--tile-stride 224`: non-overlapping tiles.
- `--tile-stride 112`: overlapping tiles, more detail coverage but roughly 4x more tiles.
- `--max-tiles 0`: use all tiles.
- `--max-tiles 16`: use a deterministic subset of 16 tiles per image for a faster experiment.
- `--vit-image-size 224`: ViT input size after tile crop.
- `--tile-pooling attention`: stores per-tile features and learns tile importance during temporal training.
- `--tile-pooling mean`: averages tile features during precompute; faster/smaller cache, but less expressive.
- `--embedding-dtype float16`: stores cached embeddings in half precision, cutting cache size and RAM use roughly in half.

Embedding augmentation:

```powershell
--augment-views 2 --augment-streams micro
```

This stores two extra augmented embedding variants per image in `macro_aug` / `micro_aug` inside `vit_embeddings.pt`. The clean embedding is still stored and used for validation/test. During training, augmented caches expand the training dataset by default:

```text
clean + augment view 1 + augment view 2
```

Disable augmented variants during training with:

```powershell
--no-augmented-embeddings
```

Or cap the multiplier manually:

```powershell
--embedding-augmentation-multiplier 2
```

Augmented embedding caches are larger and slower to build. Use mild augmentation first; the defaults apply brightness/contrast/saturation jitter, small rotation, horizontal flip, random crop/resize, and occasional blur.

For tiled precompute, `--num-workers` parallelizes image loading and PIL cropping. If one image is corrupt or unreadable, the script logs it, skips it, and records the failure in the cache metadata instead of crashing the whole run.

For best detail preservation, start with:

```powershell
--tile-pooling attention --tile-size 224 --tile-stride 224 --max-tiles 0
```

For faster experiments:

```powershell
--tile-pooling attention --tile-size 224 --tile-stride 224 --max-tiles 16
```

Rebuild `vit_embeddings.pt` whenever you change:

- `--camera`
- `--data-path`
- `--preplant-days`
- `--postharvest-days`
- tiling settings
- image preprocessing/backbone settings
- augmentation settings

## Full Image Training

This trains through the image model every epoch. It is much slower and uses much more VRAM.

```powershell
python ViT3\run_multiscale_training.py --excel-path labeling.xlsx --data-path data --out-dir results_ViT3_full --epochs 20 --folds 5 --window-days 31 --window-mode causal --stream micro --preplant-days 30 --postharvest-days 30 --date-tolerance-days 7 --batch-size 2 --accumulation-steps 16 --num-workers 4 --camera AUTO --pretrained --device cuda --log-interval 25
```

CUDA automatic mixed precision is enabled by default to reduce VRAM use and speed up ViT training. Disable it only for debugging:

```powershell
--no-amp
```

Additional optimization switches:

```powershell
--lr-scheduler cosine
--warmup-ratio 0.05
--eta-min 1e-6
--dropout 0.3
--weight-decay 3e-4
--temporal-model lstm
--temporal-layers 2
--temporal-feature-hidden-dim 32
--loss ordinal
--ordinal-power 2
--exclude-offseason
--compile-model
--compile-mode reduce-overhead
--no-fused-optimizer
--drop-last
```

For overfitting, try `--dropout 0.25` to `--dropout 0.5` and `--weight-decay 3e-4` to `--weight-decay 1e-3`. The defaults are `--dropout 0.1` and `--weight-decay 1e-4`.

The default loss is ordinal CDF regression:

```powershell
--loss ordinal --ordinal-power 2
```

This keeps one output logit per class but compares cumulative class distributions, so far-away stage mistakes are penalized more than neighboring-stage mistakes. To reproduce the previous soft cross-entropy behavior:

```powershell
--loss soft_ce
```

To train only on phenological stages and remove all `OffSeason` images:

```powershell
--exclude-offseason
```

Cosine LR decay with 5% linear warmup is enabled by default. Disable scheduling with:

```powershell
--lr-scheduler none
```

Fused AdamW is tried by default on CUDA and falls back automatically if your PyTorch build does not support it. `torch.compile` is opt-in because Windows CUDA/PyTorch setups can vary; try it after the normal run works:

```powershell
python ViT3\run_multiscale_training.py --excel-path labeling.xlsx --data-path data --out-dir results_ViT3_cache --embedding-cache results_ViT3_cache\vit_embeddings.pt --epochs 5 --folds 1 --window-days 31 --window-mode causal --stream micro --device cuda --compile-model --compile-mode reduce-overhead
```

DataLoader performance options:

```powershell
--pin-memory
--persistent-workers
--prefetch-factor 2
```

These are enabled where applicable. For cached embedding training on Windows, `--num-workers 0` can still be the best choice because worker processes may duplicate the large in-memory embedding cache.

Effective batch size is:

```text
batch-size * accumulation-steps
```

For example:

```text
2 * 16 = 32
```

If you hit CUDA out-of-memory:

1. Reduce `--batch-size`.
2. Reduce `--window-days`.
3. Use the cached embedding workflow.

## Output Files

Training writes:

```text
results_*/
  multiscale_daily_metadata.csv
  fold_assignments.csv
  all_history.csv
  all_test_metrics.csv
  all_test_predictions.csv
  all_test_confusion_matrix.csv
  all_test_confusion_matrix_normalized.csv
  all_test_confusion_matrix.png
  tensorboard/
  last_checkpoint.pt
  fold_1/
    history.csv
    test_metrics.json
    test_predictions.csv
    test_confusion_matrix.csv
    test_confusion_matrix_normalized.csv
    test_confusion_matrix.png
    best_model.pt
    last_checkpoint.pt
```

Each fold uses disjoint groups:

```text
derived train groups -> update model weights
validation groups -> choose best_model.pt
test groups -> final held-out evaluation after training
```

The test set is not used for checkpoint selection.

`last_checkpoint.pt` is a resumable training checkpoint saved after every epoch. It contains:

- model weights,
- optimizer state,
- AMP scaler state,
- completed epoch,
- fold ID,
- best validation score and epoch,
- fold history.

Resume after an interruption with:

```powershell
python ViT3\run_multiscale_training.py --excel-path labeling.xlsx --data-path data --out-dir results_ViT3_cache --embedding-cache results_ViT3_cache\vit_embeddings.pt --resume-checkpoint results_ViT3_cache\last_checkpoint.pt --epochs 20 --device cuda
```

You can also resume from a fold-specific checkpoint:

```powershell
--resume-checkpoint results_ViT3_cache\fold_3\last_checkpoint.pt
```

The script skips earlier folds, restores the interrupted fold, and continues from the next epoch. Disable per-epoch resume checkpoints with:

```powershell
--no-save-last-checkpoint
```

Important metrics:

- `accuracy`: exact class accuracy.
- `plus_minus_1_accuracy`: counts neighboring stage predictions as acceptable.
- `date_window_accuracy`: gives partial credit according to the prediction's calendar distance from the true phenology window.

For phenology, `plus_minus_1_accuracy` is often more informative than strict accuracy.

By default, `best_model.pt` is selected using validation `date_window_accuracy`. Override with:

```powershell
--checkpoint-metric accuracy
```

## TensorBoard

TensorBoard logging is enabled by default when the `tensorboard` package is installed. Logs are written to:

```text
<out-dir>/tensorboard
```

Open the dashboard with:

```powershell
tensorboard --logdir results_ViT3_cache\tensorboard
```

Then open the printed local URL in your browser.

The dashboard includes:

- train and validation metrics per epoch,
- final held-out test metrics per fold,
- model parameter counts,
- split sizes and group IDs,
- metadata label counts,
- CUDA peak VRAM per epoch when using GPU,
- hyperparameter summaries.

Disable TensorBoard logging with:

```powershell
--no-tensorboard
```

## Single-Image Inference

For cached/tiled embedding checkpoints, use:

```powershell
python ViT3\infer_cached_window.py --checkpoint results_ViT3_cache\fold_1\best_model.pt --embedding-cache results_ViT3_cache\vit_embeddings.pt --image-path "path\to\target_10X.jpeg" --planting-date 2023-11-15 --device cuda
```

If you have the previous 30 days in a folder, pass the folder too. The script parses dates from filenames and uses any images available from `target_date - 30` through the target date:

```powershell
python ViT3\infer_cached_window.py --checkpoint results_ViT3_cache\fold_1\best_model.pt --embedding-cache results_ViT3_cache\vit_embeddings.pt --image-path "path\to\2024_05_10_10X.jpeg" --image-dir "path\to\station_day_images" --planting-date 2023-11-15 --device cuda
```

If the target filename does not contain a date, provide it manually:

```powershell
python ViT3\infer_cached_window.py --checkpoint results_ViT3_cache\fold_1\best_model.pt --embedding-cache results_ViT3_cache\vit_embeddings.pt --image-path "path\to\target.jpeg" --image-dir "path\to\station_day_images" --target-date 2024-05-10 --planting-date 2023-11-15 --device cuda
```

If the checkpoint was trained with the default days-since-planting metadata, pass `--planting-date` during inference. Without it, the script uses zeros for that metadata feature.

If the checkpoint was trained with weather metadata, pass the same cache and station code:

```powershell
python ViT3\infer_cached_window.py --checkpoint results_ViT3_cache\fold_1\best_model.pt --embedding-cache results_ViT3_cache\vit_embeddings.pt --image-path "path\to\2024_05_10_10X.jpeg" --image-dir "path\to\station_day_images" --planting-date 2023-11-15 --station-code 02.02 --weather-cache results_ViT3_weather\meteostat_weather_cache.csv --device cuda
```

Missing previous days are masked out by default. To repeat the target image for missing days instead:

```powershell
--repeat-missing
```

To inspect exactly which dates were found for the causal window, add:

```powershell
--debug-window
```

For a quick check with a full-image checkpoint:

```powershell
python ViT3\infer_single_image.py --checkpoint results_ViT3_full\fold_1\best_model.pt --image-path "path\to\10X.jpeg" --planting-date 2023-11-15 --device cuda
```

For a `10X` image:

```powershell
python ViT3\infer_single_image.py --checkpoint results_ViT3_full\fold_1\best_model.pt --image-path "path\to\10X.jpeg" --stream micro --planting-date 2023-11-15 --device cuda
```

Best quick test, if matching same-day images exist:

```powershell
python ViT3\infer_single_image.py --checkpoint results_ViT3_full\fold_1\best_model.pt --macro-path "path\to\1X.jpeg" --micro-path "path\to\10X.jpeg" --stream both --target-date 2024-05-10 --planting-date 2023-11-15 --device cuda
```

Note: the model is temporal. Single-image inference repeats the same image across the window, so this is only a quick diagnostic, not the ideal inference mode.

## Terminal Logging

The training script prints:

- device and VRAM,
- metadata summary,
- label counts,
- fold split,
- dataset and loader sizes,
- model parameter counts,
- batch progress,
- epoch metrics,
- checkpoint saves.

Control batch logging with:

```powershell
--log-interval 10
```

Disable batch progress logs:

```powershell
--log-interval 0
```

