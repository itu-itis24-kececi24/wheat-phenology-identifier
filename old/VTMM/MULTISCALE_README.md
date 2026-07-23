# Multi-Scale Wheat Phenology

This folder contains the multi-scale temporal implementation for wheat phenology classification.

## Files

- `framework.py`: shared project framework for Excel loading, stage columns, station-folder matching, group folds, and original PS stage definitions.
- `multiscale_phenology.py`: metadata builder, datasets, ViT feature encoders, temporal Transformer models, and soft-label loss.
- `run_multiscale_training.py`: main training script.
- `precompute_multiscale_embeddings.py`: optional frozen-backbone embedding cache builder.
- `infer_single_image.py`: quick single-image inference helper for full-image checkpoints.
- `metadata_sanity_check.py`: small script for checking label counts and 1X/10X path availability.

The multiscale metadata builder now reuses `framework.py` for Excel/date normalization, station lookup, and phenology stage boundaries. The multiscale file only adds the 1X/10X pairing, `OffSeason` windows, soft labels, temporal windows, and model definitions.

## Architecture Overview

The project is a multi-scale, temporal classifier. It predicts the phenology stage of the center day in a rolling time window.

Input for one sample:

```text
station-year daily sequence
        |
        |-- 1X image for each day  -> macro/canopy stream
        |-- 10X image for each day -> micro/leaf-spikelet stream
```

For example, with `--window-days 31`, one training sample contains:

```text
31 macro images + 31 micro images
```

The label is the stage of the middle day:

```text
day 15 in a 31-day window
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

macro + micro features per day
  -> fusion layer
  -> temporal Transformer
  -> center-day classifier
  -> OffSeason / PS0 / ... / PS7
```

The temporal Transformer attends across the days in the window, so it can use before/after context instead of classifying a day in isolation.

### Cached Embedding Model

The cached workflow freezes the image ViT. It first computes image features once:

```text
all 1X/10X JPEGs -> frozen ViT -> saved feature vectors
```

Training then loads feature vectors instead of JPEGs:

```text
cached macro feature sequence
cached micro feature sequence
  -> fusion layer
  -> temporal Transformer
  -> center-day classifier
```

This is much faster, but the image backbone does not learn during training. Use it for quick experiments and use full-image training when you want end-to-end fine-tuning.

### Missing Days

If a date is missing, the dataset can fall back to the nearest available day and marks that timestep in the mask. The temporal Transformer receives the mask so it can distinguish real dates from filled gaps.

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

By default it uses `K1`. Use `--camera K2` for the other camera, or `--camera ALL` to include both.

## Labels

The metadata builder creates:

```text
OffSeason, PS0, PS1, PS2, PS3, PS4, PS5, PS6, PS7
```

It reads phenology dates from `labeling.xlsx`, then:

- keeps 30 days before seeding and labels them as `OffSeason`,
- keeps 30 days after harvest and labels them as `OffSeason`,
- uses soft labels near transition boundaries via `--transition-days`.

You can change the off-season window with:

```powershell
--preplant-days 30 --postharvest-days 30
```

## Environment

Use the same Python environment as your notebooks.

Install minimum dependencies:

```powershell
python -m pip install pandas numpy pillow torch torchvision scikit-learn openpyxl
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

Option 1: run the helper script from `VTMM`:

```powershell
cd VTMM
python metadata_sanity_check.py
```

Option 2: run this in Python or a notebook from the project root:

```python
from VTMM.multiscale_phenology import build_multiscale_daily_dataframe

df = build_multiscale_daily_dataframe(
    "labeling.xlsx",
    "data",
    preferred_camera="K1",
    include_preplant_days=30,
    include_postharvest_days=30,
)
print(df.head())
print(df["label"].value_counts().sort_index())
print(df[["macro_path", "micro_path"]].notna().mean())
```

A healthy result should include all `PS0` to `PS7`, plus `OffSeason`. The `macro_path` and `micro_path` availability should be high, ideally close to `1.0`.

Expected labels:

```text
OffSeason
PS0
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
python VTMM\run_multiscale_training.py --excel-path labeling.xlsx --data-path data --out-dir results_smoke --epochs 1 --folds 1 --window-days 3 --preplant-days 30 --postharvest-days 30 --batch-size 1 --accumulation-steps 1 --num-workers 0 --camera K1 --device cuda --log-interval 1
```

If CUDA is not available, use:

```powershell
--device cpu
```

CPU training will be much slower.

## Recommended Fast Workflow

For practical experiments, freeze the image ViT and precompute image embeddings once.

Step 1: build cached embeddings:

```powershell
python VTMM\precompute_multiscale_embeddings.py --excel-path labeling.xlsx --data-path data --out-dir results_multicache --camera K1 --preplant-days 30 --postharvest-days 30 --batch-size 16 --num-workers 2 --pretrained --device cuda
```

Step 2: train from cached vectors:

```powershell
python VTMM\run_multiscale_training.py --excel-path labeling.xlsx --data-path data --out-dir results_multicache --embedding-cache results_multicache\vit_embeddings.pt --epochs 5 --folds 1 --window-days 7 --preplant-days 30 --postharvest-days 30 --batch-size 32 --accumulation-steps 1 --num-workers 0 --camera K1 --device cuda --log-interval 10
```

This is much faster because training no longer decodes JPEGs or runs ViT for every frame during every epoch.

Use this workflow when the image backbone should stay frozen.

Rebuild `vit_embeddings.pt` whenever you change:

- `--camera`
- `--data-path`
- `--preplant-days`
- `--postharvest-days`
- image preprocessing/backbone settings

## Full Image Training

This trains through the image model every epoch. It is much slower and uses much more VRAM.

```powershell
python VTMM\run_multiscale_training.py --excel-path labeling.xlsx --data-path data --out-dir results_full --epochs 20 --folds 5 --window-days 31 --preplant-days 30 --postharvest-days 30 --batch-size 2 --accumulation-steps 16 --num-workers 4 --camera K1 --pretrained --device cuda --log-interval 25
```

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
  all_history.csv
  fold_1/
    history.csv
    best_model.pt
```

Important metrics:

- `accuracy`: exact class accuracy.
- `plus_minus_1_accuracy`: counts neighboring stage predictions as acceptable.

For phenology, `plus_minus_1_accuracy` is often more informative than strict accuracy.

## Single-Image Inference

For a quick check with a full-image checkpoint:

```powershell
python VTMM\infer_single_image.py --checkpoint results_full\fold_1\best_model.pt --image-path "path\to\image.jpeg" --stream macro --device cuda
```

For a `10X` image:

```powershell
python VTMM\infer_single_image.py --checkpoint results_full\fold_1\best_model.pt --image-path "path\to\image.jpeg" --stream micro --device cuda
```

Best quick test, if matching same-day images exist:

```powershell
python VTMM\infer_single_image.py --checkpoint results_full\fold_1\best_model.pt --macro-path "path\to\1X.jpeg" --micro-path "path\to\10X.jpeg" --device cuda
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
