# 3D CNN Wheat Phenology

This folder is a separate 3D CNN experiment. It keeps the same metadata, labels, folds, and metrics as the `CNN` folder, but the default model is different:

```text
previous 30 days + target day
  -> raw image tensor [days, channels, height, width]
  -> 3D CNN over time + height + width
  -> OffSeason / PS1 / ... / PS7
```

The default command uses only the `10X` close-up stream:

```text
--stream micro
--window-days 31
--window-mode causal
```

## Labels And Metadata

CNN3D now uses the shifted label definition:

```text
OffSeason = before 2-Cikis and after 9-Hasat
PS1 = 2-Cikis -> 3-Cimlenme
...
PS7 = 8-Olgunlasma -> 9-Hasat
```

The `1-Ekim` date is still used to define the season window, but planting-to-emergence is no longer treated as a visible crop stage.

The label table `kamera` / `Camera` column is also respected when `--camera AUTO` is used. Explicit `--camera K1`, `--camera K2`, or `--camera ALL` overrides the table.

CNN3D can optionally feed normalized days-since-planting into the neural network:

```powershell
--use-days-since-planting
```

When enabled, the 3D CNN visual feature is concatenated with a small metadata MLP before classification. Weather and coordinate metadata are not currently used by CNN3D.

## Why This Exists

A 3D CNN treats the daily crop sequence like a short video. Its kernels slide through:

```text
time x image height x image width
```

That can help it learn visual changes across days directly, instead of first extracting one feature per image and then using a temporal Transformer.

The tradeoff is VRAM and overfitting risk. Start small.

## Important Difference

This folder is now CNN3D-only. It does **not** use precomputed embeddings or the old 2D CNN + temporal Transformer baseline:

```text
image sequence -> 3D CNN
```

There is no precompute step for this model. Training and inference load the actual image window.

## Smoke Test

From the project root:

```powershell
python CNN3D\run_multiscale_training.py --excel-path labeling.xlsx --data-path data --out-dir results_cnn3d_smoke --epochs 1 --folds 1 --window-days 7 --window-mode causal --stream micro --batch-size 1 --accumulation-steps 1 --num-workers 0 --camera AUTO --device cuda --log-interval 1
```

The label table can be Excel or a CSV export with the same columns. These are equivalent:

```powershell
python CNN3D\run_multiscale_training.py --excel-path labeling.xlsx ...
python CNN3D\run_multiscale_training.py --label-path labeling.csv ...
```

CSV loading accepts comma or semicolon separators and common Turkish/Windows encodings.

If VRAM is tight:

```powershell
--window-days 3
--cnn3d-base-channels 16
--batch-size 1
```

## Fold Split

For `CNN3D`, validation/test station counts are edited in [run_multiscale_training.py](run_multiscale_training.py), not passed as command-line arguments:

```python
VALIDATION_FOLD_STATIONS = 1
TEST_FOLD_STATIONS = 2
```

Training station count is derived automatically:

```text
train = total station groups - validation - test
```

## Full Experiment

```powershell
python CNN3D\run_multiscale_training.py --excel-path labeling.xlsx --data-path data --out-dir results_cnn3d_full --epochs 20 --folds 5 --window-days 31 --window-mode causal --stream micro --batch-size 1 --accumulation-steps 16 --num-workers 4 --camera AUTO --device cuda --log-interval 25
```

Useful knobs:

```powershell
--image-size 224
--cnn3d-base-channels 16
--cnn3d-base-channels 24
--cnn3d-feature-dim 256
--cnn3d-dropout 0.25
--use-days-since-planting
--temporal-feature-hidden-dim 32
```

## Training Augmentation

CNN3D can apply augmentation only to the training split. Validation and test images always use deterministic resize/normalize transforms.

Enable the default augmentation recipe:

```powershell
--augment
```

This adds random resized crop, horizontal flip, small rotation, color jitter, mild blur, and random erasing. The random transform is synchronized across each temporal window so the 3D CNN does not see artificial frame-to-frame camera motion. Useful knobs:

```powershell
--augment-crop-scale-min 0.75
--augment-hflip-prob 0.5
--augment-rotation-degrees 5
--augment-color-jitter 0.2
--augment-blur-prob 0.1
--augment-erasing-prob 0.1
```

For strong overfitting, a good first run is:

```powershell
python CNN3D\run_multiscale_training.py --label-path labeling3.csv --data-path data --out-dir results_cnn3d_aug --epochs 25 --folds 7 --window-days 15 --window-mode causal --stream micro --batch-size 1 --accumulation-steps 16 --num-workers 4 --camera AUTO --device cuda --log-interval 25 --cnn3d-base-channels 8 --cnn3d-feature-dim 128 --cnn3d-dropout 0.5 --use-days-since-planting --augment
```

## Training Visualizations

At the end of training, the script saves PNG graphs under:

```text
<out-dir>/plots/
```

Generated plots:

```text
epoch_accuracy.png
epoch_date_window_accuracy.png
epoch_plus_minus_1_accuracy.png
epoch_loss.png
final_test_metrics_by_fold.png
final_test_mean_std.png
```

These show train/validation progress across epochs and final held-out test scores for each fold.

Disable PNG plots with:

```powershell
--no-plots
```

## Inference

Use [infer_image.py](infer_image.py) for CNN3D inference. It has two modes:

```text
folder window: previous dated images + target image
single-image diagnostic: repeat the target image across the window
```

### Folder Window Inference

This is the better deployment-like mode. The script parses dates from filenames and builds:

```text
target_date - 30
...
target_date
```

Example for a `10X` folder:

```powershell
python CNN3D\infer_image.py --checkpoint results_cnn3d_full\fold_1\best_model.pt --image-path "C:\path\to\02_02-2015_03_04-10_00-K1-10X.jpeg" --image-dir "C:\path\to\data\02.02\2015\K1\10X" --stream micro --device cuda --debug-window
```

The filename can use the project pattern:

```text
02_02-2015_03_04-10_00-K1-10X.jpeg
```

which is parsed as:

```text
2015-03-04
```

If a date is missing, the script uses a zero image and marks that day as missing in the temporal mask. To repeat the target image for missing days instead:

```powershell
--repeat-missing
```

If the filename does not contain a date:

```powershell
--target-date 2015-03-04
```

### Single-Image Diagnostic

If you do not pass `--image-dir`, the script repeats the target image across the whole temporal window:

```powershell
python CNN3D\infer_image.py --checkpoint results_cnn3d_full\fold_1\best_model.pt --image-path "path\to\10X.jpeg" --stream micro --device cuda
```

This is useful for checking whether the checkpoint loads and produces a sensible class distribution, but it is not the ideal biological inference mode because the model cannot see real temporal change.

### Both Streams

For a checkpoint trained with `--stream both`, provide both target images and folders:

```powershell
python CNN3D\infer_image.py --checkpoint results_cnn3d_full\fold_1\best_model.pt --macro-path "path\to\target_1X.jpeg" --micro-path "path\to\target_10X.jpeg" --macro-dir "path\to\1X_folder" --micro-dir "path\to\10X_folder" --stream both --device cuda --debug-window
```
