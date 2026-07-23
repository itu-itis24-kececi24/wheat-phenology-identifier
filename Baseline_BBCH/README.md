# Metadata + ExG BBCH Baseline

This is a deliberately lightweight, non-DINO baseline. It predicts BBCH stages
using:

- normalized latitude, longitude, and elevation;
- temperature, precipitation, weather missingness, daily GDD, and cumulative GDD;
- days since sowing;
- ExG (`2g - r - b`) distribution statistics from the selected daily photo;
- strictly causal rolling summaries over the previous 21 calendar days.

It reuses DINOv3's label parsing, camera selection, weather/GDD implementation,
location table, and balanced LOSO fold generator. No learned image representation
or future image is used.

## Train all LOSO folds

```powershell
.\.venv\Scripts\python.exe Baseline_BBCH\run_metadata_exg_training.py `
  --label-path labeling_bbch_iso_dates.csv `
  --data-path data `
  --out-dir results_metadata_exg_bbch `
  --stream micro `
  --camera AUTO `
  --window-days 21 `
  --validation-groups 2 `
  --expected-stations 15
```

The first run creates `exg_features.csv`. Later runs reuse entries whose image
path, size, and modification time have not changed. The Meteostat cache is stored
under the result directory unless `--weather-cache` points to an existing cache.
The run fails if weather is missing for every row rather than silently presenting
a zero-weather experiment as a weather baseline. Use `--allow-missing-weather`
only for pipeline smoke tests.

For a smoke test, add `--only-fold 1`. Each fold selects the number of boosting
iterations using validation macro-F1 and evaluates its held-out station once.
Outputs include per-fold pickled models, metrics and predictions, combined LOSO
predictions, the complete feature table, and `summary.json`.

This baseline intentionally uses hard interval labels. Its purpose is to measure
how much performance comes from calendar/geography/weather/greenness without a
learned visual backbone.

## Parallel execution

ExG extraction is distributed across unique uncached images, and independent LOSO
folds are trained in parallel. Both stages use all detected CPU cores by default.
Native BLAS/OpenMP threads are limited to one inside each fold worker to prevent
nested oversubscription. Override the process counts when needed:

```powershell
--exg-workers 32 --fold-workers 15
```
