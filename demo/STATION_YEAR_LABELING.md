# Station-year assisted labeling

`label_station_year.py` uses the trained demo temporal model and fine-tuned DINOv3
backbone to propose phenology milestone dates for one station-year. It encodes
each 10X image once, predicts every observed date with the model's 21-day causal
window, applies monotonic Viterbi decoding, and derives transition dates.

The command is review-only. It never modifies the canonical label CSV/XLSX.

## Example

```powershell
.\.venv\Scripts\python.exe demo\label_station_year.py `
  --data-path data `
  --station-code 02.03 `
  --year 2015 `
  --camera K1 `
  --sowing-date 2015-10-20 `
  --harvest-date 2016-06-25 `
  --output-dir proposed_labels\02.03_2015
```

You can instead point at a station, station-year, camera, or `10X` directory:

```powershell
.\.venv\Scripts\python.exe demo\label_station_year.py `
  --station-dir D:\new_data\31.01 `
  --station-code 31.01 `
  --year 2025 `
  --camera K2 `
  --sowing-date 2025-11-03 `
  --harvest-date 2026-06-18 `
  --latitude 39.92 `
  --longitude 32.85 `
  --elevation 900 `
  --output-dir proposed_labels\31.01_2025
```

Latitude, longitude, and elevation are needed together only for a station family
that is not already configured in the project.

## Outputs

- `daily_predictions.csv`: raw and monotonic predictions, probabilities, image
  paths, and temporal-window coverage.
- `milestone_proposals.csv`: proposed dates, confidence, provenance, and the
  observations bracketing each transition.
- `label_row_proposal.csv`: one compact row using the canonical milestone column
  names.
- `summary.json`: run configuration, limitations, and proposals.
- `image_embeddings.pt`: resumable image-feature cache. Unchanged images are not
  re-encoded on subsequent runs.

The checkpoint predicts intervals from BBCH0 through BBCH8. Supply
`--sowing-date` whenever known because it is also an input feature. Without it,
the script uses zero for that feature and marks the first observed BBCH0 date as
a heuristic. The checkpoint has no post-harvest class, so harvest remains blank
unless `--harvest-date` is supplied. All generated dates require human review.
