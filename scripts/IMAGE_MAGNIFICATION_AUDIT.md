# Classical 1X/10X image audit

`audit_image_magnification.py` is a read-only dataset audit. It checks the
`1X`/`10X` folder against the filename and, by default, compares every image to
nearby trusted images from both streams using SIFT feature matching, RANSAC, and
homography scale estimation. It never moves or renames files.

Install the project dependencies, then try one station first:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\audit_image_magnification.py `
  --data-path data `
  --station 02.03 `
  --year 2015 `
  --output magnification_audit_02_03.csv `
  --workers 1
```

For a fast metadata-only check that does not require OpenCV:

```powershell
.\.venv\Scripts\python.exe scripts\audit_image_magnification.py `
  --data-path data `
  --output magnification_filename_audit.csv `
  --filename-only
```

To write only actionable discrepancies and omit consistent or visually
inconclusive rows:

```powershell
.\.venv\Scripts\python.exe scripts\audit_image_magnification.py `
  --data-path data `
  --output magnification_discrepancies.csv `
  --only-discrepancies
```

This keeps `LIKELY_MISFILED`, `LIKELY_FILENAME_ERROR`,
`FILENAME_FOLDER_MISMATCH`, and `MISSING_FILENAME_LABEL` rows. The complete scan
counts are still printed to the console.

After reviewing the station-level output, run the full visual audit:

```powershell
.\.venv\Scripts\python.exe scripts\audit_image_magnification.py `
  --data-path data `
  --output magnification_audit.csv
```

The most important statuses are:

- `LIKELY_MISFILED`: visual or filename evidence indicates the opposite stream.
- `LIKELY_FILENAME_ERROR`: the image visually agrees with its folder, but its
  filename says the opposite stream.
- `FILENAME_FOLDER_MISMATCH`: filename and folder disagree, but registration is
  inconclusive.
- `VISUAL_INCONCLUSIVE`: no sufficiently strong same-scale registration.
- `CONSISTENT`: folder, filename, and visual registration agree.
- `FILENAME_FOLDER_CONSISTENT`: filename and folder agree in `--filename-only`
  mode; no claim about visual evidence is made.

Do not treat `visual_confidence` as a calibrated probability. Review the first
report and tune `--minimum-score`, `--minimum-confidence`, `--max-days`, or
`--reference-count` if a camera or station has unusually sparse imagery.
