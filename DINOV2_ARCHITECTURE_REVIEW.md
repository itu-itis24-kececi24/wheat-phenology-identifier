# DINOv2 Wheat Phenology Architecture Review

Scope: `DINOv2` and `DINOv2_Gated` only. Review date: 2026-07-10.

## Verdict

The strongest deployment baseline is still the 10X-only `DINOv2` model with tiled frozen features and a causal temporal head. It matches the expected deployment input and has fewer ways to overfit.

`DINOv2_Gated` is a valid A/B candidate when both cameras will exist at deployment. It should only replace the micro-only baseline if it improves mean held-out-station macro F1 and quadratic weighted kappa across exactly the same folds. Higher accuracy on one favorable fold is not enough.

The current data scan resolves 3,396 labeled daily rows, 16 station-years, and 15 physical stations. `27.05_2016` is present in `labeling3.csv` but has no corresponding 2016/2017 image folders. `02.06_2016` is now recovered despite its validated K2 files being physically stored under K1 folders.

## Corrections Implemented

- Folds default to physical-station grouping. Multiple years from `02.03` cannot cross train/validation/test boundaries.
- Same-day 1X and 10X images are paired by nearest capture timestamp instead of filesystem traversal order.
- Status CSV paths fall back to a cached basename lookup, recovering validated files stored under a mismatched camera directory.
- Cumulative GDD now includes every calendar day, including dates with no camera image.
- Cached timestep masks check whether an embedding really exists, not only whether a path string exists.
- The gated model masks missing streams, starts from neutral 50/50 weights, and can train with optional modality dropout.
- The last partial gradient-accumulation group is scaled by its real size.
- Model, DataLoader, Python, NumPy, Torch, and CUDA random state are reproducible and resumable.
- New Transformer runs use pre-norm blocks; old checkpoints remain post-norm unless their checkpoint says otherwise.
- Hybrid soft cross-entropy plus ordinal CDF loss is available.
- Ordinal CDF loss now uses the correct `C - 1` thresholds.
- Precompute supports AMP/TF32 and skips corrupt full images as well as corrupt tiled images.
- Gated cached inference now accepts separate macro and micro windows instead of silently using one image for both views.

## Recommended Controlled Experiments

Change one factor at a time and reuse `--fold-seed 42 --seed 42`.

1. Establish the micro-only baseline:
   - `--stream micro`
   - `--temporal-model transformer`
   - `--temporal-aggregation cls`
   - `--temporal-layers 2`
   - `--temporal-ffn-multiplier 2`
   - `--loss hybrid --ordinal-ce-weight 0.5`
   - `--dropout 0.2 --weight-decay 3e-4`

2. Tune learning rate first: `3e-5`, `1e-4`, `3e-4`.

3. Compare temporal capacity:
   - 2 layers / FFN multiplier 2
   - 4 layers / FFN multiplier 2
   - 4 layers / FFN multiplier 4

4. Compare tiling:
   - stride 224 for the compute-efficient baseline
   - stride 112 for overlapping detail coverage
   - keep `--max-tiles 0`; capped tile selection loses exact spatial coordinates

5. Compare metadata as strict ablations:
   - days since planting only
   - days since planting plus weather/GDD
   - image only

6. Compare gated fusion using identical folds and hyperparameters:
   - `--stream both --modality-dropout 0`
   - `--stream both --modality-dropout 0.1`

7. Compare objectives:
   - hybrid CE weight 0.25, 0.5, 0.75
   - pure ordinal
   - pure soft CE

Use validation macro F1 as the main model-selection metric when class balance matters. Report exact accuracy, date-window accuracy, QWK, mean absolute stage error, and per-class recall alongside it.

## Methods Most Likely To Improve Accuracy Next

1. Fine-tune the last 2-4 DINOv2 blocks after the cached head converges. Use a backbone LR around 10-50 times smaller than the temporal-head LR and layer-wise LR decay. Frozen features are fast, but domain adaptation is the largest unused source of visual improvement.

2. Train on complete station-year sequences and predict every valid day with one causal pass. The current overlapping 31-day windows heavily duplicate examples and do not directly enforce consistent progression.

3. Add monotonic sequence decoding. Wheat stages should rarely move backward. An HMM, constrained Viterbi decoder, or learned transition matrix can remove isolated backward jumps without changing the image model.

4. Replace independent tile scoring with a small tile Transformer or macro-to-micro cross-attention. Keep true 2D tile coordinates in the cache rather than reconstructing a grid from tile count.

5. Add auxiliary biological targets if annotations can be created: emergence presence, node visibility, heading/spike presence, anthesis evidence, green-to-yellow ratio, and days to the next boundary. This teaches the representation which visual evidence supports a stage.

6. Self-supervise DINOv2 on all wheat tiles before supervised training. Crop-camera DINO/MAE adaptation can reduce the gap from general internet imagery without requiring more labels.

7. Ensemble the best 3-5 physical-station folds only after choosing hyperparameters on validation results. Average class probabilities, then optionally apply the monotonic decoder.

## Experiments To Deprioritize

- DINOv2 Large/Giant before the base model is well tuned. The labeled station diversity is too small for size alone to be a reliable gain.
- Embedding SMOTE as the first imbalance fix. Interpolating entire temporal windows from different station-years can create biologically implausible trajectories. Prefer balanced sampling or class-aware loss first.
- Increasing epochs without early validation evidence. Best-checkpoint selection already protects the final model; longer runs are useful only while validation macro F1 or QWK continues improving.
- Random image-level train/test splits. Adjacent days and fixed-camera backgrounds make those scores overly optimistic.

## Evaluation Protocol

Maintain two separate claims:

- Unseen-station generalization: `--fold-group-by station`.
- Known-station, future-year generalization: `--fold-group-by station_year`, reported separately and explicitly.

Do not mix the two protocols in one average. For final deployment, retrain on all development stations with the selected recipe, reserve one untouched station or season for the final audit, and retain the fold ensemble if inference cost is acceptable.
