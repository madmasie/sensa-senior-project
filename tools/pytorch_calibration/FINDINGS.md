# FINDINGS — EPA AQS bulk data quirks

Notes gathered while debugging `fetch_public_data.py` against EPA AQS bulk
hourly files. Recorded so future contributors don't rediscover these the hard
way.

Source: https://aqs.epa.gov/aqsweb/airdata/download_files.html

## 1. Bulk file naming: numeric code vs. group name

EPA AQS bulk hourly files use **two different naming schemes** in the same
directory:

| Parameter            | Param code | Bulk file                  |
|----------------------|------------|----------------------------|
| PM2.5 FRM/FEM        | 88101      | `hourly_88101_{year}.zip`  |
| PM2.5 continuous     | 88502      | `hourly_88502_{year}.zip`  |
| SO2                  | 42401      | `hourly_42401_{year}.zip`  |
| NO2                  | 42602      | `hourly_42602_{year}.zip`  |
| Outdoor temperature  | 62101      | `hourly_TEMP_{year}.zip`   |
| Relative humidity    | 62201      | `hourly_RH_DP_{year}.zip`  |

- **Criteria pollutants** are published one file per numeric parameter code.
- **Meteorological parameters** are bundled into **named group files**
  (`TEMP`, `RH_DP`, `WIND`, `PRESS`). `hourly_62101_{year}.zip` does NOT exist
  and returns **404**.

Verify any URL before assuming: a `curl -I` HEAD request returns `200` for a
real file and `404` for a non-existent one.

Handled in code by `_FILE_TOKEN` in `ingestion/epa_aqs.py`, which maps each
param code to its actual bulk-file token.

## 2. Group files contain multiple parameter codes

A group file holds several parameters at once — e.g. `RH_DP` bundles relative
humidity (62201) **and** dewpoint (62103). The criteria-pollutant files hold
only one.

Therefore the loader **must filter the CSV by `parameter_code`** after reading,
or parameters get silently mixed/mislabeled. `load_aqs_state()` now does this
(a harmless no-op for single-parameter files).

## 3. Header naming is inconsistent: spaces vs. underscores

AQS CSV headers are not stable. Some files use spaces (`State Code`), others
use underscores (`State_Code`). `pd.read_csv(usecols=...)` matches against the
**raw** header, so a hardcoded list of either style will break on the other.

`load_aqs_state()` now peeks at the header row first, normalizes every column
name (lowercase, underscores), and keys `usecols`/`dtype` to the file's actual
names. On a genuine mismatch it raises a `ValueError` listing missing vs. found
columns instead of pandas' opaque `Usecols do not match columns` error.

## 4. California has NO 88101 / 88502 co-location — use Oregon

The script pairs PM2.5 FRM/FEM (88101, reference target) with PM2.5 non-FRM
continuous (88502, optical-sensor analog) at the *same site and UTC hour*.

**In California these two parameters are never co-located** — they are run at
separate sites by separate programs. A nationwide scan of the cached 2021/2022
bulk files (filtering only by state) found:

| Year | 88101∩88502 co-temporal site-hours | States with co-location |
|------|------------------------------------|-------------------------|
| 2021 | 72,397                             | Oregon, Washington, Iowa |
| 2022 | 95,694                             | Oregon (dominant), WA, IA |

CA contributes **zero**. The script's default `--state CA` therefore always
fails at `merge_parameters` with "No co-located rows found".

**Use `--state OR`.** Oregon has many sites with ~8,600 paired hours each
(near-complete annual hourly coverage). Example that works:

```bash
python fetch_public_data.py --state OR
```

This yields ~14k paired rows across 3 sites for 2021–2022. Note the RH join is
the limiting factor (62201 co-located at only ~3 OR sites) — add more years or
drop RH if a larger dataset is needed.

The "NOTE ON CALIFORNIA DATA" section in `fetch_public_data.py`'s docstring is
misleading and predates this finding; CA is not a viable state for this script.

`merge_parameters` now prints row/site counts after every join step, so an
empty result can be traced to the exact parameter that caused it.

## Quick reference: checking what's available

```bash
# HEAD request — 200 = exists, 404 = not found
curl -s -o /dev/null -w "%{http_code}\n" -I \
  https://aqs.epa.gov/aqsweb/airdata/hourly_TEMP_2021.zip
```

If `fetch_public_data.py` prints `No rows for state ... param XXXXX`, the param
code is absent from the (correct) downloaded file — check the code is right for
that group file rather than assuming the download failed.

## 5. Transfer-learning calibration — `finetune.py`

### Summary

`finetune.py` combines the two calibration data sources with transfer learning
instead of training a single model on one source:

1. **Pretrain** on EPA AQS data (`train_public.py`) — a large, diverse set of
   co-located optical (88502) + reference (88101) pairs. Establishes the
   general optical-PM → reference-PM correction.
2. **Fine-tune** on local SEN55 + BAM co-location data (`finetune.py`) — small
   but exactly our sensor. Nudges the pretrained weights to close the
   SEN55-specific gap.

Key implementation choices:

- **3-feature model** (`pm2_5`, `temp`, `rh`) — the schema the AQS pretrained
  model uses; the local CSV's other 5 SEN55 channels are dropped.
- **Scaler is reused, never refit** — the AQS `MinMaxScaler` is applied to the
  local data so inputs stay in the range the pretrained weights expect.
- **Low learning rate** (`1e-4`, ~10× below from-scratch) so local data refines
  rather than overwrites the AQS knowledge.
- **Order is fixed**: AQS first, local last. The dataset trained on last has
  the final say, and that must be our sensor.
- Optional `--freeze-conv` adapts only the regression head for very small
  local datasets.
- Reports test-set metrics **before and after** fine-tuning to quantify the
  actual gain. Exports straight to the firmware headers; `calibrate.cpp`
  auto-detects the 3-feature model.

### Pros

- Uses the large AQS set for volume/diversity *and* local data for
  sensor-specific accuracy — neither source alone is sufficient.
- Correct transfer-learning order avoids catastrophic forgetting of the
  local (deployment-relevant) data.
- Single end-to-end model — no two-stage cascade, so no error propagation
  between stages and no doubled int8/arena cost on the ESP32.
- Reuses the entire `src/` stack (`dataset`, `train.validate`, `evaluate`,
  `export`) — little new surface area to maintain.
- Scaler reuse keeps the input domain consistent with pretraining (the most
  common transfer-learning mistake, avoided by design).
- Before/after evaluation makes the benefit measurable, not assumed.
- Build-safe: until the model exists, the firmware runs in passthrough.

### Cons / limitations

- **3 features only.** Drops `pm1`, `pm4`, `pm10`, `voc`, `nox` — the
  8-feature local model (`main.py`) could exploit particle-size-distribution
  structure that this model cannot see.
- **AQS optical monitor ≠ SEN55.** The 88502 instruments are research-grade;
  the domain gap is closed only as well as the scarce local data allows.
- **Pretrained prior is Oregon-specific.** Per finding #4, usable AQS
  co-location is dominated by Oregon — the pretrained model carries a Pacific
  NW climate/aerosol bias that fine-tuning must overcome.
- **Local data is scarce.** Fine-tuning on a few hundred paired hours risks
  overfitting; `--freeze-conv` mitigates this but limits how much the model
  can adapt.
- **Splitting eats scarce data.** Holding out val + test (`0.15` + `0.15`)
  from an already-small local set leaves little for training.
- **Scaler range mismatch.** Local conditions outside the AQS min/max are
  squashed toward 0/1 by the reused scaler (and clamped again on-device).
- **BatchNorm on small data.** Full fine-tuning re-estimates BatchNorm
  running stats on the small local set — noisier than the AQS estimates.
- **Site/unit specific.** The fine-tuned model is calibrated to this SEN55
  unit and deployment environment; another unit or location needs new
  co-location data and a re-run.
- **Export needs `onnx2tf`** (not in the nix env) — requires the separate
  venv described in `flake.nix`'s shellHook.

## 6. Local PurpleAir + SEN55 data pairing — `prepare_purpleair_data.py`

The local co-location campaign produces two artifacts:

- `data/sen55_YYYY-MM-DD_HH-MM-SS.pkl` — ~10-minute SEN55 capture (~360 rows
  at ~0.6 Hz), one DataFrame per file with a `DatetimeIndex` and the eight
  SEN55 channels.
- `data/purpleairdata.csv` — one row per `(filename, pm2.5_1, pm2.5_2)`. The
  filename points at the matching `.pkl`; `pm2.5_1` / `pm2.5_2` are the two
  PurpleAir laser channels' averages over the same window.

`prepare_purpleair_data.py` produces the paired CSV the training pipeline
already understands:

1. For each row in `purpleairdata.csv`, open the matching `.pkl`.
2. **Average all 8 SEN55 channels over the file's full window** — the
   PurpleAir reference is already a window average, so this is the correct
   pairing point.
3. **PurpleAir reference = mean(pm2.5_1, pm2.5_2)**, matching the EPA
   PurpleAir correction equations.
4. **Skip files with <30 samples** — anything tiny is a glitched capture
   (one of ours was a 2-row / 1.6-second session — would have produced
   nonsense averages).
5. Output schema matches `prepare_data.py`'s BAM output:
   `timestamp, pm1, pm2_5, pm4, pm10, temp, rh, voc, nox, bam_pm2_5`,
   plus extra `pa_pm2_5_a`, `pa_pm2_5_b`, `source_file` columns for
   inspection (ignored by the training code).

The CSV is consumed by both `main.py` (8-feature from-scratch path) and
`finetune.py` (3-feature transfer-learning path) without modification.

## 7. Cross-validation for tiny datasets — `finetune.py --cv K`

### Why predictions.png only had 2 points

`plot_predictions(...)` in `src/evaluate.py` is called on the **test
loader**, not the full dataset. With 17 paired rows and
`val_frac=test_frac=0.15` in `config.yaml`, the split is 13/2/2.
The 2 test points are exactly what got plotted — not a bug.

### Why a test split exists at all

Three roles:

| Split | Role | Does the model see it? |
|---|---|---|
| Train | Gradient updates | Yes, every batch every epoch |
| Validation | Early stopping, LR scheduling, "best" checkpoint selection | Indirectly via the selection criterion |
| Test | Final, single, unbiased accuracy report | No, until the end |

Validation is *not* unbiased: `if val_loss < best_val_loss: torch.save(...)`
cherry-picks the model that did best on it. Reporting val MAE as the
final accuracy is optimistic. Test is the closest thing to an unbiased
generalization estimate.

### The problem at N = 17

A 2-sample test set can swing by µg/m³ just by changing the random seed
— the reported test MAE is statistical theater. The fixed-split test
landed on the two highest-PM windows in our run, producing a misleading
12 µg/m³ MAE.

### K-fold CV is the fix

`python finetune.py --cv K` (added 2026-05-25). For each fold:

1. **Reload the AQS pretrained weights fresh** — every fold starts from
   the same initial state, otherwise earlier folds leak information into
   later ones.
2. Hold out this fold's rows as the test set.
3. Split the remainder into train+val using `data.val_frac` proportionally.
4. Fine-tune with the AQS scaler reused (never refit) — same invariant as
   the single-run path.
5. Predict on the held-out fold and store predictions indexed by the
   original row in the dataset.

After all folds, every sample has been predicted exactly once by a model
that never saw it during training. Aggregate MAE/RMSE/R²/MBE are computed
over the full held-out vector.

- `--cv N` (or any K >= N) → leave-one-out.
- CV mode **does not export** — there are K models, none trained on the
  full dataset. For deployment, rerun without `--cv`.
- Per-fold checkpoints land in `models_finetuned/cv_folds/`, plot in
  `models_finetuned/cv_predictions.png`.

### The CV result on 17 samples

| Method | LOO MAE | RMSE | MBE | R² |
|---|---|---|---|---|
| Fixed split (`--freeze-conv`) | 12.10 | 12.29 | +12.10 | -1232 |
| 5-fold CV (`--cv 5 --freeze-conv`) | 3.80 | 6.06 | +2.80 | -1.25 |
| Leave-one-out (`--cv 17 --freeze-conv`) | **3.52** | 5.47 | +2.52 | -0.84 |

The fixed-split 12 was an artifact of the test set landing on the worst
two samples. The CV aggregate is the honest number.

### Implementation notes

- `finetune_loop` was extended with `checkpoint_path` (per-fold checkpoint
  path so folds don't overwrite each other) and `verbose=True` (silences
  per-epoch noise during CV). Non-breaking — both default to the old
  behavior.
- Pretrained state is loaded once into memory (`pretrained_state = torch.
  load(...)`) and `model.load_state_dict(pretrained_state)` is called at
  the top of each fold to reset.

## 8. Linear-regression baseline beats the NN at N = 17 — `linear_calibration.py`

### The problem with the NN at this scale

The 17 paired windows form two clusters: 11 at ~2.7 µg/m³ and 6 at
~11–14 µg/m³, with **nothing in between**. The MSE loss is dominated by
the 11 low-PM samples, so the fine-tuned NN fits that cluster tightly
and then *over-corrects* the high cluster (predicts 15–27 where truth
is 11–14). A 2,273-parameter (or even 545-param with `--freeze-conv`)
model is doing curve-fitting through essentially 2 data points.

### The linear-baseline result

A 2-coefficient affine fit `y_ref = a*pm2_5 + b`, leave-one-out CV on
the same 17 samples:

| Approach | LOO MAE | RMSE | MBE | R² |
|---|---|---|---|---|
| Raw SEN55 (no cal) | 2.62 | 2.94 | -2.62 | 0.470 |
| Fine-tuned NN (LOO) | 3.52 | 5.47 | +2.52 | -0.837 |
| **Linear (LOO)** | **0.70** | **0.97** | **+0.04** | **0.942** |

The linear fit:

- Beats the NN by **5×** on MAE,
- Beats raw SEN55 by **4×**,
- Meets EPA interim low-cost-sensor guidelines (MAE < 5, R² > 0.80,
  |MBE| < 5),
- Has 0.34% the parameter count of the NN.

### Deployment formula (fit on all 17 samples)

```
pm2_5_calibrated ≈ 1.3590 × pm2_5_sen55 + 0.8337
```

Two multiplies and one add in firmware. Coefficients saved to
`models_linear/linear_calibration.json`.

### When to use which

| Regime | Recommended model |
|---|---|
| N < ~50 paired windows, narrow PM range | **Linear baseline** — the data won't support more capacity. |
| N ≳ 100, wider PM range, RH variation within a single PM regime | NN starts paying off (multi-feature non-linear corrections become recoverable from the data). |

### Modularity guarantee

`linear_calibration.py` is fully self-contained:

- Does not import from `src/`.
- Does not modify `finetune.py`, `main.py`, or any other module.
- Writes only to `models_linear/`.
- Deleting it has zero effect on the rest of the pipeline.

`--features pm2_5 rh temp` runs the same script as a multivariate linear
regression (one extra coefficient per feature), useful once humidity
varies within a single PM regime.

## 9. How `finetune.py` uses AQS data — clarification

`finetune.py` does **not** read any AQS CSV. The AQS information enters
only through two precomputed artifacts that `train_public.py` already
produced:

```
models_public/best_model_public.pt   ← AQS-pretrained weights
models_public/scaler.pkl             ← MinMaxScaler fit on AQS ranges
```

| Artifact | What AQS contributed | How `finetune.py` uses it |
|---|---|---|
| `best_model_public.pt` | ~14k AQS site-hours of `(pm2_5_optical, temp, rh) → pm2_5_reference` mappings, compressed into 2,273 weights | Initial model state at the start of training (and reset to this state at the top of each CV fold). Local fine-tuning nudges these weights with a 10× lower LR. |
| `scaler.pkl` | The AQS feature min/max | Reused to normalize local SEN55 inputs into the same scaled space the pretrained weights expect. **Never refit** — `PM25CalibrationDataset(..., scaler=aqs_scaler, fit_scaler=False)`. |

What `finetune.py` does NOT do:

- Open `tools/data/public/paired_public.csv`.
- See any AQS row during gradient updates.
- Mix AQS rows into the local dataset.

The AQS information is "frozen" into the initial weights and the scaler;
every gradient update is computed purely from the local paired data.

### Why this design, not joint training

If AQS + local rows were concatenated, the 14k AQS rows would drown out
the 17 local rows in the gradient. By baking AQS into pretrained weights
first and fine-tuning on local data with a low LR, the local data has
the *final say* (which is correct — the deployment sensor is the SEN55),
while AQS-learned structure is preserved as a prior.

### Dependency chain

```
EPA AQS website
   └→ fetch_public_data.py → tools/data/public/paired_public.csv
        └→ train_public.py → models_public/best_model_public.pt
                            + models_public/scaler.pkl
             └→ finetune.py  (the only stage that touches local data)
```

`finetune.py` is the third link — it only ever opens the artifacts the
second link produced.
