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
