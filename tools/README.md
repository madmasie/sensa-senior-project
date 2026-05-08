# Sensa Tools

Python utilities for data collection, data sharing, and ML model training.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [uart\_logger.py — Record SEN55 data over serial](#uart_loggerpy)
3. [sensa\_client.py — Live BLE monitor](#sensa_clientpy)
4. [Data sharing workflow](#data-sharing-workflow)
   - [One-time setup](#one-time-setup-per-machine)
   - [data\_uploader.py — Push sessions to the cloud](#data_uploaderpy)
   - [data\_sync.py — Pull sessions from the cloud](#data_syncpy)
5. [pytorch\_calibration — Train the PM2.5 calibration model](#pytorch_calibration)
   - [Workflow A: Public EPA AQS data (no hardware needed)](#workflow-a-public-epa-aqs-data)
   - [Workflow B: Local SEN55 + BAM co-location](#workflow-b-local-sen55--bam-co-location)
   - [Combining both: upgrading from public to local](#combining-both-upgrading-from-public-to-local)

---

## Prerequisites

Install Python dependencies for each script as needed:

```bash
# Data collection and sharing
pip install pyserial pandas bleak pyyaml

# ML training pipeline (run from tools/pytorch_calibration/)
pip install -r pytorch_calibration/requirements.txt
```

**rclone** is required for data sharing. Install once per machine:
- Windows/Mac/Linux: https://rclone.org/install/

---

## uart_logger.py

Records SEN55 sensor data over a USB serial connection and saves each session
as a timestamped `.pkl` file. The recording is controlled by sentinel strings
(`START_RECORDING` / `STOP_RECORDING`) sent by the firmware over UART.

```
Dependencies: pip install pyserial pandas
```

### Usage

```bash
# Linux / Mac
python uart_logger.py --port /dev/ttyUSB0

# Windows
python uart_logger.py --port COM3

# Custom output folder (default: ./data)
python uart_logger.py --port COM3 --out ~/sensa-recordings
```

| Argument | Default | Description |
|---|---|---|
| `--port` | *(required)* | Serial port (`/dev/ttyUSB0`, `COM3`, etc.) |
| `--baud` | `115200` | Baud rate — must match `monitor_speed` in `platformio.ini` |
| `--out` | `./data` | Directory where `.pkl` files are written |

Press **Ctrl-C** to stop. Any in-progress session is saved before exit.

### Output format

Each session is saved as `<out>/sen55_YYYY-MM-DD_HH-MM-SS.pkl` — a pandas
DataFrame with a `DatetimeIndex` and columns:

| Column | Unit |
|---|---|
| `pm1` | µg/m³ |
| `pm2_5` | µg/m³ |
| `pm4` | µg/m³ |
| `pm10` | µg/m³ |
| `temp` | °C |
| `rh` | % |
| `voc` | index (1–500) |
| `nox` | index (1–500) |

> **Where to save:** Point `--out` to a folder **outside** the git repo
> (e.g. `~/sensa-recordings`). `.pkl` files are gitignored by design — use
> `data_uploader.py` to share them with teammates.

---

## sensa_client.py

Connects to the Sensa ESP32 over BLE and prints live PM2.5 readings and
classification labels as they arrive.

```
Dependencies: pip install bleak
```

### Usage

```bash
python sensa_client.py
```

Make sure the ESP32 is powered, advertising, and within BLE range (~10 m).
The sensor takes ~40 s to warm up before data begins arriving.

---

## Data sharing workflow

Raw `.pkl` files are not committed to git. Instead, a shared cloud folder
(Google Drive by default) acts as the exchange point. One teammate uploads;
others download.

```
[Recording machine]                [Training machine]
  uart_logger.py                     data_sync.py
       ↓ .pkl files                        ↑ .pkl files
  ~/sensa-recordings    →→rclone→→   Cloud folder   →→rclone→→   data/raw/
  data_uploader.py
```

### One-time setup (per machine)

**Step 1 — Install rclone**

```bash
# Linux / Mac
curl https://rclone.org/install.sh | sudo bash

# Windows
# Download the .exe from https://rclone.org/downloads/ and add to PATH
```

**Step 2 — Configure the shared remote**

```bash
rclone config
```

Walk through the interactive prompts:
1. **New remote** → name it `sensa`
2. **Storage type** → `Google Drive` (option 15 or search "drive")
3. Follow the browser-based authentication
4. When asked for a shared folder, use the Google Drive folder link the team shares

> All team members must name the remote `sensa` (this is set in `data_config.yaml`).
> The team lead creates the Google Drive folder and shares it with everyone.

**Step 3 — Create your local config**

```bash
cd tools/
cp data_config.example.yaml data_config.yaml
```

Edit `data_config.yaml`:

```yaml
recording_dir: ~/sensa-recordings    # ← change this to wherever uart_logger saves files
rclone_remote: sensa:sensa-data/raw  # ← keep this the same as the team's setup
training_data_dir: pytorch_calibration/data/raw  # ← leave as-is
```

`data_config.yaml` is gitignored — your local paths never end up in the repo.

**Step 4 — Verify rclone works**

```bash
rclone ls sensa:sensa-data/
```

If you see a file listing (or an empty result with no error), you are connected.

---

### data_uploader.py

Scan your recording folder, select sessions interactively, and push them to
the shared cloud folder.

```bash
python data_uploader.py               # interactive session selector
python data_uploader.py --all         # upload everything without prompting
python data_uploader.py --dry-run     # show what would be uploaded
```

**Example session:**

```
Recording directory : /home/alice/sensa-recordings
rclone remote       : sensa:sensa-data/raw

  #   Filename                          Size     Recorded
  ────────────────────────────────────────────────────────
    1   sen55_2024-03-15_09-30-00.pkl   45.2 KB  2024-03-15  09:30
    2   sen55_2024-03-15_14-15-22.pkl   67.8 KB  2024-03-15  14:15
    3   sen55_2024-03-16_10-00-00.pkl   52.1 KB  2024-03-16  10:00

Select sessions to upload:
  [a]     — upload all
  [1,3]   — upload sessions 1 and 3
  [1-3]   — upload sessions 1 through 3
  [Enter] — cancel

Your choice: 2-3
```

---

### data_sync.py

Pull sessions from the shared cloud folder into your local training data directory.

```bash
python data_sync.py               # download only new sessions (default)
python data_sync.py --all         # re-download everything
python data_sync.py --dry-run     # show what would be downloaded
```

**Example session:**

```
rclone remote      : sensa:sensa-data/raw
Training data dir  : .../pytorch_calibration/data/raw

  Remote sessions : 4
  Already local   : 2
  New (to sync)   : 2

  New sessions available:
    + sen55_2024-03-16_10-00-00.pkl
    + sen55_2024-03-16_11-30-00.pkl

  Already downloaded (2 files — skipped):
    ✓ sen55_2024-03-15_09-30-00.pkl
    ✓ sen55_2024-03-15_14-15-22.pkl

Downloading 2 file(s) → .../pytorch_calibration/data/raw

  ← sen55_2024-03-16_10-00-00.pkl
  ← sen55_2024-03-16_11-30-00.pkl

Done. Next steps:
  python pytorch_calibration/prepare_data.py
  python pytorch_calibration/main.py
```

---

## pytorch_calibration

Trains a small 1D convolutional neural network (SensaCalibNet) that corrects
PM2.5 readings from the SEN55 optical counter to match a reference BAM sensor.
The trained model is exported to TFLite and deployed on the ESP32-S3.

```
Dependencies: pip install -r pytorch_calibration/requirements.txt
```

There are two independent paths to a trained model. Choose based on what
hardware is available right now. Both produce a TFLite file that deploys
identically on the ESP32.

| | Workflow A — Public data | Workflow B — Local co-location |
|---|---|---|
| **Hardware needed** | None | SEN55 running beside a BAM for several days |
| **Data source** | EPA AQS public dataset | Your own `uart_logger.py` recordings |
| **Data volume** | 50,000–500,000 paired hours | ~500–2,000 paired hours typical |
| **Model inputs** | `pm2_5_optical, temp, rh` (3 features) | `pm1, pm2_5, pm4, pm10, temp, rh, voc, nox` (8 features) |
| **Entry point** | `train_public.py` | `main.py` |
| **Model output** | `models_public/` | `models/` |
| **When to use** | Before BAM access; immediately deployable | After lab co-location; better accuracy |

Both pipelines share the same underlying model architecture and training code.
Running Workflow B later simply replaces the firmware model — the ESP32 firmware
does not need to change, only the `.tflite` and `.h` files.

---

### Workflow A: Public EPA AQS data

No BAM access required. The EPA hosts co-located reference + continuous monitor
data from hundreds of California sites. We download, match by site and hour,
apply a humidity correction, and train on the result.

**Step 1 — Install dependencies**

```bash
cd tools/
pip install -r pytorch_calibration/requirements.txt
```

**Step 2 — Download and pair the EPA AQS data**

Run from inside `tools/pytorch_calibration/`:

```bash
# Default: California, 2021–2022 (~1–2 GB download, cached after first run)
python fetch_public_data.py

# Choose specific years
python fetch_public_data.py --state CA --years 2020 2021 2022

# Save to a named subfolder (useful when building multi-year datasets)
python fetch_public_data.py --state CA --years 2021 --out-dir data/public/CA_2021
python fetch_public_data.py --state CA --years 2022 --out-dir data/public/CA_2022

# Also download SO2 and NO2 (needed to activate gas-interference corrections)
python fetch_public_data.py --include-gas
```

This writes `data/public/paired_public.csv` (or `<out-dir>/paired_public.csv`)
with columns: `site_id, timestamp, pm2_5_optical, temp, rh, pm2_5_reference`.

**What the download contains:**

| AQS Parameter | Code | Role in training |
|---|---|---|
| PM2.5 FRM/FEM (BAM / filter) | 88101 | Target — "true" PM2.5 the model learns to predict |
| PM2.5 continuous non-FRM | 88502 | Feature — optical monitor reading (SEN55 analog) |
| Outdoor temperature | 62101 | Feature + temperature correction |
| Relative humidity | 62201 | Feature + humidity correction |
| SO2 (optional, `--include-gas`) | 42401 | Feature + SO2 interference correction |
| NO2 (optional, `--include-gas`) | 42602 | Feature + NO2 interference correction |

Sites are matched by the AQS site identifier (state-county-site FIPS). Only hours
where all required parameters have valid readings are kept.

**Physics-based corrections** (applied before ML training, configured in
`config.yaml` under `public_data.corrections`):

| Correction | Default | Formula |
|---|---|---|
| Humidity | `kappa: 0.5` | `PM_corrected = PM_raw / (1 + kappa × RH / (100 - RH))` |
| Temperature | `alpha: 0.0` (disabled) | `PM_corrected = PM_raw × (1 + alpha × (T - T_ref))` |
| SO2 | `beta: 0.0` (disabled) | `PM_corrected = PM_raw - beta × SO2_ppb` |
| NO2 | `gamma: 0.0` (disabled) | `PM_corrected = PM_raw - gamma × NO2_ppb` |

To add a new correction factor, subclass `CorrectionFactor` in
`ingestion/corrections.py` and register it in `CorrectionPipeline._FACTOR_MAP`.

**Step 3 — Train**

```bash
# Train on the default paired_public.csv
python train_public.py

# Train on a specific file or directory
python train_public.py --data data/public/CA_2022/paired_public.csv

# Combine multiple years or regions (concatenated before training)
python train_public.py --data data/public/CA_2021 data/public/CA_2022

# Mix of files and directories
python train_public.py --data data/public/CA_2021 \
                               data/public/CA_2022/paired_public.csv

# Train without exporting to TFLite
python train_public.py --no-export
```

**Step 4 — Deploy**

```
Copy models_public/calibration_public.tflite → firmware project
Copy include/calib_scaler_public.h            → include/
```

At inference time on the SEN55, feed these three values to the model:

| SEN55 channel | Model input |
|---|---|
| `pm2_5` | `pm2_5_optical` (input 0) |
| `temp` | `temp` (input 1) |
| `rh` | `rh` (input 2) |
| `pm1, pm4, pm10, voc, nox` | not used by this model |

**Outputs (Workflow A)**

| File | Description |
|---|---|
| `models_public/best_model_public.pt` | Best PyTorch checkpoint |
| `models_public/calibration_public.onnx` | Intermediate ONNX model |
| `models_public/calibration_public.tflite` | TFLite model → deploy to ESP32 |
| `models_public/scaler.pkl` | Fitted MinMaxScaler |
| `models_public/predictions.png` | Scatter + residual diagnostic plots |
| `include/calib_scaler_public.h` | C header with normalisation constants |

---

### Workflow B: Local SEN55 + BAM co-location

Use this workflow once you have access to a BAM machine in the lab and can
physically run the SEN55 alongside it.  This produces an 8-feature model
specific to your unit and gives better accuracy than Workflow A.

**Requirements:**
- SEN55 deployed within 1–2 m of a BAM machine (or within 1–2 km of an EPA
  AQS monitoring station).
- Several days of continuous paired readings across varied PM conditions
  (clean days, traffic periods, any smoke events). ~500+ paired hours is
  the practical minimum for a well-generalised model.

**Step 1 — Record SEN55 data**

Leave `uart_logger.py` running while the SEN55 is co-located with the BAM:

```bash
# Linux / Mac
python uart_logger.py --port /dev/ttyUSB0 --out ~/sensa-recordings

# Windows
python uart_logger.py --port COM3 --out ~/sensa-recordings
```

Each session is saved as `sen55_YYYY-MM-DD_HH-MM-SS.pkl`. The script buffers
readings between `START_RECORDING` / `STOP_RECORDING` sentinels from the firmware
(see uart_logger.py docstring for details).

**Step 2 — Obtain the BAM reference CSV**

The BAM machine produces hourly PM2.5 values. Export or copy them into:

```
pytorch_calibration/data/raw/bam_reference.csv
```

Required columns:

```
timestamp,bam_pm2_5
2024-03-15 09:00:00,12.4
2024-03-15 10:00:00,14.1
...
```

> **Alternative — use a nearby EPA AQS station:**
> If the BAM machine in the lab is unavailable, an EPA AQS station within
> ~1–2 km of your SEN55 deployment is a free substitute.
> 1. Download hourly PM2.5 data for California from
>    https://aqs.epa.gov/aqsweb/airdata/download_files.html (parameter 88101).
> 2. Filter to the nearest site by latitude/longitude.
> 3. Rename `Date GMT` + `Time GMT` → `timestamp`, `Sample Measurement` → `bam_pm2_5`.
> The station must be physically close — spatial correlation drops off quickly
> for PM2.5.

**Step 3 — Share recordings with teammates** *(optional)*

```bash
# Push your recordings to the shared cloud folder
python data_uploader.py

# Pull recordings your teammates collected
python data_sync.py
```

This syncs `.pkl` files via rclone into `pytorch_calibration/data/raw/`.
See the [Data sharing workflow](#data-sharing-workflow) section for setup.

**Step 4 — Pair SEN55 data with the BAM reference**

```bash
cd tools/pytorch_calibration/
python prepare_data.py
```

`prepare_data.py` loads all `sen55_*.pkl` files from `data/raw/`, resamples the
1–2 Hz SEN55 stream to hourly means, and joins on timestamp with the BAM CSV.
Output: `data/paired/paired_dataset.csv`.

```bash
# Custom paths:
python prepare_data.py --raw-dir path/to/raw --out-dir path/to/paired
```

**Step 5 — Train and export**

```bash
# Full pipeline: train + evaluate + export to TFLite
python main.py

# Train only (skip TFLite export)
python main.py --no-export

# Re-export an existing checkpoint without retraining
python main.py --export-only

# Verify the pipeline with synthetic data (no real data needed)
python main.py --demo
```

All tunable parameters (learning rate, model size, batch size) live in
`config.yaml` — no Python editing required.

**Step 6 — Deploy**

```
Copy models/calibration.tflite → firmware project
Copy include/calib_scaler.h    → include/
```

This model uses all 8 SEN55 channels:
`pm1, pm2_5, pm4, pm10, temp, rh, voc, nox → calibrated PM2.5`

**Outputs (Workflow B)**

| File | Description |
|---|---|
| `models/best_model.pt` | Best PyTorch checkpoint (by validation loss) |
| `models/calibration.onnx` | Intermediate ONNX model |
| `models/calibration.tflite` | Final TFLite model → deploy to ESP32 |
| `models/scaler.pkl` | Fitted MinMaxScaler (needed for export) |
| `models/predictions.png` | Scatter + residual diagnostic plots |
| `include/calib_scaler.h` | Auto-generated C header with normalisation constants |

---

### Combining both: upgrading from public to local

The recommended sequence when starting with no BAM access:

```
Phase 1 (now):
  python fetch_public_data.py        ← download EPA AQS data
  python train_public.py             ← train 3-feature model
  → deploy models_public/*.tflite to firmware
  → sensor is calibrated with publicly-trained model

Phase 2 (after BAM access + co-location):
  python prepare_data.py             ← pair local SEN55 data with BAM
  python main.py                     ← train 8-feature model
  → replace firmware model with models/*.tflite
  → sensor is now calibrated to your specific unit
```

The firmware interface does not change between phases — only the `.tflite`
and `.h` files are swapped. The 8-feature local model uses all SEN55 channels
and is expected to outperform the 3-feature public model once ~500+ paired
hours of local data are available.

---

### Accuracy targets (EPA guidelines for low-cost PM2.5 sensors)

| Metric | Target |
|---|---|
| MAE | < 5 µg/m³ |
| RMSE | < 7 µg/m³ |
| R² | > 0.80 |
| MBE | within ±5 µg/m³ |

These targets are checked automatically at the end of every training run.

### GPU note

Training runs on NVIDIA GPU automatically if CUDA is available. On Windows
with an AMD GPU, PyTorch does not support AMD (ROCm is Linux-only). Training
falls back to CPU, which is fast enough — expect under 1 minute per epoch
for this model.
