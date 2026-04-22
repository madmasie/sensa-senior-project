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

Trains a small 1D convolutional neural network that corrects PM2.5 readings
from the low-cost SEN55 optical counter to match a reference BAM sensor.
The trained model is exported to TFLite and deployed on the ESP32-S3.

```
Dependencies: pip install -r pytorch_calibration/requirements.txt
```

### Full workflow

```
1. Collect data
   uart_logger.py  →  ~/sensa-recordings/sen55_*.pkl
   (run alongside BAM to get paired readings)

2. Share data with teammates
   data_uploader.py → cloud storage
   data_sync.py     ← pull to pytorch_calibration/data/raw/

3. Pair SEN55 with BAM reference
   python pytorch_calibration/prepare_data.py

4. Train and export
   python pytorch_calibration/main.py

5. Deploy
   Copy models/calibration.tflite → firmware project
   Copy include/calib_scaler.h    → include/
```

### Quick smoke test (no hardware required)

Verify the pipeline works end-to-end before you have real data:

```bash
cd tools/
python pytorch_calibration/main.py --demo
```

This generates 2,000 synthetic sensor readings, trains the model, evaluates it,
and prints accuracy metrics. Use it to confirm that all dependencies are installed
and the export pipeline is functional.

### prepare_data.py

Pairs SEN55 `.pkl` files with the BAM reference CSV and outputs a single
`paired_dataset.csv` ready for training.

**Reference CSV format** (`data/raw/bam_reference.csv`):

```
timestamp,bam_pm2_5
2024-03-15 09:00:00,12.4
2024-03-15 10:00:00,14.1
...
```

The BAM reports hourly averages. `prepare_data.py` aggregates the 1–2 Hz SEN55
stream into matching hourly means before merging.

```bash
python pytorch_calibration/prepare_data.py
# Options:
python pytorch_calibration/prepare_data.py --raw-dir path/to/raw --out-dir path/to/paired
```

> **Using an EPA AQS station instead of a BAM:**
> Download hourly PM2.5 data for your state from
> https://aqs.epa.gov/aqsweb/airdata/download_files.html,
> filter to the nearest monitoring station, and rename the columns to
> `timestamp` and `bam_pm2_5`. The station must be within ~1–2 km of your
> SEN55 deployment for the values to be spatially comparable.

### main.py

```bash
# Full pipeline: train + evaluate + export to TFLite
python pytorch_calibration/main.py

# Train only (no export)
python pytorch_calibration/main.py --no-export

# Export an existing checkpoint to TFLite (no retraining)
python pytorch_calibration/main.py --export-only

# Smoke test with synthetic data
python pytorch_calibration/main.py --demo
```

All tunable parameters (learning rate, model size, batch size, etc.) are in
`pytorch_calibration/config.yaml` — no Python editing required for routine tuning.

### Outputs

| File | Description |
|---|---|
| `models/best_model.pt` | Best PyTorch checkpoint (by validation loss) |
| `models/calibration.onnx` | Intermediate ONNX model |
| `models/calibration.tflite` | Final TFLite model → deploy to ESP32 |
| `models/scaler.pkl` | Fitted MinMaxScaler (needed for export) |
| `models/predictions.png` | Scatter + residual diagnostic plots |
| `include/calib_scaler.h` | Auto-generated C header with normalisation constants |

### Accuracy targets (EPA guidelines for low-cost PM2.5 sensors)

| Metric | Target |
|---|---|
| MAE | < 5 µg/m³ |
| RMSE | < 7 µg/m³ |
| R² | > 0.80 |
| MBE | within ±5 µg/m³ |

### GPU note

Training runs on NVIDIA GPU automatically if CUDA is available. On Windows
with an AMD GPU, PyTorch does not support AMD (ROCm is Linux-only). Training
falls back to CPU, which is fast enough — expect under 1 minute per epoch
for this model.
