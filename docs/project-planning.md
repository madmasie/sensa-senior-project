# Project Planning — Sensa

## What We're Building
A wearable air-quality monitor that:
- Senses PM1/2.5/4/10, VOC, NOx, temperature, and humidity via SEN55
- Runs on-device ML (no cloud) to classify exposure level in real time
- Transmits results to phone/PC over BLE
- Operates for ≥8–12 hours on battery

---

## Phase 1 — Scope & Hardware
- **Sensor:** Sensirion SEN55 (PM + VOC/NOx + T/RH, all-in-one)
- **MCU:** ESP32-S3 (TFLite Micro, FreeRTOS, BLE, OTA)
- **Battery:** 2000 mAh LiPo + charger IC
- **AI constraints:**
  - Sample rate: 1–2 Hz; window: 5–10 s; overlap: 50%
  - Feature set: ~20–30 features/window
  - Model types: tiny Random Forest, 1D-CNN, DS-CNN (TFLite Micro, int8)
  - Footprint: ≤30–100 KB weights; ≤128 KB RAM at runtime
  - Latency: feature extraction <50–100 ms, inference <10–30 ms, end-to-end <150 ms

**Deliverables:** BOM, block diagram, AI constraints doc

---

## Phase 2 — Data Collection
- Co-locate prototype beside a reference PM2.5 sensor across varied environments (indoor, cooking, traffic, outdoors, dusty lab)
- ≥5 sessions × 30–60 min; log timestamp, PM1/2.5/10, T, RH, battery from device and reference
- Align timestamps; label reference PM2.5 into AQI exposure classes

**Deliverables:** CSV logs + labeled dataset (train/val/test split by session, not by row)

---

## Phase 3 — Feature Engineering
- **Preprocessing:** warm-up discard (first 30–60 s), outlier clamp, EMA smoothing, T/RH compensation
- **Features per window:**
  - Stats: mean, median, std, IQR, min/max
  - Dynamics: slope, rolling variance, % time above thresholds
  - Shape: short-lag autocorr, skew/kurtosis (optional)

**Deliverables:** Python notebook → feature table + MCU-portable code

---

## Phase 4 — Model Training
- Baselines: threshold/rule model, tiny Random Forest
- If needed: 1D-CNN or DS-CNN on filtered sequence
- Cross-validate by session/location (no data leakage)
- Metrics: accuracy, per-class F1, confusion matrix; MAE vs reference PM
- Apply int8 post-training quantization (TFLite Micro)
- Pick smallest model meeting ≥85% accuracy on test sessions

**Deliverables:** Trained model file(s), metrics report, model card (inputs, features, size, latency)

---

## Phase 5 — Embedded AI Pipeline
- Data path: SEN55 read → preprocess → ring buffer → TFLite Micro inference → class label
- Static tensor arena (document RAM usage)
- BLE GATT: expose PM values, class label, battery, config (sampling rate, window)
- Logging: circular flash buffer; on-demand dump to phone/PC

**Deliverables:** Firmware producing live class labels at 1–2 Hz

---

## Phase 6 — Companion App / Dashboard
- Minimal UI: PM readings, class label, battery, CSV export
- Config toggles: sampling rate, window length; event markers (e.g., "cooking", "traffic")

**Deliverables:** Simple viewer + exporter (Flutter app or Python desktop)

---

## Phase 7 — Calibration & Evaluation
- **Calibration:** fit linear/PLS correction using T/RH as covariates if bias vs reference is detected; apply before feature extraction
- **Accuracy:** confusion matrix + per-class F1 on test set only
- **Latency:** median and p95 around feature extraction and inference
- **Battery:** log runtime and average current from full to empty
- **Stress tests:** fast step changes (incense on/off), motion, temperature swings
- **Lab validation:** Cal Poly Air Measurements Lab (13-201), contact Qu Bing (ENVE dept head)

**Deliverables:** Evaluation report (accuracy, latency, battery, failure cases)
