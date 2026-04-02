# AGENTS.md

## Project Overview
Sensa is a wearable air quality monitor running embedded AI on an ESP32-S3. It reads PM1/2.5/4/10, VOC index, NOx index, temperature, and humidity from a Sensirion SEN55 sensor and performs on-device calibration and classification using TensorFlow Lite Micro.

## Repo Structure
```
src/
  main.cpp                      # Entry point (setup/loop)
  sensor/
    sen55_sensor.cpp            # SEN55 I2C driver
    mock_sensor.cpp             # Mock sensor for development/testing
  processing/
    buffer.cpp                  # Rolling sample buffer (ring buffer)
    features.cpp                # Feature extraction over windows
    pipeline.cpp                # End-to-end processing pipeline
    classify.cpp                # Air quality classification logic
  ml/
    calibrate.cpp               # On-device sensor calibration model
include/
  types.h                       # Shared types: Reading, Features, Classification
  sensor/sensor.h
  processing/
    features.h
    classify.h
    pipeline.h
  ml/
    calibrate.h
```

## Language & Toolchain
- C++17, Arduino framework, PlatformIO
- Target board: ESP32-S3 (`rymcu-esp32-s3-devkitc-1`)
- Build: `pio run`
- Upload: `pio run --target upload`
- Serial monitor: `pio device monitor`

## Key Types (include/types.h)
- `Reading` — raw timestamped sensor sample (PM1/2.5/4/10, temp, RH, VOC, NOx)
- `Features` — aggregated statistics over a rolling window (~20–30 features)
- `Classification` — enum: GOOD, MODERATE, UNHEALTHY, VERY_UNHEALTHY, HAZARDOUS, UNKNOWN

## AQI Classification Thresholds (PM2.5 µg/m³)
| Class            | Range (µg/m³)   |
|------------------|-----------------|
| GOOD             | 0 – 9.0         |
| MODERATE         | 9.1 – 35.4      |
| UNHEALTHY        | 35.5 – 125.4    |
| VERY_UNHEALTHY   | 125.5 – 225.4   |
| HAZARDOUS        | 225.5+          |

## AI / ML Constraints
- Sample rate: 1–2 Hz; window: 5–10 s; overlap: 50%
- Feature set: ~20–30 features per window (mean, median, std, IQR, min/max, slope, rolling variance, % above threshold, short-lag autocorr)
- Model types: tiny Random Forest, 1D-CNN, or DS-CNN (TFLite Micro, int8 quantized)
- Edge footprint: ≤30–100 KB weights; ≤128 KB RAM at runtime
- Inference latency target: <10–30 ms; feature extraction: <50–100 ms; end-to-end: <150 ms
- Accuracy target: ≥85% class agreement vs. reference sensor on unseen sessions
- Static tensor arena — document RAM usage; no dynamic allocation in inference path

## Data Pipeline (on-device)
```
SEN55 read → preprocess (warm-up discard, outlier clamp, EMA smoothing, T/RH compensation)
           → ring buffer → feature extraction → TFLite Micro inference → class label
           → BLE GATT notify (PM values, class label, battery, config characteristic)
```

## BLE / Communication
- BLE GATT service exposes: PM readings, classification label, battery level, config (sampling rate, window length)
- For early development: stream raw data to PC via serial monitor before BLE integration

## Power / Battery
- Target: ≥8–12 hours continuous at 1–2 Hz updates
- Low-power pattern: MCU + sensor sleep between reads; wake every ~60 s, take 10 s of data, run ML, transmit over BLE
- Refer to SEN55 low-power operation guide for reduced-power modes

## Guidelines for AI Agents
- Do not modify `include/types.h` shared types without updating all dependent files
- Prefer stack allocation over heap allocation (embedded constraints)
- Avoid dynamic memory allocation (`new`/`malloc`) in hot paths
- Keep functions small and focused; limited RAM/flash on microcontroller
- Serial output is for debugging only — guard with `#ifdef DEBUG`
- All sensor reads are non-blocking; do not use `delay()` in production code
- Cross-validate ML models by session/location to avoid data leakage
- Apply int8 post-training quantization before deploying any TFLite model
