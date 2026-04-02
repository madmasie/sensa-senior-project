# AGENTS.md

## Project Overview
Sensa is a wearable air quality monitor running embedded AI on an ESP32-S3. It reads PM1/2.5/4/10, VOC index, NOx index, temperature, and humidity from a Sensirion SEN55 sensor and performs on-device calibration and classification using TensorFlow Lite Micro.

## Team
This project is built by a team of **electrical engineers**, not software engineers. All code must include comments that explain what the code is doing and why — do not assume familiarity with C++ patterns, embedded conventions, or build tooling.

## Repo Structure
```
src/
  main.cpp                      # Entry point (setup/loop)
  sensor/
    sen55_sensor.cpp            # SEN55 I2C driver
    mock_sensor.cpp             # Mock sensor for development/testing
  processing/
    buffer.cpp                  # (stub) Rolling sample buffer — logic lives in lib/
    feature_extraction.cpp      # (stub) Feature extraction — logic lives in lib/
    pipeline.cpp                # End-to-end processing pipeline
    classify.cpp                # Air quality classification logic
  ml/
    calibrate.cpp               # On-device sensor calibration model
include/
  types.h                       # Shared types: Reading, Features, Classification
  sensor/sensor.h
  processing/
    feature_extraction.h        # NOTE: named feature_extraction.h, NOT features.h
    classify.h
    pipeline.h
  ml/
    calibrate.h
lib/
  buffer/src/buffer.h           # RingBuffer<N> — portable, no Arduino dependency
  features/src/
    feature_extraction.h        # Feature extraction — portable, no Arduino dependency
  classify/src/
    classify.h                  # classify() — portable, no Arduino dependency
    classify.cpp
test/
  native/
    test_buffer/test_main.cpp   # GoogleTest suite for RingBuffer
    test_feature_extraction/test_main.cpp # GoogleTest suite for feature extraction
    test_classify/test_main.cpp # GoogleTest suite for classify()
```

## Language & Toolchain
- C++17, Arduino framework, PlatformIO
- Target board: ESP32-S3 (`rymcu-esp32-s3-devkitc-1`)
- Build: `pio run`
- Upload: `pio run --target upload`
- Serial monitor: `pio device monitor`
- Run tests: `~/.platformio/penv/bin/pio test -e native`

## Key Types (include/types.h)
- `Reading` — raw timestamped sensor sample (PM1/2.5/4/10, temp, RH, VOC, NOx)
- `FeatureVector` — aggregated statistics over a rolling window (~20–30 features)
- `Classification` — enum: GOOD, MODERATE, UNHEALTHY, VERY_UNHEALTHY, HAZARDOUS, UNKNOWN

## AQI Classification Thresholds (PM2.5 µg/m³)
| Class          | Range (µg/m³) |
|----------------|---------------|
| GOOD           | 0 – 9.0       |
| MODERATE       | 9.1 – 35.4    |
| UNHEALTHY      | 35.5 – 125.4  |
| VERY_UNHEALTHY | 125.5 – 225.4 |
| HAZARDOUS      | 225.5+        |

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

## Unit Testing Setup
- Framework: **GoogleTest** via PlatformIO native environment
- Portable logic lives in `lib/` (no Arduino headers) and is tested on the host machine
- Test files live in `test/native/test_<name>/test_main.cpp`
- Each test file must include its own `main()`:
  ```cpp
  int main(int argc, char **argv) {
      ::testing::InitGoogleTest(&argc, argv);
      if (RUN_ALL_TESTS()) ;
      return 0;
  }
  ```
- Run with: `~/.platformio/penv/bin/pio test -e native`

### Known Gotchas (Ubuntu 24.04 / GCC 13)
- **Never name a project header `features.h`** — it shadows glibc's system `features.h` and causes cascading build failures. Use a unique name like `feature_extraction.h`.
- The `lib/<name>/src/` include path is added to the compiler search path, so any filename that matches a system header will silently shadow it.
- `std::sqrt(float)` is ambiguous on GCC 13 — always cast to `double`: `(float)std::sqrt((double)x)`

## Guidelines for AI Agents

### General
- Do not modify `include/types.h` shared types without updating all dependent files
- Prefer stack allocation over heap allocation (embedded constraints)
- Avoid dynamic memory allocation (`new`/`malloc`) in hot paths
- Keep functions small and focused; limited RAM/flash on microcontroller
- Serial output is for debugging only — guard with `#ifdef DEBUG`
- All sensor reads are non-blocking; do not use `delay()` in production code
- Cross-validate ML models by session/location to avoid data leakage
- Apply int8 post-training quantization before deploying any TFLite model

### Code Style
- **All code must be commented.** The team are electrical engineers, not software engineers. Explain what each function does, what its parameters mean, and why non-obvious decisions were made.
- Comments should be written for someone who understands circuits and signals but may not know C++ idioms or embedded patterns.

### When You Don't Know Something
- **Do not guess or make assumptions** about library APIs, PlatformIO behavior, hardware specs, or toolchain details.
- Instead, **provide a ready-to-use ChatGPT prompt** that the team can paste to get a verified answer, then wait for the result before proceeding.
- Format the prompt clearly so it includes: the platform/toolchain version, the exact error or question, and what a good answer should address.
- This strategy has already been used successfully in this project for PlatformIO + GCC 13 unit testing issues.
