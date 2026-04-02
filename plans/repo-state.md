# Repository State — Sensa

_Last updated: 2026-04-01_

---

## What's Done

### Build & Test Infrastructure
- `platformio.ini` configured for both ESP32-S3 target and native desktop testing
- GoogleTest running via `~/.platformio/penv/bin/pio test -e native` (23/23 passing)
- CMake removed — PlatformIO native is the single test runner
- `-Iinclude` added to native build flags so `types.h` is visible to all lib code

### Library Code (`lib/`) — Tested, No Arduino Dependency
- `lib/buffer/src/buffer.h` — `RingBuffer<N>` circular buffer, fully implemented and tested
- `lib/features/src/feature_extraction.h` — `extract_features()`, computes mean/std/timestamps over a window, fully implemented and tested
- `lib/classify/src/classify.h` + `classify.cpp` — `classify()`, threshold-based PM2.5 → Classification mapping, fully implemented and tested

### Shared Types (`include/types.h`)
- `Reading` struct — PM1/2.5/4/10, temp, RH, VOC, NOx, timestamp
- `FeatureVector` struct — window stats (mean, std, timestamps, sample count)
- `Classification` enum — GOOD, MODERATE, UNHEALTHY, VERY_UNHEALTHY, HAZARDOUS, UNKNOWN

### Firmware (`src/`) — Implemented
| File | Status |
|------|--------|
| `src/main.cpp` | setup()/loop() driving pipeline at 1 Hz, serial output |
| `src/sensor/mock_sensor.cpp` | MockSensor — oscillating PM2.5, no hardware needed |
| `src/processing/classify.cpp` | Stub — delegates to `lib/classify/` |
| `src/processing/buffer.cpp` | Stub — logic lives in `lib/buffer/` |
| `src/processing/feature_extraction.cpp` | Stub — logic lives in `lib/features/` |
| `src/processing/pipeline.cpp` | pipeline_init() / pipeline_tick() — fully wired |

### Headers (`include/`) — Implemented
| File | Status |
|------|--------|
| `include/sensor/sensor.h` | ISensor abstract base class |
| `include/processing/classify.h` | classify() declaration |
| `include/processing/pipeline.h` | pipeline_init() / pipeline_tick() declarations |

### Tests (`test/native/`)
- `test_buffer/` — 6 tests covering push, wrap, ordering, clear
- `test_feature_extraction/` — 5 tests covering mean, std, timestamps, env means
- `test_classify/` — 12 tests covering every AQI threshold boundary and midpoint

### Documentation
- `AGENTS.md` — coding guidelines, gotchas, ChatGPT prompt strategy
- `docs/project-planning.md` — phase-by-phase project plan
- `docs/customer-specs.md` — requirements, AQI thresholds, BOM
- `docs/project-requirements.md` — full engineering requirements doc (Overleaf source)

---

## What's Empty / Stub Only

| File | Status |
|------|--------|
| `src/sensor/sen55_sensor.cpp` | Empty — SEN55 I2C driver not started |
| `src/ml/calibrate.cpp` | Empty — calibration model not started |
| `include/processing/feature_extraction.h` | Empty stub |
| `include/ml/calibrate.h` | Empty stub |

---

## Known Issues / Gotchas
- Never name a header `features.h` — shadows glibc's system header on GCC 13
- `std::sqrt(float)` is ambiguous on GCC 13 — always use `(float)std::sqrt((double)x)`
- See `AGENTS.md` for full list
