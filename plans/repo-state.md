# Repository State — Sensa

_Last updated: 2026-04-01_

---

## What's Done

### Build & Test Infrastructure
- `platformio.ini` configured for both ESP32-S3 target and native desktop testing
- GoogleTest running via `~/.platformio/penv/bin/pio test -e native` (11/11 passing)
- CMake removed — PlatformIO native is the single test runner

### Library Code (`lib/`) — Tested, No Arduino Dependency
- `lib/buffer/src/buffer.h` — `RingBuffer<N>` circular buffer, fully implemented and tested
- `lib/features/src/feature_extraction.h` — `extract_features()`, computes mean/std/timestamps over a window, fully implemented and tested

### Shared Types (`include/types.h`)
- `Reading` struct — PM1/2.5/4/10, temp, RH, VOC, NOx, timestamp
- `Features` struct — window stats (mean, std, timestamps, sample count)
- `Classification` enum — GOOD, MODERATE, UNHEALTHY, VERY_UNHEALTHY, HAZARDOUS, UNKNOWN

### Tests (`test/native/`)
- `test_buffer/` — 6 tests covering push, wrap, ordering, clear
- `test_feature_extraction/` — 5 tests covering mean, std, timestamps, env means

### Documentation
- `AGENTS.md` — coding guidelines, gotchas, ChatGPT prompt strategy
- `docs/project-planning.md` — phase-by-phase project plan
- `docs/customer-specs.md` — requirements, AQI thresholds, BOM
- `docs/project-requirements.md` — full engineering requirements doc (Overleaf source)

---

## What's Empty / Stub Only

### `src/` — All firmware files are stubs (0 bytes or placeholder)
| File | Status |
|------|--------|
| `src/main.cpp` | Placeholder only — no real setup/loop logic |
| `src/sensor/sen55_sensor.cpp` | Empty — SEN55 I2C driver not started |
| `src/sensor/mock_sensor.cpp` | Empty — mock sensor not started |
| `src/processing/buffer.cpp` | Empty — logic lives in `lib/buffer/` |
| `src/processing/feature_extraction.cpp` | Empty — logic lives in `lib/features/` |
| `src/processing/pipeline.cpp` | Empty — pipeline not started |
| `src/processing/classify.cpp` | Empty — classifier not started |
| `src/ml/calibrate.cpp` | Empty — calibration model not started |

### `include/` — All headers are empty stubs
| File | Status |
|------|--------|
| `include/sensor/sensor.h` | Empty |
| `include/processing/feature_extraction.h` | Empty |
| `include/processing/classify.h` | Empty |
| `include/processing/pipeline.h` | Empty |
| `include/ml/calibrate.h` | Empty |

---

## Known Issues / Gotchas
- Never name a header `features.h` — shadows glibc's system header on GCC 13
- `std::sqrt(float)` is ambiguous on GCC 13 — always use `(float)std::sqrt((double)x)`
- See `AGENTS.md` for full list
