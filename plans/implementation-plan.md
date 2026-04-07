# Implementation Plan — Sensa Firmware

_Ordered by dependency. Each item lists what needs to exist before you can start it._

---

## ✅ Done

| Step | File | Notes |
|------|------|-------|
| Shared types | `include/types.h` | `Reading`, `FeatureVector`, `Classification` |
| Ring buffer | `lib/buffer/src/buffer.h` | `RingBuffer<N>`, tested |
| Feature extraction | `lib/features/src/feature_extraction.h` | `extract_features()`, tested |
| Sensor interface | `include/sensor/sensor.h` | `ISensor` abstract base class |
| Mock sensor | `src/sensor/mock_sensor.cpp` | Oscillating PM2.5, no hardware needed |
| Classification | `lib/classify/src/classify.cpp` | Threshold-based, tested |
| Pipeline | `src/processing/pipeline.cpp` | `pipeline_init()` / `pipeline_tick()` |
| Main loop | `src/main.cpp` | 1 Hz serial output via mock sensor |
| SEN55 sensor driver | `src/sensor/sen55_sensor.cpp` | Real hardware, I2C confirmed working at 0x69 |
| End-to-end hardware validation | `src/main.cpp` | Classification printing correctly after warm-up; raw sensor values logged each tick |
| BLE GATT service | `src/ble/ble_service.cpp` | Advertises as "Sensa"; notifies PM2.5 (float) and classification label (uint8) on each tick |
| BLE full Reading characteristic | `src/ble/ble_service.cpp` | Added `...26aa` characteristic — sends all 9 sensor fields as packed 36-byte payload (uint32 ts + 8× float32) |
| Pipeline unit tests | `test/native/test_pipeline/` | 6 tests covering warm-up, buffer-filling, classification, and last-reading state |
| Web dashboard | `webapp/` | React + TypeScript + Vite; connects via Web Bluetooth (Chrome/Edge); live metric cards + charts for all 8 sensor fields |

---

## 3. BLE GATT — (new file, e.g. `src/ble/ble_service.cpp`)
**Depends on:** pipeline (done), working hardware  
**What it does:** Broadcasts PM readings and classification label over Bluetooth to a phone/PC.  
**What to implement:**
- ~~BLE service with characteristics for PM2.5, class label, battery, config~~ ✅ PM2.5 + label + full Reading done
- Battery level characteristic
- Config characteristic (sampling rate, window length)
- Replace `delay(1000)` in `main.cpp` with non-blocking timer once BLE is added

---

## 4. ML Calibration Model — `src/ml/calibrate.cpp`
**Depends on:** data collection (off-device), trained TFLite model  
**What it does:** Applies a trained correction to raw PM readings to compensate for sensor drift and T/RH cross-sensitivity.  
**What to implement:**
- Load TFLite model into static tensor arena
- Run inference on a `Reading` → return corrected PM2.5
- Document RAM usage of tensor arena
