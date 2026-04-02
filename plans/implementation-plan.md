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

---

## 2. SEN55 Sensor Driver — `src/sensor/sen55_sensor.cpp`
**Depends on:** `include/sensor/sensor.h` (done), SEN55 physically wired  
**What it does:** Reads real PM, VOC, NOx, T/RH values from the SEN55 over I2C.  
**What to implement:**
- Subclass `ISensor`; implement `read(Reading& out) -> bool`
- I2C init (address 0x69, 100 kbit/s)
- Start measurement command
- Read measurement command → populate a `Reading` struct
- Warm-up discard (first 30–60 s of readings are unreliable)
- Reference: [Sensirion embedded-i2c-sen5x driver](https://github.com/Sensirion/embedded-i2c-sen5x)
- Swap `MockSensor` for `Sen55Sensor` in `main.cpp` once working

---

## 3. BLE GATT — (new file, e.g. `src/ble/ble_service.cpp`)
**Depends on:** pipeline (done), working hardware  
**What it does:** Broadcasts PM readings and classification label over Bluetooth to a phone/PC.  
**What to implement:**
- BLE service with characteristics for PM2.5, class label, battery, config
- Notify on new classification result
- Replace `delay(1000)` in `main.cpp` with non-blocking timer once BLE is added

---

## 4. ML Calibration Model — `src/ml/calibrate.cpp`
**Depends on:** data collection (off-device), trained TFLite model  
**What it does:** Applies a trained correction to raw PM readings to compensate for sensor drift and T/RH cross-sensitivity.  
**What to implement:**
- Load TFLite model into static tensor arena
- Run inference on a `Reading` → return corrected PM2.5
- Document RAM usage of tensor arena
