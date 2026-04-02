# Implementation Plan — Sensa Firmware

_Ordered by dependency. Each item lists what needs to exist before you can start it._

---

## 1. Mock Sensor — `src/sensor/mock_sensor.cpp`
**Depends on:** `include/types.h` (done)  
**What it does:** Returns fake but realistic `Reading` values so the full pipeline can be tested on the ESP32 without the SEN55 physically connected.  
**What to implement:**
- A function that returns a `Reading` with hardcoded or slowly varying PM2.5 values
- Timestamp using `millis()`

---

## 2. SEN55 Sensor Driver — `src/sensor/sen55_sensor.cpp`
**Depends on:** `include/sensor/sensor.h`, SEN55 I2C wiring  
**What it does:** Reads real PM, VOC, NOx, T/RH values from the SEN55 over I2C.  
**What to implement:**
- I2C init (address 0x69, 100 kbit/s)
- Start measurement command
- Read measurement command → populate a `Reading` struct
- Warm-up discard (first 30–60 s of readings are unreliable)
- Reference: [Sensirion embedded-i2c-sen5x driver](https://github.com/Sensirion/embedded-i2c-sen5x)

---

## 3. Sensor Interface Header — `include/sensor/sensor.h`
**Depends on:** `include/types.h` (done)  
**What it does:** Defines a common interface so `pipeline.cpp` can call either the real sensor or the mock without caring which one it is.  
**What to implement:**
- Abstract base class or function pointer struct with a `read(Reading& out) -> bool` signature

---

## 4. Classification — `src/processing/classify.cpp`
**Depends on:** `include/types.h` (done), `include/processing/classify.h`  
**What it does:** Maps a `Features` struct to a `Classification` enum using PM2.5 thresholds.  
**What to implement:**
- Threshold-based rules using AQI breakpoints from `docs/customer-specs.md`
- This is the baseline model — no ML needed yet

---

## 5. Pipeline — `src/processing/pipeline.cpp`
**Depends on:** buffer (done), feature_extraction (done), classify (step 4), sensor interface (step 3)  
**What it does:** Wires everything together: read → buffer → extract features → classify → return label.  
**What to implement:**
- `pipeline_init()` — set up buffer and sensor
- `pipeline_tick()` — call once per sample interval; returns a `Classification`

---

## 6. Main Loop — `src/main.cpp`
**Depends on:** pipeline (step 5)  
**What it does:** Arduino `setup()` / `loop()` that drives the pipeline and prints results to serial.  
**What to implement:**
- `setup()`: init serial, init pipeline
- `loop()`: call `pipeline_tick()` every ~1 s, print PM2.5 + classification to serial

---

## 7. BLE GATT — (new file TBD)
**Depends on:** pipeline (step 5), working hardware  
**What it does:** Broadcasts PM readings and classification label over Bluetooth to a phone/PC.  
**What to implement:**
- BLE service with characteristics for PM2.5, class label, battery, config
- Notify on new classification result

---

## 8. ML Calibration Model — `src/ml/calibrate.cpp`
**Depends on:** data collection (off-device), trained TFLite model  
**What it does:** Applies a trained correction to raw PM readings to compensate for sensor drift and T/RH cross-sensitivity.  
**What to implement:**
- Load TFLite model into static tensor arena
- Run inference on a `Reading` → return corrected PM2.5
- Document RAM usage of tensor arena

---

## Suggested Order to Start
1. Mock sensor → get data flowing through buffer + feature extraction on real hardware
2. Serial output in `main.cpp` → verify readings look right
3. Classify → see labels printed to serial
4. SEN55 driver → swap mock for real sensor
5. BLE → wireless output
6. ML calibration → accuracy improvement
