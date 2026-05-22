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
<<<<<<< HEAD
| Custom PCB support | `platformio.ini`, `src/main.cpp`, `include/sensor/sen55_sensor.h` | `sensa-pcb` build env; `SENSA_PCB` flag enables STATUS_LED (GPIO16), motor (GPIO10), buzzer (GPIO11); dev board RGB LED excluded from PCB build |
=======
| Web dashboard | `webapp/` | React + TypeScript + Vite; connects via Web Bluetooth (Chrome/Edge); live metric cards + charts for all 8 sensor fields |
| ML calibration integration | `src/ml/calibrate.cpp` | TFLite Micro inference module; wired into `pipeline_tick()` so PM2.5 is corrected before buffering. Builds in passthrough mode until the model is trained/exported. |
| TFLite → C-array export | `tools/pytorch_calibration/src/export.py` | `export_tflite_as_c_header()` embeds the `.tflite` as `include/calib_model.h`; `main.py` runs it automatically. |
| Transfer-learning trainer | `tools/pytorch_calibration/finetune.py` | Pretrain on AQS → fine-tune on local SEN55+BAM. Produces the 3-feature combined model + firmware headers. `calibrate.cpp` auto-detects 3- vs 8-feature. |
>>>>>>> d9d4ec6 (add fine tuning transfer learning from aqs -> local data)

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
**Status:** Firmware integration done — running in passthrough until a model exists.
**What it does:** Applies a trained correction to raw PM2.5 to compensate for sensor drift and T/RH cross-sensitivity.
**Done:**
- `calibrate.cpp` / `include/ml/calibrate.h` — TFLite Micro module: loads the
  int8 model into a 16 KB static tensor arena, runs inference on a `Reading`,
  returns corrected PM2.5. Reports `arena_used_bytes()` over serial at boot.
- Wired into `pipeline_tick()` — PM2.5 corrected before it enters the buffer.
- `export.py` now emits `include/calib_model.h` (the model as a C byte array).
- Builds today via an `__has_include` guard: passthrough until the model and
  TFLite Micro library are both present.

**Remaining to make it actually calibrate:**
- Produce the model. Recommended (transfer learning): `train_public.py` to
  pretrain on EPA AQS, collect local SEN55+BAM co-location data, then
  `finetune.py` to fine-tune — this writes `calib_model.h` + `calib_scaler.h`.
  (Alternative: `main.py` for the 8-feature local-only model.)
- Verify and uncomment the TFLite Micro `lib_deps` in `platformio.ini`
  (ChatGPT prompt is in that file).
- After the first real build, trim `kTensorArenaSize` in `calibrate.cpp` using
  the `arena_used_bytes()` value printed at boot.
