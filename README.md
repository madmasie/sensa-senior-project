# Sensa  
**Wearable Air Quality Monitoring System with Embedded Edge AI**

---

## Overview

**Sensa** is a wearable air-quality monitoring device that measures particulate matter (PM), gaseous pollutants, temperature, and humidity in real time. Unlike many existing portable monitors, Sensa performs **on-device (edge) AI calibration and classification**, eliminating reliance on cloud processing while improving accuracy in dynamic, real-world environments.

The system integrates a **Sensirion SEN55 environmental sensor** with an **ESP32-S3 microcontroller**, enabling low-power sensing, wireless communication, and embedded machine learning. Sensa is designed to be **portable, affordable, and privacy-preserving**, making professional-grade air-quality awareness accessible for everyday users.

---

## Project Goals

- Measure PM (PM1, PM2.5, PM4, PM10), VOC index, NOx index, temperature, and humidity
- Perform **local sensor calibration and air-quality classification** using embedded AI
- Operate without cloud dependency for low latency and user privacy
- Achieve a **full-day battery life** (≥ 12 hours)
- Remain compact, wearable, and affordable (target cost < $100)
- Validate accuracy using laboratory-grade reference equipment

---

## System Architecture (High Level)
- TODO


---

## Hardware

- **Microcontroller:** ESP32-S3  
  - Dual-core MCU  
  - Wi-Fi + Bluetooth Low Energy  
  - FreeRTOS support  
  - TensorFlow Lite Micro compatible  

- **Environmental Sensor:** Sensirion SEN55  
  - PM1 / PM2.5 / PM4 / PM10  
  - VOC Index (1–500)  
  - NOx Index (1–500)  
  - Temperature & Relative Humidity  
  - I²C interface  

- **Power:** Rechargeable Li-ion battery  
- **Enclosure:** Lightweight, ventilated, wearable form factor  

---

## Running Tests

Unit tests run on your PC (no hardware needed) using GoogleTest via PlatformIO's native environment.

```bash
~/.platformio/penv/bin/pio test -e native
```

Tests live in `test/native/`. Each subdirectory is an independent test suite:
- `test_buffer/` — tests for the ring buffer
- `test_feature_extraction/` — tests for feature extraction

To add a new test suite, create `test/native/test_<name>/test_main.cpp` and follow the pattern in an existing test file.

### Firmware
- **Language:** C++  
- **Framework:** Arduino framework on ESP32  
- **Build System:** PlatformIO  
- **Key Responsibilities:**
  - Sensor interfacing (I²C)
  - Data buffering and rolling averages
  - Feature extraction
  - AI-based calibration and classification
  - BLE / Wi-Fi communication
  - Power management (sleep cycles)

### Embedded AI
- **Framework:** TensorFlow Lite Micro
- **Models:** Lightweight calibration and regression/classification models  
  - Random Forest / SVR / small neural networks
- **Purpose:** Correct sensor drift and environmental cross-sensitivity in real time

