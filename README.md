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

The firmware supports two hardware targets selectable at build time.

### Dev Board (default)
- **Board:** RYMCU ESP32-S3-DevKitC-1
- Used for early development and testing
- Built-in addressable RGB LED on GPIO38 used as a heartbeat indicator
- Build: `pio run`

### Custom Sensa PCB
- **MCU module:** ESP32-S3-WROOM-1
- **I²C:** SDA = GPIO8, SCL = GPIO9 (10 kΩ pull-ups to 3.3 V on-board)
- **Alerts:** vibration motor via MOSFET on GPIO10; piezo buzzer on GPIO11
- **Status LED:** GPIO16 (`STATUS_LED`) — blinks as heartbeat, flashes on boot
- **Charge LED:** `LED_CHG` driven by MCP73831 charger IC (hardware-only, no firmware needed)
- **USB:** native USB-C (GPIO19/20) with ESD protection and MCP73831 LiPo charging
- **Power:** single-cell LiPo → boost to 5 V for SEN55; 3.3 V rail for ESP32 logic
- Build: `pio run -e sensa-pcb`

#### Alert behavior (PCB only)
When air quality exceeds a threshold, the vibration motor and piezo buzzer fire together in short pulses. The number of pulses indicates severity:

| Classification | Pulses |
|---|---|
| MODERATE | 1 |
| UNHEALTHY | 2 |
| VERY_UNHEALTHY | 3 |
| HAZARDOUS | 3 |

A 30-second cooldown prevents repeated alerts while air quality stays bad.

### Environmental Sensor (both targets)
- **Sensirion SEN55** over I²C
  - PM1 / PM2.5 / PM4 / PM10
  - VOC Index (1–500)
  - NOx Index (1–500)
  - Temperature & Relative Humidity

---

## Building & Flashing

Choose the command set that matches your hardware:

**Dev board (RYMCU ESP32-S3-DevKitC-1):**
```bash
pio run --target upload   # compile and flash
pio device monitor        # open serial output
```

**Custom Sensa PCB (ESP32-S3-WROOM-1):**
```bash
pio run -e sensa-pcb --target upload   # compile and flash
pio device monitor                     # open serial output
```

The `-e sensa-pcb` flag selects the PCB build environment, which enables the STATUS_LED, vibration motor, and buzzer. Without it, you get the dev board build.

### First-time flashing the custom PCB

The first time you flash a brand-new PCB, the ESP32 may not automatically enter programming mode. If the upload hangs or times out, follow these steps:

1. Hold down the **BOOT button** on the PCB (connected to GPIO0)
2. While holding BOOT, press and release the **RESET button** (EN)
3. Release the **BOOT button**
4. Now run the upload command — the ESP32 is now waiting for firmware

After the first successful flash, future uploads should work without this — just plug in and run the command normally.

### If you get a "port not found" or "permission denied" error (Linux/WSL)

The PCB shows up as `/dev/ttyACM0` (or similar). If you see a permission error:
```bash
sudo chmod 666 /dev/ttyACM0
```

---

## Flashing from WSL (Windows + WSL2)

WSL2 can't access USB devices directly — you need to forward the ESP32's USB port from Windows into WSL using **usbipd**.

**One-time setup (Windows, run as Administrator):**
```powershell
winget install usbipd
```

**Every time you plug in the ESP32:**

1. In PowerShell (as Administrator), find the ESP32's bus ID:
   ```powershell
   usbipd list
   ```
   Look for something like `USB Serial` or `CP210x` or `USB JTAG`.

2. Bind it (one-time per device):
   ```powershell
   usbipd bind --busid <BUSID>
   ```

3. Attach it to WSL:
   ```powershell
   usbipd attach --wsl --busid <BUSID>
   ```

4. In WSL, verify it shows up:
   ```bash
   ls /dev/ttyUSB* /dev/ttyACM*
   ```
   You should see something like `/dev/ttyACM0`.

5. If you get a permission denied error on the port:
   ```bash
   sudo chmod 666 /dev/ttyACM0
   ```

Now `pio run --target upload` and `pio device monitor` will work normally from WSL.

**To detach when done** (PowerShell):
```powershell
usbipd detach --busid <BUSID>
```

---



## Web Dashboard

A browser-based dashboard lives in `webapp/`. It connects directly to the ESP32 over **Web Bluetooth** (no drivers, no Python) and displays live readings and charts for all sensor fields.

**Requirements:** Chrome or Edge (Web Bluetooth is not supported in Firefox or Safari).

```bash
cd webapp
npm install
npm run dev   # opens at http://localhost:5173
```

The dashboard subscribes to three BLE characteristics:
- `...26a8` — PM2.5 float (kept for Python client compat)
- `...26a9` — classification label (uint8)
- `...26aa` — full `Reading` struct as a packed 36-byte payload (all PM, temp, RH, VOC, NOx fields)

---

## Unit Tests

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

