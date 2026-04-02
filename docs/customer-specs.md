# Customer Specs — Sensa

## Target Users
People who want real-time awareness of their air quality exposure:
- Urban commuters, construction/industrial workers, lab researchers
- Health-conscious individuals tracking daily environmental conditions

---

## Customer Requirements

| Requirement    | Target                                                                 |
|----------------|------------------------------------------------------------------------|
| Price          | ≤$100 total BOM                                                        |
| Battery life   | ≥8–12 hours continuous (full workday)                                  |
| Accuracy       | ≥85% class agreement vs. reference sensor; self-calibrating on-device  |
| Latency        | Real-time updates; end-to-end <150 ms                                  |
| Connectivity   | BLE to phone/PC; serial monitor for early dev                          |
| Usability      | Minimal setup; intuitive app with color/visual indicators              |
| Portability    | Wearable, lightweight, ventilated enclosure                            |
| Privacy        | All processing local — no cloud dependency                             |
| Sustainability | Rechargeable battery; durable components                               |

---

## Engineering Specs

### Core Components

| Component         | Spec / Notes                                                                 |
|-------------------|------------------------------------------------------------------------------|
| PM Sensor         | SEN55 — PM1/2.5/4/10, VOC Index, NOx Index, T/RH; I²C @ 0x69; 5V           |
| MCU               | ESP32-S3 — TFLite Micro, FreeRTOS, BLE/Wi-Fi, OTA; <3 in² board             |
| Battery           | Li-ion 3.7V 2000 mAh; target ≥12 h at 1–2 Hz                               |
| Enclosure         | ~3×3×3 in, <2 lbs; ventilated; clip/strap mount                             |
| AI Model          | TinyML — Random Forest or small NN (1–2 layers); TFLite Micro int8           |
| Communication     | BLE GATT (PM, class label, battery, config); Adaptive Frequency Hopping      |

### Power Strategy
- MCU + sensor sleep between reads
- Wake every ~60 s → take 10 s of data → run ML → transmit over BLE → sleep
- Target: <1% BLE packet loss; ~1 Mbps in noisy RF environments

### AQI Classification Thresholds (PM2.5 µg/m³)

| Class          | Range (µg/m³) | Health Guidance                                              |
|----------------|---------------|--------------------------------------------------------------|
| Good           | 0 – 9.0       | Great for outdoor activity                                   |
| Moderate       | 9.1 – 35.4    | Sensitive groups limit prolonged exertion                    |
| Unhealthy      | 35.5 – 125.4  | Sensitive groups avoid; others limit exertion                |
| Very Unhealthy | 125.5 – 225.4 | Sensitive groups avoid all outdoor activity                  |
| Hazardous      | 225.5+        | Everyone avoids outdoor activity                             |

Source: [NPS Air Quality & Human Health](https://www.nps.gov/subjects/air/humanhealth-pm.htm)

### VOC Concern Levels

| Range (ppb)   | Level              |
|---------------|--------------------|
| 0 – 500       | Low concern        |
| 500 – 1,500   | Moderate concern   |
| 1,500 – 3,000 | High concern       |
| 3,000+        | Very high concern  |

Source: [Evergreen Air Quality](https://evergreenairquality.com/blog/what-level-of-voc-is-dangerous/)

---

## BOM (Approximate)

| Part                          | Cost     |
|-------------------------------|----------|
| SEN55 sensor                  | ~$40     |
| SEN5x jumper wire (JST GHR)   | ~$8      |
| ESP32-S3 dev board            | ~$18     |
| Li-ion battery 3.7V 2000 mAh  | ~$12     |
| Misc (resistors, PCB, etc.)   | ~$20     |
| **Total (est.)**              | **~$98** |
