#include <Arduino.h>

#include "processing/pipeline.h"
#include "sensor/mock_sensor.h"

// Use the mock sensor until the SEN55 driver is implemented.
// To switch to the real sensor: replace MockSensor with Sen55Sensor here.
static MockSensor sensor;

// Non-blocking 1 Hz timer.
// We store the last time we ran and compare against millis() each loop.
// This avoids delay(), which blocks the CPU and prevents any other work
// (e.g. BLE, power management) from running between samples.
static uint32_t last_tick_ms = 0;
static const uint32_t INTERVAL_MS = 1000;

void setup() {
    Serial.begin(115200);
    pipeline_init(&sensor);
}

void loop() {
    uint32_t now = millis();

    // Unsigned subtraction handles the ~49-day millis() rollover correctly.
    if (now - last_tick_ms < INTERVAL_MS) return;
    last_tick_ms = now;

    Classification result = pipeline_tick();

    Serial.print("Classification: ");
    switch (result) {
        case Classification::GOOD:          Serial.println("GOOD");          break;
        case Classification::MODERATE:      Serial.println("MODERATE");      break;
        case Classification::UNHEALTHY:     Serial.println("UNHEALTHY");     break;
        case Classification::VERY_UNHEALTHY:Serial.println("VERY_UNHEALTHY");break;
        case Classification::HAZARDOUS:     Serial.println("HAZARDOUS");     break;
        default:                            Serial.println("UNKNOWN");       break;
    }
}
