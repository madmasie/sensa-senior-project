#include <Arduino.h>

#include "processing/pipeline.h"
#include "sensor/sen55_sensor.h"

// Real SEN55 sensor on SDA=8, SCL=9.
// To revert to the mock sensor: swap Sen55Sensor for MockSensor
// and include "sensor/mock_sensor.h" instead.
static Sen55Sensor sensor;

// Non-blocking 1 Hz timer.
// We store the last time we ran and compare against millis() each loop.
// This avoids delay(), which blocks the CPU and prevents any other work
// (e.g. BLE, power management) from running between samples.
static uint32_t last_tick_ms = 0;
static const uint32_t INTERVAL_MS = 1000;

// Standard pin for the RGB LED on ESP32-S3-DevKitC-1
#define RGB_LED_PIN 38

void setup() {
    Serial.begin(115200);
    // On ESP32-S3 with USB-JTAG, Serial is USB CDC — wait up to 3 s for the
    // host (PC) to open the port, otherwise the first messages are lost.
    unsigned long t = millis();
    while (!Serial && (millis() - t) < 3000);

    if (!sensor.begin()) {
        Serial.println("ERROR: SEN55 failed to start. Check wiring (SDA=8, SCL=9).");
    }
    pipeline_init(&sensor);
}

void loop() {
    // Three fast blinks (Color: Green)
    for (int i = 0; i < 3; i++) {
        // neopixelWrite(pin, red, green, blue) - values from 0 to 255
        // digitalWrite(RGB_LED_PIN, HIGH);
        neopixelWrite(RGB_LED_PIN, 0, 64, 0); 
        delay(100);
        
        // Turn off
        // digitalWrite(RGB_LED_PIN, LOW);
        neopixelWrite(RGB_LED_PIN, 0, 0, 0);  
        delay(100);
    }
    
    // One long pause to create the asymmetrical pattern
    delay(1000);
    uint32_t now = millis();

    // Unsigned subtraction handles the ~49-day millis() rollover correctly.
    if (now - last_tick_ms < INTERVAL_MS) return;
    last_tick_ms = now;

    Classification result = pipeline_tick();

    Serial.print("Classification: ");
    switch (result) {
        case Classification::GOOD:
            Serial.println("GOOD");
            break;
        case Classification::MODERATE:
            Serial.println("MODERATE");
            break;
        case Classification::UNHEALTHY:
            Serial.println("UNHEALTHY");
            break;
        case Classification::VERY_UNHEALTHY:
            Serial.println("VERY_UNHEALTHY");
            break;
        case Classification::HAZARDOUS:
            Serial.println("HAZARDOUS");
            break;
        default:
            Serial.println("UNKNOWN");
            break;
    }
}
