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

// Standard pin for the RGB LED on ESP32-S3-DevKitC-1
#define RGB_LED_PIN 38

void setup() {
    Serial.begin(115200);
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
    Serial.println("Hello world");
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
