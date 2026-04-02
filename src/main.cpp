#include <Arduino.h>

#include "processing/pipeline.h"
#include "sensor/mock_sensor.cpp"  // include the mock directly until the real driver exists

// Use the mock sensor until the SEN55 driver is implemented.
// To switch to the real sensor: replace MockSensor with Sen55Sensor here.
static MockSensor sensor;

void setup() {
    Serial.begin(115200);
    pipeline_init(&sensor);
}

void loop() {
    Classification result = pipeline_tick();

    // Print PM2.5 label to serial so we can verify the pipeline is working.
    // Labels map to: 0=GOOD, 1=MODERATE, 2=UNHEALTHY, 3=VERY_UNHEALTHY, 4=HAZARDOUS, 255=UNKNOWN
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

    delay(1000);  // 1 Hz sample rate
}
