#include <Arduino.h>

#include "ble/ble_service.h"
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

#ifdef SENSA_PCB
// ── Custom PCB pin assignments ───────────────────────────────────────────────

// STATUS_LED: a standard GPIO-driven LED on the PCB.
// HIGH = on, LOW = off.
#define STATUS_LED_PIN 16

// GPIO10 drives a MOSFET gate that switches the vibration motor on the low side.
// Writing HIGH turns the motor on; LOW turns it off.
#define MOTOR_PIN  10

// GPIO11 drives a piezo buzzer for audible alerts.
#define BUZZER_PIN 11

// How long (ms) to pulse the motor and buzzer when an alert fires.
#define ALERT_PULSE_MS 500

// Pulse the motor and buzzer together for a short burst.
// Called whenever air quality is UNHEALTHY or worse.
static void trigger_alert() {
    digitalWrite(MOTOR_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    delay(ALERT_PULSE_MS);
    digitalWrite(MOTOR_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
}

#else
// ── Dev board: built-in addressable RGB LED on GPIO38 ────────────────────────
// The dev board uses a NeoPixel-style RGB LED driven by neopixelWrite().
// The custom PCB has no such LED, so this block is excluded from that build.
#define RGB_LED_PIN 38

// Three quick green blinks — visual heartbeat so we know the board is alive.
static void blink_heartbeat() {
    for (int i = 0; i < 3; i++) {
        neopixelWrite(RGB_LED_PIN, 0, 64, 0);  // dim green
        delay(100);
        neopixelWrite(RGB_LED_PIN, 0, 0, 0);   // off
        delay(100);
    }
    delay(1000);  // long pause to make the pattern asymmetric / recognizable
}
#endif

void setup() {
    Serial.begin(115200);
    // On ESP32-S3 with USB CDC, wait up to 3 s for the host to open the port
    // so the first log messages aren't lost.
    unsigned long t = millis();
    while (!Serial && (millis() - t) < 3000);

#ifdef SENSA_PCB
    // Configure all PCB output pins, defaulting to off.
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);

    pinMode(MOTOR_PIN, OUTPUT);
    digitalWrite(MOTOR_PIN, LOW);

    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    // Brief flash on STATUS_LED to confirm the firmware has booted.
    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(200);
    digitalWrite(STATUS_LED_PIN, LOW);

    // Note: LED_CHG (charge indicator) is driven by the MCP73831 charger IC
    // directly — it is not connected to a GPIO and requires no firmware code.

    Serial.println("[Sensa] Running on custom PCB.");
#else
    Serial.println("[Sensa] Running on dev board.");
#endif

    if (!sensor.begin()) {
        Serial.println("ERROR: SEN55 failed to start. Check wiring (SDA=8, SCL=9).");
    }
    pipeline_init(&sensor);
    ble_init();
}

void loop() {
#ifdef SENSA_PCB
    // On the PCB, blink STATUS_LED once per cycle as a heartbeat.
    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(50);
    digitalWrite(STATUS_LED_PIN, LOW);
#else
    // On the dev board, use the built-in RGB LED as a heartbeat instead.
    blink_heartbeat();
#endif

    uint32_t now = millis();

    // Unsigned subtraction handles the ~49-day millis() rollover correctly.
    if (now - last_tick_ms < INTERVAL_MS) return;
    last_tick_ms = now;

    Classification result = pipeline_tick();
    // Pass the full reading so BLE clients receive all sensor fields.
    ble_notify(pipeline_last_reading(), result);

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
#ifdef SENSA_PCB
            trigger_alert();
#endif
            break;
        case Classification::VERY_UNHEALTHY:
            Serial.println("VERY_UNHEALTHY");
#ifdef SENSA_PCB
            trigger_alert();
#endif
            break;
        case Classification::HAZARDOUS:
            Serial.println("HAZARDOUS");
#ifdef SENSA_PCB
            trigger_alert();
#endif
            break;
        default:
            Serial.println("UNKNOWN");
            break;
    }
}
