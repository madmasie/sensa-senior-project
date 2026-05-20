#include <Arduino.h>

#include "ble/ble_service.h"
#include "processing/pipeline.h"
#include "sensor/sen55_sensor.h"

static Sen55Sensor sensor;

// 1 Hz pipeline timer — fires once per second without blocking the CPU.
static uint32_t last_tick_ms = 0;
static const uint32_t INTERVAL_MS = 1000;

#ifdef SENSA_PCB
// ── Custom PCB pin assignments ───────────────────────────────────────────────
#define STATUS_LED_PIN 16
#define MOTOR_PIN      10
#define BUZZER_PIN     11

// LEDC (ESP32 PWM) config for the passive piezo buzzer.
// The buzzer needs a square wave at its resonant frequency to produce sound.
// 3 kHz is a typical resonant frequency for small piezo transducers.
#define BUZZER_CHANNEL   0
#define BUZZER_FREQ_HZ   3000
#define BUZZER_RESOLUTION 8   // 8-bit duty cycle (0–255)
#define BUZZER_DUTY      128  // 50% duty cycle = clean square wave

// How long the motor+buzzer pulse lasts (ms).
#define ALERT_PULSE_MS 500

// Minimum time between alerts (ms). Prevents continuous firing while air is bad.
#define ALERT_COOLDOWN_MS 30000

// Tracks when the last alert fired so we can enforce the cooldown.
static uint32_t last_alert_ms = 0;

// Tracks when the alert pulse started so we can turn it off non-blocking.
// 0 means no alert is currently active.
static uint32_t alert_start_ms = 0;

// Start an alert pulse if the cooldown has elapsed.
// Does not use delay() — the pulse is turned off in loop() after ALERT_PULSE_MS.
static void maybe_trigger_alert() {
    uint32_t now = millis();
    // Enforce cooldown: don't re-trigger if we alerted recently.
    if (now - last_alert_ms < ALERT_COOLDOWN_MS) return;

    digitalWrite(MOTOR_PIN, HIGH);
    ledcWrite(BUZZER_CHANNEL, BUZZER_DUTY);  // start square wave tone
    alert_start_ms = now;   // record when the pulse started
    last_alert_ms  = now;
}

#else
// ── Dev board: built-in addressable RGB LED on GPIO38 ────────────────────────
#define RGB_LED_PIN 38

static void blink_heartbeat() {
    for (int i = 0; i < 3; i++) {
        neopixelWrite(RGB_LED_PIN, 0, 64, 0);
        delay(100);
        neopixelWrite(RGB_LED_PIN, 0, 0, 0);
        delay(100);
    }
    delay(1000);
}
#endif

void setup() {
    Serial.begin(115200);
    unsigned long t = millis();
    while (!Serial && (millis() - t) < 3000);

#ifdef SENSA_PCB
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);
    pinMode(MOTOR_PIN, OUTPUT);
    digitalWrite(MOTOR_PIN, LOW);

    // Set up LEDC PWM for the passive piezo buzzer.
    // ledcSetup configures the channel frequency and resolution.
    // ledcAttachPin connects the channel to the physical GPIO.
    // The buzzer stays silent until we set a non-zero duty cycle.
    ledcSetup(BUZZER_CHANNEL, BUZZER_FREQ_HZ, BUZZER_RESOLUTION);
    ledcAttachPin(BUZZER_PIN, BUZZER_CHANNEL);
    ledcWrite(BUZZER_CHANNEL, 0);  // silent by default

    // Brief boot flash on STATUS_LED to confirm firmware started.
    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(200);
    digitalWrite(STATUS_LED_PIN, LOW);

    // LED_CHG is driven by the MCP73831 charger IC — no firmware needed.
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
    uint32_t now = millis();

#ifdef SENSA_PCB
    // Turn off the alert pulse once ALERT_PULSE_MS has elapsed.
    // This replaces the old delay()-based approach so the CPU isn't blocked.
    if (alert_start_ms != 0 && (now - alert_start_ms >= ALERT_PULSE_MS)) {
        digitalWrite(MOTOR_PIN, LOW);
        ledcWrite(BUZZER_CHANNEL, 0);  // stop tone (duty = 0 → no output)
        alert_start_ms = 0;
    }

    // Single short blink on STATUS_LED as a heartbeat once per pipeline tick.
    // We do this here so it's visible even when the sensor is still warming up.
    if (now - last_tick_ms >= INTERVAL_MS) {
        digitalWrite(STATUS_LED_PIN, HIGH);
        delay(50);
        digitalWrite(STATUS_LED_PIN, LOW);
    }
#else
    blink_heartbeat();
#endif

    // 1 Hz pipeline tick.
    if (now - last_tick_ms < INTERVAL_MS) return;
    last_tick_ms = now;

    Classification result = pipeline_tick();
    ble_notify(pipeline_last_reading(), result);

    Serial.print("Classification: ");
    switch (result) {
        case Classification::GOOD:          Serial.println("GOOD");          break;
        case Classification::MODERATE:      Serial.println("MODERATE");      break;
        case Classification::UNHEALTHY:     Serial.println("UNHEALTHY");
#ifdef SENSA_PCB
            maybe_trigger_alert();
#endif
            break;
        case Classification::VERY_UNHEALTHY: Serial.println("VERY_UNHEALTHY");
#ifdef SENSA_PCB
            maybe_trigger_alert();
#endif
            break;
        case Classification::HAZARDOUS:     Serial.println("HAZARDOUS");
#ifdef SENSA_PCB
            maybe_trigger_alert();
#endif
            break;
        default:                            Serial.println("UNKNOWN");       break;
    }
}
