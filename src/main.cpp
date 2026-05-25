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
#define BUZZER_DUTY      255  // 100% duty cycle = maximum volume for demo

// How long each motor+buzzer pulse lasts (ms).
#define ALERT_PULSE_MS  400
// Gap between pulses in a multi-burst alert (ms).
#define ALERT_GAP_MS    100
// Minimum time between alerts (ms). Reduced for demo so it fires frequently.
#define ALERT_COOLDOWN_MS 5000

// Tracks when the last alert fired so we can enforce the cooldown.
static uint32_t last_alert_ms = 0;

// --- Non-blocking burst state machine ---
// Instead of using delay(), we track where we are in the burst pattern
// and advance one step per loop() iteration based on elapsed time.
static uint8_t  burst_total    = 0;  // how many pulses to fire total
static uint8_t  burst_count    = 0;  // how many pulses fired so far
static bool     burst_on       = false; // true = currently in a pulse, false = in a gap
static uint32_t burst_timer_ms = 0;  // when the current pulse/gap started

// Start a burst alert with `pulses` pulses if the cooldown has elapsed.
static void maybe_trigger_alert(uint8_t pulses) {
    uint32_t now = millis();
    if (now - last_alert_ms < ALERT_COOLDOWN_MS) return;

    // Kick off the first pulse immediately.
    burst_total    = pulses;
    burst_count    = 0;
    burst_on       = true;
    burst_timer_ms = now;
    last_alert_ms  = now;

    digitalWrite(MOTOR_PIN, HIGH);
    ledcWrite(BUZZER_CHANNEL, BUZZER_DUTY);
}

// Called every loop() iteration to advance the burst state machine.
// Turns the motor/buzzer on and off at the right times without blocking.
static void update_alert() {
    if (burst_total == 0) return;  // no active alert

    uint32_t now = millis();

    if (burst_on) {
        // Currently in a pulse — check if it's time to turn off.
        if (now - burst_timer_ms >= ALERT_PULSE_MS) {
            digitalWrite(MOTOR_PIN, LOW);
            ledcWrite(BUZZER_CHANNEL, 0);
            burst_on       = false;
            burst_timer_ms = now;
            burst_count++;

            // If we've fired all pulses, we're done.
            if (burst_count >= burst_total) burst_total = 0;
        }
    } else {
        // Currently in a gap — check if it's time to fire the next pulse.
        if (now - burst_timer_ms >= ALERT_GAP_MS) {
            digitalWrite(MOTOR_PIN, HIGH);
            ledcWrite(BUZZER_CHANNEL, BUZZER_DUTY);
            burst_on       = true;
            burst_timer_ms = now;
        }
    }
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
    // Advance the burst state machine — turns motor/buzzer on/off at the right times.
    update_alert();

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
        case Classification::MODERATE:      Serial.println("MODERATE");
#ifdef SENSA_PCB
            maybe_trigger_alert(1);  // 1 short burst
#endif
            break;
        case Classification::UNHEALTHY:     Serial.println("UNHEALTHY");
#ifdef SENSA_PCB
            maybe_trigger_alert(2);  // 2 short bursts
#endif
            break;
        case Classification::VERY_UNHEALTHY: Serial.println("VERY_UNHEALTHY");
#ifdef SENSA_PCB
            maybe_trigger_alert(3);  // 3 short bursts
#endif
            break;
        case Classification::HAZARDOUS:     Serial.println("HAZARDOUS");
#ifdef SENSA_PCB
            maybe_trigger_alert(3);  // 3 short bursts (same as VERY_UNHEALTHY max)
#endif
            break;
        default:                            Serial.println("UNKNOWN");       break;
    }
}
