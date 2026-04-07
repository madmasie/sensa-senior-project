#include <gtest/gtest.h>

#include "pipeline.h"
#include "sensor/sensor.h"
#include "types.h"

// ── Stub sensor ──────────────────────────────────────────────────────────────
// A minimal ISensor implementation we control in tests.
// We can set how many reads fail (simulating warm-up) and what value to return.
struct StubSensor : public ISensor {
    int   fail_count = 0;   // return false this many times before succeeding
    float pm2_5     = 10.0f;

    bool read(Reading& out) override {
        if (fail_count > 0) { fail_count--; return false; }
        out = Reading{};
        out.pm2_5 = pm2_5;
        return true;
    }
};

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(Pipeline, ReturnsUnknownWithNoSensor) {
    // pipeline_tick() before pipeline_init() should never crash — just return UNKNOWN
    pipeline_init(nullptr);
    EXPECT_EQ(Classification::UNKNOWN, pipeline_tick());
}

TEST(Pipeline, ReturnsUnknownWhileSensorFailing) {
    // If the sensor returns false (e.g. during warm-up), result must be UNKNOWN
    StubSensor s;
    s.fail_count = 3;
    pipeline_init(&s);
    EXPECT_EQ(Classification::UNKNOWN, pipeline_tick());
    EXPECT_EQ(Classification::UNKNOWN, pipeline_tick());
    EXPECT_EQ(Classification::UNKNOWN, pipeline_tick());
}

TEST(Pipeline, ReturnsUnknownWhileBufferFilling) {
    // Even after the sensor starts returning data, we need a full window (10 samples)
    // before a classification is possible.
    StubSensor s;
    pipeline_init(&s);
    // Push 9 readings — buffer holds 10, so it's not full yet
    for (int i = 0; i < 9; i++) {
        EXPECT_EQ(Classification::UNKNOWN, pipeline_tick()) << "tick " << i;
    }
}

TEST(Pipeline, ClassifiesAfterFullWindow) {
    // After 10 successful reads the buffer is full and we should get a real label.
    // pm2_5 = 5.0 is well within GOOD range (< 12.1 µg/m³).
    StubSensor s;
    s.pm2_5 = 5.0f;
    pipeline_init(&s);
    Classification result = Classification::UNKNOWN;
    for (int i = 0; i < 10; i++) result = pipeline_tick();
    EXPECT_NE(Classification::UNKNOWN, result);
    EXPECT_EQ(Classification::GOOD, result);
}

TEST(Pipeline, LastReadingUpdatesOnSuccess) {
    // pipeline_last_reading() should reflect the most recent accepted sample
    StubSensor s;
    s.pm2_5 = 42.0f;
    pipeline_init(&s);
    pipeline_tick();
    EXPECT_FLOAT_EQ(42.0f, pipeline_last_reading().pm2_5);
}

TEST(Pipeline, LastReadingZeroBeforeAnySuccess) {
    // Before any successful read, pipeline_last_reading() returns a zeroed Reading
    StubSensor s;
    s.fail_count = 999;
    pipeline_init(&s);
    pipeline_tick();
    EXPECT_FLOAT_EQ(0.0f, pipeline_last_reading().pm2_5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (RUN_ALL_TESTS()) ;
    return 0;
}
