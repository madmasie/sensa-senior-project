#include <gtest/gtest.h>
#include "feature_extraction.h"

// Helper: builds a buffer with n readings where pm2_5 starts at pm_start
// and increases by pm_step each sample. Other fields are fixed constants.
static RingBuffer<10> make_buf(float pm_start, float pm_step, int n) {
    RingBuffer<10> buf;
    Reading r{};
    r.temp_c = 22.0f; r.rh = 50.0f; r.voc_index = 100.0f; r.nox_index = 10.0f;
    for (int i = 0; i < n; i++) {
        r.ts_ms = i * 1000;  // 1 reading per second
        r.pm2_5 = pm_start + i * pm_step;
        buf.push(r);
    }
    return buf;
}

TEST(FeatureExtraction, RequiresTwoSamples) {
    // A single sample isn't enough to compute meaningful statistics
    RingBuffer<10> buf;
    Reading r{}; r.ts_ms = 0;
    buf.push(r);
    FeatureVector f{};
    EXPECT_FALSE(extract_features(buf, f));
}

TEST(FeatureExtraction, Mean) {
    // pm2_5 values: 10, 20, 30 → mean should be 20
    auto buf = make_buf(10.0f, 10.0f, 3);
    FeatureVector f{};
    ASSERT_TRUE(extract_features(buf, f));
    EXPECT_NEAR(20.0f, f.pm2_5_mean, 0.01f);
}

TEST(FeatureExtraction, Std) {
    // pm2_5 values: 10, 20, 30 → std deviation ≈ 8.165
    // Calculated as: sqrt(((10-20)^2 + (20-20)^2 + (30-20)^2) / 3)
    auto buf = make_buf(10.0f, 10.0f, 3);
    FeatureVector f{};
    extract_features(buf, f);
    EXPECT_NEAR(8.165f, f.pm2_5_std, 0.01f);
}

TEST(FeatureExtraction, WindowTimestamps) {
    // Verify the window start/end times and sample count are recorded correctly
    auto buf = make_buf(10.0f, 0.0f, 5);  // 5 samples, 1 second apart
    FeatureVector f{};
    extract_features(buf, f);
    EXPECT_EQ(0u,    f.window_start_ms);
    EXPECT_EQ(4000u, f.window_end_ms);
    EXPECT_EQ(5,     f.sample_count);
}

TEST(FeatureExtraction, EnvMeans) {
    // Verify temperature, humidity, VOC, and NOx means are computed correctly
    auto buf = make_buf(10.0f, 0.0f, 3);
    FeatureVector f{};
    extract_features(buf, f);
    EXPECT_NEAR(22.0f,  f.temp_mean, 0.01f);
    EXPECT_NEAR(50.0f,  f.rh_mean,   0.01f);
    EXPECT_NEAR(100.0f, f.voc_mean,  0.01f);
    EXPECT_NEAR(10.0f,  f.nox_mean,  0.01f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (RUN_ALL_TESTS()) ;
    return 0;
}
