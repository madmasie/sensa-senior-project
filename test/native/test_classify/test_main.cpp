#include <gtest/gtest.h>

#include "classify.h"

// Helper: build a FeatureVector with a given pm2_5_mean, other fields zeroed.
static FeatureVector fv(float pm) {
    FeatureVector f{};
    f.pm2_5_mean = pm;
    return f;
}

TEST(Classify, Good) { EXPECT_EQ(Classification::GOOD, classify(fv(0.0f))); }
TEST(Classify, GoodBoundary) { EXPECT_EQ(Classification::GOOD, classify(fv(9.0f))); }
TEST(Classify, Moderate) { EXPECT_EQ(Classification::MODERATE, classify(fv(9.1f))); }
TEST(Classify, ModerateMid) { EXPECT_EQ(Classification::MODERATE, classify(fv(20.0f))); }
TEST(Classify, ModBoundary) { EXPECT_EQ(Classification::MODERATE, classify(fv(35.4f))); }
TEST(Classify, Unhealthy) { EXPECT_EQ(Classification::UNHEALTHY, classify(fv(35.5f))); }
TEST(Classify, UnhealthyMid) { EXPECT_EQ(Classification::UNHEALTHY, classify(fv(80.0f))); }
TEST(Classify, UnhBoundary) { EXPECT_EQ(Classification::UNHEALTHY, classify(fv(125.4f))); }
TEST(Classify, VeryUnhealthy) { EXPECT_EQ(Classification::VERY_UNHEALTHY, classify(fv(125.5f))); }
TEST(Classify, VeryUnhBound) { EXPECT_EQ(Classification::VERY_UNHEALTHY, classify(fv(225.4f))); }
TEST(Classify, Hazardous) { EXPECT_EQ(Classification::HAZARDOUS, classify(fv(225.5f))); }
TEST(Classify, HazardousHigh) { EXPECT_EQ(Classification::HAZARDOUS, classify(fv(500.0f))); }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (RUN_ALL_TESTS())
        ;
    return 0;
}
