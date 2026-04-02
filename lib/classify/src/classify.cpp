#include "classify.h"

Classification classify(const FeatureVector& f) {
    float pm = f.pm2_5_mean;

    if (pm <= 9.0f) return Classification::GOOD;
    if (pm <= 35.4f) return Classification::MODERATE;
    if (pm <= 125.4f) return Classification::UNHEALTHY;
    if (pm <= 225.4f) return Classification::VERY_UNHEALTHY;
    if (pm > 225.4f) return Classification::HAZARDOUS;

    return Classification::UNKNOWN;
}
