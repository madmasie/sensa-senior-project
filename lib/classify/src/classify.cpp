#include "classify.h"

Classification classify(const FeatureVector& f) {
    float pm = f.pm2_5_mean;

    // Guard against invalid readings (negative or NaN)
    if (pm < 0.0f || pm != pm) return Classification::UNKNOWN;

    // EPA AQI PM2.5 breakpoints (µg/m³)
    if (pm <= 9.0f)   return Classification::GOOD;
    if (pm <= 35.4f)  return Classification::MODERATE;
    if (pm <= 125.4f) return Classification::UNHEALTHY;
    if (pm <= 225.4f) return Classification::VERY_UNHEALTHY;
    return Classification::HAZARDOUS;
}
