#pragma once
#include <cstdint>
#include <cmath>
#include "types.h"
#include "buffer.h"

/*
 * extract_features()
 * Computes a Features struct from the readings currently stored in a RingBuffer.
 *
 * This is the "signal processing" step of the pipeline. Instead of feeding raw
 * sensor samples directly into the ML model, we summarize the window with
 * statistics (mean, std, etc.) that capture both the level and the variability
 * of the signal. This makes the model smaller and more robust.
 *
 * Parameters:
 *   buf  — the ring buffer holding the current window of readings
 *   out  — the Features struct to fill in (passed by reference, modified in place)
 *
 * Returns true on success, false if the buffer has fewer than 2 samples
 * (need at least 2 to compute meaningful statistics).
 *
 * NOTE: This function is a template so it works with any buffer size N.
 * The compiler generates the actual code when you call it with a specific N.
 */
template <uint16_t N>
bool extract_features(const RingBuffer<N>& buf, Features& out) {
    uint16_t n = buf.count();
    if (n < 2) return false;  // not enough data yet

    // Record the time span of this window
    out.window_start_ms = buf[0].ts_ms;
    out.window_end_ms   = buf[n - 1].ts_ms;
    out.sample_count    = n;

    // --- Pass 1: compute means ---
    // Sum all values, then divide by count. Same as averaging a set of
    // voltage measurements on an oscilloscope.
    float sum_pm = 0, sum_temp = 0, sum_rh = 0, sum_voc = 0, sum_nox = 0;
    for (uint16_t i = 0; i < n; i++) {
        sum_pm   += buf[i].pm2_5;
        sum_temp += buf[i].temp_c;
        sum_rh   += buf[i].rh;
        sum_voc  += buf[i].voc_index;
        sum_nox  += buf[i].nox_index;
    }
    out.pm2_5_mean = sum_pm   / n;
    out.temp_mean  = sum_temp / n;
    out.rh_mean    = sum_rh   / n;
    out.voc_mean   = sum_voc  / n;
    out.nox_mean   = sum_nox  / n;

    // --- Pass 2: compute PM2.5 standard deviation ---
    // Std deviation measures how spread out the PM values are around the mean.
    // High std = rapidly changing air quality; low std = stable conditions.
    // Formula: sqrt( sum((x - mean)^2) / n )
    float sq_sum = 0;
    for (uint16_t i = 0; i < n; i++) {
        float d = buf[i].pm2_5 - out.pm2_5_mean;
        sq_sum += d * d;
    }
    // Cast to double before sqrt to avoid GCC 13 ambiguous overload error,
    // then cast back to float for storage.
    out.pm2_5_std = (float)std::sqrt((double)(sq_sum / n));

    return true;
}
