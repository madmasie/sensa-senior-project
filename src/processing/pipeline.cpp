#include "processing/pipeline.h"
#include "classify.h"
#include "buffer.h"
#include "feature_extraction.h"

// Window size: 10 samples at 1 Hz = 10 seconds of data.
// Adjust N to change the window length.
static RingBuffer<10> s_buf;

// Pointer to whichever sensor was passed to pipeline_init()
static ISensor* s_sensor = nullptr;

void pipeline_init(ISensor* sensor) {
    s_sensor = sensor;
    s_buf.clear();
}

Classification pipeline_tick() {
    if (!s_sensor) return Classification::UNKNOWN;

    // Read one sample from the sensor and add it to the rolling window
    Reading r{};
    if (!s_sensor->read(r)) return Classification::UNKNOWN;
    s_buf.push(r);

    // Wait until the buffer has a full window before classifying.
    // Early readings would be based on too few samples to be meaningful.
    if (!s_buf.full()) return Classification::UNKNOWN;

    // Summarise the window into a feature vector, then classify
    FeatureVector f{};
    if (!extract_features(s_buf, f)) return Classification::UNKNOWN;

    return classify(f);
}
