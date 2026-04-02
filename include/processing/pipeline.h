#pragma once
#include "sensor/sensor.h"
#include "types.h"

/*
 * pipeline_init()
 * Sets up the processing pipeline with the given sensor.
 * Call this once in setup().
 *
 * Parameters:
 *   sensor — pointer to a sensor instance (real SEN55 or mock)
 */
void pipeline_init(ISensor* sensor);

/*
 * pipeline_tick()
 * Reads one sample from the sensor, pushes it into the ring buffer, and —
 * once the buffer has enough data — extracts features and returns a classification.
 *
 * Call this once per sample interval (e.g. every 1 s in loop()).
 *
 * Returns the current Classification, or UNKNOWN if the buffer isn't full yet.
 */
Classification pipeline_tick();
