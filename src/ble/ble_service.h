#pragma once
#include "types.h"

/*
 * ble_init()
 *
 * Sets up the BLE GATT server with a single "Air Quality" service.
 * Call once in setup() after Serial is ready.
 *
 * The device will advertise as "Sensa" and a phone/PC can connect
 * to read or receive notifications on the characteristics below.
 *
 * Service UUID:       4fafc201-1fb5-459e-8fcc-c5c9c331914b
 * Characteristics:
 *   PM2.5  (notify):  beb5483e-36e1-4688-b7f5-ea07361b26a8  — float, µg/m³
 *   Label  (notify):  beb5483e-36e1-4688-b7f5-ea07361b26a9  — uint8 (Classification enum)
 */
void ble_init();

/*
 * ble_notify(pm2_5, result)
 *
 * Pushes a new PM2.5 value and classification label to any connected BLE client.
 * Call this once per pipeline_tick() that returns a valid result.
 *
 * pm2_5  — raw PM2.5 reading in µg/m³
 * result — Classification enum value
 */
void ble_notify(float pm2_5, Classification result);
