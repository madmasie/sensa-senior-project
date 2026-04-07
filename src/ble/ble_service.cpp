#include "ble/ble_service.h"

#include <Arduino.h>
#include <BLE2902.h>
#include <BLECharacteristic.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

// UUIDs identify this service and its characteristics to BLE clients.
// These are randomly generated — they just need to be unique and consistent
// between the firmware and whatever app reads them.
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHAR_UUID_PM25      "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define CHAR_UUID_LABEL     "beb5483e-36e1-4688-b7f5-ea07361b26a9"

// Pointers to the two characteristics so ble_notify() can update them later.
static BLECharacteristic* s_pm25_char  = nullptr;
static BLECharacteristic* s_label_char = nullptr;

// Tracks whether a client is currently connected.
// We only bother sending notifications when someone is listening.
static bool s_connected = false;

// BLE connection callbacks — called automatically by the BLE stack
// when a client connects or disconnects.
class ConnectionCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer*)    override { s_connected = true;  Serial.println("[BLE] Client connected."); }
    void onDisconnect(BLEServer* srv) override {
        s_connected = false;
        Serial.println("[BLE] Client disconnected. Restarting advertising...");
        // Restart advertising so a new client can connect after the previous one leaves.
        srv->startAdvertising();
    }
};

void ble_init() {
    // Initialize the BLE stack and set the device name that appears when scanning.
    BLEDevice::init("Sensa");

    // Create the GATT server and register our connection callbacks.
    BLEServer* server = BLEDevice::createServer();
    server->setCallbacks(new ConnectionCallbacks());

    // Create the Air Quality service.
    BLEService* service = server->createService(SERVICE_UUID);

    // PM2.5 characteristic — clients can subscribe to receive updates (NOTIFY).
    // READ allows a client to poll the value without waiting for a notification.
    s_pm25_char = service->createCharacteristic(
        CHAR_UUID_PM25,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    // BLE2902 is the standard CCCD descriptor — without it, clients can't
    // enable notifications (they get "Write Not Permitted" when trying to subscribe).
    s_pm25_char->addDescriptor(new BLE2902());

    // Classification label characteristic — same pattern as PM2.5.
    s_label_char = service->createCharacteristic(
        CHAR_UUID_LABEL,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    s_label_char->addDescriptor(new BLE2902());

    // Start the service, then begin advertising so clients can find the device.
    service->start();
    BLEDevice::startAdvertising();

    Serial.println("[BLE] Advertising as 'Sensa'.");
}

void ble_notify(float pm2_5, Classification result) {
    // No client connected — nothing to send.
    if (!s_connected) return;

    // Write the PM2.5 float as raw bytes into the characteristic and notify.
    // The client reads these 4 bytes and interprets them as an IEEE 754 float.
    s_pm25_char->setValue(reinterpret_cast<uint8_t*>(&pm2_5), sizeof(pm2_5));
    s_pm25_char->notify();

    // Write the classification label as a single byte (0–4, or 255 for UNKNOWN).
    uint8_t label = static_cast<uint8_t>(result);
    s_label_char->setValue(&label, sizeof(label));
    s_label_char->notify();
}
