"""
sensa_client.py
---------------
Connects to the Sensa BLE device and prints PM2.5 + classification label
as notifications arrive (once per second after warm-up).

Requirements:
    pip install bleak

Usage:
    python sensa_client.py
"""

import asyncio
import struct
from bleak import BleakScanner, BleakClient

# Must match the UUIDs in ble_service.cpp
CHAR_UUID_PM25  = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CHAR_UUID_LABEL = "beb5483e-36e1-4688-b7f5-ea07361b26a9"

# Maps the uint8 label byte to a human-readable string (matches Classification enum)
LABELS = {0: "GOOD", 1: "MODERATE", 2: "UNHEALTHY", 3: "VERY_UNHEALTHY", 4: "HAZARDOUS", 255: "UNKNOWN"}

# Latest values — notifications for PM2.5 and label arrive independently,
# so we store the most recent of each and print together.
latest = {"pm2_5": None, "label": None}

def on_pm25(_, data: bytearray):
    # Data is 4 raw bytes — unpack as a little-endian float
    latest["pm2_5"] = struct.unpack("<f", bytes(data))[0]
    print_latest()

def on_label(_, data: bytearray):
    latest["label"] = LABELS.get(data[0], f"?({data[0]})")
    print_latest()

def print_latest():
    if latest["pm2_5"] is not None and latest["label"] is not None:
        print(f"PM2.5: {latest['pm2_5']:.1f} µg/m³  |  {latest['label']}")

async def main():
    print("Scanning for 'Sensa'...")
    device = await BleakScanner.find_device_by_name("Sensa", timeout=10)
    if device is None:
        print("ERROR: Could not find 'Sensa'. Make sure the ESP32 is powered and advertising.")
        return

    print(f"Found Sensa at {device.address}. Connecting...")
    async with BleakClient(device) as client:
        print("Connected. Waiting for data (warm-up takes ~40 s)...\n")
        await client.start_notify(CHAR_UUID_PM25,  on_pm25)
        await client.start_notify(CHAR_UUID_LABEL, on_label)
        # Run until Ctrl+C
        await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDisconnected.")
