import { useState, useCallback, useRef } from "react";

// Must match UUIDs in ble_service.cpp
const SERVICE_UUID   = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
const CHAR_PM25      = "beb5483e-36e1-4688-b7f5-ea07361b26a8";
const CHAR_LABEL     = "beb5483e-36e1-4688-b7f5-ea07361b26a9";

export type AqiLabel = "GOOD" | "MODERATE" | "UNHEALTHY" | "VERY_UNHEALTHY" | "HAZARDOUS" | "UNKNOWN";

const LABEL_MAP: Record<number, AqiLabel> = {
  0: "GOOD", 1: "MODERATE", 2: "UNHEALTHY",
  3: "VERY_UNHEALTHY", 4: "HAZARDOUS", 255: "UNKNOWN",
};

export interface SensaReading {
  ts: number;   // Date.now()
  pm25: number;
}

export interface SensaState {
  connected: boolean;
  pm25: number | null;
  label: AqiLabel;
  history: SensaReading[];  // rolling window for the chart
}

const HISTORY_MAX = 60; // keep last 60 data points

export function useSensa() {
  const [state, setState] = useState<SensaState>({
    connected: false,
    pm25: null,
    label: "UNKNOWN",
    history: [],
  });

  // Hold a ref to the device so we can disconnect later
  const deviceRef = useRef<BluetoothDevice | null>(null);

  const connect = useCallback(async () => {
    if (!navigator.bluetooth) {
      alert("Web Bluetooth is not supported in this browser. Use Chrome or Edge.");
      return;
    }

    try {
      const device = await navigator.bluetooth.requestDevice({
        filters: [{ name: "Sensa" }],
        optionalServices: [SERVICE_UUID],
      });

      deviceRef.current = device;

      device.addEventListener("gattserverdisconnected", () => {
        setState(s => ({ ...s, connected: false }));
      });

      const server  = await device.gatt!.connect();
      const service = await server.getPrimaryService(SERVICE_UUID);

      // --- PM2.5 characteristic ---
      const pm25Char = await service.getCharacteristic(CHAR_PM25);
      await pm25Char.startNotifications();
      pm25Char.addEventListener("characteristicvaluechanged", (e: Event) => {
        // 4 raw bytes → IEEE 754 little-endian float (matches firmware)
        const view = (e.target as BluetoothRemoteGATTCharacteristic).value!;
        const pm25 = view.getFloat32(0, /*littleEndian=*/true);
        const ts   = Date.now();
        setState(s => ({
          ...s,
          pm25,
          history: [...s.history.slice(-(HISTORY_MAX - 1)), { ts, pm25 }],
        }));
      });

      // --- Label characteristic ---
      const labelChar = await service.getCharacteristic(CHAR_LABEL);
      await labelChar.startNotifications();
      labelChar.addEventListener("characteristicvaluechanged", (e: Event) => {
        const view  = (e.target as BluetoothRemoteGATTCharacteristic).value!;
        const label = LABEL_MAP[view.getUint8(0)] ?? "UNKNOWN";
        setState(s => ({ ...s, label }));
      });

      setState(s => ({ ...s, connected: true }));
    } catch (err) {
      console.error("BLE connect failed:", err);
    }
  }, []);

  const disconnect = useCallback(() => {
    deviceRef.current?.gatt?.disconnect();
    deviceRef.current = null;
    setState(s => ({ ...s, connected: false }));
  }, []);

  return { state, connect, disconnect };
}
