import { useState, useCallback, useRef } from "react";

// Must match UUIDs in ble_service.cpp
const SERVICE_UUID    = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
const CHAR_PM25       = "beb5483e-36e1-4688-b7f5-ea07361b26a8";
const CHAR_LABEL      = "beb5483e-36e1-4688-b7f5-ea07361b26a9";
const CHAR_READING    = "beb5483e-36e1-4688-b7f5-ea07361b26aa";

export type AqiLabel = "GOOD" | "MODERATE" | "UNHEALTHY" | "VERY_UNHEALTHY" | "HAZARDOUS" | "UNKNOWN";

const LABEL_MAP: Record<number, AqiLabel> = {
  0: "GOOD", 1: "MODERATE", 2: "UNHEALTHY",
  3: "VERY_UNHEALTHY", 4: "HAZARDOUS", 255: "UNKNOWN",
};

export interface SensaReading {
  ts: number;       // Date.now() — wall clock time of receipt
  pm1: number;
  pm25: number;
  pm4: number;
  pm10: number;
  tempC: number;
  rh: number;
  voc: number;
  nox: number;
}

export interface SensaState {
  connected: boolean;
  label: AqiLabel;
  latest: SensaReading | null;
  history: SensaReading[];  // rolling window for charts
}

const HISTORY_MAX = 60;

// Unpack the 36-byte Reading payload sent by the firmware.
// Layout (little-endian): uint32 ts_ms, then 8x float32
function unpackReading(view: DataView): Omit<SensaReading, "ts"> {
  return {
    pm1:   view.getFloat32(4,  true),
    pm25:  view.getFloat32(8,  true),
    pm4:   view.getFloat32(12, true),
    pm10:  view.getFloat32(16, true),
    tempC: view.getFloat32(20, true),
    rh:    view.getFloat32(24, true),
    voc:   view.getFloat32(28, true),
    nox:   view.getFloat32(32, true),
  };
}

export function useSensa() {
  const [state, setState] = useState<SensaState>({
    connected: false,
    label: "UNKNOWN",
    latest: null,
    history: [],
  });

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

      // --- Full Reading characteristic (all sensor fields) ---
      const readingChar = await service.getCharacteristic(CHAR_READING);
      await readingChar.startNotifications();
      readingChar.addEventListener("characteristicvaluechanged", (e: Event) => {
        const view = (e.target as BluetoothRemoteGATTCharacteristic).value!;
        const reading: SensaReading = { ts: Date.now(), ...unpackReading(view) };
        setState(s => ({
          ...s,
          latest: reading,
          history: [...s.history.slice(-(HISTORY_MAX - 1)), reading],
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

      // Subscribe to PM2.5 too (keeps Python client working, no-op here)
      const pm25Char = await service.getCharacteristic(CHAR_PM25);
      await pm25Char.startNotifications();

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
