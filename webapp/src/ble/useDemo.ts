import { useState, useRef, useCallback } from "react";
import type { SensaState, SensaReading, AqiLabel } from "./useSensa";

const HISTORY_MAX = 60;

function fakeReading(prev: SensaReading | null): SensaReading {
  // Walk each value with a small random step so charts look like real sensor data
  const walk = (v: number, step: number, min: number, max: number) =>
    Math.min(max, Math.max(min, v + (Math.random() - 0.5) * step));

  const base = prev ?? { ts: 0, pm1: 4, pm25: 8, pm4: 10, pm10: 12, tempC: 22, rh: 45, voc: 100, nox: 10 };
  return {
    ts:    Date.now(),
    pm1:   walk(base.pm1,   1,  0, 50),
    pm25:  walk(base.pm25,  2,  0, 80),
    pm4:   walk(base.pm4,   1,  0, 80),
    pm10:  walk(base.pm10,  2,  0, 100),
    tempC: walk(base.tempC, 0.2, 15, 35),
    rh:    walk(base.rh,    1,  20, 90),
    voc:   walk(base.voc,   10, 1, 500),
    nox:   walk(base.nox,   3,  1, 500),
  };
}

function labelFor(pm25: number): AqiLabel {
  if (pm25 <= 12)   return "GOOD";
  if (pm25 <= 35.4) return "MODERATE";
  if (pm25 <= 55.4) return "UNHEALTHY";
  if (pm25 <= 150)  return "VERY_UNHEALTHY";
  return "HAZARDOUS";
}

export function useDemo() {
  const [state, setState] = useState<SensaState>({
    connected: false,
    label: "UNKNOWN",
    latest: null,
    history: [],
  });

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startDemo = useCallback(() => {
    // Seed with an initial reading immediately so charts aren't empty
    const r0 = fakeReading(null);
    setState({ connected: true, label: labelFor(r0.pm25), latest: r0, history: [r0] });

    intervalRef.current = setInterval(() => {
      setState(s => {
        const r = fakeReading(s.latest);
        return {
          ...s,
          label: labelFor(r.pm25),
          latest: r,
          history: [...s.history.slice(-(HISTORY_MAX - 1)), r],
        };
      });
    }, 1000);
  }, []);

  const stopDemo = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setState(s => ({ ...s, connected: false }));
  }, []);

  return { state, startDemo, stopDemo };
}
