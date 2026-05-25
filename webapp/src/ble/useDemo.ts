import { useState, useRef, useCallback } from "react";
import type { SensaState, SensaReading, AqiLabel } from "./useSensa";

const HISTORY_MAX = 60;

// PM2.5 centre values (µg/m³) for each AQI phase, in order.
// The demo cycles through all 5 levels, spending PHASE_DURATION_S seconds each.
const PHASES: { label: AqiLabel; pm25: number }[] = [
  { label: "GOOD",          pm25:   5 },
  { label: "MODERATE",      pm25:  20 },
  { label: "UNHEALTHY",     pm25:  80 },
  { label: "VERY_UNHEALTHY",pm25: 175 },
  { label: "HAZARDOUS",     pm25: 260 },
];

const PHASE_DURATION_S = 3; // seconds per AQI level → full cycle = 15 s

function fakeReading(pm25Target: number, prev: SensaReading | null): SensaReading {
  const jitter = (range: number) => (Math.random() - 0.5) * range;
  const pm25 = Math.max(0, pm25Target + jitter(4));
  return {
    ts:    Date.now(),
    pm25,
    pm1:   pm25 * 0.6  + jitter(1),
    pm4:   pm25 * 1.2  + jitter(1),
    pm10:  pm25 * 1.5  + jitter(2),
    tempC: (prev?.tempC ?? 22) + jitter(0.4),
    rh:    (prev?.rh    ?? 45) + jitter(2),
    voc:   100 + pm25 * 0.5 + jitter(10),
    nox:   10  + pm25 * 0.1 + jitter(3),
  };
}

export function useDemo() {
  const [state, setState] = useState<SensaState>({
    connected: false,
    label: "UNKNOWN",
    latest: null,
    history: [],
  });

  const intervalRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const phaseRef     = useRef(0);   // current phase index
  const ticksInPhase = useRef(0);   // how many 1-s ticks spent in this phase

  const startDemo = useCallback(() => {
    phaseRef.current     = 0;
    ticksInPhase.current = 0;

    const phase0 = PHASES[0];
    const r0 = fakeReading(phase0.pm25, null);
    setState({ connected: true, label: phase0.label, latest: r0, history: [r0] });

    intervalRef.current = setInterval(() => {
      // Advance phase after PHASE_DURATION_S ticks
      ticksInPhase.current++;
      if (ticksInPhase.current >= PHASE_DURATION_S) {
        ticksInPhase.current = 0;
        phaseRef.current = (phaseRef.current + 1) % PHASES.length;
      }

      const { label, pm25 } = PHASES[phaseRef.current];

      setState(s => {
        const r = fakeReading(pm25, s.latest);
        return {
          ...s,
          label,
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
