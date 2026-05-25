import { useState, useRef, useCallback } from "react";
import type { SensaState, SensaReading, AqiLabel } from "./useSensa";

const HISTORY_MAX = 60;

// Each phase: which AQI level and how many seconds to stay there.
// The sequence is hand-crafted to feel like a realistic outdoor walk —
// starts clean, drifts up, dips back, climbs again, etc.
// Timings are randomized at runtime within the [min, max] range below.
const SEQUENCE: { label: AqiLabel; pm25: number; minS: number; maxS: number }[] = [
  { label: "GOOD",           pm25:   5, minS: 8,  maxS: 14 },
  { label: "MODERATE",       pm25:  20, minS: 3,  maxS:  6 },
  { label: "GOOD",           pm25:   5, minS: 4,  maxS:  8 },
  { label: "MODERATE",       pm25:  20, minS: 6,  maxS: 12 },
  { label: "UNHEALTHY",      pm25:  80, minS: 3,  maxS:  5 },
  { label: "MODERATE",       pm25:  20, minS: 2,  maxS:  4 },
  { label: "UNHEALTHY",      pm25:  80, minS: 5,  maxS: 10 },
  { label: "VERY_UNHEALTHY", pm25: 175, minS: 3,  maxS:  6 },
  { label: "UNHEALTHY",      pm25:  80, minS: 2,  maxS:  5 },
  { label: "VERY_UNHEALTHY", pm25: 175, minS: 6,  maxS: 10 },
  { label: "HAZARDOUS",      pm25: 260, minS: 4,  maxS:  8 },
  { label: "VERY_UNHEALTHY", pm25: 175, minS: 3,  maxS:  6 },
  { label: "UNHEALTHY",      pm25:  80, minS: 4,  maxS:  8 },
  { label: "MODERATE",       pm25:  20, minS: 5,  maxS: 10 },
  { label: "GOOD",           pm25:   5, minS: 8,  maxS: 14 },
];

function randInt(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

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

  const intervalRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const phaseRef       = useRef(0);
  const ticksInPhase   = useRef(0);
  const phaseDuration  = useRef(randInt(SEQUENCE[0].minS, SEQUENCE[0].maxS));

  const startDemo = useCallback(() => {
    phaseRef.current      = 0;
    ticksInPhase.current  = 0;
    phaseDuration.current = randInt(SEQUENCE[0].minS, SEQUENCE[0].maxS);

    const p0 = SEQUENCE[0];
    const r0 = fakeReading(p0.pm25, null);
    setState({ connected: true, label: p0.label, latest: r0, history: [r0] });

    intervalRef.current = setInterval(() => {
      ticksInPhase.current++;
      if (ticksInPhase.current >= phaseDuration.current) {
        ticksInPhase.current = 0;
        phaseRef.current = (phaseRef.current + 1) % SEQUENCE.length;
        const next = SEQUENCE[phaseRef.current];
        phaseDuration.current = randInt(next.minS, next.maxS);
      }

      const { label, pm25 } = SEQUENCE[phaseRef.current];

      setState(s => {
        const r = fakeReading(pm25, s.latest);
        return { ...s, label, latest: r, history: [...s.history.slice(-(HISTORY_MAX - 1)), r] };
      });
    }, 1000);
  }, []);

  const stopDemo = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setState(s => ({ ...s, connected: false }));
  }, []);

  return { state, startDemo, stopDemo };
}
