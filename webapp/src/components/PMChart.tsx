import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer,
} from "recharts";
import type { SensaReading } from "../ble/useSensa";
import styles from "./PMChart.module.css";

// EPA PM2.5 breakpoints — only shown when dataKey === "pm25"
const PM25_BREAKPOINTS = [
  { value: 12,   label: "Good/Mod",     color: "#eab308" },
  { value: 35.4, label: "Mod/Unhealthy",color: "#f97316" },
  { value: 55.4, label: "Unhealthy",    color: "#ef4444" },
];

interface Props {
  history: SensaReading[];
  title: string;
  dataKey: keyof SensaReading;
  unit: string;
  color?: string;
  showBreakpoints?: boolean;
}

export function PMChart({ history, title, dataKey, unit, color = "#a855f7", showBreakpoints = false }: Props) {
  const data = history.map(r => ({
    ...r,
    time: new Date(r.ts).toLocaleTimeString(),
  }));

  return (
    <div className={styles.container}>
      <div className={styles.title}>{title}</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="time" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 11 }} unit={unit} width={52} />
          <Tooltip
            formatter={(v) => [`${Number(v).toFixed(1)} ${unit}`, title]}
            contentStyle={{ background: "var(--code-bg)", border: "1px solid var(--border)" }}
          />
          {showBreakpoints && PM25_BREAKPOINTS.map(bp => (
            <ReferenceLine key={bp.value} y={bp.value} stroke={bp.color}
              strokeDasharray="4 2" label={{ value: bp.label, fontSize: 10, fill: bp.color }} />
          ))}
          <Line type="monotone" dataKey={dataKey as string} stroke={color}
            dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
