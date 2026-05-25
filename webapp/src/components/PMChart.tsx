import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from "recharts";
import type { SensaReading } from "../ble/useSensa";
import styles from "./PMChart.module.css";

interface Props {
  history: SensaReading[];
  title: string;
  dataKey: keyof SensaReading;
  unit: string;
  color?: string;
  showBreakpoints?: boolean; // kept for API compat, no longer used
}

export function PMChart({ history, title, dataKey, unit, color = "#a855f7" }: Props) {
  const data = history.map(r => ({
    ...r,
    time: new Date(r.ts).toLocaleTimeString(),
  }));

  return (
    <div className={styles.container}>
      <div className={styles.title}>{title}</div>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="time" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 11 }} unit={unit} width={52} />
          <Tooltip
            formatter={(v) => [`${Number(v).toFixed(1)} ${unit}`, title]}
            contentStyle={{ background: "var(--code-bg)", border: "1px solid var(--border)" }}
          />
          <Line type="monotone" dataKey={dataKey as string} stroke={color}
            dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
