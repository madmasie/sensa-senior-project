import type { AqiLabel } from "../ble/useSensa";
import styles from "./AqiBadge.module.css";

const CONFIG: Record<AqiLabel, { label: string; color: string }> = {
  GOOD:          { label: "Good",          color: "#22c55e" },
  MODERATE:      { label: "Moderate",      color: "#eab308" },
  UNHEALTHY:     { label: "Unhealthy",     color: "#f97316" },
  VERY_UNHEALTHY:{ label: "Very Unhealthy",color: "#ef4444" },
  HAZARDOUS:     { label: "Hazardous",     color: "#7c3aed" },
  UNKNOWN:       { label: "Unknown",       color: "#6b7280" },
};

export function AqiBadge({ label }: { label: AqiLabel }) {
  const { label: text, color } = CONFIG[label];
  return (
    <span className={styles.badge} style={{ background: color }}>
      {text}
    </span>
  );
}
