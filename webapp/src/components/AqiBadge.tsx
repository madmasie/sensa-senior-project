import type { AqiLabel } from "../ble/useSensa";
import styles from "./AqiBadge.module.css";

const AQI_CONFIG: Record<AqiLabel, { label: string; description: string; color: string }> = {
  GOOD:          { label: "Good",          description: "Air quality is satisfactory.",                color: "#22c55e" },
  MODERATE:      { label: "Moderate",      description: "Acceptable, but some pollutants present.",   color: "#eab308" },
  UNHEALTHY:     { label: "Unhealthy",     description: "Everyone may begin to experience effects.",  color: "#f97316" },
  VERY_UNHEALTHY:{ label: "Very Unhealthy",description: "Health alert — avoid prolonged exposure.",   color: "#ef4444" },
  HAZARDOUS:     { label: "Hazardous",     description: "Emergency conditions. Stay indoors.",        color: "#7c3aed" },
  UNKNOWN:       { label: "Unknown",       description: "Waiting for sensor data…",                   color: "#6b7280" },
};

export function AqiBadge({ label }: { label: AqiLabel }) {
  const { label: text, color } = AQI_CONFIG[label];
  return (
    <span className={styles.badge} style={{ background: color }}>
      {text}
    </span>
  );
}

export function AqiBanner({ label }: { label: AqiLabel }) {
  const { label: text, description, color } = AQI_CONFIG[label];
  return (
    <div className={styles.banner} style={{ background: color }}>
      <span className={styles.bannerLabel}>{text}</span>
      <span className={styles.bannerDesc}>{description}</span>
    </div>
  );
}
