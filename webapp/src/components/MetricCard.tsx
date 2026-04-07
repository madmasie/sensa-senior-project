import styles from "./MetricCard.module.css";

interface Props {
  label: string;
  value: string | number | null;
  unit?: string;
  children?: React.ReactNode; // slot for badge or extra content
}

export function MetricCard({ label, value, unit, children }: Props) {
  return (
    <div className={styles.card}>
      <div className={styles.label}>{label}</div>
      <div className={styles.value}>
        {value ?? "—"}
        {unit && value !== null && <span className={styles.unit}>{unit}</span>}
      </div>
      {children}
    </div>
  );
}
