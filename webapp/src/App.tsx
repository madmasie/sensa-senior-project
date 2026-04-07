import { useSensa } from "./ble/useSensa";
import { AqiBadge } from "./components/AqiBadge";
import { MetricCard } from "./components/MetricCard";
import { PMChart } from "./components/PMChart";
import "./App.css";

export default function App() {
  const { state, connect, disconnect } = useSensa();
  const { connected, pm25, label, history } = state;

  return (
    <div className="app">
      <header className="header">
        <h1>Sensa</h1>
        <p className="subtitle">Air Quality Monitor</p>
        <button
          className={`btn ${connected ? "btn-disconnect" : "btn-connect"}`}
          onClick={connected ? disconnect : connect}
        >
          {connected ? "Disconnect" : "Connect via Bluetooth"}
        </button>
        {connected && <span className="status-dot" title="Connected" />}
      </header>

      <section className="grid">
        <MetricCard label="PM2.5" value={pm25 !== null ? pm25.toFixed(1) : null} unit="µg/m³">
          <AqiBadge label={label} />
        </MetricCard>

        {/* Placeholder cards — wire up when firmware exposes more characteristics */}
        <MetricCard label="PM1.0"  value={null} unit="µg/m³" />
        <MetricCard label="PM4.0"  value={null} unit="µg/m³" />
        <MetricCard label="PM10"   value={null} unit="µg/m³" />
        <MetricCard label="VOC Index"  value={null} />
        <MetricCard label="NOx Index"  value={null} />
        <MetricCard label="Temperature" value={null} unit="°C" />
        <MetricCard label="Humidity"    value={null} unit="%" />
      </section>

      <section className="charts">
        <PMChart
          history={history}
          title="PM2.5 over time"
          dataKey="pm25"
          unit="µg/m³"
        />
      </section>

      {!connected && (
        <p className="hint">
          Requires Chrome or Edge — Web Bluetooth is not supported in Firefox or Safari.
        </p>
      )}
    </div>
  );
}
