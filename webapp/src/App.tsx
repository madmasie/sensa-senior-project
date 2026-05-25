import { useSensa } from "./ble/useSensa";
import { useDemo } from "./ble/useDemo";
import { AqiBadge, AqiBanner } from "./components/AqiBadge";
import { MetricCard } from "./components/MetricCard";
import { PMChart } from "./components/PMChart";
import "./App.css";

export default function App() {
  const ble  = useSensa();
  const demo = useDemo();

  // Use demo state when demo is active, otherwise BLE state
  const isDemo = demo.state.connected;
  const { state, connect, disconnect } = isDemo
    ? { state: demo.state, connect: demo.startDemo, disconnect: demo.stopDemo }
    : ble;

  const { connected, label, latest, history } = state;

  const fmt = (v: number | undefined, decimals = 1) =>
    v !== undefined ? v.toFixed(decimals) : null;

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
        {!isDemo && !ble.state.connected && (
          <button className="btn btn-demo" onClick={demo.startDemo}>
            Demo
          </button>
        )}
        {isDemo && (
          <button className="btn btn-disconnect" onClick={demo.stopDemo}>
            Stop Demo
          </button>
        )}
        {connected && <span className="status-dot" title="Connected" />}
      </header>

      <AqiBanner label={label} />

      <section className="grid">
        <MetricCard label="PM2.5" value={fmt(latest?.pm25)} unit="µg/m³">
          <AqiBadge label={label} />
        </MetricCard>
        <MetricCard label="PM1.0"  value={fmt(latest?.pm1)}   unit="µg/m³" />
        <MetricCard label="PM4.0"  value={fmt(latest?.pm4)}   unit="µg/m³" />
        <MetricCard label="PM10"   value={fmt(latest?.pm10)}  unit="µg/m³" />
        <MetricCard label="VOC Index"   value={fmt(latest?.voc, 0)} />
        <MetricCard label="NOx Index"   value={fmt(latest?.nox, 0)} />
        <MetricCard label="Temperature" value={fmt(latest?.tempC)} unit="°C" />
        <MetricCard label="Humidity"    value={fmt(latest?.rh)}    unit="%" />
      </section>

      <section className="charts">
        <PMChart history={history} title="PM2.5" dataKey="pm25" unit="µg/m³" showBreakpoints color="#a855f7" />
        <PMChart history={history} title="PM1.0" dataKey="pm1"  unit="µg/m³" color="#6366f1" />
        <PMChart history={history} title="PM10"  dataKey="pm10" unit="µg/m³" color="#ec4899" />
        <PMChart history={history} title="VOC Index" dataKey="voc" unit="" color="#f59e0b" />
        <PMChart history={history} title="NOx Index" dataKey="nox" unit="" color="#ef4444" />
        <PMChart history={history} title="Temperature" dataKey="tempC" unit="°C" color="#22c55e" />
        <PMChart history={history} title="Humidity"    dataKey="rh"    unit="%" color="#38bdf8" />
      </section>

      {!connected && (
        <p className="hint">
          Requires Chrome or Edge — Web Bluetooth is not supported in Firefox or Safari.
        </p>
      )}
    </div>
  );
}
