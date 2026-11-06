import { useEffect, useState } from "react";
import KpiCard from "./KpiCard.jsx";
import LineChart from "./LineChart.jsx";
import AlertList from "./AlertList.jsx";
import { useSession } from "../state/session.jsx";

const FREQS = [["M", "MONTHLY"], ["Q", "QUARTERLY"], ["W", "WEEKLY"]];

/** Headline KPIs, a chart of the chosen KPI, and the list of unusual changes found. */
export default function Overview({ compact = false, onInvestigate }) {
  const { client, summary, expired } = useSession();
  const sid = summary.session_id;
  const [freq, setFreq] = useState("M");
  const [kpi, setKpi] = useState(summary.kpis[0]?.key);
  const [heads, setHeads] = useState(null);
  const [series, setSeries] = useState(null);
  const [alerts, setAlerts] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => { setKpi(summary.kpis[0]?.key); }, [sid]);   // new dataset: start with its first KPI

  const fail = (e) => (e.status === 404 ? expired() : setErr(e.message));

  useEffect(() => {
    let live = true;
    setHeads(null);
    client.headlines(sid, "M").then((r) => live && setHeads(r)).catch((e) => live && fail(e));
    return () => { live = false; };
  }, [client, sid]);

  useEffect(() => {
    if (!kpi) return undefined;
    let live = true;
    setSeries(null);
    setAlerts(null);
    setErr("");
    Promise.all([client.series(sid, kpi, freq), client.alerts(sid, { freq, kpi, top_n: 8 })])
      .then(([s, a]) => { if (live) { setSeries(s); setAlerts(a); } })
      .catch((e) => live && fail(e));
    return () => { live = false; };
  }, [client, sid, kpi, freq]);

  return (
    <div className="overview">
      <div className="kpi-grid">
        {heads
          ? heads.headlines.map((h, i) => <KpiCard key={h.key} headline={h} index={i} active={h.key === kpi} onSelect={setKpi} />)
          : Array.from({ length: Math.min(4, summary.kpis.length) }, (_, i) => <div key={i} className="skeleton kpi-skel" aria-busy="true" />)}
      </div>
      {heads?.problems?.length > 0 && <p className="warn">⚠ {heads.problems[0]}</p>}

      <div className="seg mono" role="group" aria-label="Time step">
        {FREQS.map(([k, t]) => (
          <button key={k} type="button" aria-pressed={freq === k} className={freq === k ? "on" : ""} onClick={() => setFreq(k)}>{t}</button>
        ))}
      </div>

      {err && <p className="error" role="alert">{err}</p>}
      {series ? <LineChart series={series} events={alerts?.events || []} /> : <div className="skeleton" style={{ height: 380 }} aria-busy="true" />}

      {!compact && (
        <>
          <h3 className="mono eyebrow">WHAT CHANGED · {alerts ? `${alerts.total} FOUND` : "LOOKING…"}</h3>
          <AlertList data={alerts} loading={!alerts && !err} onInvestigate={onInvestigate} />
        </>
      )}
    </div>
  );
}
