import { useEffect, useRef, useState } from "react";
import AlertList from "./AlertList.jsx";
import Investigation from "./Investigation.jsx";
import { useSession } from "../state/session.jsx";

const FREQS = [["M", "MONTHLY"], ["Q", "QUARTERLY"], ["W", "WEEKLY"]];

/** Pick an unusual change, press Investigate, and see the verdict with its evidence. */
export default function RootCause({ rc, setRc, focus, clearFocus }) {
  const { client, summary, expired } = useSession();
  const sid = summary.session_id;
  const { kpi, freq, alerts, selected, result, error, running } = rc;
  const resultRef = useRef(null);
  const [useAi, setUseAi] = useState(true);
  const [aiOn, setAiOn] = useState(false);
  const patch = (c) => setRc((r) => ({ ...r, ...c }));

  useEffect(() => {
    client.health().then((h) => setAiOn(!!h?.llm?.configured)).catch(() => {});
  }, [client]);

  // load the alert list whenever the KPI or time step changes
  useEffect(() => {
    let live = true;
    patch({ alerts: null, error: "" });
    client.alerts(sid, { freq, kpi: kpi || undefined, top_n: 10 })
      .then((a) => live && patch({ alerts: a }))
      .catch((e) => live && (e.status === 404 ? expired() : patch({ error: e.message })));
    return () => { live = false; };
  }, [client, sid, kpi, freq]);

  const investigate = async (e) => {
    patch({ selected: e.id, result: null, error: "", running: true });
    try {
      const a = await client.alertWhy(sid, e.kpi, String(e.start).slice(0, 10), e.freq, useAi && aiOn);
      patch({ result: { answer: a, headline: e.headline }, running: false });
      setTimeout(() => resultRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" }), 80);
    } catch (err) {
      if (err.status === 404 && /session/i.test(err.message)) return expired();
      patch({ error: err.message, running: false });
    }
  };

  // arriving from the Overview's Investigate button
  useEffect(() => {
    if (!focus) return;
    patch({ kpi: focus.kpi, freq: focus.freq });
    investigate(focus);
    clearFocus();
  }, [focus]);                                           // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="rootcause">
      <p className="lede">Pick an unusual change. InsightTrace splits it into groups, tests each group with five checks, and tells you if one is a real cause, is not, or if the evidence is too weak to say.</p>
      <div className="rc-controls">
        <label><span className="mono dim">KPI</span>
          <select value={kpi} onChange={(e) => patch({ kpi: e.target.value, selected: "", result: null })}>
            <option value="">All headline KPIs</option>
            {summary.kpis.map((k) => <option key={k.key} value={k.key}>{k.label}</option>)}
          </select>
        </label>
        <div className="seg mono" role="group" aria-label="Time step">
          {FREQS.map(([k, t]) => (
            <button key={k} type="button" aria-pressed={freq === k} className={freq === k ? "on" : ""} onClick={() => patch({ freq: k, selected: "", result: null })}>{t}</button>
          ))}
        </div>
        {aiOn && <label className="toggle"><input type="checkbox" checked={useAi} onChange={(e) => setUseAi(e.target.checked)} /> <span className="mono">AI WORDING</span></label>}
      </div>

      {error && <p className="error" role="alert">{error}</p>}
      <AlertList data={alerts} loading={!alerts && !error} onInvestigate={investigate} selectedId={selected} runningId={running ? selected : ""} />

      <div ref={resultRef}>
        {running && (
          <div className="glass thinking-big" aria-busy="true">
            <div className="thinking mono"><i /><i /><i /> SPLITTING THE CHANGE INTO GROUPS AND RUNNING THE FIVE CHECKS ON EACH…</div>
            <p className="dim">This can take 10 to 30 seconds on a large file.</p>
          </div>
        )}
        {result && !running && <Investigation answer={result.answer} headline={result.headline} />}
      </div>
    </div>
  );
}
