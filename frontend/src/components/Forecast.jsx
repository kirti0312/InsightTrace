import { useEffect, useState } from "react";
import LineChart from "./LineChart.jsx";
import AnswerCard from "./AnswerCard.jsx";
import { useSession } from "../state/session.jsx";
import { forecastSeries } from "../lib/analyst.js";
import { FREQ_OPTIONS, HORIZONS, coverageInfo, freqUnit, errorText, forecastArgs, skillInfo, statusMeaning, statusTone, tableRows } from "../lib/forecastui.js";

/** Next periods with an honest range, and the back-test that shows how far to trust it. */
export default function Forecast({ fs, setFs }) {
  const { client, summary, expired } = useSession();
  const sid = summary.session_id;
  const { kpi, freq, horizon, result, error, running } = fs;
  const [useAi, setUseAi] = useState(true);
  const [aiOn, setAiOn] = useState(false);
  const patch = (c) => setFs((s) => ({ ...s, ...c }));

  useEffect(() => {
    client.health().then((h) => setAiOn(!!h?.llm?.configured)).catch(() => {});
  }, [client]);

  const run = async () => {
    const { args, error: bad } = forecastArgs({ kpi: kpi || summary.kpis[0]?.key, freq, horizon });
    if (!args) return patch({ error: bad });
    patch({ running: true, error: "" });
    try {
      patch({ result: await client.tool(sid, "forecast", args, useAi && aiOn), running: false });
    } catch (e) {
      if (e.status === 404 && /session/i.test(e.message)) return expired();
      patch({ error: e.message, running: false });
    }
  };

  useEffect(() => { if (!result && !running) run(); }, []);       // eslint-disable-line react-hooks/exhaustive-deps

  const fc = result?.forecast;
  const series = fc && forecastSeries(fc);
  const cov = fc && coverageInfo(fc.validation);
  const skill = fc && skillInfo(fc.validation);
  const rows = fc ? tableRows(fc) : [];

  return (
    <div className="forecast">
      <p className="lede">The next periods, with a range. The range is checked against the past: InsightTrace pretends it is standing at earlier dates, forecasts forward, and counts how often the real value landed inside.</p>

      <form className="fc-controls" onSubmit={(e) => { e.preventDefault(); run(); }}>
        <label><span className="mono dim">KPI</span>
          <select value={kpi || summary.kpis[0]?.key} onChange={(e) => patch({ kpi: e.target.value })}>
            {summary.kpis.map((k) => <option key={k.key} value={k.key}>{k.label}</option>)}
          </select>
        </label>
        <label><span className="mono dim">TIME STEP</span>
          <select value={freq} onChange={(e) => patch({ freq: e.target.value })}>
            {FREQ_OPTIONS.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
          </select>
        </label>
        <label><span className="mono dim">LOOK AHEAD</span>
          <select value={horizon} onChange={(e) => patch({ horizon: e.target.value })}>
            {HORIZONS.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
          </select>
        </label>
        {aiOn && <label className="toggle"><input type="checkbox" checked={useAi} onChange={(e) => setUseAi(e.target.checked)} /> <span className="mono">AI WORDING</span></label>}
        <button type="submit" className="btn mono" disabled={running}>{running ? "FORECASTING…" : "FORECAST"}</button>
      </form>

      {error && <p className="error" role="alert">{error}</p>}
      {running && <div className="skeleton" style={{ height: 320 }} aria-busy="true" />}

      {result && !running && (
        <div className="fc-result">
          {fc ? (
            <>
              <div className="verdict-panel glass">
                <div>
                  <p className="mono eyebrow">CONFIDENCE</p>
                  <span className={`verdict-big badge-${statusTone(fc.status)}`}>{fc.status.replace(/_/g, " ")}</span>
                </div>
                <div className="verdict-text">
                  <p className="verdict-head">{fc.label}: next {fc.points.length} {freqUnit(fc.freq, fc.points.length)}</p>
                  <p className="dim">{statusMeaning(fc.status)}</p>
                  {fc.reason && <p className="dim">{fc.reason}</p>}
                </div>
              </div>

              {series && <LineChart series={series} />}

              <div className="fc-grid">
                <div className="glass fc-table">
                  <p className="mono eyebrow">THE NUMBERS · 80% RANGE</p>
                  <div className="table-wrap tall">
                    <table>
                      <thead><tr><th>Period</th><th>Forecast</th><th>Likely between</th></tr></thead>
                      <tbody>{rows.map((r) => <tr key={r.period}><td>{r.period}</td><td><b>{r.forecast}</b></td><td>{r.range}</td></tr>)}</tbody>
                    </table>
                  </div>
                </div>

                <div className="glass backtest">
                  <p className="mono eyebrow">BACK-TEST · HOW MUCH TO TRUST IT</p>
                  {cov ? (
                    <div className="bt-item">
                      <p className="bt-title">Did the range hold? <span className={`mk-${cov.tone}`}>{cov.verdict}</span></p>
                      <div className="cov-bar" role="img" aria-label={`The ${cov.target}% range held ${cov.actual}% of the time`}>
                        <span className="cov-fill" style={{ width: `${Math.min(100, cov.actual)}%` }} />
                        <span className="cov-target" style={{ left: `${cov.target}%` }} />
                      </div>
                      <p className="dim">It was promised to hold {cov.target}% of the time and actually held {cov.actual}% across {cov.checks} past forecasts.</p>
                    </div>
                  ) : <p className="dim">No back-test was possible: too little history.</p>}
                  {skill && (
                    <div className="bt-item">
                      <p className="bt-title">Better than a simple baseline? <span className={`mk-${skill.tone}`}>{skill.better ? "YES" : "NO"}</span></p>
                      <p className="dim">{skill.text}. {errorText(fc.validation)}.</p>
                    </div>
                  )}
                </div>
              </div>
              {fc.notes?.length > 0 && <ul className="notes">{fc.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
            </>
          ) : null}
          <AnswerCard item={{ id: 0, question: "Forecast", mode: "tool", answer: result }} bare noChart />
        </div>
      )}
    </div>
  );
}
