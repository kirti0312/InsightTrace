import { useEffect, useRef, useState } from "react";
import LineChart from "./LineChart.jsx";
import { argChips, forecastSeries, paragraphs, sourceInfo, splitCitations, statusInfo, verdictInfo } from "../lib/analyst.js";

function Badge({ info }) {
  return info ? <span className={`badge badge-${info.tone} mono`}>{info.label}</span> : null;
}

function Cited({ text, onCite }) {
  return paragraphs(text).map((line, i) => (
    <p key={i} className="ans-line">
      {splitCitations(line).map((seg, j) =>
        seg.type === "text" ? (
          <span key={j}>{seg.value}</span>
        ) : (
          seg.ids.map((id) => (
            id === "F" ? (
              <span key={id + j} className="cite mono static" title="Forecast figures">F</span>
            ) : (
              <button key={id + j} type="button" className="cite mono" onClick={() => onCite(id)} title={`Show evidence ${id}`}>
                {id}
              </button>
            )
          ))
        ),
      )}
    </p>
  ));
}

function Record({ rec, hot }) {
  const ref = useRef(null);
  useEffect(() => {
    if (hot) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [hot]);
  const v = verdictInfo(rec.verdict);
  const facts = Object.values(rec.facts || {});
  return (
    <li ref={ref} id={`rec-${rec.id}`} className={`rec${hot ? " hot" : ""}`}>
      <div className="rec-top">
        <span className="cite mono static">{rec.id}</span>
        <b>{rec.title}</b>
        <Badge info={v} />
      </div>
      <p className="rec-text">{rec.text}</p>
      {facts.length > 0 && (
        <dl className="facts">
          {facts.map((f, i) => (
            <div key={i}><dt className="mono">{f.label}</dt><dd>{f.display}</dd></div>
          ))}
        </dl>
      )}
      {rec.checks?.length > 0 && (
        <ul className="checks">
          {rec.checks.map((c, i) => (
            <li key={i} className={`check check-${String(c.status).toLowerCase()}`}>
              <span className="mono check-status">{String(c.status).toUpperCase()}</span> <b>{String(c.name).replace(/_/g, " ")}</b> — {c.detail}
            </li>
          ))}
        </ul>
      )}
      {rec.sql?.length > 0 && (
        <details className="sql">
          <summary className="mono">SQL USED</summary>
          {rec.sql.map((q, i) => (
            <pre key={i}>{q.sql}{q.params?.length ? `\n-- parameters: ${JSON.stringify(q.params)}` : ""}</pre>
          ))}
        </details>
      )}
    </li>
  );
}

/** One question and its answer: wording with clickable citations, how it was computed, and the evidence behind it. */
export default function AnswerCard({ item, bare = false, noChart = false }) {
  const [open, setOpen] = useState(false);
  const [hot, setHot] = useState("");
  const a = item.answer;
  const cite = (id) => {
    setOpen(true);
    setHot(id);
    setTimeout(() => setHot(""), 2400);
  };

  const ev = a?.evidence;
  const verdict = ev && verdictInfo(ev.conclusion);
  const status = a && statusInfo(a.status);
  const src = a && sourceInfo(a.explanation);
  const fc = !noChart && a?.forecast && forecastSeries(a.forecast);
  const chips = a ? argChips(a.args) : [];

  return (
    <article className="qa">
      {!bare && <div className="q-bubble"><span className="mono dim">{item.mode === "tool" ? "QUICK TOOL" : "YOU ASKED"}</span><p>{item.question}</p></div>}
      <div className="a-card glass" aria-live="polite">
        {item.pending && <div className="thinking mono" aria-busy="true"><i /><i /><i /> WORKING OUT THE ANSWER FROM YOUR DATA…</div>}
        {item.error && <p className="error" role="alert">{item.error}</p>}
        {a && (
          <>
            <div className="a-head">
              <Badge info={verdict} />
              <Badge info={status} />
              {!noChart && a.forecast && <Badge info={{ label: a.forecast.status.replace(/_/g, " "), tone: a.forecast.status === "OK" ? "answer" : "unknown" }} />}
            </div>
            <div className={`a-text${status ? " muted" : ""}`}><Cited text={a.text} onCite={cite} /></div>
            {fc && <LineChart series={fc} />}
            {!noChart && a.forecast?.notes?.length > 0 && <ul className="notes">{a.forecast.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
            {a.notes?.length > 0 && <ul className="notes">{a.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
            {ev?.actions?.length > 0 && (
              <div className="actions">
                <p className="mono eyebrow">WHAT TO CHECK NEXT · NOT PROVEN</p>
                <ul>{ev.actions.map((x, i) => <li key={i}>{x.text}</li>)}</ul>
              </div>
            )}
            <footer className="a-foot">
              {a.tool && <span className="mono dim">TOOL · {a.tool.toUpperCase()}</span>}
              {chips.map((c) => <span key={c} className="chip mono">{c}</span>)}
              {src && <span className={`src mono src-${src.tone}`} title={src.why || ""}>{src.label}</span>}
              {ev?.records?.length > 0 && (
                <button type="button" className="link-btn mono" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
                  {open ? "HIDE EVIDENCE" : `SHOW EVIDENCE (${ev.records.length})`}
                </button>
              )}
            </footer>
            {src?.tone === "plain" && src.why && <p className="why-plain mono">{src.why}</p>}
            {open && ev && (
              <div className="evidence">
                <ol className="recs">{ev.records.map((r) => <Record key={r.id} rec={r} hot={hot === r.id} />)}</ol>
                {ev.limits?.length > 0 && <ul className="notes limits">{ev.limits.map((l) => <li key={l}>{l}</li>)}</ul>}
              </div>
            )}
          </>
        )}
      </div>
    </article>
  );
}
