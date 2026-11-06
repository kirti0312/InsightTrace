import { useRef, useState } from "react";
import { useSession } from "../state/session.jsx";
import { plural } from "../lib/format.js";

export default function UploadPanel({ compact = false }) {
  const { summary, busy, error, loadDemo, upload, clear } = useSession();
  const input = useRef(null);
  const [drag, setDrag] = useState(false);

  const pick = (files) => {
    const f = files?.[0];
    if (f) upload(f);
  };

  return (
    <section className="upload glass" aria-label="Load your data">
      {!compact && <p className="mono eyebrow">STEP 1 · LOAD YOUR DATA</p>}
      <div
        className={`dropzone${drag ? " is-drag" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files); }}
      >
        <p className="drop-title">Drop a CSV or Excel file here</p>
        <p className="drop-sub">It needs a date column and a sales (revenue) column. Other columns such as region, product or channel are used to explain changes.</p>
        <input ref={input} type="file" hidden accept=".csv,.tsv,.txt,.xlsx" onChange={(e) => { pick(e.target.files); e.target.value = ""; }} />
        <button type="button" className="btn mono" disabled={busy} onClick={() => input.current?.click()}>
          {busy ? "READING…" : "CHOOSE A FILE"}
        </button>
      </div>

      <div className="demo-row">
        <span className="mono dim">NO FILE? TRY A DEMO</span>
        <button type="button" className="btn ghost mono" disabled={busy} onClick={() => loadDemo("synthetic")}>SYNTHETIC BUSINESS</button>
        <button type="button" className="btn ghost mono" disabled={busy} onClick={() => loadDemo("superstore")}>SUPERSTORE</button>
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      {summary && (
        <div className="summary" aria-live="polite">
          <p className="mono eyebrow">LOADED · {summary.name.toUpperCase()}</p>
          <ul className="chips">
            <li className="chip mono">{plural(summary.rows, "row")}</li>
            <li className="chip mono">{summary.date_from} → {summary.date_to}</li>
            <li className="chip mono">{plural(summary.kpis.length, "KPI")}</li>
            {summary.dimensions.map((d) => (
              <li key={d.name} className="chip mono">{d.name} · {d.values}</li>
            ))}
          </ul>
          {summary.warnings.map((w) => (
            <p key={w} className="warn">⚠ {w}</p>
          ))}
          <button type="button" className="link-btn mono" onClick={clear}>REMOVE THIS DATA</button>
        </div>
      )}
    </section>
  );
}
