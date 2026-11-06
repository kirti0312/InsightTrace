import { alertTag } from "../lib/format.js";

export default function AlertList({ data, loading, onInvestigate, selectedId, runningId }) {
  if (loading) return <div className="skeleton" style={{ height: 120 }} aria-busy="true" />;
  if (!data) return null;
  if (!data.events.length)
    return <p className="empty glass mono">NO UNUSUAL CHANGES FOUND AT THIS SENSITIVITY. THE DATA LOOKS CALM.</p>;
  return (
    <ol className="alerts">
      {data.events.map((e, i) => (
        <li key={e.id} className={`alert glass sev-${e.severity}${selectedId === e.id ? " selected" : ""}`} style={{ "--i": i }}>
          <div className="alert-top">
            <span className="tag mono">{alertTag(e)}</span>
            <span className={`sev mono sev-${e.severity}`}>{String(e.severity).toUpperCase()}</span>
          </div>
          <p>{e.headline}</p>
          {onInvestigate ? (
            <button type="button" className={`btn mono${selectedId === e.id ? "" : " ghost"}`} disabled={!!runningId} onClick={() => onInvestigate(e)}>
              {runningId === e.id ? "INVESTIGATING…" : selectedId === e.id ? "INVESTIGATED ✓ · RUN AGAIN" : "INVESTIGATE →"}
            </button>
          ) : null}
        </li>
      ))}
      {data.total > data.events.length && (
        <li className="mono more">{data.total - data.events.length} MORE ALERTS NOT SHOWN</li>
      )}
    </ol>
  );
}
