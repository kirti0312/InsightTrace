import { changeText, direction } from "../lib/format.js";

const ARROW = { up: "▲", down: "▼", flat: "■", none: "" };

function Delta({ label, value, points }) {
  const dir = direction(value);
  return (
    <span className={`delta delta-${dir}`}>
      <span className="delta-arrow" aria-hidden="true">{ARROW[dir]}</span>
      <span>{changeText(value, points)}</span>
      <span className="delta-label">{label}</span>
    </span>
  );
}

export default function KpiCard({ headline, active, onSelect, index = 0 }) {
  const pts = !!headline.change_is_points;
  return (
    <button
      type="button"
      className={`kpi-card glass${active ? " is-active" : ""}`}
      style={{ "--i": index }}
      onClick={() => onSelect?.(headline.key)}
      aria-pressed={!!active}
    >
      <span className="mono kpi-label">{headline.label}</span>
      <span className="kpi-value">{headline.value_display}</span>
      <Delta label="vs last period" value={headline.change_vs_previous} points={pts} />
      <Delta label="vs last year" value={headline.change_vs_year_ago} points={pts} />
    </button>
  );
}
