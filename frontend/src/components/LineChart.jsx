import { useMemo, useState } from "react";
import { bandPath, indexForDate, linePath, nearestIndex, niceTicks, valueDomain, xTickIndexes } from "../lib/chart.js";
import { alertTag, compact, periodLabel } from "../lib/format.js";

const W = 880, H = 380, M = { l: 64, r: 22, t: 34, b: 44 };

/** KPI line with the range the detector expected, flagged periods, and callout tags for alerts. */
export default function LineChart({ series, events = [], onPickEvent }) {
  const [hover, setHover] = useState(-1);
  const [table, setTable] = useState(false);

  const g = useMemo(() => {
    const pts = series?.points || [];
    const dom = valueDomain(pts);
    if (!pts.length || !dom) return null;
    const ticks = niceTicks(dom[0], dom[1], 5);
    const lo = ticks[0], hi = ticks[ticks.length - 1];
    const x = (i) => M.l + (pts.length === 1 ? (W - M.l - M.r) / 2 : (i / (pts.length - 1)) * (W - M.l - M.r));
    const y = (v) => M.t + (1 - (v - lo) / (hi - lo || 1)) * (H - M.t - M.b);
    const xs = pts.map((_, i) => x(i));
    const f = (k) => pts.map((p) => (p[k] == null ? null : y(p[k])));
    return { pts, ticks, xs, x, y, ys: f("value"), lows: f("lower"), highs: f("upper") };
  }, [series]);

  if (!g) return <div className="chart-empty mono">NO DATA TO DRAW FOR THIS KPI</div>;
  const { pts, ticks, xs, y, ys } = g;
  const fmt = series.fmt;
  const ff = Number.isInteger(series.forecastFrom) && series.forecastFrom > 0 && series.forecastFrom < pts.length ? series.forecastFrom : -1;
  const isFc = ff > 0;

  const tags = events
    .map((e) => ({ e, i: indexForDate(pts, e.start) }))
    .filter((t) => t.i >= 0)
    .slice(0, 3);

  const onMove = (ev) => {
    const r = ev.currentTarget.getBoundingClientRect();
    setHover(nearestIndex(xs, ((ev.clientX - r.left) / r.width) * W));
  };
  const hp = hover >= 0 ? pts[hover] : null;

  return (
    <figure className="chart glass">
      <figcaption className="chart-head">
        <span className="mono">{series.label.toUpperCase()}</span>
        <span className="chart-legend mono">
          {isFc ? (
            <><i className="lg lg-line" /> HISTORY <i className="lg lg-dash" /> FORECAST <i className="lg lg-band" /> 80% RANGE</>
          ) : (
            <><i className="lg lg-line" /> ACTUAL <i className="lg lg-band" /> EXPECTED RANGE <i className="lg lg-flag" /> UNUSUAL</>
          )}
        </span>
        <button type="button" className="link-btn mono" onClick={() => setTable((v) => !v)} aria-expanded={table}>
          {table ? "HIDE TABLE" : "VIEW AS TABLE"}
        </button>
      </figcaption>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${series.label} over time, ${pts.length} periods, ${pts.filter((p) => p.flagged).length} flagged as unusual`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(-1)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
            <text className="axis" x={M.l - 10} y={y(t) + 4} textAnchor="end">{compact(t, fmt)}</text>
          </g>
        ))}
        {xTickIndexes(pts).map((i) => (
          <text key={i} className="axis" x={xs[i]} y={H - 14} textAnchor="middle">{periodLabel(pts[i].period, series.freq)}</text>
        ))}
        <path className="band" d={bandPath(xs, g.lows, g.highs)} />
        {isFc ? (
          <>
            <line className="split" x1={(xs[ff - 1] + xs[ff]) / 2} x2={(xs[ff - 1] + xs[ff]) / 2} y1={M.t} y2={H - M.b} />
            <text className="axis split-label" x={(xs[ff - 1] + xs[ff]) / 2 + 8} y={M.t + 12}>FORECAST →</text>
            <path className="line" d={linePath(xs.slice(0, ff), ys.slice(0, ff))} pathLength="1" />
            <path className="line fc" d={linePath(xs.slice(ff - 1), ys.slice(ff - 1))} />
          </>
        ) : (
          <path className="line" d={linePath(xs, ys)} pathLength="1" />
        )}
        {pts.map((p, i) =>
          p.flagged && ys[i] != null ? <circle key={i} className="flag" cx={xs[i]} cy={ys[i]} r="5" /> : null,
        )}
        {tags.map(({ e, i }, n) => {
          const tx = Math.min(Math.max(xs[i], M.l + 70), W - M.r - 80);
          const ty = M.t + 4 + n * 30;
          return (
            <g key={e.id} className="callout" onClick={() => onPickEvent?.(e)} tabIndex={0}
               onKeyDown={(k) => (k.key === "Enter" ? onPickEvent?.(e) : null)}>
              <line x1={xs[i]} x2={tx} y1={ys[i] ?? y(0)} y2={ty + 20} />
              <rect x={tx - 74} y={ty} width="148" height="20" />
              <text x={tx} y={ty + 14} textAnchor="middle">{alertTag(e)}</text>
            </g>
          );
        })}
        {hp && (
          <g className="hover">
            <line x1={xs[hover]} x2={xs[hover]} y1={M.t} y2={H - M.b} />
            {ys[hover] != null && <circle cx={xs[hover]} cy={ys[hover]} r="4.5" />}
          </g>
        )}
      </svg>

      {hp && (
        <div className="tooltip mono" style={{ left: `${(xs[hover] / W) * 100}%` }}>
          <b>{periodLabel(hp.period, series.freq)}</b>
          <span>{hp.display}</span>
          {hp.expected != null && <span>EXPECTED {compact(hp.expected, fmt)}</span>}
          {hp.flagged && <span className="tt-flag">UNUSUAL</span>}
        </div>
      )}

      {table && (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Period</th><th>Actual</th><th>Expected</th><th>Unusual?</th></tr></thead>
            <tbody>
              {pts.map((p) => (
                <tr key={p.period} className={p.flagged ? "row-flag" : ""}>
                  <td>{periodLabel(p.period, series.freq)}</td>
                  <td>{p.display}</td>
                  <td>{p.expected == null ? "n/a" : compact(p.expected, fmt)}</td>
                  <td>{p.flagged ? `Yes (${p.direction})` : "No"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {series.notes?.length > 0 && <p className="chart-note mono">{series.notes[0]}</p>}
    </figure>
  );
}
