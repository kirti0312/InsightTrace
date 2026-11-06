// Pure chart maths for the SVG line chart (no React, tested with `node --test`).
import { isNum, parseISO } from "./format.js";

/** "Nice" axis ticks covering [min, max]: niceTicks(0, 87) -> [0, 20, 40, 60, 80, 100]. */
export function niceTicks(min, max, count = 5) {
  if (!isNum(min) || !isNum(max)) return [];
  if (min === max) {
    const pad = Math.abs(min) * 0.1 || 1;
    min -= pad;
    max += pad;
  }
  const span = max - min;
  const raw = span / Math.max(1, count - 1);
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / pow;
  const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * pow;
  const start = Math.floor(min / step) * step;
  const out = [];
  for (let i = 0; i < 12; i++) {
    const v = Number((start + i * step).toFixed(10));
    out.push(v);
    if (v >= max) break;                 // the last tick must reach the top of the data
  }
  return out;
}

/** Value range of everything the chart draws (values, expected range). Returns [min, max] or null. */
export function valueDomain(points) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of points || []) {
    for (const k of ["value", "lower", "upper"]) {
      if (isNum(p[k])) {
        lo = Math.min(lo, p[k]);
        hi = Math.max(hi, p[k]);
      }
    }
  }
  return Number.isFinite(lo) ? [lo, hi] : null;
}

/** SVG path text for a line, with gaps where the value is missing. xs/ys are arrays of pixel positions or null. */
export function linePath(xs, ys) {
  let d = "";
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    if (!isNum(xs[i]) || !isNum(ys[i])) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${xs[i].toFixed(1)},${ys[i].toFixed(1)}`;
    pen = true;
  }
  return d;
}

/** Closed band between lower[] and upper[] (the range the detector expected). Skips gaps. */
export function bandPath(xs, lows, highs) {
  const idx = [];
  for (let i = 0; i < xs.length; i++) if (isNum(xs[i]) && isNum(lows[i]) && isNum(highs[i])) idx.push(i);
  if (idx.length < 2) return "";
  const top = idx.map((i, n) => `${n ? "L" : "M"}${xs[i].toFixed(1)},${highs[i].toFixed(1)}`).join("");
  const bottom = idx.slice().reverse().map((i) => `L${xs[i].toFixed(1)},${lows[i].toFixed(1)}`).join("");
  return `${top}${bottom}Z`;
}

/** Index of the point closest to a pixel x position (for hover). */
export function nearestIndex(xs, px) {
  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < xs.length; i++) {
    if (!isNum(xs[i])) continue;
    const d = Math.abs(xs[i] - px);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/** Which chart point an ISO date falls on (the last point at or before it; clamped). */
export function indexForDate(points, iso) {
  const t = parseISO(iso)?.getTime();
  if (t == null || !points.length) return -1;
  let idx = 0;
  for (let i = 0; i < points.length; i++) {
    const pt = parseISO(points[i].period)?.getTime();
    if (pt != null && pt <= t) idx = i;
  }
  return idx;
}

/** x tick positions: January of each year when the data spans several years, otherwise about 6 evenly spaced points. */
export function xTickIndexes(points, maxTicks = 7) {
  const n = points.length;
  if (n === 0) return [];
  const years = [];
  points.forEach((p, i) => {
    const d = parseISO(p.period);
    if (d && d.getUTCMonth() === 0 && d.getUTCDate() <= 7) years.push(i);
  });
  if (years.length >= 2 && years.length <= maxTicks) return years;
  const step = Math.max(1, Math.round((n - 1) / (maxTicks - 1)));
  const out = [];
  for (let i = 0; i < n; i += step) out.push(i);
  return out;
}
