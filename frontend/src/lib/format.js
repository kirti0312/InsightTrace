// Small, pure text helpers. No React, no browser: tested with `node --test`.

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function isNum(x) {
  return typeof x === "number" && Number.isFinite(x);
}

/** 1234567 -> "1.2M", 15300 -> "15.3k", 0.191 with fmt "pct" -> "19.1%". Used for chart axes. */
export function compact(value, fmt = "int") {
  if (!isNum(value)) return "n/a";
  if (fmt === "pct") return `${(value * 100).toFixed(1).replace(/\.0$/, "")}%`;
  const a = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (a >= 1e9) return `${sign}${(a / 1e9).toFixed(a >= 1e10 ? 0 : 1)}B`;
  if (a >= 1e6) return `${sign}${(a / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e3) return `${sign}${(a / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  return `${sign}${a.toFixed(a < 10 && a % 1 ? 1 : 0)}`;
}

/** A change shown next to a KPI. `isPoints`: the number is a difference between two percentages. */
export function changeText(change, isPoints = false) {
  if (!isNum(change)) return "n/a";
  const sign = change > 0 ? "+" : change < 0 ? "-" : "";
  return isPoints ? `${sign}${Math.abs(change * 100).toFixed(1)} pts` : `${sign}${Math.abs(change * 100).toFixed(1)}%`;
}

/** "up" | "down" | "flat" | "none" for colouring and arrows (never the only signal: the text has the sign too). */
export function direction(change) {
  if (!isNum(change)) return "none";
  if (Math.abs(change) < 0.0005) return "flat";
  return change > 0 ? "up" : "down";
}

/** "2024-03-01" -> a Date at UTC midnight (no time-zone surprises). */
export function parseISO(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ""));
  if (!m) return null;
  return new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
}

/** "2024-03-01" -> "Mar 2024" (month and quarter charts) or "1 Mar 2024" (weekly). */
export function periodLabel(iso, freq = "M") {
  const d = parseISO(iso);
  if (!d) return "";
  const mon = MONTHS[d.getUTCMonth()];
  if (freq === "W") return `${d.getUTCDate()} ${mon} ${d.getUTCFullYear()}`;
  if (freq === "Q") return `Q${Math.floor(d.getUTCMonth() / 3) + 1} ${d.getUTCFullYear()}`;
  return `${mon} ${d.getUTCFullYear()}`;
}

/** Short callout text for an alert, e.g. "FELL 26% VS EXPECTED". Built from numbers, not from the LLM. */
export function alertTag(event) {
  if (!event) return "";
  const word = event.direction === "down" ? "FELL" : "ROSE";
  const v = Math.abs(Number(event.pct_vs_expected));
  if (!Number.isFinite(v)) return word;
  return event.is_points ? `${word} ${(v * 100).toFixed(1)} PTS VS EXPECTED` : `${word} ${(v * 100).toFixed(0)}% VS EXPECTED`;
}

export function plural(n, one, many) {
  return `${n.toLocaleString("en-US")} ${n === 1 ? one : many ?? one + "s"}`;
}
