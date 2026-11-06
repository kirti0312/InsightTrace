// Pure helpers for the Forecast tab.
import { compact, periodLabel } from "./format.js";

const CODES = { month: "M", quarter: "Q", week: "W", M: "M", Q: "Q", W: "W" };
/** The API reports the step as "M" / "Q" / "W"; the tool arguments use "month" / "quarter" / "week". */
export function freqCode(f) {
  return CODES[f] || "M";
}
export function freqUnit(f, n = 2) {
  const u = { M: ["month", "months"], Q: ["quarter", "quarters"], W: ["week", "weeks"] }[freqCode(f)];
  return n === 1 ? u[0] : u[1];
}

export const FREQ_OPTIONS = [["month", "Monthly"], ["quarter", "Quarterly"], ["week", "Weekly"]];
export const HORIZONS = [["", "Default"], ["3", "3 periods"], ["6", "6 periods"], ["12", "12 periods"], ["18", "18 periods"], ["24", "24 periods"]];

const MEANING = {
  OK: "In past tests this forecast beat the simple baseline and its range held about as often as promised. Still an estimate, not a promise.",
  LOW_CONFIDENCE: "In past tests this forecast was not clearly better than repeating last year, or its range was too narrow. Use it as a rough guide only.",
  UNKNOWN: "There is not enough history to check this forecast, so no numbers are claimed.",
};
export function statusMeaning(status) {
  return MEANING[status] || "";
}
export function statusTone(status) {
  return status === "OK" ? "answer" : "unknown";
}

/** Did the range hold as often as promised? e.g. an 80% range that held 35% of the time is too narrow. */
export function coverageInfo(v) {
  if (!v || typeof v.coverage !== "number") return null;
  const target = Math.round((v.level ?? 0.8) * 100);
  const actual = Math.round(v.coverage * 100);
  const gap = actual - target;
  const verdict = gap >= -10 ? "ABOUT RIGHT" : "TOO NARROW";
  return { target, actual, verdict, tone: gap >= -10 ? "pass" : "fail", checks: v.n_checks ?? 0 };
}

/** Is the forecast more accurate than "repeat the same period last year"? */
export function skillInfo(v) {
  if (!v || typeof v.skill !== "number") return null;
  const pct = Math.round(Math.abs(v.skill) * 100);
  if (v.skill > 0.02) return { better: true, tone: "pass", text: `${pct}% more accurate than repeating last year` };
  if (v.skill < -0.02) return { better: false, tone: "fail", text: `${pct}% less accurate than repeating last year` };
  return { better: false, tone: "unknown", text: "about as accurate as repeating last year" };
}

export function errorText(v) {
  if (!v || typeof v.mae !== "number") return "";
  return `${(v.mae * 100).toFixed(1)}% typical error (repeating last year: ${typeof v.mae_naive === "number" ? (v.mae_naive * 100).toFixed(1) + "%" : "n/a"})`;
}

export function tableRows(fc) {
  if (!fc || !Array.isArray(fc.points)) return [];
  const f = freqCode(fc.freq);
  return fc.points.map((p) => ({
    period: periodLabel(p.period, f),
    forecast: compact(p.forecast, fc.fmt),
    range: `${compact(p.lo, fc.fmt)} to ${compact(p.hi, fc.fmt)}`,
  }));
}

/** Form values -> tool arguments. */
export function forecastArgs({ kpi, freq, horizon }) {
  if (!kpi) return { args: null, error: "Choose a KPI." };
  const args = { kpi, freq: freq || "month" };
  if (horizon) args.horizon = Number(horizon);
  return { args, error: "" };
}
