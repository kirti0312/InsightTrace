// Pure helpers for the Analyst tab (tested with `node --test`).
import { compact } from "./format.js";

/** Split answer text into plain pieces and citation pieces: "Sales fell. [E1, E2]" -> text, cite(E1,E2). */
export function splitCitations(text) {
  const out = [];
  const re = /\[((?:E\d+|F)(?:\s*,\s*(?:E\d+|F))*)\]/g;
  let last = 0;
  let m;
  const s = String(text ?? "");
  while ((m = re.exec(s))) {
    if (m.index > last) out.push({ type: "text", value: s.slice(last, m.index) });
    out.push({ type: "cite", ids: m[1].split(",").map((x) => x.trim()) });
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push({ type: "text", value: s.slice(last) });
  return out;
}

/** Paragraphs = non-empty lines. */
export function paragraphs(text) {
  return String(text ?? "").split(/\n+/).map((l) => l.trimEnd()).filter((l) => l.trim());
}

const VERDICTS = {
  CAUSE: { label: "CAUSE", tone: "cause" },
  NOT_A_CAUSE: { label: "NOT A CAUSE", tone: "not" },
  UNKNOWN: { label: "UNKNOWN", tone: "unknown" },
  ANSWER: { label: "ANSWER", tone: "answer" },
};
export function verdictInfo(v) {
  return VERDICTS[v] || null;
}

const STATUS = {
  ok: null,
  unsupported: { label: "CANNOT ANSWER THIS", tone: "unknown" },
  no_llm: { label: "AI NOT CONNECTED", tone: "unknown" },
  llm_error: { label: "AI UNAVAILABLE", tone: "unknown" },
  failed: { label: "COULD NOT RUN", tone: "unknown" },
  invalid: { label: "NOT UNDERSTOOD", tone: "unknown" },
  error: { label: "ERROR", tone: "unknown" },
};
export function statusInfo(s) {
  return s in STATUS ? STATUS[s] : { label: String(s || "").toUpperCase(), tone: "unknown" };
}

/** Where the wording came from: the AI (checked by the validator) or the plain, code-built summary. */
export function sourceInfo(explanation) {
  if (!explanation) return null;
  return explanation.source === "llm"
    ? { label: "AI WORDING · CHECKED AGAINST THE NUMBERS", tone: "ai" }
    : { label: "PLAIN EVIDENCE SUMMARY", tone: "plain", why: explanation.reason || "" };
}

const nice = (s) => String(s).replace(/_/g, " ");
export function argChips(args) {
  const out = [];
  for (const [k, v] of Object.entries(args || {})) {
    if (v === null || v === undefined || v === "") continue;
    if (typeof v === "object") {
      for (const [a, b] of Object.entries(v)) out.push(`${a} = ${b}`);
    } else out.push(`${nice(k)}: ${v}`);
  }
  return out;
}

/** A forecast result as a chart series: history (actual values) followed by the forecast with its band. */
export function forecastSeries(fc, keepHistory = 24) {
  if (!fc || !Array.isArray(fc.points)) return null;
  const hist = (fc.history || []).slice(-keepHistory).map((h) => ({
    period: h.period, value: h.value, display: compact(h.value, fc.fmt),
    expected: null, lower: null, upper: null, flagged: false, direction: "",
  }));
  // join the band to the last real value so the shaded area starts where the line does
  if (hist.length) {
    const last = hist[hist.length - 1];
    last.lower = last.value;
    last.upper = last.value;
  }
  const fut = fc.points.map((p) => ({
    period: p.period, value: p.forecast, display: compact(p.forecast, fc.fmt),
    expected: p.forecast, lower: p.lo, upper: p.hi, flagged: false, direction: "",
  }));
  return { kpi: fc.kpi, forecastFrom: hist.length, label: `${fc.label} · forecast`, fmt: fc.fmt, freq: { month: "M", quarter: "Q", week: "W" }[fc.freq] || fc.freq || "M", points: [...hist, ...fut], notes: [] };
}

/** Example questions built from the loaded file (real KPI and column names, nothing invented). */
export function suggestQuestions(summary) {
  if (!summary) return [];
  const k = summary.kpis[0]?.label?.toLowerCase() || "revenue";
  const dim = summary.dimensions[0]?.name;
  const year = String(summary.date_to || "").slice(0, 4);
  const q = [];
  if (year) q.push(`What was ${k} in ${year}?`);
  if (dim) q.push(`Which ${dim} had the highest ${k}?`);
  q.push(`Show the ${k} trend by month`);
  q.push(`What unusual changes happened in ${k}?`);
  q.push(`Forecast ${k} for the next 6 months`);
  return q.slice(0, 5);
}

export const QUICK_TOOLS = [
  { key: "kpi_value", label: "KPI value", fields: ["kpi", "start", "end"] },
  { key: "breakdown", label: "Break down by a column", fields: ["kpi", "dimension", "start", "end"] },
  { key: "trend", label: "Trend over time", fields: ["kpi", "start", "end"] },
  { key: "alerts", label: "Unusual changes", fields: ["kpi"] },
  { key: "why", label: "Why did it change?", fields: ["kpi", "start", "end"], needs: ["start", "end"] },
  { key: "forecast", label: "Forecast", fields: ["kpi"] },
];

/** Form values -> tool arguments: only the fields this tool uses, empty ones dropped. Returns {args, error}. */
export function buildToolArgs(tool, form) {
  const spec = QUICK_TOOLS.find((t) => t.key === tool);
  if (!spec) return { args: null, error: "Choose a tool." };
  const args = {};
  for (const f of spec.fields) {
    const v = String(form[f] ?? "").trim();
    if (v) args[f] = v;
  }
  if (spec.fields.includes("kpi") && !args.kpi && tool !== "alerts") return { args: null, error: "Choose a KPI." };
  if (spec.fields.includes("dimension") && !args.dimension) return { args: null, error: "Choose a column." };
  for (const n of spec.needs || []) if (!args[n]) return { args: null, error: "This tool needs a start and an end (a year such as 2024 is fine)." };
  return { args, error: "" };
}
