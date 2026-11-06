import { useState } from "react";
import { QUICK_TOOLS, buildToolArgs } from "../lib/analyst.js";

/** Run one of the fixed tools directly (no AI needed): pick a tool, a KPI, a column and dates. */
export default function QuickTools({ summary, busy, onRun }) {
  const [tool, setTool] = useState("kpi_value");
  const [form, setForm] = useState({ kpi: summary.kpis[0]?.key || "", dimension: summary.dimensions[0]?.name || "", start: "", end: "" });
  const [err, setErr] = useState("");
  const spec = QUICK_TOOLS.find((t) => t.key === tool);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const run = (e) => {
    e.preventDefault();
    const { args, error } = buildToolArgs(tool, form);
    setErr(error);
    if (args) onRun(tool, args, `${spec.label}: ${Object.values(args).join(", ")}`);
  };

  return (
    <form className="quick glass" onSubmit={run}>
      <p className="mono eyebrow">QUICK TOOLS · NO AI NEEDED</p>
      <div className="quick-grid">
        <label><span className="mono">TOOL</span>
          <select value={tool} onChange={(e) => { setTool(e.target.value); setErr(""); }}>
            {QUICK_TOOLS.map((t) => <option key={t.key} value={t.key}>{t.label}</option>)}
          </select>
        </label>
        {spec.fields.includes("kpi") && (
          <label><span className="mono">KPI</span>
            <select value={form.kpi} onChange={set("kpi")}>
              {tool === "alerts" && <option value="">All KPIs</option>}
              {summary.kpis.map((k) => <option key={k.key} value={k.key}>{k.label}</option>)}
            </select>
          </label>
        )}
        {spec.fields.includes("dimension") && (
          <label><span className="mono">COLUMN</span>
            <select value={form.dimension} onChange={set("dimension")}>
              {summary.dimensions.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
            </select>
          </label>
        )}
        {spec.fields.includes("start") && (
          <label><span className="mono">FROM</span><input value={form.start} onChange={set("start")} placeholder="2024 or 2024-03" /></label>
        )}
        {spec.fields.includes("end") && (
          <label><span className="mono">TO</span><input value={form.end} onChange={set("end")} placeholder="2024 or 2024-06" /></label>
        )}
        <button type="submit" className="btn mono" disabled={busy}>RUN</button>
      </div>
      {err && <p className="error" role="alert">{err}</p>}
    </form>
  );
}
