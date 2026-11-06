import test from "node:test";
import assert from "node:assert/strict";
import { splitCitations, paragraphs, verdictInfo, statusInfo, sourceInfo, argChips, forecastSeries, suggestQuestions, buildToolArgs } from "../src/lib/analyst.js";

test("citations are split out", () => {
  assert.deepEqual(splitCitations("Sales fell. [E1] Then X [E2, E3] ok."), [
    { type: "text", value: "Sales fell. " }, { type: "cite", ids: ["E1"] },
    { type: "text", value: " Then X " }, { type: "cite", ids: ["E2", "E3"] }, { type: "text", value: " ok." }]);
  assert.deepEqual(splitCitations("no cites [note] here"), [{ type: "text", value: "no cites [note] here" }]);
  assert.deepEqual(splitCitations(null), []);
  assert.deepEqual(splitCitations("[E10]"), [{ type: "cite", ids: ["E10"] }]);
  assert.deepEqual(splitCitations("Up. [F]"), [{ type: "text", value: "Up. " }, { type: "cite", ids: ["F"] }]);
});

test("paragraphs drop blank lines", () => {
  assert.deepEqual(paragraphs("a\n\nb \n  \nc"), ["a", "b", "c"]);
  assert.deepEqual(paragraphs(undefined), []);
});

test("verdict, status, source", () => {
  assert.equal(verdictInfo("NOT_A_CAUSE").label, "NOT A CAUSE");
  assert.equal(verdictInfo(""), null);
  assert.equal(statusInfo("ok"), null);
  assert.equal(statusInfo("no_llm").label, "AI NOT CONNECTED");
  assert.equal(statusInfo("weird").label, "WEIRD");
  assert.equal(sourceInfo(null), null);
  assert.equal(sourceInfo({ source: "llm" }).tone, "ai");
  assert.equal(sourceInfo({ source: "fallback", reason: "r" }).why, "r");
});

test("argument chips", () => {
  assert.deepEqual(argChips({ kpi: "revenue", top_n: 5, start: "", filters: { Region: "West" }, order: null }), ["kpi: revenue", "top n: 5", "Region = West"]);
  assert.deepEqual(argChips(null), []);
});

test("forecast becomes a chart series with a joined band", () => {
  const fc = { kpi: "revenue", label: "Revenue", fmt: "money", freq: "month",
    history: [{ period: "2024-11-01", value: 100 }, { period: "2024-12-01", value: 120 }],
    points: [{ period: "2025-01-01", forecast: 110, lo: 90, hi: 130 }] };
  const s = forecastSeries(fc);
  assert.equal(s.points.length, 3);
  assert.equal(s.freq, "M");
  assert.equal(s.forecastFrom, 2);
  assert.equal(s.points[1].lower, 120);        // band starts at the last real value
  assert.equal(s.points[2].upper, 130);
  assert.equal(s.points[0].lower, null);
  assert.equal(forecastSeries(null), null);
  assert.equal(forecastSeries({ history: [], points: [] }).points.length, 0);
  assert.equal(forecastSeries({ ...fc, history: Array.from({ length: 50 }, (_, i) => ({ period: "2020-01-01", value: i })) }, 10).points.length, 11);
});

test("suggested questions use the real names", () => {
  const q = suggestQuestions({ kpis: [{ label: "Revenue" }], dimensions: [{ name: "Region" }], date_to: "2024-12-31" });
  assert.equal(q[0], "What was revenue in 2024?");
  assert.ok(q.some((x) => x.includes("Region")));
  assert.ok(q.length <= 5);
  assert.deepEqual(suggestQuestions(null), []);
  assert.ok(suggestQuestions({ kpis: [], dimensions: [], date_to: "" }).length >= 2);
});

test("tool arguments", () => {
  assert.deepEqual(buildToolArgs("breakdown", { kpi: "revenue", dimension: "Region", start: "", end: " 2024 ", junk: "x" }), { args: { kpi: "revenue", dimension: "Region", end: "2024" }, error: "" });
  assert.match(buildToolArgs("breakdown", { kpi: "revenue" }).error, /column/);
  assert.match(buildToolArgs("trend", {}).error, /KPI/);
  assert.match(buildToolArgs("why", { kpi: "revenue", start: "2024" }).error, /start and an end/);
  assert.equal(buildToolArgs("alerts", {}).error, "");
  assert.equal(buildToolArgs("nope", {}).args, null);
});
