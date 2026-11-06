import test from "node:test";
import assert from "node:assert/strict";
import { freqCode, freqUnit, statusMeaning, statusTone, coverageInfo, skillInfo, errorText, tableRows, forecastArgs, HORIZONS } from "../src/lib/forecastui.js";

test("status text and tone", () => {
  assert.match(statusMeaning("LOW_CONFIDENCE"), /rough guide/);
  assert.equal(statusMeaning("???"), "");
  assert.equal(statusTone("OK"), "answer");
  assert.equal(statusTone("LOW_CONFIDENCE"), "unknown");
});

test("coverage says whether the range was too narrow", () => {
  assert.deepEqual(coverageInfo({ level: 0.8, coverage: 0.347, n_checks: 69 }), { target: 80, actual: 35, verdict: "TOO NARROW", tone: "fail", checks: 69 });
  assert.equal(coverageInfo({ level: 0.8, coverage: 0.78 }).verdict, "ABOUT RIGHT");
  assert.equal(coverageInfo(null), null);
  assert.equal(coverageInfo({}), null);
});

test("skill compares with repeating last year", () => {
  assert.equal(skillInfo({ skill: -0.273 }).better, false);
  assert.match(skillInfo({ skill: -0.273 }).text, /27% less accurate/);
  assert.match(skillInfo({ skill: 0.31 }).text, /31% more accurate/);
  assert.equal(skillInfo({ skill: 0.01 }).tone, "unknown");
  assert.equal(skillInfo({}), null);
  assert.match(errorText({ mae: 0.1998, mae_naive: 0.1569 }), /20.0% typical error \(repeating last year: 15.7%\)/);
  assert.equal(errorText({}), "");
});

test("table rows are readable", () => {
  const rows = tableRows({ fmt: "money", freq: "month", points: [{ period: "2025-01-01", forecast: 439803, lo: 267870, hi: 534983 }] });
  assert.deepEqual(rows, [{ period: "Jan 2025", forecast: "440k", range: "268k to 535k" }]);
  assert.deepEqual(tableRows(null), []);
});

test("arguments", () => {
  assert.deepEqual(forecastArgs({ kpi: "revenue", freq: "quarter", horizon: "6" }), { args: { kpi: "revenue", freq: "quarter", horizon: 6 }, error: "" });
  assert.deepEqual(forecastArgs({ kpi: "revenue", freq: "", horizon: "" }).args, { kpi: "revenue", freq: "month" });
  assert.match(forecastArgs({}).error, /KPI/);
  assert.equal(HORIZONS[0][0], "");
});

test("the API's M/Q/W and the tool's month/quarter/week both work", () => {
  assert.equal(freqCode("Q"), "Q"); assert.equal(freqCode("quarter"), "Q"); assert.equal(freqCode("zzz"), "M");
  assert.equal(freqUnit("Q", 3), "quarters"); assert.equal(freqUnit("week", 1), "week");
  assert.equal(tableRows({ fmt: "money", freq: "Q", points: [{ period: "2025-01-01", forecast: 1, lo: 1, hi: 2 }] })[0].period, "Q1 2025");
});
