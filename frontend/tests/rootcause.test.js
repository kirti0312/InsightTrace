import test from "node:test";
import assert from "node:assert/strict";
import { parseShare, checkMark, candidates, checkColumns, bar, headlineRecord, VERDICT_MEANING } from "../src/lib/rootcause.js";

const ev = { records: [
  { id: "E1", kind: "headline", title: "Revenue change", text: "Revenue fell.", facts: {}, checks: [] },
  { id: "E2", kind: "cause", verdict: "CAUSE", title: "Customer = A, B", entities: ["A", "B"], facts: { share: { display: "59%" }, contribution: { display: "-486,145" } },
    checks: [{ name: "Size", status: "pass" }, { name: "Differs", status: "pass" }] },
  { id: "E3", kind: "rejected", verdict: "NOT_A_CAUSE", title: "Segment = X", entities: [], facts: { share: { display: "100%" } },
    checks: [{ name: "Size", status: "fail" }, { name: "Started", status: "unknown" }] },
  { id: "E4", kind: "unknown", title: "Nothing clear", facts: {}, checks: [] },
] };

test("share parsing", () => {
  assert.equal(parseShare("59%"), 59);
  assert.equal(parseShare("-4.5%"), -4.5);
  assert.equal(parseShare("n/a"), null);
  assert.equal(parseShare(undefined), null);
});

test("check marks always carry a word as well as a symbol", () => {
  assert.deepEqual(checkMark("pass"), { mark: "✓", label: "PASS", tone: "pass" });
  assert.equal(checkMark("FAIL").label, "FAIL");
  assert.equal(checkMark("unknown").label, "UNSURE");
  assert.equal(checkMark(undefined).tone, "unknown");
});

test("candidates keep order, counts and verdicts", () => {
  const c = candidates(ev);
  assert.deepEqual(c.map((x) => x.id), ["E2", "E3"]);      // the "unknown" record is not a candidate
  assert.equal(c[0].passed, 2); assert.equal(c[0].total, 2); assert.equal(c[0].share, 59); assert.equal(c[0].contribution, "-486,145");
  assert.equal(c[1].passed, 0); assert.equal(c[1].total, 2);
  assert.deepEqual(candidates(null), []);
});

test("grid columns, bars, headline", () => {
  assert.deepEqual(checkColumns(candidates(ev)), ["Size", "Differs", "Started"]);
  assert.equal(bar(59), 59); assert.equal(bar(250), 100); assert.equal(bar(0), 2); assert.equal(bar(-30), 30); assert.equal(bar(null), 0);
  assert.equal(headlineRecord(ev).id, "E1");
  assert.equal(headlineRecord({ records: [] }), null);
  assert.ok(VERDICT_MEANING.UNKNOWN.includes("Nothing is claimed"));
});
