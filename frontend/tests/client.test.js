import test from "node:test";
import assert from "node:assert/strict";
import { createClient, ApiError, START_HINT } from "../src/api/client.js";

function fake(status, body) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    return { ok: status < 400, status, json: async () => { if (body === undefined) throw new Error("no json"); return body; } };
  };
  return { fetchImpl, calls };
}

test("GET builds the right url and query", async () => {
  const { fetchImpl, calls } = fake(200, { ok: true });
  const c = createClient({ baseUrl: "https://api.example.com/", fetchImpl });
  await c.series("a b", "revenue", "M", "high");
  assert.equal(calls[0].url, "https://api.example.com/api/sessions/a%20b/series?kpi=revenue&freq=M&sensitivity=high");
  assert.equal(calls[0].init.method, "GET");
});

test("empty query values are dropped", async () => {
  const { fetchImpl, calls } = fake(200, {});
  const c = createClient({ fetchImpl });
  await c.alerts("s1", { freq: "M", kpi: undefined, top_n: 5 });
  assert.equal(calls[0].url, "/api/sessions/s1/alerts?freq=M&top_n=5");
});

test("POST sends JSON", async () => {
  const { fetchImpl, calls } = fake(200, { text: "hi" });
  const c = createClient({ fetchImpl });
  const r = await c.ask("s1", "why did sales fall?", false);
  assert.deepEqual(JSON.parse(calls[0].init.body), { question: "why did sales fall?", use_llm: false });
  assert.equal(calls[0].init.headers["Content-Type"], "application/json");
  assert.equal(r.text, "hi");
  await c.demo("synthetic");
  assert.deepEqual(JSON.parse(calls[1].init.body), { name: "synthetic" });
});

test("upload sends form data without a JSON header", async () => {
  const { fetchImpl, calls } = fake(200, {});
  const c = createClient({ fetchImpl });
  const file = new File(["a,b\n1,2"], "x.csv");
  await c.upload(file);
  assert.ok(calls[0].init.body instanceof FormData);
  assert.equal(calls[0].init.body.get("file").name, "x.csv");
  assert.equal(calls[0].init.headers["Content-Type"], undefined);
});

test("server errors become ApiError with the server's message", async () => {
  const c = createClient({ fetchImpl: fake(400, { detail: "Unknown KPI 'x'." }).fetchImpl });
  await assert.rejects(c.summary("s"), (e) => e instanceof ApiError && e.status === 400 && e.message === "Unknown KPI 'x'.");
  const c2 = createClient({ fetchImpl: fake(404, { detail: "Session expired." }).fetchImpl });
  await assert.rejects(c2.summary("s"), (e) => e.status === 404);
  const c3 = createClient({ fetchImpl: fake(429, {}).fetchImpl });
  await assert.rejects(c3.ask("s", "q"), (e) => /wait a minute/.test(e.message));
  const c4 = createClient({ fetchImpl: fake(500, undefined).fetchImpl });
  await assert.rejects(c4.health(), (e) => e.status === 500 && /error \(500\)/.test(e.message));
  const c5 = createClient({ fetchImpl: fake(422, { detail: [{ msg: "field required" }] }).fetchImpl });
  await assert.rejects(c5.ask("s", ""), (e) => e.message === "field required");
});

test("network failure gives the start-the-API hint", async () => {
  const c = createClient({ fetchImpl: async () => { throw new TypeError("Failed to fetch"); } });
  await assert.rejects(c.health(), (e) => e.status === 0 && e.message === START_HINT);
});
