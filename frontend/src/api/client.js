// The only file that talks to the backend. `createClient` takes a base URL and a fetch function,
// so tests can pass a fake fetch and nothing here depends on the browser.

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export const START_HINT =
  "Cannot reach the InsightTrace API. Start it with:  uvicorn api.main:app --port 8000";

function detailText(body, status) {
  const d = body && body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d) && d.length) return d.map((x) => x.msg || String(x)).join("; ");
  if (status === 429) return "Too many AI requests. Please wait a minute and try again.";
  return `The server answered with an error (${status}).`;
}

export function createClient({ baseUrl = "", fetchImpl } = {}) {
  const base = String(baseUrl || "").replace(/\/+$/, "");
  const f = fetchImpl || ((...a) => globalThis.fetch(...a));

  async function call(method, path, { body, form, query } = {}) {
    let url = `${base}/api${path}`;
    if (query) {
      const q = Object.entries(query).filter(([, v]) => v !== undefined && v !== null && v !== "");
      if (q.length) url += `?${new URLSearchParams(q.map(([k, v]) => [k, String(v)])).toString()}`;
    }
    const init = { method, headers: {} };
    if (form) init.body = form;
    else if (body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    let res;
    try {
      res = await f(url, init);
    } catch {
      throw new ApiError(START_HINT, 0);
    }
    let data = null;
    try {
      data = await res.json();
    } catch {
      /* not JSON */
    }
    if (!res.ok) throw new ApiError(detailText(data, res.status), res.status);
    return data;
  }

  const s = (sid) => `/sessions/${encodeURIComponent(sid)}`;
  return {
    health: () => call("GET", "/health"),
    tools: () => call("GET", "/tools"),
    demo: (name) => call("POST", "/demo", { body: { name } }),
    upload: (file) => {
      const form = new FormData();
      form.append("file", file, file.name);
      return call("POST", "/upload", { form });
    },
    summary: (sid) => call("GET", `${s(sid)}/summary`),
    headlines: (sid, freq = "M") => call("GET", `${s(sid)}/headlines`, { query: { freq } }),
    alerts: (sid, opts = {}) => call("GET", `${s(sid)}/alerts`, { query: opts }),
    series: (sid, kpi, freq = "M", sensitivity = "normal") =>
      call("GET", `${s(sid)}/series`, { query: { kpi, freq, sensitivity } }),
    alertWhy: (sid, kpi, start, freq = "M", use_llm = true) =>
      call("POST", `${s(sid)}/alerts/why`, { body: { kpi, start, freq, use_llm } }),
    tool: (sid, tool, args = {}, use_llm = true) =>
      call("POST", `${s(sid)}/tool`, { body: { tool, args, use_llm } }),
    ask: (sid, question, use_llm = true) => call("POST", `${s(sid)}/ask`, { body: { question, use_llm } }),
    remove: (sid) => call("DELETE", s(sid)),
  };
}
