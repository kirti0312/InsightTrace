"""The API's logic, with no web framework in it (so it can be tested without a server).

api/main.py only maps URLs to the methods of `Service`. Everything that matters is here:

  - SESSIONS: one uploaded file = one session (an in-memory Dataset). Kept in memory only, never
    written to disk; evicted after `ttl_seconds` of no use and when more than `max_sessions` exist.
    A restart of the server forgets everything (the user uploads again).
  - LIMITS: upload size, number of sessions, questions per minute per session (the free Gemini
    quota is small), question length (in llm/analyst.py).
  - ERRORS: every expected problem is an ApiError(status, message) with a message that is safe to
    show. Unexpected problems are NOT turned into messages here (main.py returns a generic 500, so
    no internals or file contents leak).
  - SAFETY: the API key never appears in any response. The LLM can only choose among the fixed tools
    (llm/tools.py); SQL is read-only and guarded (llm/sqlguard.py).

Honest limits
-------------
  - In-memory sessions suit ONE server process (the free hosting plans). With several workers or a
    restart, sessions are lost / not shared.
  - There are no user accounts: whoever has the session id can use that session. Ids are random
    (128 bits) and never listed.
  - Heavy calls (a "why" verification) take seconds and block that session while they run.
  - Uploaded files are read by pandas / openpyxl: only upload files you trust on a shared server.
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from engine.detect import SENSITIVITY, config_for, detect_all, detect_kpi
from engine.evidence import build_evidence
from engine.ingest import IngestError, load_dataset
from engine.kpi import FREQ, available_kpis, format_value, headline
from engine.verify import to_plain, verify_event
from llm.analyst import AnalystAnswer, answer_question, run_tool_direct
from llm.client import llm_status
from llm.explain import explain_evidence
from llm.tools import TOOLS

ROOT = Path(__file__).resolve().parent.parent
DEMOS = {"synthetic": ROOT / "data" / "synthetic_demo.csv",
         "superstore": ROOT / "data" / "Sample_-_Superstore.csv"}
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx"}
ALERT_FREQS = ("W", "M", "Q")
MAX_ALERTS = 50


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class Session:
    id: str
    ds: object
    name: str
    created: float
    last_used: float
    shared: bool = False                      # a demo dataset shared by several sessions (read-only)
    lock: threading.RLock = field(default_factory=threading.RLock)
    asks: deque = field(default_factory=deque)


class Service:
    def __init__(self, llm=None, loader=load_dataset, max_sessions: int = 20, ttl_seconds: float = 7200,
                 max_upload_bytes: int = 25 * 1024 * 1024, asks_per_minute: int = 20, clock=time.time,
                 demos: dict | None = None):
        self.llm = llm
        self.loader = loader
        self.max_sessions, self.ttl = max_sessions, ttl_seconds
        self.max_upload_bytes, self.asks_per_minute = max_upload_bytes, asks_per_minute
        self.clock = clock
        self.demos = dict(DEMOS if demos is None else demos)
        self._sessions: dict[str, Session] = {}
        self._demo_cache: dict[str, object] = {}
        self._guard = threading.RLock()

    # ------------------------------------------------------------------ sessions
    def _purge(self):
        now = self.clock()
        for sid in [k for k, s in self._sessions.items() if now - s.last_used > self.ttl]:
            del self._sessions[sid]

    def _new_session(self, ds, name: str, shared: bool = False) -> Session:
        with self._guard:
            self._purge()
            while len(self._sessions) >= self.max_sessions:          # evict the least recently used
                oldest = min(self._sessions.values(), key=lambda s: s.last_used)
                del self._sessions[oldest.id]
            now = self.clock()
            s = Session(secrets.token_urlsafe(16), ds, name, now, now, shared)
            self._sessions[s.id] = s
            return s

    def session(self, sid: str) -> Session:
        with self._guard:
            self._purge()
            s = self._sessions.get(sid)
            if s is None:
                raise ApiError(404, "This session was not found or has expired. Upload the file again.")
            s.last_used = self.clock()
            return s

    def delete(self, sid: str) -> dict:
        with self._guard:
            self.session(sid)
            del self._sessions[sid]
        return {"deleted": True}

    def n_sessions(self) -> int:
        with self._guard:
            self._purge()
            return len(self._sessions)

    # ------------------------------------------------------------------ health / lists
    def health(self) -> dict:
        if self.llm is not None:
            llm = {"configured": True, "model": getattr(self.llm, "model", getattr(self.llm, "name", "?")),
                   "reason": "ok"}
        else:
            st = llm_status()
            llm = {"configured": False, "model": None, "reason": st["reason"]}
        return {"ok": True, "llm": llm, "demos": [k for k, p in self.demos.items() if Path(p).exists()],
                "max_upload_mb": round(self.max_upload_bytes / 1024 / 1024, 1)}

    def tools(self) -> list[dict]:
        return [{"name": n, "help": t.help, "params": t.params, "required": list(t.required)}
                for n, t in TOOLS.items()]

    # ------------------------------------------------------------------ loading
    def upload(self, data: bytes, filename: str) -> dict:
        name = Path(str(filename or "")).name
        if not name:
            raise ApiError(400, "The file has no name.")
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ApiError(400, f"Unsupported file type. Upload one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
        if not data:
            raise ApiError(400, "The file is empty.")
        if len(data) > self.max_upload_bytes:
            raise ApiError(413, f"The file is too large (limit {self.max_upload_bytes // 1024 // 1024} MB).")
        try:
            ds = self.loader(data, filename=name)
        except IngestError as err:
            raise ApiError(400, f"Could not read this file: {err}") from None
        except Exception:                                       # pandas / openpyxl problems: no internals shown
            raise ApiError(400, "Could not read this file. Check that it is a valid CSV or Excel table "
                                "with a date column and a sales (revenue) column.") from None
        return self._summary(self._new_session(ds, name))

    def demo(self, name: str) -> dict:
        path = self.demos.get(name)
        if path is None or not Path(path).exists():
            raise ApiError(404, f"Unknown demo '{name}'. Available: "
                                f"{', '.join(k for k, p in self.demos.items() if Path(p).exists()) or 'none'}.")
        with self._guard:
            if name not in self._demo_cache:
                self._demo_cache[name] = self.loader(str(path))
            ds = self._demo_cache[name]
        return self._summary(self._new_session(ds, Path(path).name, shared=True))

    # ------------------------------------------------------------------ read-only views
    def _summary(self, s: Session) -> dict:
        ds, sc = s.ds, s.ds.schema
        return {"session_id": s.id, "name": s.name, "rows": int(ds.n_rows),
                "date_from": ds.min_date.strftime("%Y-%m-%d"), "date_to": ds.max_date.strftime("%Y-%m-%d"),
                "kpis": [{"key": k.key, "label": k.label, "fmt": k.fmt, "additive": k.additive}
                         for k in available_kpis(ds)],
                "dimensions": [{"name": d, "values": int(sc.cardinality.get(d, 0))} for d in sc.dimensions],
                "warnings": list(ds.warnings), "aliases": dict(sc.aliases)}

    def summary(self, sid: str) -> dict:
        return self._summary(self.session(sid))

    def headlines(self, sid: str, freq: str = "M") -> dict:
        s = self.session(sid)
        if freq not in FREQ:
            raise ApiError(400, f"freq must be one of {', '.join(FREQ)}.")
        out, problems = [], []
        with s.lock:
            for k in available_kpis(s.ds):
                try:
                    h = headline(s.ds, k.key, freq)
                    h["value_display"] = format_value(h["value"], h["fmt"])
                    out.append(to_plain(h))
                except Exception as err:
                    problems.append(f"{k.label}: {err}")
        return {"freq": freq, "headlines": out, "problems": problems}

    def alerts(self, sid: str, freq: str = "M", kpi: str | None = None, primary_only: bool = True,
               sensitivity: str = "normal", top_n: int = 20) -> dict:
        s = self.session(sid)
        if freq not in ALERT_FREQS:
            raise ApiError(400, f"freq must be one of {', '.join(ALERT_FREQS)}.")
        if sensitivity not in SENSITIVITY:
            raise ApiError(400, f"sensitivity must be one of {', '.join(SENSITIVITY)}.")
        top_n = max(1, min(int(top_n), MAX_ALERTS))
        keys = [k.key for k in available_kpis(s.ds)]
        if kpi and kpi not in keys:
            raise ApiError(400, f"Unknown KPI '{kpi}'. Available: {', '.join(keys)}.")
        with s.lock:
            try:
                events, notes = detect_all(s.ds, freq, [kpi] if kpi else None, config=config_for(sensitivity),
                                           primary_only=primary_only)
            except (ValueError, IngestError) as err:
                raise ApiError(400, str(err)) from None
        rows = []
        for e in events[:top_n]:
            d = to_plain(e)
            d["headline"] = e.headline()
            d["id"] = f"{e.kpi}:{e.freq}:{e.start:%Y-%m-%d}"
            rows.append(d)
        return {"freq": freq, "sensitivity": sensitivity, "total": len(events), "events": rows, "notes": notes}

    def series(self, sid: str, kpi: str, freq: str = "M", sensitivity: str = "normal") -> dict:
        """One KPI over time for a chart: value per complete period, the range the detector expected, and
        which periods it flagged. (An incomplete last period is left out, as in the detector.)"""
        s = self.session(sid)
        if freq not in ALERT_FREQS:
            raise ApiError(400, f"freq must be one of {', '.join(ALERT_FREQS)}.")
        if sensitivity not in SENSITIVITY:
            raise ApiError(400, f"sensitivity must be one of {', '.join(SENSITIVITY)}.")
        kinds = {k.key: k for k in available_kpis(s.ds)}
        if kpi not in kinds:
            raise ApiError(400, f"Unknown KPI '{kpi}'. Available: {', '.join(kinds)}.")
        with s.lock:
            try:
                res = detect_kpi(s.ds, kpi, freq, None, config_for(sensitivity))
            except (ValueError, IngestError) as err:
                raise ApiError(400, str(err)) from None
        k = kinds[kpi]
        points = []
        for r in res.frame.itertuples(index=False):
            def num(x):
                return None if x is None or x != x else float(x)
            points.append({"period": r.period.strftime("%Y-%m-%d"), "value": num(r.value),
                           "display": format_value(r.value, k.fmt) if num(r.value) is not None else "n/a",
                           "expected": num(r.expected), "lower": num(r.lower), "upper": num(r.upper),
                           "flagged": bool(r.flagged), "direction": str(r.direction or "")})
        return {"kpi": kpi, "label": k.label, "fmt": k.fmt, "freq": freq, "sensitivity": sensitivity,
                "points": points, "notes": list(res.notes)}

    # ------------------------------------------------------------------ answers (may use the LLM)
    def _llm_for(self, s: Session, use_llm: bool):
        """The LLM to use for this request (None = plain, always-safe wording). Counts toward the rate limit."""
        if not use_llm or self.llm is None:
            return None
        now = self.clock()
        while s.asks and now - s.asks[0] > 60:
            s.asks.popleft()
        if len(s.asks) >= self.asks_per_minute:
            raise ApiError(429, "Too many AI requests in a minute. Wait a little, or turn the AI wording off "
                                "(use_llm = false) to get the plain answer.")
        s.asks.append(now)
        return self.llm

    def ask(self, sid: str, question: str, use_llm: bool = True) -> dict:
        s = self.session(sid)
        with s.lock:
            llm = self._llm_for(s, use_llm)
            return answer_question(s.ds, question, llm).to_dict()

    def tool(self, sid: str, tool: str, args: dict | None = None, use_llm: bool = True) -> dict:
        s = self.session(sid)
        with s.lock:
            llm = self._llm_for(s, use_llm)
            ans = run_tool_direct(s.ds, tool, args or {}, llm)
        if ans.status == "failed":
            raise ApiError(400, ans.text)
        return ans.to_dict()

    def alert_why(self, sid: str, kpi: str, start: str, freq: str = "M", use_llm: bool = True) -> dict:
        """Verify one detected alert with its own window and growth-adjusted baseline (verify_event)."""
        s = self.session(sid)
        if freq not in ALERT_FREQS:
            raise ApiError(400, f"freq must be one of {', '.join(ALERT_FREQS)}.")
        if kpi not in [k.key for k in available_kpis(s.ds)]:
            raise ApiError(400, f"Unknown KPI '{kpi}'.")
        with s.lock:
            try:
                events, _ = detect_all(s.ds, freq, [kpi])
            except (ValueError, IngestError) as err:
                raise ApiError(400, str(err)) from None
            match = [e for e in events if f"{e.start:%Y-%m-%d}" == str(start)[:10]]
            if not match:
                raise ApiError(404, "No alert for that KPI starts on that date. Reload the alerts list.")
            event = match[0]
            llm = self._llm_for(s, use_llm)
            bundle = build_evidence(verify_event(s.ds, event), event)
            expl = explain_evidence(bundle, llm)
        return AnalystAnswer("", "ok", expl.text, "why", {"kpi": kpi, "start": str(start)[:10], "freq": freq},
                             expl, bundle, None, []).to_dict()
