"""The analyst's tools: a FIXED set of safe functions that answer questions about the loaded file.

The idea, in plain words
------------------------
The AI analyst never touches the data itself. When a user asks a question, the LLM only picks ONE
tool from this file and fills in its arguments (as JSON). The tool is ordinary Python/SQL: it checks
the arguments, runs the numbers, and returns EVIDENCE RECORDS (E1, E2, ...) in the same shape the
verification step uses. The explanation step and the validator then work exactly as before, so every
number the user sees was produced by Python/SQL and is cited.

The 8 tools
-----------
  kpi_value     one KPI over a date range (optionally vs the same dates a year earlier)
  breakdown     a KPI split by one category column (top / bottom N members)
  trend         a KPI over time (month / quarter / week)
  contributors  who contributed to the change in a KPI between two windows (exact decomposition)
  alerts        unusual changes found by the change detector
  why           verified causes of a change (the 5-check verification + evidence)
  forecast      the validated forecast with its uncertainty range
  sql           a last resort: ONE read-only SELECT on the table `sales`, checked by llm/sqlguard.py

Safety
------
  - Arguments are validated: unknown arguments are rejected; KPI and column names are matched
    against the file (with friendly aliases like "sales" = revenue); filter values are matched
    against the real values in the file; dates must be ISO (2024-03-31, 2024-03 or 2024) and are
    clipped to the data range, with a note.
  - A bad call raises AnalystError with a message written for the planner, so it can fix and retry.
  - Nothing here writes to the data. The only free-form input is the `sql` tool, guarded in sqlguard.py.

Honest limits
-------------
  - The planner (LLM) may pick the wrong tool or arguments for a question. The evidence shown
    always states exactly what was computed (KPI, dates, filters), so a wrong choice is visible.
  - `breakdown` shares are shown only for sum KPIs (parts add up to the total).
  - One tool call per question: multi-step questions ("compare West and East, then forecast") are
    answered with the single best tool, or declined.
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.decompose import (DecomposeError, _grouped, _total, decompose, one_year_earlier)
from engine.detect import detect_all
from engine.evidence import (EvidenceBundle, Fact, Record, _d, _fact, _money_signed, _ok, _where,
                             build_evidence)
from engine.forecast import ForecastError, forecast_kpi, history_for
from engine.ingest import IngestError, q
from engine.kpi import available_kpis, format_value
from engine.verify import verify_window
from llm.sqlguard import SqlGuardError, clean_select

MAX_SQL_ROWS = 200
MAX_ROW_RECORDS = 10
MAX_TREND_ROWS = 24


class AnalystError(ValueError):
    """A tool call that cannot be run. The message is written so the planner can fix it."""


# ----------------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------------
class AnswerBundle(EvidenceBundle):
    """Evidence for a plain question (not a cause verification): conclusion is "ANSWER", so the
    validator asks for citations and correct numbers but no cause-hedging."""

    def summary_text(self, max_words: int = 190) -> str:
        """Plain answer built only from the records (no LLM); short enough to pass the validator."""
        lines, used, cut = [], 0, False
        for r in self.records:
            line = f"{r.text} [{r.id}]"
            n = len(line.split())
            if lines and used + n > max_words:
                cut = True
                break
            lines.append(line)
            used += n
        if cut:
            lines.append("More rows are in the evidence table.")
        lines.append(self.limits[0])
        return "\n".join(lines)


@dataclass
class ToolResult:
    tool: str
    args: dict
    bundle: EvidenceBundle | None = None
    forecast: object | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ToolSpec:
    func: object
    help: str
    params: dict          # name -> short description (shown to the planner)
    required: tuple = ()


# ----------------------------------------------------------------------------
# Argument helpers
# ----------------------------------------------------------------------------
KPI_ALIASES = {
    "sales": "revenue", "sale": "revenue", "turnover": "revenue", "income": "revenue",
    "margin": "profit_margin", "profit_margin_%": "profit_margin", "margins": "profit_margin",
    "profits": "profit", "earnings": "profit",
    "units": "quantity", "units_sold": "quantity", "qty": "quantity", "volume": "quantity",
    "aov": "avg_order_value", "average_order_value": "avg_order_value", "order_value": "avg_order_value",
    "basket_size": "avg_order_value",
    "order_count": "orders", "number_of_orders": "orders", "transactions": "orders", "order": "orders",
    "discount": "avg_discount", "discounts": "avg_discount", "average_discount": "avg_discount",
}
FREQ_NAMES = {"m": "M", "month": "M", "monthly": "M", "months": "M",
              "q": "Q", "quarter": "Q", "quarterly": "Q", "quarters": "Q",
              "w": "W", "week": "W", "weekly": "W", "weeks": "W"}
UNIT_WORD = {"M": "month", "Q": "quarter", "W": "week"}


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def resolve_kpi(ds, value):
    if not isinstance(value, str) or not value.strip():
        raise AnalystError("Argument 'kpi' is required (for example: revenue, profit, orders).")
    avail = {k.key: k for k in available_kpis(ds)}
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    key = KPI_ALIASES.get(key, key)
    if key in avail:
        return avail[key]
    for k in avail.values():                                   # a label such as "Average order value"
        if _norm(k.label) == _norm(value) or _norm(k.key) == _norm(value):
            return k
    raise AnalystError(f"KPI '{value}' is not available for this file. Available KPIs: {', '.join(avail)}.")


def resolve_dimension(ds, value) -> str:
    dims = list(ds.schema.dimensions)
    if not isinstance(value, str) or not value.strip():
        raise AnalystError(f"Argument 'dimension' is required. Available columns: {', '.join(dims)}.")
    for d in dims:
        if d == value:
            return d
    for d in dims:
        if _norm(d) == _norm(value):
            return d
    raise AnalystError(f"'{value}' is not a category column in this file. Available columns: {', '.join(dims)}.")


_members_cache: dict = {}


def _members(ds, col: str) -> list[str]:
    df = ds.query(f"SELECT DISTINCT CAST({q(col)} AS VARCHAR) AS v FROM sales")
    return [str(x) for x in df["v"].dropna().tolist()]


def resolve_filters(ds, filters) -> dict:
    """{"region": "west"} -> {"Region": ["West"]}, checked against the real values in the file."""
    if filters in (None, {}, []):
        return {}
    if not isinstance(filters, dict):
        raise AnalystError("Argument 'filters' must be an object like {\"Region\": \"West\"}.")
    out: dict = {}
    for col, val in filters.items():
        dim = resolve_dimension(ds, col)
        vals = val if isinstance(val, (list, tuple)) else [val]
        if not vals:
            raise AnalystError(f"The filter for '{dim}' is empty.")
        members = _members(ds, dim)
        picked = []
        for v in vals:
            s = str(v).strip()
            hit = None
            if s in members:
                hit = s
            else:
                low = [m for m in members if m.lower() == s.lower()]
                if len(low) == 1:
                    hit = low[0]
                else:
                    sub = [m for m in members if s and s.lower() in m.lower()]
                    if len(sub) == 1:
                        hit = sub[0]
                    elif len(sub) > 1:
                        raise AnalystError(f"'{s}' matches several values of {dim}: {', '.join(sub[:5])}. "
                                           f"Use the exact value.")
            if hit is None:
                close = difflib.get_close_matches(s, members, n=5, cutoff=0.5)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                raise AnalystError(f"'{s}' is not a value of {dim} in this file.{hint}")
            if hit not in picked:
                picked.append(hit)
        out[dim] = picked
    return out


_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_ISO_M = re.compile(r"^(\d{4})-(\d{1,2})$")
_ISO_Y = re.compile(r"^(\d{4})$")
_DATE_HELP = "Dates must be written as 2024-03-31, 2024-03 (a whole month) or 2024 (a whole year)."


def _parse_date(value, is_end: bool) -> pd.Timestamp:
    s = str(value).strip()
    try:
        if m := _ISO.match(s):
            return pd.Timestamp(int(m[1]), int(m[2]), int(m[3]))
        if m := _ISO_M.match(s):
            ts = pd.Timestamp(int(m[1]), int(m[2]), 1)
            return ts + pd.offsets.MonthEnd(0) if is_end else ts
        if m := _ISO_Y.match(s):
            return pd.Timestamp(int(m[1]), 12 if is_end else 1, 31 if is_end else 1)
    except (ValueError, OverflowError):
        pass
    raise AnalystError(f"'{value}' is not a valid date. {_DATE_HELP}")


def resolve_window(ds, a: dict, notes: list[str], need: bool = False):
    """(start, end) from args 'start'/'end', clipped to the data range."""
    s, e = a.get("start"), a.get("end")
    if need and (s in (None, "") or e in (None, "")):
        raise AnalystError("Both 'start' and 'end' dates are required. " + _DATE_HELP)
    start = _parse_date(s, False) if s not in (None, "") else pd.Timestamp(ds.min_date)
    end = _parse_date(e, True) if e not in (None, "") else pd.Timestamp(ds.max_date)
    lo, hi = pd.Timestamp(ds.min_date).normalize(), pd.Timestamp(ds.max_date).normalize()
    if end < start:
        raise AnalystError("The end date is before the start date.")
    if end < lo or start > hi:
        raise AnalystError(f"This file has no data in that range. Its data runs from {_d(lo)} to {_d(hi)}.")
    if start < lo:
        notes.append(f"The start date was moved to {_d(lo)}, where the data begins.")
        start = lo
    if end > hi:
        notes.append(f"The end date was moved to {_d(hi)}, where the data ends.")
        end = hi
    return start, end


def _top_n(a: dict, notes: list[str], default: int = 5) -> int:
    v = a.get("top_n", default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
        raise AnalystError("'top_n' must be a whole number from 1 to 10.")
    v = int(v)
    if v < 1 or v > 10:
        notes.append("The number of rows shown was limited to between 1 and 10.")
        v = min(10, max(1, v))
    return v


def _freq(a: dict, default: str = "M") -> str:
    f = a.get("freq", default)
    key = FREQ_NAMES.get(str(f).strip().lower())
    if key is None:
        raise AnalystError("'freq' must be one of: month, quarter, week.")
    return key


def _entities(names) -> list[str]:
    """Names from the file that could break the number checks: they contain digits or '. '."""
    out = []
    for n in names:
        n = str(n)
        if len(n) >= 3 and (any(c.isdigit() for c in n) or ". " in n) and not re.fullmatch(r"[\d,.\s]+", n):
            out.append(n)
    return list(dict.fromkeys(out))


def _value(kpi, nd) -> float:
    num, den = nd
    if kpi.den:
        return float("nan") if (den is None or not np.isfinite(den) or den == 0) else num / den
    return float(num)


def _show(kpi, v) -> str:
    return format_value(v, kpi.fmt) if _ok(v) else "n/a"


def _span(a, b) -> str:
    return f"{_d(a)} to {_d(b)}"


def _where_text(filters: dict) -> str:
    return f" for {_where(filters)}" if filters else ""


def _bundle(kpi, window, records, limits, notes, compare=None) -> AnswerBundle:
    w = (_d(window[0]), _d(window[1]))
    c = (_d(compare[0]), _d(compare[1])) if compare else w
    return AnswerBundle(kpi.key if kpi else "", kpi.label if kpi else "Data", w, c, "ANSWER", records, [],
                        [*limits, *notes])


BASE_LIMIT = ("These figures come straight from your file and describe what happened; "
              "they do not explain why it happened.")


def _rec(rid, kind, title, text, facts=None, sql=None, entities=()) -> Record:
    return Record(rid, kind, title, "", text, facts or {}, [], {}, sql or [], list(entities))


# ----------------------------------------------------------------------------
# Tool 1: kpi_value
# ----------------------------------------------------------------------------
def tool_kpi_value(ds, a: dict) -> ToolResult:
    notes: list[str] = []
    kpi = resolve_kpi(ds, a.get("kpi"))
    filters = resolve_filters(ds, a.get("filters"))
    start, end = resolve_window(ds, a, notes)
    nd, sq = _total(ds, kpi, start, end, filters)
    v = _value(kpi, nd)
    facts: dict = {}
    _fact(facts, "value", v if _ok(v) else None, _show(kpi, v), f"{kpi.label} in the window")
    records = [_rec("E1", "headline", kpi.label,
                    f"{kpi.label}{_where_text(filters)} was {_show(kpi, v)} from {_span(start, end)}.",
                    facts, [sq], _entities([m for vs in filters.values() for m in vs]))]
    limits = [BASE_LIMIT]
    cmp_win = None
    if a.get("compare") in (True, "true", "yes", 1):
        cs, ce = one_year_earlier(start, end)
        if ce < pd.Timestamp(ds.min_date):
            limits.append("There is no data for the same dates a year earlier, so no comparison was made.")
        else:
            cmp_win = (cs, ce)
            bnd, sq2 = _total(ds, kpi, cs, ce, filters)
            b = _value(kpi, bnd)
            f2: dict = {}
            _fact(f2, "base", b if _ok(b) else None, _show(kpi, b), "value a year earlier")
            if _ok(v) and _ok(b):
                chg = v - b
                _fact(f2, "change", chg, _money_signed(chg, kpi.fmt), "change versus a year earlier")
                pct = ""
                if kpi.fmt != "pct" and b != 0:
                    _fact(f2, "pct", chg / abs(b), f"{chg / abs(b):+.1%}", "relative change")
                    pct = f", {f2['pct'].display}"
                tail = f"; the change is {f2['change'].display}{pct}"
            else:
                tail = "; the change cannot be computed"
            records.append(_rec("E2", "row", "A year earlier",
                                f"A year earlier ({_span(cs, ce)}) it was {_show(kpi, b)}{tail}.",
                                f2, [sq2]))
    return ToolResult("kpi_value", a, _bundle(kpi, (start, end), records, limits, notes, cmp_win), None, notes)


# ----------------------------------------------------------------------------
# Tool 2: breakdown
# ----------------------------------------------------------------------------
def tool_breakdown(ds, a: dict) -> ToolResult:
    notes: list[str] = []
    kpi = resolve_kpi(ds, a.get("kpi"))
    dim = resolve_dimension(ds, a.get("dimension"))
    filters = resolve_filters(ds, a.get("filters"))
    start, end = resolve_window(ds, a, notes)
    n = _top_n(a, notes)
    order = str(a.get("order", "desc")).lower()
    if order not in ("desc", "asc"):
        raise AnalystError("'order' must be 'desc' (highest first) or 'asc' (lowest first).")
    df, q1 = _grouped(ds, kpi, dim, start, end, filters)
    if kpi.den:
        df["value"] = np.where(df["den"].fillna(0) != 0, df["num"] / df["den"], np.nan)
    else:
        df["value"] = df["num"]
    df = df.dropna(subset=["value"])
    if df.empty:
        raise AnalystError("No rows match those dates and filters.")
    nd, q2 = _total(ds, kpi, start, end, filters)
    total = _value(kpi, nd)
    df = df.sort_values(["value", "member"], ascending=[order == "asc", True], kind="mergesort")
    shown = df.head(n)
    tf: dict = {}
    _fact(tf, "total", total if _ok(total) else None, _show(kpi, total), "total")
    _fact(tf, "shown", len(shown), f"{len(shown)}", "members shown")
    _fact(tf, "members", len(df), f"{len(df)}", "members in total")
    head = (f"{kpi.label} by {dim}{_where_text(filters)} from {_span(start, end)}: the total is {_show(kpi, total)}; "
            f"showing the {'highest' if order == 'desc' else 'lowest'} {len(shown)} of {len(df)}.")
    records = [_rec("E1", "headline", f"{kpi.label} by {dim}", head, tf, [q2],
                    _entities([m for vs in filters.values() for m in vs]))]
    for _, r in shown.iterrows():
        f: dict = {}
        _fact(f, "value", float(r["value"]), _show(kpi, r["value"]), f"{kpi.label} for this member")
        text = f"{r['member']}: {f['value'].display}"
        if kpi.additive and _ok(total) and total != 0:
            share = float(r["value"]) / total
            _fact(f, "share", share, f"{share:.0%}", "share of the total")
            text += f" ({f['share'].display} of the total)"
        records.append(_rec(f"E{len(records) + 1}", "row", str(r["member"]), text + ".", f, [q1],
                            _entities([r["member"]])))
    limits = [BASE_LIMIT]
    if not kpi.additive:
        limits.append("Shares of the total are not shown for this KPI because its parts do not simply add up.")
    return ToolResult("breakdown", a, _bundle(kpi, (start, end), records, limits, notes), None, notes)


# ----------------------------------------------------------------------------
# Tool 3: trend
# ----------------------------------------------------------------------------
def _plabel(ts, freq: str) -> str:
    ts = pd.Timestamp(ts)
    return ts.strftime("%b %Y") if freq == "M" else f"{UNIT_WORD[freq]} starting {_d(ts)}"


def tool_trend(ds, a: dict) -> ToolResult:
    notes: list[str] = []
    kpi = resolve_kpi(ds, a.get("kpi"))
    freq = _freq(a)
    filters = resolve_filters(ds, a.get("filters"))
    try:
        s, _, hist_notes = history_for(ds, kpi.key, freq, filters)
    except ForecastError as err:
        raise AnalystError(str(err)) from err
    notes += hist_notes
    if a.get("start") not in (None, "") or a.get("end") not in (None, ""):
        w0, w1 = resolve_window(ds, a, notes)
        s = s[(s.index >= w0) & (s.index <= w1)]
    s = s.dropna()
    if s.empty:
        raise AnalystError("There are no complete periods in that range to show a trend.")
    first, last = s.index[0], s.index[-1]
    hi, lo = s.idxmax(), s.idxmin()
    f: dict = {}
    _fact(f, "periods", len(s), f"{len(s)}", "periods shown")
    _fact(f, "first", float(s.iloc[0]), _show(kpi, s.iloc[0]), "first value")
    _fact(f, "last", float(s.iloc[-1]), _show(kpi, s.iloc[-1]), "latest value")
    _fact(f, "highest", float(s.max()), _show(kpi, s.max()), "highest value")
    _fact(f, "lowest", float(s.min()), _show(kpi, s.min()), "lowest value")
    text = (f"{kpi.label}{_where_text(filters)} by {UNIT_WORD[freq]}, {len(s)} complete periods from "
            f"{_plabel(first, freq)} to {_plabel(last, freq)}: it went from {f['first'].display} to "
            f"{f['last'].display}, with a high of {f['highest'].display} ({_plabel(hi, freq)}) and a low of "
            f"{f['lowest'].display} ({_plabel(lo, freq)}).")
    records = [_rec("E1", "headline", f"{kpi.label} by {UNIT_WORD[freq]}", text, f, [],
                    _entities([m for vs in filters.values() for m in vs]))]
    tail = s.tail(MAX_TREND_ROWS)
    for ts, v in tail.items():
        rf: dict = {}
        _fact(rf, "value", float(v), _show(kpi, v), kpi.label)
        records.append(_rec(f"E{len(records) + 1}", "row", _plabel(ts, freq),
                            f"{_plabel(ts, freq)}: {rf['value'].display}.", rf, []))
    limits = [BASE_LIMIT]
    if len(s) > len(tail):
        limits.append("Only the most recent periods are listed row by row.")
    end_of_last = pd.Period(last, freq={"M": "M", "Q": "Q", "W": "W-SUN"}[freq]).end_time
    sq = _trend_sql(kpi, freq, filters, first, end_of_last)
    records[0].sql = [sq]
    return ToolResult("trend", a, _bundle(kpi, (first, end_of_last), records, limits, notes), None, notes)


def _trend_sql(kpi, freq, filters, start, end) -> dict:
    """The query behind the trend (the same one engine.kpi.kpi_series runs, limited to these dates)."""
    from engine.kpi import FREQ
    where, params = _trend_where(filters, start, end)
    sql = (f"SELECT date_trunc('{FREQ[freq][0]}', order_date) AS period, {kpi.num} AS num, "
           f"{kpi.den if kpi.den else 'NULL'} AS den FROM sales{where} GROUP BY 1 ORDER BY 1")
    return {"sql": sql, "params": [str(p) for p in params]}


def _trend_where(filters: dict, start, end):
    clauses, params = ["order_date >= ?", "order_date <= ?"], [pd.Timestamp(start).date(), pd.Timestamp(end).date()]
    for col, vals in filters.items():
        clauses.append(f"{q(col)} IN ({', '.join('?' for _ in vals)})")
        params.extend(str(v) for v in vals)
    return " WHERE " + " AND ".join(clauses), params


# ----------------------------------------------------------------------------
# Tool 4: contributors
# ----------------------------------------------------------------------------
def tool_contributors(ds, a: dict) -> ToolResult:
    notes: list[str] = []
    kpi = resolve_kpi(ds, a.get("kpi"))
    dim = resolve_dimension(ds, a.get("dimension"))
    filters = resolve_filters(ds, a.get("filters"))
    start, end = resolve_window(ds, a, notes, need=True)
    n = _top_n(a, notes)
    cs, ce = a.get("compare_start"), a.get("compare_end")
    if (cs in (None, "")) != (ce in (None, "")):
        raise AnalystError("Give both 'compare_start' and 'compare_end', or neither (then the same dates a "
                           "year earlier are used).")
    if cs not in (None, ""):
        compare = (_parse_date(cs, False), _parse_date(ce, True))
        if compare[1] < compare[0]:
            raise AnalystError("The comparison end date is before its start date.")
    else:
        compare = one_year_earlier(start, end)
    if compare[1] < pd.Timestamp(ds.min_date) or compare[0] > pd.Timestamp(ds.max_date):
        raise AnalystError("There is no data in the comparison window. Its dates must fall inside "
                           f"{_span(ds.min_date, ds.max_date)}.")
    try:
        dec = decompose(ds, kpi.key, dim, (start, end), compare, filters)
    except DecomposeError as err:
        raise AnalystError(str(err)) from err
    pts = kpi.fmt == "pct"
    f: dict = {}
    _fact(f, "change", dec.change, _money_signed(dec.change, kpi.fmt), "change between the windows")
    _fact(f, "before", dec.base_total, _show(kpi, dec.base_total), "value in the comparison window")
    _fact(f, "after", dec.cur_total, _show(kpi, dec.cur_total), "value in the main window")
    text = (f"{kpi.label}{_where_text(filters)} changed by {f['change'].display}, from {f['before'].display} "
            f"({_span(*compare)}) to {f['after'].display} ({_span(start, end)}).")
    records = [_rec("E1", "headline", f"{kpi.label} change", text, f, dec.queries[2:],
                    _entities([m for vs in filters.values() for m in vs]))]
    for _, r in dec.top(n).iterrows():
        rf: dict = {}
        _fact(rf, "contribution", float(r["contribution"]), _money_signed(float(r["contribution"]), kpi.fmt),
              "contribution to the change")
        t = f"{r['member']}: {rf['contribution'].display}"
        sh = r["share"]
        if _ok(sh):
            _fact(rf, "share", float(sh), f"{abs(sh):.0%}", "share of the change")
            t += (f" ({rf['share'].display} of the change)" if sh >= 0 else
                  f" (offsetting {rf['share'].display} of the change)")
        records.append(_rec(f"E{len(records) + 1}", "row", str(r["member"]), t + ".", rf, dec.queries[:2],
                            _entities([r["member"]])))
    limits = [BASE_LIMIT + " A big contributor is only a lead to check; use 'why' to verify it."]
    limits += [x for x in dec.notes if not any(c.isdigit() for c in x)]
    return ToolResult("contributors", a, _bundle(kpi, (start, end), records, limits, notes, compare), None, notes)


# ----------------------------------------------------------------------------
# Tool 5: alerts
# ----------------------------------------------------------------------------
def tool_alerts(ds, a: dict) -> ToolResult:
    notes: list[str] = []
    freq = _freq(a)
    kpi = resolve_kpi(ds, a["kpi"]) if a.get("kpi") not in (None, "") else None
    n = _top_n(a, notes)
    primary_only = a.get("primary_only", True) not in (False, "false", "no", 0)
    events, dnotes = detect_all(ds, freq, [kpi.key] if kpi else None, primary_only=primary_only)
    window = (pd.Timestamp(ds.min_date), pd.Timestamp(ds.max_date))
    shown = events[:n]
    if not shown:
        rec = _rec("E1", "headline", "Alerts",
                   "No unusual changes were found in this file at the normal sensitivity.", {}, [])
        return ToolResult("alerts", a, _bundle(kpi, window, [rec], [BASE_LIMIT], notes), None, notes)
    f: dict = {}
    _fact(f, "found", len(events), f"{len(events)}", "unusual changes found")
    _fact(f, "shown", len(shown), f"{len(shown)}", "shown")
    records = [_rec("E1", "headline", "Alerts",
                    f"{len(events)} unusual change(s) were found; showing the strongest {len(shown)}.", f, [])]
    for e in shown:
        ef: dict = {}
        _fact(ef, "pct", e.pct_vs_expected, (f"{e.pct_vs_expected * 100:+.1f} pts" if e.is_points
                                              else f"{e.pct_vs_expected:+.1%}"), "change versus expected")
        _fact(ef, "observed", e.observed, format_value(e.observed, e.fmt), "observed")
        _fact(ef, "expected", e.expected, format_value(e.expected, e.fmt), "expected")
        records.append(_rec(f"E{len(records) + 1}", "row", f"{e.label} {e.direction}", e.headline() + ".", ef, []))
    limits = [BASE_LIMIT + " An alert says WHEN something changed; use 'why' to check what drove it."]
    limits += [x for x in dnotes if not any(c.isdigit() for c in x)][:2]
    return ToolResult("alerts", a, _bundle(kpi, window, records, limits, notes), None, notes)


# ----------------------------------------------------------------------------
# Tool 6: why (verified causes)
# ----------------------------------------------------------------------------
def tool_why(ds, a: dict) -> ToolResult:
    notes: list[str] = []
    kpi = resolve_kpi(ds, a.get("kpi"))
    filters = resolve_filters(ds, a.get("filters"))
    start, end = resolve_window(ds, a, notes, need=True)
    try:
        res = verify_window(ds, kpi.key, (start, end), None, filters)
    except (ValueError, IngestError) as err:
        raise AnalystError(str(err)) from err
    bundle = build_evidence(res)
    bundle.limits += notes
    return ToolResult("why", a, bundle, None, notes)


# ----------------------------------------------------------------------------
# Tool 7: forecast
# ----------------------------------------------------------------------------
def tool_forecast(ds, a: dict) -> ToolResult:
    notes: list[str] = []
    kpi = resolve_kpi(ds, a.get("kpi"))
    freq = _freq(a)
    filters = resolve_filters(ds, a.get("filters"))
    h = a.get("horizon")
    if h is not None:
        if isinstance(h, bool) or not isinstance(h, (int, float)) or int(h) != h or not 1 <= int(h) <= 24:
            raise AnalystError("'horizon' must be a whole number of periods from 1 to 24.")
        h = int(h)
    try:
        fc = forecast_kpi(ds, kpi.key, freq, filters, h)
    except (ForecastError, IngestError) as err:
        raise AnalystError(str(err)) from err
    return ToolResult("forecast", a, None, fc, notes)


# ----------------------------------------------------------------------------
# Tool 8: sql (read-only, guarded)
# ----------------------------------------------------------------------------
def _cell(v) -> tuple[str, float | None]:
    """(text, number or None) for one result cell."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT:
        return "null", None
    if isinstance(v, (bool, np.bool_)):
        return ("true" if v else "false"), None
    if isinstance(v, (int, np.integer)):
        return str(int(v)), float(v)
    if isinstance(v, (float, np.floating)):
        return f"{float(v):,.2f}", float(v)
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "isoformat"):
        try:
            return _d(v), None
        except (ValueError, TypeError):
            pass
    s = " ".join(str(v).split())
    return (s if len(s) <= 60 else s[:57] + "..."), None


def tool_sql(ds, a: dict) -> ToolResult:
    query = a.get("query")
    try:
        core, wrapped = clean_select(query, MAX_SQL_ROWS)
    except SqlGuardError as err:
        raise AnalystError(f"The SQL was rejected: {err}") from err
    try:
        df = ds.query(wrapped)
    except Exception as err:                                  # database errors: tell the planner, no crash
        raise AnalystError("The database could not run this query: " + " ".join(str(err).split())[:200]) from err
    truncated = len(df) > MAX_SQL_ROWS
    df = df.head(MAX_SQL_ROWS)
    sqlinfo = [{"sql": core, "params": []}]
    if df.empty:
        text = "The query returned no rows."
    elif truncated:
        text = f"The query returned more than {MAX_SQL_ROWS} rows; only the first {MAX_SQL_ROWS} are kept."
    else:
        text = f"The query returned {len(df)} rows."
    f: dict = {}
    if not df.empty and not truncated:
        _fact(f, "rows", len(df), f"{len(df)}", "rows returned")
    window = (pd.Timestamp(ds.min_date), pd.Timestamp(ds.max_date))
    records = [_rec("E1", "headline", "SQL result", text, f, sqlinfo)]
    for _, row in df.head(MAX_ROW_RECORDS).iterrows():
        parts, facts, ents = [], {}, []
        for col, v in row.items():
            shown, num = _cell(v)
            parts.append(f"{col} = {shown}")
            if num is not None:
                _fact(facts, str(col), num, shown, str(col))
            elif isinstance(v, str):
                ents.append(shown)
        records.append(_rec(f"E{len(records) + 1}", "row", f"Row {len(records)}", "; ".join(parts) + ".",
                            facts, sqlinfo, _entities(ents)))
    limits = [BASE_LIMIT + " This answer came from a custom read-only SQL query, shown in the evidence."]
    if len(df) > MAX_ROW_RECORDS:
        limits.append("Only the first rows are listed.")
    return ToolResult("sql", a, _bundle(None, window, records, limits, []), None, [])


# ----------------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------------
_COMMON = {
    "kpi": "revenue | profit | orders | quantity | profit_margin | avg_order_value | avg_discount (only those in the file)",
    "start": "ISO date 2024-03-01, month 2024-03, or year 2024",
    "end": "ISO date, month or year (inclusive; a month/year means its last day)",
    "filters": 'optional, e.g. {"Region": "West"}; values must exist in the file',
}
TOOLS: dict[str, ToolSpec] = {
    "kpi_value": ToolSpec(tool_kpi_value, "One KPI over a date range, optionally compared with the same dates a year earlier.",
                          {**_COMMON, "compare": "true to compare with the same dates a year earlier"}, ("kpi",)),
    "breakdown": ToolSpec(tool_breakdown, "A KPI split by one category column (top or bottom members).",
                          {**_COMMON, "dimension": "a category column of the file",
                           "top_n": "1-10 (default 5)", "order": "desc (default, highest first) or asc"},
                          ("kpi", "dimension")),
    "trend": ToolSpec(tool_trend, "A KPI over time, by month, quarter or week (complete periods only).",
                      {**_COMMON, "freq": "month (default) | quarter | week"}, ("kpi",)),
    "contributors": ToolSpec(tool_contributors, "WHO contributed to the change in a KPI between two windows (exact split by one column).",
                             {**_COMMON, "dimension": "a category column of the file", "top_n": "1-10 (default 5)",
                              "compare_start": "optional baseline start (default: same dates a year earlier)",
                              "compare_end": "optional baseline end"}, ("kpi", "dimension", "start", "end")),
    "alerts": ToolSpec(tool_alerts, "Unusual changes found by the change detector (what and when).",
                       {"kpi": "optional: only this KPI", "freq": "month (default) | quarter | week",
                        "top_n": "1-10 (default 5)", "primary_only": "true (default) = headline KPIs only"}, ()),
    "why": ToolSpec(tool_why, "WHY a KPI changed in a window: tests the biggest contributors with 5 checks and says CAUSE / NOT_A_CAUSE / UNKNOWN.",
                    {k: _COMMON[k] for k in ("kpi", "start", "end", "filters")}, ("kpi", "start", "end")),
    "forecast": ToolSpec(tool_forecast, "A validated forecast with an uncertainty range for the next periods.",
                         {"kpi": _COMMON["kpi"], "freq": "month (default) | quarter | week",
                          "horizon": "periods ahead, 1-24 (default depends on freq)", "filters": _COMMON["filters"]},
                         ("kpi",)),
    "sql": ToolSpec(tool_sql, "LAST RESORT when no other tool fits: ONE read-only SELECT on the table sales "
                              "(columns are listed in the dataset context; no JOIN, no comments, no subqueries except FROM (SELECT ...)).",
                    {"query": "a single SELECT statement"}, ("query",)),
}


def describe_tools() -> str:
    lines = []
    for name, spec in TOOLS.items():
        lines.append(f"- {name}: {spec.help}")
        for p, d in spec.params.items():
            lines.append(f"    {p}{'*' if p in spec.required else ''}: {d}")
    lines.append("(* = required)")
    return "\n".join(lines)


def run_tool(ds, tool, args=None) -> ToolResult:
    """Validate and run one tool. Raises AnalystError (message for the planner) if it cannot be run."""
    name = str(tool or "").strip().lower()
    spec = TOOLS.get(name)
    if spec is None:
        raise AnalystError(f"Unknown tool '{tool}'. Available tools: {', '.join(TOOLS)}.")
    args = {} if args is None else args
    if not isinstance(args, dict):
        raise AnalystError("'args' must be a JSON object.")
    unknown = sorted(set(args) - set(spec.params))
    if unknown:
        raise AnalystError(f"Unknown argument(s) for {name}: {', '.join(map(str, unknown))}. "
                           f"Allowed: {', '.join(spec.params)}.")
    missing = [p for p in spec.required if args.get(p) in (None, "")]
    if missing:
        raise AnalystError(f"Missing required argument(s) for {name}: {', '.join(missing)}.")
    res = spec.func(ds, dict(args))
    res.tool = name
    return res
