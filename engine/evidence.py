"""Evidence records: the ONLY thing the LLM (and the UI) is allowed to talk about.

The idea, in plain words
------------------------
verify.py produces a rich result object. This module flattens it into a small, numbered
list of EVIDENCE RECORDS:

    E1  headline    "Revenue fell ... in Aug-Nov 2024"
    E2  cause       "Customer 242, Customer 317 ... explain 59% of the change"   (+ its 5 checks)
    E3  rejected    "Technology": looked big, but fell WITH the whole business
    E4  unknown     (when nothing passed) "the cause is not identifiable from this file"

Every number in a record is a FACT with a raw value and a display string ("59%", "4.5x").
The rules for the next step (llm/):
    - the LLM may only explain using these records and must cite them ([E2]);
    - a validator rejects any number in the LLM text that is not one of these displays.
So the LLM can never invent a figure; it can only reword evidence made by Python/SQL.

This module also gives:
    - a rule-based `summary_text()` that needs NO LLM (used as the safe fallback), and
    - rule-based ACTIONS: suggestions to check, each linked to the records it is based on.
      Actions are hypotheses to test, never claims of cause.

Honest limits
-------------
  - "CAUSE" means "the evidence supports it", not "proved". Every bundle carries limits that
    say so, and the wording never says "caused by" without that qualifier.
  - Actions are generic rules by KPI and dimension, not business advice tailored to your firm.
  - Names of categories, customers and so on may contain digits ("Customer 242"). They are
    listed in `entity_names()` so the validator can ignore digits inside names.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.kpi import format_value
from engine.verify import (CAUSE, NOT_A_CAUSE, UNKNOWN, Cause, SliceVerdict, WindowResult,
                           _fmt_move, to_plain)


NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
CITE_RE = re.compile(r"\[E\d+(?:\s*,\s*E\d+)*\]")


def strip_citations(text: str) -> str:
    """Remove [E2]-style citation tags (their digits are record ids, not evidence numbers)."""
    return CITE_RE.sub(" ", text)


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------
@dataclass
class Fact:
    value: float | str | None      # the raw number (or text) from Python/SQL
    display: str                   # exactly how it may be written in text
    label: str = ""                # what it is, in words


@dataclass
class Record:
    id: str                        # "E1", "E2", ...
    kind: str                      # headline / cause / rejected / unknown
    title: str
    verdict: str                   # CAUSE / NOT_A_CAUSE / UNKNOWN / "" for the headline
    text: str                      # one deterministic sentence built from the facts
    facts: dict[str, Fact] = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)       # name, status, detail
    filters: dict = field(default_factory=dict)
    sql: list[dict] = field(default_factory=list)          # evidence queries (SQL + parameters)
    entities: list[str] = field(default_factory=list)      # names that may contain digits


@dataclass
class Action:
    text: str
    basis: list[str]               # record ids this suggestion rests on
    kind: str                      # "check" (evidence-supported lead) or "collect" (needs more data)


@dataclass
class EvidenceBundle:
    kpi: str
    label: str
    window: tuple[str, str]
    compare: tuple[str, str]
    conclusion: str
    records: list[Record]
    actions: list[Action]
    limits: list[str]

    def get(self, rid: str) -> Record:
        for r in self.records:
            if r.id == rid:
                return r
        raise KeyError(rid)

    def causes(self) -> list[Record]:
        return [r for r in self.records if r.kind == "cause"]

    def entity_names(self) -> set[str]:
        return {e for r in self.records for e in r.entities}

    def allowed_displays(self) -> set[str]:
        """Every number string the explanation is allowed to contain."""
        out = set()
        for r in self.records:
            out |= {f.display for f in r.facts.values()}
        return out

    def allowed_numbers(self) -> set[str]:
        """Every number token that appears in the records: fact displays, record sentences and
        check sentences. A validator should accept only these (digits inside entity names are
        handled separately with entity_names())."""
        out = set()
        for r in self.records:
            texts = [r.text] + [f.display for f in r.facts.values()] + [c["detail"] for c in r.checks]
            for t in texts:
                for e in sorted(self.entity_names(), key=len, reverse=True):
                    t = t.replace(e, " ")
                out |= {m.replace(",", "") for m in NUM_RE.findall(t)}
        out |= {m.replace(",", "") for t in self.window + self.compare for m in NUM_RE.findall(t)}
        return out

    def to_dict(self) -> dict:
        return to_plain(self)

    def for_llm(self) -> dict:
        """Compact JSON for the LLM: records with facts and checks, no SQL, no raw placebo lists."""
        recs = []
        for r in self.records:
            recs.append({"id": r.id, "kind": r.kind, "title": r.title, "verdict": r.verdict, "text": r.text,
                         "facts": {k: {"display": f.display, "label": f.label} for k, f in r.facts.items()},
                         "checks": [{"name": c["name"], "status": c["status"], "detail": c["detail"]}
                                    for c in r.checks]})
        return {"kpi": self.label, "window": list(self.window), "compare_window": list(self.compare),
                "conclusion": self.conclusion, "records": recs,
                "actions": [{"text": a.text, "basis": a.basis, "kind": a.kind} for a in self.actions],
                "limits": self.limits}

    def summary_text(self) -> str:
        """Plain explanation built only from the records (no LLM). Safe fallback."""
        lines = []
        for r in self.records:
            if r.kind in ("headline", "cause", "unknown"):
                lines.append(f"{r.text} [{r.id}]")
        rej = [r for r in self.records if r.kind == "rejected"]
        if rej:
            lines.append("Looked at but not supported as causes: "
                         + "; ".join(f"{r.title} [{r.id}]" for r in rej) + ".")
        if self.actions:
            lines.append("Suggested next steps (to check, not proven): " + " ".join(a.text for a in self.actions))
        lines.append(self.limits[0])
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%d %b %Y").lstrip("0")


def _span(w) -> str:
    return f"{_d(w[0])} to {_d(w[1])}"


def _ok(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(float(x))


def _money_signed(x: float, fmt: str) -> str:
    if fmt == "pct":
        return f"{x * 100:+.1f} pts"
    return ("-" if x < 0 else "+") + format_value(abs(x), fmt)


def _where(filters: dict) -> str:
    return " and ".join(f"{k} = {', '.join(v)}" for k, v in filters.items())


def _fact(facts: dict, key: str, value, display: str, label: str):
    facts[key] = Fact(value if not isinstance(value, (np.floating, np.integer)) else float(value),
                      display, label)


# Which check-value keys become facts, how they are shown, and what they mean
def _check_facts(facts: dict, checks, pct: bool):
    for c in checks:
        v = c.values
        if c.name == "Size and distinctiveness":
            if _ok(v.get("normal_share")):
                _fact(facts, "normal_share", v["normal_share"], f"{v['normal_share']:.0%}",
                      "the group's normal share of the business")
            if _ok(v.get("lift")):
                _fact(facts, "lift", v["lift"], f"{v['lift']:.1f}x", "how many times its normal weight")
        elif c.name == "Differs from the rest":
            for k, lab in (("group_move", "how much the group moved"), ("rest_move", "how much the rest moved"),
                           ("gap", "gap between group and rest")):
                if _ok(v.get(k)):
                    _fact(facts, k, v[k], _fmt_move(v[k], pct), lab)
        elif c.name == "Started inside the window":
            if _ok(v.get("pre_gap")):
                _fact(facts, "pre_gap", v["pre_gap"], _fmt_move(v["pre_gap"], pct),
                      "gap in the period just before the window")
        elif c.name == "Not a yearly repeat":
            if _ok(v.get("previous_year_gap")):
                _fact(facts, "previous_year_gap", v["previous_year_gap"], _fmt_move(v["previous_year_gap"], pct),
                      "gap in the same window a year earlier")
        elif c.name == "Unusual for its own history":
            if _ok(v.get("n_placebo")):
                _fact(facts, "n_placebo", v["n_placebo"], f"{int(v['n_placebo'])}",
                      "ordinary past windows compared")
            if _ok(v.get("z")):
                _fact(facts, "z", abs(v["z"]), f"{abs(v['z']):.1f}", "spreads beyond ordinary swings")


def _entities(filters: dict) -> list[str]:
    return [m for vals in filters.values() for m in vals]


# ----------------------------------------------------------------------------
# Actions (rules, not LLM)
# ----------------------------------------------------------------------------
_PRODUCT_DIMS = {"Product Name", "Sub-Category", "Category", "Product ID"}
_PLACE_DIMS = {"Region", "State", "City", "Country", "Postal Code"}
_PEOPLE_DIMS = {"Customer Name", "Customer ID", "Segment"}


def _action_for(res: WindowResult, cause: Cause, rid: str) -> Action:
    down = cause.contribution < 0
    key = res.kpi
    who = _where({k: v for k, v in cause.filters.items()})
    dims = set(cause.filters)
    margin_like = any(t in key for t in ("margin", "discount", "profit"))
    if dims & {"Customer Name", "Customer ID"}:
        t = (f"Contact the customers ({who}) to find out why their orders changed, and check for account or "
             f"contract changes." if down else f"Find out what these customers ({who}) respond to and repeat it.")
    elif dims & _PRODUCT_DIMS:
        if down and margin_like:
            t = f"Review discounts, price changes and costs for {who} in this window."
        elif down:
            t = f"Check stock availability, pricing and competitor activity for {who} in this window."
        else:
            t = f"Find out what drove the rise for {who} (promotion, pricing, demand) and whether it can be repeated."
    elif dims & _PLACE_DIMS:
        t = (f"Review local operations, pricing and competitor activity for {who}." if down
             else f"Find out what worked for {who} and whether it can be repeated elsewhere.")
    else:
        t = (f"Look into {who} first: it is where most of this change happened." if down
             else f"Look into {who}: it is where most of this rise happened.")
    return Action(t, [rid], "check")


# ----------------------------------------------------------------------------
# Building the bundle
# ----------------------------------------------------------------------------
def build_evidence(res: WindowResult, event=None, max_rejected: int = 4) -> EvidenceBundle:
    """Turn a verify.WindowResult (optionally with the detect.ChangeEvent that triggered it)
    into numbered evidence records, actions and limits."""
    pct = res.fmt == "pct"
    records: list[Record] = []
    actions: list[Action] = []

    def new_id() -> str:
        return f"E{len(records) + 1}"

    # --- E1 headline
    hf: dict[str, Fact] = {}
    arrow = "fell" if res.change < 0 else "rose"
    _fact(hf, "change", res.change, _money_signed(res.change, res.fmt), "change versus the baseline")
    size = f"{abs(res.change) * 100:.1f} pts" if pct else format_value(abs(res.change), res.fmt)
    text = (f"{res.label} {arrow} by {size} "
            f"from {_span(res.window)}, compared with the baseline ({_span(res.compare)})")
    if event is not None:
        _fact(hf, "observed", event.observed, format_value(event.observed, event.fmt), "observed value")
        _fact(hf, "expected", event.expected, format_value(event.expected, event.fmt), "expected value")
        _fact(hf, "peak_z", abs(event.peak_z), f"{abs(event.peak_z):.1f}", "how unusual the peak period was")
        text += (f"; observed {hf['observed'].display} against an expected {hf['expected'].display}")
    text += "."
    records.append(Record(new_id(), "headline", f"{res.label} change", "", text, hf))

    # --- the explicit UNKNOWN record (placed right after the headline)
    if res.conclusion == UNKNOWN:
        rid = new_id()
        records.append(Record(rid, "unknown", "Cause unknown", UNKNOWN,
                              "The cause of this change is UNKNOWN from the data in this file: no region, product, "
                              "customer or other slice passed all the checks.", {}))
        actions.append(Action("Collect information that is not in this file for these dates (promotions, stock "
                              "levels, marketing spend, pricing changes, outside events) and add it as columns.",
                              [rid], "collect"))

    # --- causes
    for cause in res.causes:
        leaf: SliceVerdict = cause.leaf
        rid = new_id()
        f: dict[str, Fact] = {}
        _fact(f, "share", cause.share_of_window_change, f"{abs(cause.share_of_window_change):.0%}",
              "share of the whole change explained")
        _fact(f, "contribution", cause.contribution, _money_signed(cause.contribution, res.fmt),
              "contribution to the change")
        _check_facts(f, leaf.checks, pct)
        passed = sum(1 for c in leaf.checks if c.status == "pass")
        _fact(f, "checks_passed", passed, f"{passed}", "checks passed")
        _fact(f, "checks_total", len(leaf.checks), f"{len(leaf.checks)}", "checks run")
        where = _where(cause.filters)
        text = (f"The evidence supports {where} as a cause: it accounts for {f['share'].display} of the change "
                f"({f['contribution'].display}) and passed {f['checks_passed'].display} of "
                f"{f['checks_total'].display} checks")
        if "lift" in f:
            text += f", being {f['lift'].display} its normal weight in the change"
        text += "."
        records.append(Record(rid, "cause", where, CAUSE, text, f,
                              [{"name": c.name, "status": c.status, "detail": c.detail} for c in leaf.checks],
                              {k: list(v) for k, v in cause.filters.items()}, leaf.queries,
                              _entities(cause.filters)))
        actions.append(_action_for(res, cause, rid))

    # --- rejected / unknown candidates (best first by share), to show what was ruled out
    shown = 0
    for v in sorted(res.candidates, key=lambda x: -(abs(x.share_of_change) if _ok(x.share_of_change) else 0)):
        if v.verdict == CAUSE or not v.members or shown >= max_rejected:
            continue
        if any(set(v.members) <= set(r.entities) and v.dimension in r.filters for r in records if r.kind == "cause"):
            continue
        rid = new_id()
        f = {}
        if _ok(v.share_of_change):
            _fact(f, "share", v.share_of_change, f"{abs(v.share_of_change):.0%}", "share of the change it appears in")
        _check_facts(f, v.checks, pct)
        where = _where({v.dimension: v.members})
        kind = "rejected" if v.verdict == NOT_A_CAUSE else "unknown"
        why = v.reason
        text = (f"{where} is NOT supported as a cause: {why}" if kind == "rejected"
                else f"{where} could not be confirmed as a cause: {why}")
        records.append(Record(rid, "rejected", where, v.verdict, text, f,
                              [{"name": c.name, "status": c.status, "detail": c.detail} for c in v.checks],
                              {v.dimension: list(v.members)}, v.queries, list(v.members)))
        shown += 1

    limits = ["The evidence supports these findings; it does not prove them. Something not in this file "
              "(a competitor, weather, a data error) could also be involved."]
    limits += list(res.notes)
    n_pl = [c.values.get("n_placebo") for cz in res.causes for c in cz.leaf.checks
            if c.name == "Unusual for its own history" and c.values.get("n_placebo") is not None]
    if n_pl and min(n_pl) < 12:
        limits.append("The history available for the 'unusual for its own history' check is short, "
                      "so treat borderline findings with extra care.")
    return EvidenceBundle(res.kpi, res.label, (_d(res.window[0]), _d(res.window[1])),
                          (_d(res.compare[0]), _d(res.compare[1])), res.conclusion, records, actions, limits)
