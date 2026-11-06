"""Tests for engine/evidence.py (Step 7, part 2).

    pytest tests/test_evidence.py -v

Pure tests build small verify results by hand (fast, no database). The database tests use the
synthetic dataset: case A (a verified cause) and U (UNKNOWN).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.detect import detect_all  # noqa: E402
from engine.evidence import NUM_RE, build_evidence, strip_citations  # noqa: E402
from engine.ingest import load_dataset  # noqa: E402
from engine.verify import (CAUSE, NOT_A_CAUSE, UNKNOWN, Cause, Check, SliceVerdict,  # noqa: E402
                           WindowResult, verify_event, verify_window)

SYNTHETIC = ROOT / "data" / "synthetic_demo.csv"
needs_synthetic = pytest.mark.skipif(not SYNTHETIC.exists(), reason="synthetic file missing")

W = (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-08-31"))
C = (pd.Timestamp("2023-06-01"), pd.Timestamp("2023-08-31"))


def checks(n_placebo=20):
    return [
        Check("Size and distinctiveness", "pass", "5 customers explain 80% of the change.",
              {"share_of_change": 0.8, "normal_share": 0.07, "lift": 11.9}),
        Check("Differs from the rest", "pass", "The group moved -90.0% while the rest moved -5.0%.",
              {"group_move": -0.9, "rest_move": -0.05, "gap": -0.85}),
        Check("Started inside the window", "pass", "Gap before was -2.0%.", {"pre_gap": -0.02, "gap": -0.85}),
        Check("Not a yearly repeat", "pass", "Last year gap was +1.0%.", {"previous_year_gap": 0.01}),
        Check("Unusual for its own history", "pass", "Gap is 6.0 spreads away.",
              {"n_placebo": n_placebo, "z": -6.0, "placebo_gaps": [0.1, float("nan")]}),
    ]


def make_cause(filters, contribution=-100000.0, share=0.8, ck=None):
    leaf = SliceVerdict(next(iter(filters)), list(next(iter(filters.values()))), filters, CAUSE, "ok",
                        ck or checks(), contribution, share, 5.0, [{"sql": "SELECT 1 WHERE a = ?", "params": ["x"]}])
    return Cause(filters, share, contribution, [leaf])


def make_rejected(dim, members, share=0.9, verdict=NOT_A_CAUSE):
    return SliceVerdict(dim, members, {dim: members}, verdict, f"{dim} fell with everything else.",
                        [Check("Size and distinctiveness", "fail", "Not distinctive.",
                               {"normal_share": 0.8, "lift": 1.1})], -1.0, share, 1.1, [])


def make_result(causes=(), candidates=(), kpi="revenue", label="Revenue", fmt="money", change=-150000.0,
                conclusion=None, notes=()):
    conclusion = conclusion or (CAUSE if causes else UNKNOWN)
    return WindowResult(kpi, label, fmt, W, C, change, conclusion, "reason text", list(causes),
                        list(candidates), list(notes))


def numbers_in(text, entities=()):
    text = strip_citations(text)
    for e in sorted(entities, key=len, reverse=True):
        text = text.replace(e, " ")
    return {m.replace(",", "") for m in NUM_RE.findall(text)}


# ============================================================ pure tests
def test_records_are_numbered_in_order_with_headline_first():
    res = make_result([make_cause({"Customer Name": ["Customer 242", "Customer 317"]})],
                      [make_rejected("Category", ["Technology"])])
    b = build_evidence(res)
    assert [r.id for r in b.records] == ["E1", "E2", "E3"]
    assert [r.kind for r in b.records] == ["headline", "cause", "rejected"]
    assert b.conclusion == CAUSE and b.get("E2").verdict == CAUSE
    with pytest.raises(KeyError):
        b.get("E9")


def test_cause_record_numbers_come_from_the_result_and_are_formatted():
    b = build_evidence(make_result([make_cause({"Customer Name": ["Customer 242"]}, -123456.0, 0.8)]))
    f = b.get("E2").facts
    assert f["share"].display == "80%" and f["share"].value == pytest.approx(0.8)
    assert f["contribution"].display == "-123,456"
    assert f["lift"].display == "11.9x" and f["normal_share"].display == "7%"
    assert f["checks_passed"].display == "5" and f["checks_total"].display == "5"
    assert f["n_placebo"].display == "20" and f["z"].display == "6.0"
    assert f["group_move"].display == "-90.0%" and f["gap"].display == "-85.0%"
    assert len(b.get("E2").checks) == 5


def test_percent_kpis_are_written_in_points():
    ck = checks()
    ck[1] = Check("Differs from the rest", "pass", "x", {"group_move": -0.3, "rest_move": -0.02, "gap": -0.28})
    res = make_result([make_cause({"Region": ["West"]}, -0.045, 1.06, ck)], kpi="profit_margin",
                      label="Profit margin", fmt="pct", change=-0.0424)
    b = build_evidence(res)
    assert b.get("E1").facts["change"].display == "-4.2 pts"
    assert "4.2 pts" in b.get("E1").text and b.get("E2").facts["contribution"].display == "-4.5 pts"
    assert b.get("E2").facts["gap"].display == "-28.0 pts"


def test_unknown_result_has_an_unknown_record_and_a_collect_action_and_no_cause():
    b = build_evidence(make_result([], [make_rejected("Region", ["Central", "West"])]))
    kinds = [r.kind for r in b.records]
    assert kinds[:2] == ["headline", "unknown"] and "cause" not in kinds
    assert b.conclusion == UNKNOWN
    assert [a.kind for a in b.actions] == ["collect"] and b.actions[0].basis == ["E2"]
    assert "UNKNOWN" in b.get("E2").text


def test_actions_follow_simple_rules_and_point_at_their_record():
    cust = build_evidence(make_result([make_cause({"Customer Name": ["Customer 242"]})]))
    assert cust.actions[0].basis == ["E2"] and cust.actions[0].kind == "check"
    assert "customers" in cust.actions[0].text.lower()

    margin = build_evidence(make_result([make_cause({"Sub-Category": ["Phones"]}, -0.04, 1.0)],
                                        kpi="profit_margin", label="Profit margin", fmt="pct", change=-0.04))
    assert "discount" in margin.actions[0].text.lower()

    stock = build_evidence(make_result([make_cause({"Product Name": ["Phones Max"]})]))
    assert "stock" in stock.actions[0].text.lower()

    place = build_evidence(make_result([make_cause({"Region": ["East"]})]))
    assert "local operations" in place.actions[0].text

    up = build_evidence(make_result([make_cause({"Region": ["East"]}, 5000.0, 0.7)], change=7000.0))
    assert "worked" in up.actions[0].text and up.get("E1").text.startswith("Revenue rose")


def test_wording_never_claims_proof_or_plain_causation():
    b = build_evidence(make_result([make_cause({"Region": ["East"]})], [make_rejected("Segment", ["Consumer"])]))
    text = b.summary_text().lower()
    assert "does not prove" in text and "to check, not proven" in text
    assert "caused by" not in text and "proved that" not in text and "the cause is" not in text
    assert "evidence supports" in b.get("E2").text.lower()


def test_strip_citations_removes_tags_only():
    assert strip_citations("Revenue fell [E1] and X [E2, E3] 5%.").split() == ["Revenue", "fell", "and", "X", "5%."]


def test_every_number_in_the_summary_is_in_the_allowed_set():
    """The fallback summary uses only numbers that exist as evidence (what the validator will enforce)."""
    res = make_result([make_cause({"Customer Name": ["Customer 242", "Customer 317"]})],
                      [make_rejected("Category", ["Technology"]), make_rejected("Region", ["East"])],
                      notes=["No expected value was given."])
    b = build_evidence(res)
    allowed = b.allowed_numbers()
    used = numbers_in(b.summary_text(), b.entity_names())
    assert used and used <= allowed, f"not in evidence: {sorted(used - allowed)}"
    assert "999" not in allowed and "242" not in allowed          # digits in a name are not numbers


def test_rejected_list_is_capped_and_skips_candidates_already_inside_a_cause():
    cands = [make_rejected("Category", [f"C{i}"], share=0.9 - i * 0.05) for i in range(8)]
    cands.append(make_rejected("Customer Name", ["Customer 242"], share=0.99))      # same as the cause
    b = build_evidence(make_result([make_cause({"Customer Name": ["Customer 242"]})], cands), max_rejected=3)
    rej = [r for r in b.records if r.kind == "rejected"]
    assert len(rej) == 3 and all("Customer 242" not in r.title for r in rej)
    assert [r.facts["share"].display for r in rej] == ["90%", "85%", "80%"]       # best first


def test_nan_values_do_not_crash_and_the_export_is_valid_json():
    ck = checks()
    ck[0] = Check("Size and distinctiveness", "unknown", "cannot measure", {"lift": float("nan")})
    cause = make_cause({"Region": ["East"]}, float("nan"), float("nan"), ck)
    cause.leaf.queries = [{"sql": "SELECT 1", "params": []}]
    b = build_evidence(make_result([cause], [make_rejected("Segment", ["Consumer"], share=float("nan"))]))
    text = json.dumps(b.to_dict(), allow_nan=False)
    assert "NaN" not in text and "lift" not in b.get("E2").facts
    json.dumps(b.for_llm(), allow_nan=False)


def test_for_llm_has_facts_and_checks_but_no_sql():
    b = build_evidence(make_result([make_cause({"Customer Name": ["Customer 242"]})]))
    d = b.for_llm()
    dumped = json.dumps(d)
    assert "SELECT" not in dumped and "placebo_gaps" not in dumped
    rec = [r for r in d["records"] if r["id"] == "E2"][0]
    assert rec["facts"]["share"]["display"] == "80%" and len(rec["checks"]) == 5
    assert d["actions"] and d["limits"] and d["conclusion"] == CAUSE
    assert b.to_dict()["records"][1]["sql"][0]["params"] == ["x"]   # the UI export keeps the SQL


def test_short_history_adds_a_limit_and_notes_are_carried():
    b = build_evidence(make_result([make_cause({"Region": ["East"]}, ck=checks(n_placebo=7))],
                                   notes=["No expected value was given."]))
    assert any("short" in x for x in b.limits) and "No expected value was given." in b.limits
    assert not any("short" in x for x in build_evidence(
        make_result([make_cause({"Region": ["East"]})])).limits)


# ============================================================ synthetic data
@pytest.fixture(scope="module")
def synthetic():
    return load_dataset(SYNTHETIC)


@needs_synthetic
def test_A_evidence_names_west_phones_with_five_passed_checks(synthetic):
    b = build_evidence(verify_window(synthetic, "profit_margin", ("2024-03-01", "2024-04-30")))
    cause = b.causes()[0]
    assert cause.filters == {"Region": ["West"], "Sub-Category": ["Phones"]}
    assert cause.facts["checks_passed"].display == "5" and "pts" in b.get("E1").facts["change"].display
    assert cause.sql and all("sql" in q for q in cause.sql)
    used = numbers_in(b.summary_text(), b.entity_names())
    assert used <= b.allowed_numbers(), sorted(used - b.allowed_numbers())


@needs_synthetic
def test_U_evidence_says_unknown_and_never_names_a_cause(synthetic):
    b = build_evidence(verify_window(synthetic, "revenue", ("2024-09-01", "2024-09-30")))
    assert b.conclusion == UNKNOWN and b.causes() == []
    assert b.get("E2").kind == "unknown" and b.actions[0].kind == "collect"
    assert "supports" not in b.summary_text().split("Looked at")[0].lower().replace("evidence supports these", "")


@needs_synthetic
def test_event_evidence_carries_observed_and_expected(synthetic):
    ev = [e for e in detect_all(synthetic, "M", kpis=["revenue"], primary_only=True)[0] if e.direction == "down"][0]
    b = build_evidence(verify_event(synthetic, ev), ev)
    f = b.get("E1").facts
    assert {"observed", "expected", "peak_z", "change"} <= set(f)
    assert "observed" in b.get("E1").text and "expected" in b.get("E1").text
    used = numbers_in(b.summary_text(), b.entity_names())
    assert used <= b.allowed_numbers(), sorted(used - b.allowed_numbers())
    json.dumps(b.to_dict(), allow_nan=False)
