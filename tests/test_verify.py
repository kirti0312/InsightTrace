"""Tests for engine/verify.py.

Run from the project folder:
    pytest tests/test_verify.py -v

1. Small pure tests (no database).
2. The shipped synthetic dataset with its planted events and decoys
   (answer key: data/synthetic_ground_truth.json):
     A, B, C must be found; the unexplained September dip (U) must be UNKNOWN;
     the decoys (East March dip every year, January slump) must NOT be called causes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.decompose import DecomposeError  # noqa: E402
from engine.detect import detect_all  # noqa: E402
from engine.ingest import load_dataset  # noqa: E402
from engine.verify import (CAUSE, NOT_A_CAUSE, UNKNOWN, VerifyConfig, _intersect, _mad_sigma,  # noqa: E402
                           _pick_group, to_plain, verify_by_month, verify_event,
                           verify_subwindows, verify_window)

SYNTHETIC = ROOT / "data" / "synthetic_demo.csv"
TRUTH = ROOT / "data" / "synthetic_ground_truth.json"
needs_synthetic = pytest.mark.skipif(not (SYNTHETIC.exists() and TRUTH.exists()),
                                     reason="synthetic files missing")


# ============================================================ pure tests
def _rows(**contrib):
    return pd.DataFrame({"member": list(contrib), "contribution": list(contrib.values())})


def test_pick_group_takes_fewest_members_covering_the_change():
    rows = _rows(A=-60.0, B=-20.0, C=-10.0, D=-5.0, E=5.0)
    g = _pick_group(rows, -95.0, VerifyConfig(cover=0.6, tail_ratio=0.9))
    assert list(g["member"]) == ["A"]                     # A alone is 63% of the change


def test_pick_group_keeps_members_that_moved_nearly_as_much_as_the_biggest():
    rows = _rows(A=-20.0, B=-19.0, C=-18.0, D=-17.0, E=-16.0, F=-1.0)
    g = _pick_group(rows, -91.0, VerifyConfig(cover=0.3, tail_ratio=0.4))
    assert list(g["member"]) == ["A", "B", "C", "D", "E"]  # five similar losers are one group


def test_pick_group_ignores_members_moving_the_other_way_and_caps_the_size():
    rows = _rows(A=-10.0, B=-9.0, C=-8.0, D=-7.0, E=50.0)
    assert "E" not in set(_pick_group(rows, -34.0, VerifyConfig())["member"])
    assert len(_pick_group(rows, -34.0, VerifyConfig(max_group=2, tail_ratio=0.1))) == 2
    assert _pick_group(_rows(A=5.0), -10.0, VerifyConfig()).empty


def test_mad_sigma_and_intersect_helpers():
    assert _mad_sigma([1, 2, 3, 4, 100]) == pytest.approx(1.4826 * 1.0)   # outlier barely matters
    assert _intersect({"Region": ["West"]}, {"Sub-Category": ["Phones"]}) == {"Region": ["West"],
                                                                              "Sub-Category": ["Phones"]}
    assert _intersect({"Region": ["West", "East"]}, {"Region": ["East", "South"]}) == {"Region": ["East"]}
    assert _intersect({"Region": ["West"]}, {"Region": ["East"]}) is None


def test_to_plain_makes_valid_json():
    from engine.verify import Check
    obj = {"a": np.float64(1.5), "b": float("nan"), "c": np.int64(3), "d": pd.Timestamp("2024-03-01"),
           "e": [Check("x", "pass", "ok", {"v": np.float64(float("inf"))})], "f": np.bool_(True)}
    out = to_plain(obj)
    text = json.dumps(out, allow_nan=False)                 # would raise on NaN/inf
    assert out["b"] is None and out["c"] == 3 and out["d"] == "2024-03-01"
    assert out["e"][0]["values"]["v"] is None and out["f"] is True and "NaN" not in text


# ============================================================ synthetic data
@pytest.fixture(scope="module")
def synthetic():
    return load_dataset(SYNTHETIC)


@pytest.fixture(scope="module")
def truth():
    return {e["id"][0]: e for e in json.loads(TRUTH.read_text())["events"] if e["id"][0] in "ABC"}


def _name(customer_id: str) -> str:
    return f"Customer {int(customer_id.split('-')[1])}"


def _members_of(res, dimension):
    """All members mentioned for `dimension` across verified causes."""
    out = set()
    for c in res.causes:
        out |= set(c.filters.get(dimension, []))
    return out


@needs_synthetic
def test_A_margin_collapse_is_verified_as_west_phones(synthetic, truth):
    res = verify_window(synthetic, "profit_margin", tuple(truth["A"]["window"]))
    assert res.conclusion == CAUSE
    top = res.causes[0]
    assert top.filters.get("Region") == [truth["A"]["where"]["Region"]]
    assert top.filters.get("Sub-Category") == [truth["A"]["where"]["Sub-Category"]]
    assert top.share_of_window_change > 0.8
    assert all(c.status == "pass" for c in top.leaf.checks)


@needs_synthetic
def test_B_the_five_churned_customers_are_the_cause(synthetic, truth):
    res = verify_window(synthetic, "revenue", ("2024-06-01", "2024-08-31"))
    assert res.conclusion == CAUSE
    churned = {_name(c) for c in truth["B"]["where"]["Customer ID"]}
    found = _members_of(res, "Customer Name")
    assert found == churned, f"found {sorted(found)}, planted {sorted(churned)}"


@needs_synthetic
def test_C_stockout_is_traced_to_phones(synthetic, truth):
    res = verify_window(synthetic, "revenue", tuple(truth["C"]["window"]))
    assert res.conclusion == CAUSE
    phones = {m for c in res.causes for m in c.filters.get("Sub-Category", []) + c.filters.get("Product Name", [])}
    assert phones & {"Phones", truth["C"]["where"]["Product Name"]}


@needs_synthetic
def test_U_unexplained_september_dip_is_UNKNOWN(synthetic):
    """The ~18% dip was planted with no cause in the data. We must say UNKNOWN, not invent one."""
    res = verify_window(synthetic, "revenue", ("2024-09-01", "2024-09-30"))
    assert res.conclusion == UNKNOWN and res.causes == []
    assert "UNKNOWN" in res.reason
    assert not any(v.verdict == CAUSE for v in res.candidates)


@needs_synthetic
def test_decoy_D1_east_march_dip_is_not_called_a_cause(synthetic):
    """East Chairs dips every March: a recurring pattern, not a new cause."""
    res = verify_window(synthetic, "revenue", ("2024-03-01", "2024-03-31"), filters={"Region": ["East"]})
    assert res.conclusion == UNKNOWN and res.causes == []


@needs_synthetic
def test_decoy_D2_january_slump_is_not_called_a_cause(synthetic):
    res = verify_window(synthetic, "revenue", ("2024-01-01", "2024-01-31"))
    assert res.conclusion == UNKNOWN and res.causes == []


@needs_synthetic
def test_a_calm_period_has_nothing_to_explain(synthetic):
    res = verify_window(synthetic, "revenue", ("2023-07-01", "2023-08-31"))
    assert res.conclusion == UNKNOWN and res.causes == []


@needs_synthetic
def test_every_claim_carries_its_checks_numbers_and_sql(synthetic, truth):
    res = verify_window(synthetic, "revenue", ("2024-06-01", "2024-08-31"))
    leaf = res.causes[0].leaf
    assert [c.name for c in leaf.checks] == ["Size and distinctiveness", "Differs from the rest",
                                              "Started inside the window", "Not a yearly repeat",
                                              "Unusual for its own history"]
    for c in leaf.checks:
        assert c.detail and c.values                          # a sentence AND the numbers behind it
    assert len(leaf.queries) == 2 and all("sql" in x and "params" in x for x in leaf.queries)
    filtered = verify_window(synthetic, "revenue", ("2024-06-01", "2024-08-31"), filters={"Region": ["Central"]})
    q0 = filtered.causes[0].leaf.queries[0]
    assert "?" in q0["sql"] and "Central" in q0["params"] and "Central" not in q0["sql"]   # values are parameters
    assert leaf.checks[-1].values["n_placebo"] >= 6
    # the whole result survives JSON export
    json.dumps(to_plain(res), allow_nan=False)


@needs_synthetic
def test_failed_candidates_explain_why(synthetic):
    res = verify_window(synthetic, "revenue", ("2024-06-01", "2024-08-31"))
    rejected = [v for v in res.candidates if v.verdict == NOT_A_CAUSE]
    assert rejected and all(v.reason for v in rejected)
    # product categories fall WITH the whole business: lift too low
    cat = [v for v in res.candidates if v.dimension == "Category"][0]
    assert cat.verdict == NOT_A_CAUSE and cat.checks[0].status == "fail"


@needs_synthetic
def test_event_verification_uses_the_detected_events_expected_value(synthetic):
    events, _ = detect_all(synthetic, "M", kpis=["revenue"], primary_only=True)
    ev = [e for e in events if e.direction == "down"][0]
    res = verify_event(synthetic, ev)
    assert res.compare == (ev.compare_start, ev.compare_end)
    assert res.change == pytest.approx(ev.impact, rel=1e-6)    # observed - expected
    assert res.conclusion == CAUSE
    assert _members_of(res, "Customer Name")                  # the churned customers show up


@needs_synthetic
def test_subwindows_each_get_their_own_verdict(synthetic):
    wins = [("2024-06-01", "2024-08-31"), ("2024-09-01", "2024-09-30")]
    a, b = verify_subwindows(synthetic, "revenue", wins)
    assert a.conclusion == CAUSE and b.conclusion == UNKNOWN
    months = verify_by_month(synthetic, "revenue", "2024-09-01", "2024-09-30")
    assert len(months) == 1 and months[0].conclusion == UNKNOWN


@needs_synthetic
def test_strict_config_can_turn_a_cause_into_not_a_cause(synthetic):
    strict = VerifyConfig(min_lift=1000.0)
    res = verify_window(synthetic, "revenue", ("2024-06-01", "2024-08-31"), config=strict)
    assert res.conclusion == UNKNOWN and res.causes == []


@needs_synthetic
def test_bad_windows_give_clear_errors(synthetic):
    with pytest.raises(DecomposeError, match="outside the data"):
        verify_window(synthetic, "revenue", ("2030-01-01", "2030-02-01"))
    with pytest.raises(DecomposeError, match="outside the data"):
        verify_window(synthetic, "revenue", ("2022-01-01", "2022-02-01"))   # data starts 2022: no year before


@needs_synthetic
def test_short_history_gives_unknown_not_a_guess(synthetic):
    """Windows in the first ~1.5 years have too few placebo windows: UNKNOWN, never CAUSE."""
    res = verify_window(synthetic, "revenue", ("2023-02-01", "2023-03-31"))
    assert res.conclusion != CAUSE or all(
        c.leaf.checks[-1].values.get("n_placebo", 0) >= 6 for c in res.causes)


# ============================================================ growth must not hide a loss (found by bench/run_bench.py)
def _generated(seed):
    from bench.make_synthetic import generate
    df, truth = generate(seed)
    return load_dataset(df.to_csv(index=False).encode(), filename=f"seed{seed}.csv"), truth


def _customers(res):
    names = set()
    for c in res.causes:
        for d in [c.filters] + [dict(p.split(" = ", 1) for p in a.split(" + ")) for a in c.also_described_as]:
            v = d.get("Customer Name", [])
            names |= set(v if isinstance(v, list) else [m.strip() for m in v.split(",")])
    return names


def test_a_loss_that_ordinary_growth_cancelled_is_still_investigated():
    ds, truth = _generated(4)           # here the churn is almost exactly offset by growth versus last year
    planted = {f"Customer {int(c.split('-')[1])}"
               for c in next(e for e in truth["events"] if e["id"].startswith("B"))["where"]["Customer ID"]}
    window = ("2024-06-01", "2024-08-31")
    old = verify_window(ds, "revenue", window, config=VerifyConfig(auto_growth=False))
    assert old.conclusion == UNKNOWN and "barely changed" in old.reason       # the old behaviour: hidden
    new = verify_window(ds, "revenue", window)
    assert new.conclusion == CAUSE and len(_customers(new) & planted) >= 3
    assert any("removed from the comparison" in n for n in new.notes)


def test_growth_adjustment_does_not_invent_causes_in_a_quiet_window():
    ds, _ = _generated(4)
    res = verify_window(ds, "revenue", ("2023-07-01", "2023-08-31"))
    assert res.conclusion == UNKNOWN


@needs_synthetic
def test_growth_adjustment_leaves_windows_with_a_clear_change_untouched(synthetic):
    on = verify_window(synthetic, "revenue", ("2024-06-01", "2024-08-31"))
    off = verify_window(synthetic, "revenue", ("2024-06-01", "2024-08-31"), config=VerifyConfig(auto_growth=False))
    assert on.change == off.change and [c.describe() for c in on.causes] == [c.describe() for c in off.causes]
    assert not any("removed from the comparison" in n for n in on.notes)
