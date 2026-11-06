"""Tests for engine/decompose.py.

Run from the project folder:
    pytest tests/test_decompose.py -v

Two kinds of tests:
  1. Pure maths on small tables built in the test (no database): we can prove
     the contributions add up EXACTLY.
  2. The shipped synthetic dataset with its planted events (answer key in
     data/synthetic_ground_truth.json): did the right region/product/customer
     come out on top?
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.decompose import (BLANK_LABEL, OTHER_LABEL, DecomposeError, contributions,  # noqa: E402
                              decompose, decompose_event, decompose_subwindows,
                              monthly_windows, one_year_earlier, rank_dimensions)
from engine.detect import detect_all  # noqa: E402
from engine.ingest import load_dataset  # noqa: E402
from engine.kpi import kpi_value  # noqa: E402

SUPERSTORE = ROOT / "data" / "Sample_-_Superstore.csv"
SYNTHETIC = ROOT / "data" / "synthetic_demo.csv"
TRUTH = ROOT / "data" / "synthetic_ground_truth.json"
needs_superstore = pytest.mark.skipif(not SUPERSTORE.exists(), reason="Superstore CSV missing")
needs_synthetic = pytest.mark.skipif(not (SYNTHETIC.exists() and TRUTH.exists()),
                                     reason="synthetic files missing")


def table(**members):
    """table(East=(num, den), West=(num, den)) -> DataFrame member/num/den."""
    return pd.DataFrame({"member": list(members),
                         "num": [v[0] for v in members.values()],
                         "den": [v[1] for v in members.values()]})


def sums(**members):
    return pd.DataFrame({"member": list(members), "num": list(members.values()), "den": np.nan})


# ============================================================ pure maths
def test_sum_contributions_add_up_exactly():
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = int(rng.integers(2, 12))
        names = [f"m{i}" for i in range(k)]
        cur = sums(**dict(zip(names, rng.uniform(0, 1000, k))))
        base = sums(**dict(zip(names, rng.uniform(0, 1000, k))))
        rows, total, c, b = contributions(cur, base, (cur["num"].sum(), np.nan),
                                          (base["num"].sum(), np.nan), ratio=False)
        assert rows["contribution"].sum() == pytest.approx(total, abs=1e-9)
        assert total == pytest.approx(cur["num"].sum() - base["num"].sum())
        assert OTHER_LABEL not in set(rows["member"])          # nothing left over
        assert rows["share"].sum() == pytest.approx(1.0)


def test_ratio_contributions_add_up_exactly():
    """Margin = profit / revenue. Contributions must sum to the change in margin."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        k = int(rng.integers(2, 10))
        names = [f"m{i}" for i in range(k)]
        d1, d0 = rng.uniform(50, 500, k), rng.uniform(50, 500, k)
        n1, n0 = d1 * rng.uniform(-0.2, 0.4, k), d0 * rng.uniform(-0.2, 0.4, k)
        cur = table(**dict(zip(names, zip(n1, d1))))
        base = table(**dict(zip(names, zip(n0, d0))))
        rows, total, r1, r0 = contributions(cur, base, (n1.sum(), d1.sum()), (n0.sum(), d0.sum()), ratio=True)
        assert total == pytest.approx(n1.sum() / d1.sum() - n0.sum() / d0.sum())
        assert rows["contribution"].sum() == pytest.approx(total, abs=1e-12)
        assert OTHER_LABEL not in set(rows["member"])


def test_the_member_whose_margin_collapsed_is_the_biggest_negative():
    cur = table(East=(20.0, 100.0), West=(-30.0, 100.0), South=(21.0, 100.0))   # West: 20% -> -30%
    base = table(East=(20.0, 100.0), West=(20.0, 100.0), South=(20.0, 100.0))
    rows, total, _, _ = contributions(cur, base, (11.0, 300.0), (60.0, 300.0), ratio=True)
    assert rows["member"].iloc[0] == "West" and rows["direction"].iloc[0] == "down"
    assert rows["contribution"].iloc[0] == pytest.approx(total * rows["share"].iloc[0])
    assert rows.loc[rows["member"] == "West", "contribution"].iloc[0] < -0.15


def test_new_and_vanished_members_are_handled():
    cur = sums(A=100.0, NEW=40.0)
    base = sums(A=100.0, GONE=60.0)
    rows, total, _, _ = contributions(cur, base, (140.0, np.nan), (160.0, np.nan), ratio=False)
    by = rows.set_index("member")["contribution"]
    assert by["NEW"] == pytest.approx(40.0) and by["GONE"] == pytest.approx(-60.0) and by["A"] == 0
    assert total == pytest.approx(-20.0)


def test_leftover_becomes_a_visible_overlap_row_and_everything_still_adds_up():
    """Distinct orders: 3 members each have 10 orders, but 4 orders touch two members,
    so the true total is 26, not 30."""
    cur = sums(A=10.0, B=10.0, C=10.0)
    base = sums(A=10.0, B=10.0, C=10.0)
    rows, total, _, _ = contributions(cur, base, (26.0, np.nan), (30.0, np.nan), ratio=False)
    assert total == pytest.approx(-4.0)
    assert rows["member"].iloc[-1] == OTHER_LABEL
    assert rows["contribution"].iloc[-1] == pytest.approx(-4.0)
    assert rows["contribution"].sum() == pytest.approx(total)


def test_scaled_baseline_removes_uniform_growth():
    """Everything grew 10% except B, which stayed flat. Against the growth-adjusted
    baseline only B stands out, and the contributions add up to observed - expected."""
    base = sums(A=100.0, B=100.0, C=100.0)
    cur = sums(A=110.0, B=100.0, C=110.0)
    expected_total = 330.0
    rows, total, cur_v, base_v = contributions(cur, base, (320.0, np.nan), (300.0, np.nan),
                                               ratio=False, base_scale=expected_total / 300.0)
    assert base_v == pytest.approx(330.0)
    assert total == pytest.approx(320.0 - 330.0)
    by = rows.set_index("member")["contribution"]
    assert by["B"] == pytest.approx(-10.0) and by["A"] == pytest.approx(0.0) and by["C"] == pytest.approx(0.0)


def test_scaled_baseline_is_refused_for_ratios():
    t = table(A=(1.0, 10.0))
    with pytest.raises(DecomposeError):
        contributions(t, t, (1.0, 10.0), (1.0, 10.0), ratio=True, base_scale=1.2)


def test_undefined_ratio_gives_a_clear_error():
    t = table(A=(1.0, 10.0))
    with pytest.raises(DecomposeError, match="undefined"):
        contributions(t, t, (1.0, 0.0), (1.0, 10.0), ratio=True)
    with pytest.raises(DecomposeError, match="undefined"):
        contributions(t, t, (1.0, 10.0), (np.nan, np.nan), ratio=True)


def test_rows_are_sorted_by_size_with_overlap_row_last_and_blank_label_kept():
    cur = sums(A=5.0, B=-50.0, C=20.0)
    base = sums(A=0.0, B=0.0, C=0.0)
    rows, _, _, _ = contributions(cur, base, (-20.0, np.nan), (0.0, np.nan), ratio=False)
    assert list(rows["member"][:3]) == ["B", "C", "A"]
    assert rows["member"].iloc[-1] == OTHER_LABEL                      # sum is -25, total -20 -> +5 left over
    blank = contributions(sums(**{BLANK_LABEL: 1.0}), sums(**{BLANK_LABEL: 2.0}),
                          (1.0, np.nan), (2.0, np.nan), ratio=False)[0]
    assert blank["member"].iloc[0] == BLANK_LABEL


def test_zero_total_change_gives_nan_share_not_a_crash():
    rows, total, _, _ = contributions(sums(A=5.0, B=5.0), sums(A=10.0, B=0.0),
                                      (10.0, np.nan), (10.0, np.nan), ratio=False)
    assert total == 0 and rows["share"].isna().all()


def test_helper_windows():
    assert one_year_earlier("2024-03-01", "2024-04-30") == (pd.Timestamp("2023-03-01"), pd.Timestamp("2023-04-30"))
    assert one_year_earlier("2024-02-29", "2024-02-29")[0] == pd.Timestamp("2023-02-28")
    wins = monthly_windows("2024-06-10", "2024-08-15")
    assert wins == [(pd.Timestamp("2024-06-10"), pd.Timestamp("2024-06-30")),
                    (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-31")),
                    (pd.Timestamp("2024-08-01"), pd.Timestamp("2024-08-15"))]


# ============================================================ the database layer
@pytest.fixture(scope="module")
def synthetic():
    return load_dataset(SYNTHETIC)


@pytest.fixture(scope="module")
def truth():
    return {e["id"][0]: e for e in json.loads(TRUTH.read_text())["events"] if e["id"][0] in "ABC"}


def _name(customer_id: str) -> str:
    """CU-0242 -> 'Customer 242' (the customer-name column is the one kept in the table)."""
    return f"Customer {int(customer_id.split('-')[1])}"


@needs_synthetic
def test_revenue_by_region_adds_up_to_the_real_change(synthetic):
    win = ("2024-06-01", "2024-08-31")
    d = decompose(synthetic, "revenue", "Region", win)
    cur = kpi_value(synthetic, "revenue", *win)
    base = kpi_value(synthetic, "revenue", *one_year_earlier(*win))
    assert d.change == pytest.approx(cur - base)
    assert d.rows["contribution"].sum() == pytest.approx(cur - base, abs=1e-6)
    assert d.residual == 0 and d.baseline_kind == "last_year"
    assert d.compare == one_year_earlier(*win)


@needs_synthetic
def test_margin_by_region_adds_up_to_the_change_in_margin(synthetic):
    win = ("2024-03-01", "2024-04-30")
    d = decompose(synthetic, "profit_margin", "Sub-Category", win)
    expected = (kpi_value(synthetic, "profit_margin", *win)
                - kpi_value(synthetic, "profit_margin", *one_year_earlier(*win)))
    assert d.is_ratio and d.change == pytest.approx(expected)
    assert d.rows["contribution"].sum() == pytest.approx(expected, abs=1e-9)
    assert d.residual == 0 and OTHER_LABEL not in set(d.rows["member"])   # margin splits exactly, no leftover


@needs_synthetic
def test_distinct_orders_still_add_up_with_the_overlap_row(synthetic):
    d = decompose(synthetic, "orders", "Sub-Category", ("2024-06-01", "2024-08-31"))
    assert d.rows["contribution"].sum() == pytest.approx(d.change, abs=1e-6)


@needs_synthetic
def test_evidence_queries_are_recorded_with_parameters(synthetic):
    d = decompose(synthetic, "revenue", "Region", ("2024-06-01", "2024-08-31"), filters={"Segment": "Consumer"})
    assert len(d.queries) == 4
    assert all("sql" in x and "params" in x for x in d.queries)
    assert all("Segment" in x["sql"] and "Consumer" in x["params"] for x in d.queries)
    assert "?" in d.queries[0]["sql"] and "Consumer" not in d.queries[0]["sql"]   # values are parameters


@needs_synthetic
def test_A_margin_collapse_points_to_west_then_phones(synthetic, truth):
    """Planted event A: West Phones discount spike, Mar-Apr 2024."""
    win = tuple(truth["A"]["window"])
    by_region = decompose(synthetic, "profit_margin", "Region", win)
    assert by_region.top(1, "down")["member"].iloc[0] == truth["A"]["where"]["Region"]
    drill = decompose(synthetic, "profit_margin", "Sub-Category", win, filters={"Region": "West"})
    assert drill.top(1, "down")["member"].iloc[0] == truth["A"]["where"]["Sub-Category"]
    # and the same drill-down on profit (money) agrees
    profit = decompose(synthetic, "profit", "Sub-Category", win, filters={"Region": "West"})
    assert profit.top(1, "down")["member"].iloc[0] == "Phones"


@needs_synthetic
def test_B_churned_customers_are_among_the_biggest_losers(synthetic, truth):
    """Planted event B: the 5 largest Central customers stop ordering from June 2024."""
    churned = {_name(c) for c in truth["B"]["where"]["Customer ID"]}
    d = decompose(synthetic, "revenue", "Customer Name", ("2024-06-01", "2024-08-31"),
                  filters={"Region": "Central"})
    top5 = set(d.top(5, "down")["member"])
    assert len(top5 & churned) >= 3, f"top losers {sorted(top5)}, churned {sorted(churned)}"
    assert d.top_k_share(5) > 0.3


@needs_synthetic
def test_C_stockout_shows_up_as_phones_max_in_east_and_south(synthetic, truth):
    """Planted event C: Phones Max ~92% unavailable in East+South, 15 Oct - 30 Nov 2024."""
    win = tuple(truth["C"]["window"])
    d = decompose(synthetic, "revenue", "Product Name", win,
                  filters={"Region": truth["C"]["where"]["Region"]})
    assert d.top(1, "down")["member"].iloc[0] == truth["C"]["where"]["Product Name"]


@needs_synthetic
def test_overlapping_windows_are_split_by_subwindow(synthetic):
    """Jun-Aug (B only), Sep (U), 15 Oct-30 Nov (B + C): each window gets its own decomposition."""
    wins = [("2024-06-01", "2024-08-31"), ("2024-09-01", "2024-09-30"), ("2024-10-15", "2024-11-30")]
    parts = decompose_subwindows(synthetic, "revenue", "Region", wins)
    assert [p.window[0] for p in parts] == [pd.Timestamp(w[0]) for w in wins]
    for p, w in zip(parts, wins):
        assert p.change == pytest.approx(kpi_value(synthetic, "revenue", *w)
                                         - kpi_value(synthetic, "revenue", *one_year_earlier(*w)))
    # Central (customer churn) is the biggest regional loser in the Jun-Aug window
    assert parts[0].top(1, "down")["member"].iloc[0] == "Central"


@needs_synthetic
def test_event_decomposition_adds_up_to_the_events_impact(synthetic):
    events, _ = detect_all(synthetic, "M", kpis=["revenue"])
    ev = [e for e in events if e.direction == "down"
          and e.start <= pd.Timestamp("2024-08-31") and e.end >= pd.Timestamp("2024-06-01")][0]
    d = decompose_event(synthetic, ev, "Region")
    assert d.baseline_kind == "expected"
    assert d.change == pytest.approx(ev.impact, rel=1e-6)
    assert d.rows["contribution"].sum() == pytest.approx(ev.impact, rel=1e-6)
    ratio_ev = [e for e in detect_all(synthetic, "M", kpis=["profit_margin"])[0] if e.direction == "down"][0]
    assert decompose_event(synthetic, ratio_ev, "Region").baseline_kind == "last_year"


@needs_synthetic
def test_rank_dimensions_orders_by_concentration(synthetic):
    r = rank_dimensions(synthetic, "revenue", ("2024-06-01", "2024-08-31"), filters={"Region": "Central"})
    assert list(r.columns) == ["dimension", "n_members", "top_k_share", "concentration",
                               "top_member", "top_contribution"]
    assert "Region" not in set(r["dimension"])                      # already fixed by the filter
    conc = r["concentration"].dropna().tolist()
    assert conc == sorted(conc, reverse=True)
    assert r["dimension"].iloc[0] == "Customer Name"               # the churn is customer-shaped
    # a 3-member column has a top-3 share of 100% but is NOT 'concentrated'
    cat = r[r["dimension"] == "Category"].iloc[0]
    assert cat["top_k_share"] == pytest.approx(1.0) and cat["concentration"] == pytest.approx(1.0)


@needs_synthetic
def test_bad_inputs_give_clear_errors(synthetic):
    with pytest.raises(DecomposeError, match="not a known category"):
        decompose(synthetic, "revenue", "Nope", ("2024-06-01", "2024-08-31"))
    with pytest.raises(DecomposeError, match="No rows"):
        decompose(synthetic, "revenue", "Region", ("2030-01-01", "2030-02-01"))
    with pytest.raises(DecomposeError, match="before it starts"):
        decompose(synthetic, "revenue", "Region", ("2024-08-31", "2024-06-01"))
    with pytest.raises(Exception):                                     # unknown filter column is refused
        decompose(synthetic, "revenue", "Region", ("2024-06-01", "2024-08-31"), filters={"Bogus": "x"})


@needs_superstore
def test_superstore_decomposition_runs_and_adds_up():
    ds = load_dataset(SUPERSTORE)
    for key, dim in (("revenue", "Region"), ("profit", "Sub-Category"), ("profit_margin", "Category")):
        win = ("2017-06-01", "2017-08-31")
        d = decompose(ds, key, dim, win)
        assert d.rows["contribution"].sum() == pytest.approx(d.change, abs=1e-6)
        assert np.isfinite(d.cur_total) and np.isfinite(d.base_total)
    assert d.summary().startswith("Profit margin by Category")
