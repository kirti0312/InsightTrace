"""Tests for engine/detect.py.

Run from the project folder:
    pytest tests/test_detect.py -v

Two kinds of tests:
  1. Controlled series built in the test (fast): we know exactly what we put in,
     so we know what must (and must not) be flagged.
  2. The shipped datasets: the synthetic file with its planted events, and the
     real Superstore file (sanity checks only, since it has no answer key).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.detect import (SENSITIVITY, DetectConfig, _period_end, config_for,  # noqa: E402
                           detect_all, detect_kpi, detect_series, find_events)
from engine.ingest import load_dataset  # noqa: E402
from engine.kpi import KPI  # noqa: E402

SUPERSTORE = ROOT / "data" / "Sample_-_Superstore.csv"
SYNTHETIC = ROOT / "data" / "synthetic_demo.csv"
needs_superstore = pytest.mark.skipif(not SUPERSTORE.exists(), reason="Superstore CSV missing")
needs_synthetic = pytest.mark.skipif(not SYNTHETIC.exists(), reason="synthetic CSV missing")

SEASON = np.array([0.80, 0.82, 0.95, 1.0, 1.0, 1.0, 0.98, 1.0, 1.10, 1.15, 1.35, 1.45])


def make_series(n=36, seed=0, noise=0.06, growth=0.008):
    """A realistic monthly series: growth x strong seasonality x random noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    value = 100 * (1 + growth * t) * SEASON[t % 12] * np.exp(rng.normal(0, noise, n))
    return pd.DataFrame({"period": pd.date_range("2022-01-01", periods=n, freq="MS"),
                         "value": value})


def flagged_idx(frame):
    return [int(i) for i in np.flatnonzero(frame["flagged"])]


# ------------------------------------------------------------- clean series
def test_clean_seasonal_series_rarely_raises_false_alarms():
    flagged = assessed = 0
    for seed in range(20):
        frame, _ = detect_series(make_series(seed=seed), "money", "M")
        flagged += int(frame["flagged"].sum())
        assessed += int(frame["assessed"].sum())
    assert assessed > 300
    assert flagged / assessed < 0.03          # measured: about 1%


def test_strong_seasonality_is_not_mistaken_for_change():
    """January is ~45% below December every year. That must NOT be flagged."""
    for seed in range(5):
        frame, _ = detect_series(make_series(n=48, seed=seed, noise=0.02), "money", "M")
        assert frame["flagged"].sum() == 0
        jan = frame[frame["period"].dt.month == 1]
        assert (jan["flagged"] == False).all()  # noqa: E712


def test_first_year_is_not_judged():
    frame, _ = detect_series(make_series(), "money", "M")
    assert not frame["assessed"].iloc[:12].any()      # no baseline yet
    assert frame["expected"].iloc[:12].isna().all()


# ---------------------------------------------------------- injected changes
def test_injected_drop_is_found_and_only_there():
    found = 0
    for seed in range(10):
        data = make_series(seed=seed, noise=0.05)
        data.loc[30:32, "value"] *= 0.70                 # -30% for three months
        frame, _ = detect_series(data, "money", "M")
        idx = flagged_idx(frame)
        found += len(set(idx) & {30, 31, 32}) >= 2
        assert all(frame.loc[i, "direction"] == "down" for i in set(idx) & {30, 31, 32})
        assert not [i for i in idx if i < 28]            # nothing flagged before it
    assert found >= 9


def test_injected_spike_is_flagged_as_up():
    data = make_series(seed=3, noise=0.04)
    data.loc[28, "value"] *= 1.6
    frame, _ = detect_series(data, "money", "M")
    assert frame.loc[28, "flagged"] and frame.loc[28, "direction"] == "up"


def test_a_pattern_that_repeats_every_year_is_only_flagged_the_first_time():
    for seed in range(5):
        data = make_series(n=48, seed=seed, noise=0.04)
        for i in (17, 29, 41):                           # same odd month each year
            data.loc[i, "value"] *= 1.6
        frame, _ = detect_series(data, "money", "M")
        assert frame.loc[17, "flagged"]                  # new -> flagged
        assert not frame.loc[29, "flagged"]              # repeats -> seasonal
        assert not frame.loc[41, "flagged"]


def test_permanent_shift_is_flagged_then_accepted_without_a_later_echo():
    """-25% from month 26 for good. Flag it, accept it as the new normal, and do
    NOT raise a false 'echo' alert a year later when the base year has it too."""
    for seed in range(6):
        data = make_series(n=72, seed=seed, noise=0.04)
        data.loc[26:, "value"] *= 0.75
        frame, _ = detect_series(data, "money", "M")
        idx = flagged_idx(frame)
        assert {26, 27, 28} <= set(idx), f"seed {seed}: onset missed {idx}"
        assert [i for i in idx if i >= 36] == [], f"seed {seed}: echo alert {idx}"


def test_data_gaps_do_not_crash_and_are_not_flagged():
    data = make_series(n=36, seed=2)
    data.loc[20, "value"] = np.nan
    frame, _ = detect_series(data, "money", "M")
    assert not frame.loc[20, "flagged"]
    assert len(frame) == 36


# ----------------------------------------------------------- other KPI kinds
def test_short_series_falls_back_and_says_so():
    frame, notes = detect_series(make_series(n=10), "money", "M")
    assert (frame["method"] == "trailing").all()
    assert any("seasonality" in n for n in notes)


def test_too_little_data_gives_a_clear_error():
    with pytest.raises(ValueError, match="at least 6"):
        detect_series(make_series(n=5), "money", "M")


def test_series_with_zeros_and_negatives_uses_additive_mode():
    """Profit can go negative; logs would break. It must still work."""
    rng = np.random.default_rng(1)
    t = np.arange(36)
    value = 5 * SEASON[t % 12] + rng.normal(0, 1.5, 36) - 3     # crosses zero
    data = pd.DataFrame({"period": pd.date_range("2022-01-01", periods=36, freq="MS"),
                         "value": value})
    data.loc[30:32, "value"] -= 25                               # a clear loss
    frame, _ = detect_series(data, "money", "M")
    assert frame["expected"].notna().sum() > 0
    assert len(set(flagged_idx(frame)) & {30, 31, 32}) >= 2


def test_percentage_kpi_reports_point_changes():
    rng = np.random.default_rng(4)
    value = 0.20 + rng.normal(0, 0.005, 36)
    value[30:32] = 0.14                                           # margin drops 6 points
    data = pd.DataFrame({"period": pd.date_range("2022-01-01", periods=36, freq="MS"),
                         "value": value})
    frame, _ = detect_series(data, "pct", "M")
    assert set(flagged_idx(frame)) >= {30, 31}
    assert frame.loc[30, "pct_dev"] == pytest.approx(-0.06, abs=0.02)   # in points


def test_per_day_scale_removes_month_length_effects():
    """Value = 10 per day. Month totals differ only because months differ in length."""
    period = pd.date_range("2022-01-01", periods=36, freq="MS")
    days = np.array([p.days_in_month for p in period])
    data = pd.DataFrame({"period": period, "value": 10.0 * days})
    frame, _ = detect_series(data, "money", "M", scale=days)
    assert frame["flagged"].sum() == 0
    assessed = frame[frame["assessed"]]
    assert assessed["pct_dev"].abs().max() < 0.02
    assert np.allclose(assessed["expected"], assessed["value"], rtol=0.02)


def test_weekly_detection_works():
    n = 130
    rng = np.random.default_rng(5)
    t = np.arange(n)
    value = 50 * (1 + 0.002 * t) * (1 + 0.3 * np.sin(2 * np.pi * t / 52)) * np.exp(rng.normal(0, 0.04, n))
    value[100:104] *= 0.6
    data = pd.DataFrame({"period": pd.date_range("2022-01-03", periods=n, freq="W-MON"),
                         "value": value})
    frame, _ = detect_series(data, "money", "W")
    assert len(set(flagged_idx(frame)) & {100, 101, 102, 103}) >= 3
    assert len([i for i in flagged_idx(frame) if i < 98]) <= 2


# ------------------------------------------------------------------ episodes
def _frame(flags, directions=None, n=30):
    directions = directions or {}
    dev = np.where(np.isin(np.arange(n), flags), -10.0, 0.0)
    return pd.DataFrame({
        "period": pd.date_range("2023-01-01", periods=n, freq="MS"),
        "value": 100.0 + dev, "deviation": dev, "z": np.where(dev != 0, -4.0, 0.2),
        "flagged": np.isin(np.arange(n), flags), "assessed": True,
        "direction": ["down" if i in flags else "" for i in range(n)],
    }).assign(**({"direction": [directions.get(i, "down" if i in flags else "")
                                for i in range(n)]} if directions else {}))


REVENUE = KPI("revenue", "Revenue", "SUM(revenue)", None, "money", ("revenue",), True)


def test_close_flags_merge_into_one_episode_far_ones_do_not():
    one = find_events(_frame([20, 22]), REVENUE, "M")          # gap of one period
    two = find_events(_frame([10, 20]), REVENUE, "M")
    assert len(one) == 1 and one[0].n_periods == 3
    assert len(two) == 2


def test_opposite_directions_are_separate_episodes():
    frame = _frame([20, 21], directions={20: "down", 21: "up"})
    frame.loc[21, "deviation"] = 10.0
    assert len(find_events(frame, REVENUE, "M")) == 2


def test_episode_fields_are_consistent():
    (event,) = find_events(_frame([18, 19, 20]), REVENUE, "M")
    assert event.direction == "down" and event.n_periods == 3
    assert event.start == pd.Timestamp("2024-07-01")
    assert event.end == pd.Timestamp("2024-09-30")
    assert event.compare_start == pd.Timestamp("2023-07-01")      # same window, a year earlier
    assert event.compare_end == pd.Timestamp("2023-09-30")
    assert event.impact == pytest.approx(-30.0)
    assert event.observed == pytest.approx(event.expected + event.impact)
    assert event.pct_vs_expected == pytest.approx(event.impact / event.expected)
    assert event.severity == "high" and not event.ongoing is None
    text = event.headline()
    assert "fell" in text and "Jul 2024 to Sep 2024" in text


def test_period_end_is_the_last_day_and_respects_the_data_cap():
    assert _period_end(pd.Timestamp("2024-02-01"), "M", None) == pd.Timestamp("2024-02-29")
    assert _period_end(pd.Timestamp("2024-02-01"), "M", pd.Timestamp("2024-02-15")) == pd.Timestamp("2024-02-15")
    assert _period_end(pd.Timestamp("2024-01-01"), "Q", None) == pd.Timestamp("2024-03-31")


def test_sensitivity_presets_are_ordered_and_validated():
    assert config_for("high").z_threshold < config_for("normal").z_threshold < config_for("low").z_threshold
    assert set(SENSITIVITY) == {"high", "normal", "low"}
    assert isinstance(config_for(), DetectConfig)
    with pytest.raises(ValueError):
        config_for("extreme")


# ------------------------------------------------------ the synthetic dataset
@pytest.fixture(scope="module")
def synthetic():
    return load_dataset(SYNTHETIC)


@needs_synthetic
def test_synthetic_planted_changes_are_found(synthetic):
    events, _ = detect_all(synthetic, "M")
    down = [e for e in events if e.direction == "down"]

    def hit(kpis, start, end, onset_from=None):
        return [e for e in down if e.kpi in kpis
                and e.start <= pd.Timestamp(end) and e.end >= pd.Timestamp(start)
                and (onset_from is None or e.start >= pd.Timestamp(onset_from))]

    # A: the Mar-Apr 2024 discount spike shows up as a margin collapse
    assert hit({"profit_margin"}, "2024-03-01", "2024-04-30")
    # B: customer churn from June 2024 -> volume falls, starting Jun-Aug
    assert hit({"orders", "quantity", "revenue"}, "2024-06-01", "2024-08-31", onset_from="2024-06-01")
    # U and C: the September dip and the Oct-Nov stockout sit inside a falling revenue/orders episode
    assert hit({"orders", "quantity", "revenue"}, "2024-09-01", "2024-09-30")
    assert hit({"orders", "quantity", "revenue"}, "2024-10-01", "2024-11-30")


@needs_synthetic
def test_synthetic_clean_period_and_january_raise_no_alerts(synthetic):
    events, _ = detect_all(synthetic, "M")
    assert not [e for e in events if e.start < pd.Timestamp("2024-03-01")]   # 2023 + early 2024 are clean
    assert not [e for e in events if e.start.month == 1]                    # January slump is seasonal


@needs_synthetic
def test_drilling_into_central_shows_the_churn_clearly(synthetic):
    """Filtering to the region where the churn happened makes it much easier to see."""
    result = detect_kpi(synthetic, "revenue", "M", filters={"Region": "Central"})
    churn = [e for e in result.events if e.direction == "down"
             and e.start <= pd.Timestamp("2024-08-31") and e.end >= pd.Timestamp("2024-06-01")]
    assert churn and churn[0].pct_vs_expected < -0.25
    assert churn[0].start >= pd.Timestamp("2024-06-01")


@needs_synthetic
def test_synthetic_frame_invariants(synthetic):
    frame = detect_kpi(synthetic, "revenue", "M").frame
    assert len(frame) == 36
    assert (frame.loc[frame["flagged"], "assessed"]).all()
    judged = frame[frame["assessed"]]
    assert (judged["lower"] <= judged["upper"]).all()
    assert ((judged["lower"] <= judged["expected"]) & (judged["expected"] <= judged["upper"])).all()
    assert frame["value"].sum() == pytest.approx(
        synthetic.query("SELECT SUM(revenue) AS t FROM sales")["t"][0])


@needs_synthetic
def test_detect_all_ordering_and_primary_only(synthetic):
    events, _ = detect_all(synthetic, "M")
    flags = [e.primary for e in events]
    assert flags == sorted(flags, reverse=True)          # headline KPIs first
    only, _ = detect_all(synthetic, "M", primary_only=True)
    assert only and all(e.primary for e in only)
    sel, _ = detect_all(synthetic, "M", kpis=["orders"])
    assert {e.kpi for e in sel} <= {"orders"}


@needs_synthetic
def test_bad_frequency_is_rejected(synthetic):
    with pytest.raises(ValueError, match="freq"):
        detect_kpi(synthetic, "revenue", "D")


# ----------------------------------------------------------------- Superstore
@needs_superstore
def test_superstore_runs_cleanly_and_drops_the_unfinished_month():
    ds = load_dataset(SUPERSTORE)
    result = detect_kpi(ds, "revenue", "M")
    assert result.frame["period"].iloc[-1] == pd.Timestamp("2017-11-01")   # data ends 30 Dec
    events, _ = detect_all(ds, "M")
    assert 3 <= len(events) <= 40                       # sanity range, not an accuracy claim
    for e in events:
        assert e.start <= e.end and e.n_periods >= 1
        assert e.severity in {"low", "medium", "high"}
        assert np.isfinite(e.observed) and np.isfinite(e.expected)


@needs_superstore
def test_superstore_higher_sensitivity_never_gives_fewer_alerts():
    ds = load_dataset(SUPERSTORE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counts = {lvl: len(detect_all(ds, "M", config=config_for(lvl))[0])
                  for lvl in ("high", "normal", "low")}
    assert counts["high"] >= counts["normal"] >= counts["low"]
