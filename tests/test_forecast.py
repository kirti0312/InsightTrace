"""Tests for engine/forecast.py (Step 8).

    pytest tests/test_forecast.py -v

The key promise is CALIBRATION: an 80% band should hold about 80% of real values. We test that on
many simulated series where the truth is known, on a fully independent hold-out, and we test that
the walk-forward check never peeks at the future.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.forecast import (LOW_CONFIDENCE, OK, UNKNOWN, Forecast, ForecastConfig,  # noqa: E402
                             ForecastError, _backtest, _validate, _conformal_q, _deseason_trend, _predict,
                             _seasonal_growth, _seasonal_index, forecast_kpi, forecast_series,
                             history_for)
from engine.ingest import load_dataset  # noqa: E402

SUPERSTORE = ROOT / "data" / "Sample_-_Superstore.csv"
SYNTHETIC = ROOT / "data" / "synthetic_demo.csv"
needs_superstore = pytest.mark.skipif(not SUPERSTORE.exists(), reason="Superstore CSV missing")
needs_synthetic = pytest.mark.skipif(not SYNTHETIC.exists(), reason="synthetic CSV missing")

SEASON = np.array([0.80, 0.82, 0.95, 1.0, 1.0, 1.0, 0.98, 1.0, 1.10, 1.15, 1.35, 1.45])


def make(n=60, seed=0, noise=0.06, growth=0.008, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    v = 100 * (1 + growth * t) * SEASON[t % 12] * np.exp(rng.normal(0, noise, n))
    return pd.Series(v, index=pd.date_range(start, periods=n, freq="MS"))


# ============================================================ the two models
def test_seasonal_index_recovers_a_known_yearly_pattern():
    y = 100 * SEASON[np.arange(48) % 12]
    idx = _seasonal_index(y, 12, additive=False)
    assert idx is not None and idx.mean() == pytest.approx(1.0)
    assert np.allclose(idx, SEASON / SEASON.mean(), atol=0.01)
    assert _seasonal_index(y[:20], 12, additive=False) is None      # under two years: not estimable


def test_models_reproduce_a_noiseless_seasonal_series():
    y = 100 * (1 + 0.01 * np.arange(48)) * SEASON[np.arange(48) % 12]
    truth = 100 * (1 + 0.01 * np.arange(48, 54)) * SEASON[np.arange(48, 54) % 12]
    a = _seasonal_growth(y, 12, 6, False, (0.5, 2.0))
    b = _deseason_trend(y, 12, 6, False)
    f, name = _predict(y, 12, 6, False, (0.5, 2.0))
    assert np.max(np.abs(a / truth - 1)) < 0.08
    assert np.max(np.abs(b / truth - 1)) < 0.05
    assert np.max(np.abs(f / truth - 1)) < 0.06 and "blend" in name


def test_growth_is_clipped_so_one_wild_half_year_cannot_explode_the_forecast():
    y = 100 * SEASON[np.arange(36) % 12]
    y[-6:] *= 50                                              # absurd jump
    f = _seasonal_growth(y, 12, 6, False, (0.5, 2.0))
    assert np.all(f <= 2.0 * 100 * SEASON.max() + 1e-9)


def test_short_history_uses_seasonal_growth_only():
    y = make(20).to_numpy()
    f, name = _predict(y, 12, 3, False, (0.5, 2.0))
    assert "under 2 full years" in name and np.all(np.isfinite(f))


def test_conformal_quantiles_are_ordered_and_use_the_small_sample_rule():
    e = np.arange(-10.0, 10.0)                                # 20 errors
    lo, hi = _conformal_q(e, 0.1, 0.9)
    assert lo < 0 < hi and lo == e[1] and hi == e[18]
    lo2, hi2 = _conformal_q(np.array([-1.0, 0.0, 1.0]), 0.1, 0.9)
    assert (lo2, hi2) == (-1.0, 1.0)                           # tiny sample -> widest, never narrower


# ============================================================ calibration (the promise)
def test_band_has_about_the_stated_coverage_on_an_independent_holdout():
    """Fit on 42 months, check the 6 unseen months, over 60 simulated businesses. Truth is known."""
    inside = total = 0
    for seed in range(60):
        s = make(48, seed)
        f = forecast_series(s.iloc[:42], "M", horizon=6)
        assert f.status != UNKNOWN
        actual = s.iloc[42:].to_numpy()
        inside += int(((actual >= f.points["lo"].to_numpy()) & (actual <= f.points["hi"].to_numpy())).sum())
        total += 6
    cov = inside / total
    assert 0.68 <= cov <= 0.90, f"coverage {cov:.2f} over {total} points (target 0.80)"


def test_walk_forward_coverage_matches_the_target_with_enough_history():
    covs = []
    for seed in range(10):
        f = forecast_series(make(72, seed), "M")
        covs.append(f.validation["coverage"])
        assert f.status == OK and f.validation["n_checks"] >= 30
    assert 0.72 <= float(np.mean(covs)) <= 0.88


def test_a_wider_level_gives_a_wider_band():
    s = make(60, 1)
    f80 = forecast_series(s, "M", config=ForecastConfig(level=0.8))
    f95 = forecast_series(s, "M", config=ForecastConfig(level=0.95))
    w80 = (f80.points["hi"] - f80.points["lo"]).mean()
    w95 = (f95.points["hi"] - f95.points["lo"]).mean()
    assert w95 > w80 and f95.validation["coverage"] > f80.validation["coverage"]


def test_noisier_series_get_wider_bands():
    calm = forecast_series(make(60, 2, noise=0.03), "M")
    wild = forecast_series(make(60, 2, noise=0.15), "M")
    rel = lambda f: ((f.points["hi"] - f.points["lo"]) / f.points["forecast"]).mean()   # noqa: E731
    assert rel(wild) > 1.8 * rel(calm)


def test_walk_forward_validation_never_peeks_at_the_future():
    """Errors used to build the band at origin t0 (those whose target is before t0) must not change
    when everything from t0 onward is rewritten."""
    y = make(60, 3).to_numpy()
    t0 = 40
    y2 = y.copy()
    y2[t0:] = y2[t0:] * 7 + 1000                             # wreck the 'future'
    a, b = _backtest(y, 12, 6, False, (0.5, 2.0)), _backtest(y2, 12, 6, False, (0.5, 2.0))
    ka, kb = a.target < t0, b.target < t0
    assert ka.sum() == kb.sum() > 20
    assert np.allclose(a.err[ka], b.err[kb]) and np.array_equal(a.h[ka], b.h[kb])


def test_the_band_used_at_each_origin_depends_only_on_errors_already_known_then():
    """Rewrite everything from month 44 on. For every check whose origin AND target lie before 44,
    the band width and the inside/outside result must be identical: nothing leaked backwards."""
    y = make(60, 3).to_numpy()
    y2 = y.copy()
    t0 = 44
    y2[t0:] = y2[t0:] * 6 + 500
    cfg = ForecastConfig()
    out = []
    for arr in (y, y2):
        back = _backtest(arr, 12, 6, False, (0.5, 2.0))
        out.append(_validate(arr, 12, 6, False, True, back, cfg, detail=True)["detail"])
    a, b = out
    ka, kb = (a["target"] < t0) & (a["origin"] < t0), (b["target"] < t0) & (b["origin"] < t0)
    assert ka.sum() == kb.sum() > 20
    assert np.allclose(a["width"][ka], b["width"][kb]) and np.array_equal(a["inside"][ka], b["inside"][kb])


def test_shocks_in_the_history_widen_the_band_and_lower_confidence():
    s = make(60, 4)
    shocked = s.copy()
    shocked.iloc[38:46] *= 0.5                                # an 8-month collapse, then it recovers
    f0, f1 = forecast_series(s, "M"), forecast_series(shocked, "M")
    rel = lambda f: ((f.points["hi"] - f.points["lo"]) / f.points["forecast"]).mean()   # noqa: E731
    assert rel(f1) > rel(f0)
    assert f1.validation["coverage"] < f0.validation["coverage"]


# ============================================================ honesty about limits
def test_too_little_history_gives_unknown_not_a_guess():
    f = forecast_series(make(16), "M")
    assert f.status == UNKNOWN and f.points.empty and "at least" in f.reason
    assert "UNKNOWN" in f.summary()
    assert f.to_dict()["points"] == []


def test_mostly_empty_series_is_unknown():
    s = make(48)
    s.iloc[::2] = np.nan
    assert forecast_series(s, "M").status == UNKNOWN


def test_a_few_gaps_are_filled_and_reported():
    s = make(60, 5)
    s.iloc[20] = np.nan
    f = forecast_series(s, "M")
    assert f.status in (OK, LOW_CONFIDENCE) and any("interpolation" in n for n in f.notes)


def test_few_checks_means_low_confidence_not_ok():
    f = forecast_series(make(30, 6), "M", horizon=3)
    assert f.status == LOW_CONFIDENCE and f.validation["n_checks"] < 30
    assert "rough" in f.reason


def test_not_better_than_last_year_is_stated():
    """A pure repeating pattern with tiny noise: 'same as last year' is hard to beat (fixed seed 3)."""
    rng = np.random.default_rng(3)
    s = pd.Series(100 * SEASON[np.arange(60) % 12] * np.exp(rng.normal(0, 0.01, 60)),
                  index=pd.date_range("2020-01-01", periods=60, freq="MS"))
    f = forecast_series(s, "M")
    assert f.validation["skill"] <= 0
    assert any("not better than simply repeating last year" in n for n in f.notes)
    good = forecast_series(make(60, 0), "M")                  # a growing series: the model beats last year
    assert good.validation["skill"] > 0.1 and not any("not better" in n for n in good.notes)


def test_bad_inputs_give_clear_errors():
    with pytest.raises(ForecastError, match="monthly"):
        forecast_series(make(48), "D")
    with pytest.raises(ForecastError, match="horizon"):
        forecast_series(make(48), "M", horizon=13)
    with pytest.raises(ForecastError, match="horizon"):
        forecast_series(make(48), "M", horizon=0)


# ============================================================ shape of the output
def test_points_are_ordered_future_months_and_the_band_brackets_the_forecast():
    s = make(60, 8)
    f = forecast_series(s, "M", horizon=6)
    p = f.points
    assert list(p["h"]) == [1, 2, 3, 4, 5, 6]
    assert p["period"].iloc[0] == s.index[-1] + pd.DateOffset(months=1)
    assert list(p["period"].dt.day) == [1] * 6
    assert (p["lo"] <= p["forecast"]).all() and (p["forecast"] <= p["hi"]).all() and (p["lo"] >= 0).all()


def test_the_forecast_follows_the_yearly_pattern():
    s = make(60, 9, noise=0.03)                               # ends Dec 2024; forecast Jan..Jun 2025
    f = forecast_series(s, "M", horizon=12)
    by = f.points.set_index(f.points["period"].dt.month)["forecast"]
    assert by[12] > 1.3 * by[1]                              # December high, January low
    assert by[11] > by[2] and by[1] < by[4]


def test_negative_and_ratio_series_use_addition_and_still_work():
    rng = np.random.default_rng(10)
    t = np.arange(60)
    v = 5 * SEASON[t % 12] + rng.normal(0, 1.0, 60) - 3        # crosses zero (like profit)
    s = pd.Series(v, index=pd.date_range("2020-01-01", periods=60, freq="MS"))
    f = forecast_series(s, "M")
    assert f.status != UNKNOWN and any("addition" in n for n in f.notes)
    assert (f.points["lo"] <= f.points["forecast"]).all() and (f.points["forecast"] <= f.points["hi"]).all()
    ratio = pd.Series(0.2 + 0.03 * SEASON[t % 12] + rng.normal(0, 0.005, 60), index=s.index)
    fr = forecast_series(ratio, "M", is_ratio=True, fmt="pct", label="Profit margin")
    assert fr.status != UNKNOWN and fr.points["forecast"].between(0.1, 0.35).all()


def test_quarterly_and_weekly_are_supported():
    q = pd.Series(100 * (1 + 0.01 * np.arange(20)) * np.tile([0.9, 1.0, 1.0, 1.2], 5) *
                  np.exp(np.random.default_rng(11).normal(0, 0.03, 20)),
                  index=pd.date_range("2020-01-01", periods=20, freq="QS"))
    fq = forecast_series(q, "Q")
    assert fq.status != UNKNOWN and len(fq.points) == 4
    t = np.arange(150)
    w = pd.Series(50 * (1 + 0.3 * np.sin(2 * np.pi * t / 52)) * np.exp(np.random.default_rng(12).normal(0, 0.04, 150)),
                  index=pd.date_range("2022-01-03", periods=150, freq="W-MON"))
    fw = forecast_series(w, "W")
    assert fw.status != UNKNOWN and len(fw.points) == 13


def test_export_summary_and_facts_are_consistent_and_json_safe():
    f = forecast_series(make(60, 13), "M", label="Revenue", kpi="revenue")
    json.dumps(f.to_dict(), allow_nan=False)
    facts = f.facts()
    first = f.points.iloc[0]
    k = f"{first['period']:%Y-%m-%d}"
    assert f"forecast.{k}" in facts and facts["level"] == "80%" and "coverage" in facts
    text = f.summary()
    assert "status" in text and facts[f"forecast.{k}"] in text and facts["coverage"] in text


# ============================================================ real files
@pytest.fixture(scope="module")
def synthetic():
    return load_dataset(SYNTHETIC)


@needs_synthetic
def test_history_drops_nothing_when_the_data_is_complete_months(synthetic):
    s, kpi, notes = history_for(synthetic, "revenue", "M")
    assert len(s) == 36 and s.index[0] == pd.Timestamp("2022-01-01") and notes == []
    assert kpi.key == "revenue"


@needs_synthetic
def test_synthetic_forecast_is_honestly_low_confidence_because_of_the_planted_shocks(synthetic):
    f = forecast_kpi(synthetic, "revenue", "M")
    assert f.status == LOW_CONFIDENCE and len(f.points) == 6 and f.validation["n_checks"] >= 30
    assert f.validation["coverage"] < 0.68
    json.dumps(f.to_dict(), allow_nan=False)


@needs_synthetic
def test_a_forecast_made_before_the_shocks_exposes_them_as_outside_the_band(synthetic):
    """Fit on data up to May 2024, then look at Jun-Nov 2024 (churn, the September dip, the stock-out).
    The forecast cannot know about them; the real values must fall BELOW the band."""
    s, _, _ = history_for(synthetic, "revenue", "M")
    f = forecast_series(s.loc[:"2024-05-01"], "M", horizon=6)
    actual = s.loc["2024-06-01":"2024-11-01"].to_numpy()
    assert len(actual) == 6 and f.status != UNKNOWN
    assert int((actual < f.points["lo"].to_numpy()).sum()) >= 5


@needs_synthetic
def test_slice_forecast_and_unknown_kpi(synthetic):
    f = forecast_kpi(synthetic, "revenue", "M", filters={"Region": "East"}, horizon=3)
    assert f.filters == {"Region": "East"} and len(f.points) == 3
    with pytest.raises(ForecastError):
        forecast_kpi(synthetic, "nonsense", "M")
    with pytest.raises(ForecastError, match="monthly"):
        forecast_kpi(synthetic, "revenue", "D")


@needs_superstore
def test_superstore_drops_partial_first_and_last_months_and_forecasts():
    ds = load_dataset(SUPERSTORE)
    s, _, notes = history_for(ds, "revenue", "M")
    assert s.index[0] == pd.Timestamp("2014-02-01") and s.index[-1] == pd.Timestamp("2017-11-01")
    assert any("last period" in n for n in notes) and any("first period" in n for n in notes)
    f = forecast_kpi(ds, "revenue", "M")
    assert f.status in (OK, LOW_CONFIDENCE) and f.points["period"].iloc[0] == pd.Timestamp("2017-12-01")
    assert f.validation["n_checks"] >= 100 and 0.55 <= f.validation["coverage"] <= 0.95
