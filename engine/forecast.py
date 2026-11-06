"""Forecasting a KPI with an honest uncertainty band, validated on data the model has not seen.

The idea, in plain words
------------------------
We predict the next few periods (months by default) of a KPI such as revenue, and draw a
band around each prediction. A band is only useful if it is the RIGHT width, so we test it:
an "80% band" should contain the real value about 80% of the time.

How the prediction is made (no libraries beyond numpy/pandas)
-------------------------------------------------------------
 Model A "seasonal growth": same period last year, scaled by how much the last half-year
         grew versus the same half-year a year before (growth limited to 0.5x .. 2x).
 Model B "deseasonalised trend": remove the yearly pattern (classical decomposition), fit a
         straight line to the recent de-seasonalised values, flatten (damp) the slope so it
         does not run away, then put the yearly pattern back.
 Final    average of A and B when there are >= 2 full years; A alone with 1.5-2 years.
 No model is chosen after looking at results (that would leak information); the rule is fixed.

How the band is made
--------------------
Walk back through history ("rolling origin"): pretend it is month t, forecast the next 6
months, and compare with what really happened. Those past mistakes (divided by the recent
size of the KPI) are collected per distance ahead. The band is the 10th..90th percentile of
past mistakes, so it is as wide as this KPI's mistakes really were. (A small-sample
correction, as in conformal prediction, is used.)

How it is validated (the part that matters)
-------------------------------------------
Walk-forward: at every past month t we build the band using ONLY mistakes whose outcome was
already known at t, forecast ahead, and then check whether the real values fell inside. The
share inside should be near the band level (80%). We also report the last `horizon` months
as a hold-out, and compare against the plain "same as last year" forecast (skill).
Status: OK if coverage is within 12 points of the level with >= 30 checks; LOW_CONFIDENCE if
not (or too few checks); UNKNOWN if the history is too short to measure its own errors.

Honest limits
-------------
  - It extrapolates patterns in this one file. It does not know about promotions, price
    changes, stock-outs or the economy. Past planted/real shocks in the history (a stock-out,
    churn) make the past errors, and therefore the bands, wider.
  - Needs about 22+ months (monthly). Bands from 2-3 years of history are rough: even with a
    good model the validation has few independent points (the checks overlap).
  - Ratio KPIs (margin, average discount, order value) are forecast directly as a value
    series; the band can be loose when the ratio is near zero.
  - Bands are per period. Do not add them up to get a band for a yearly total.
  - Days are not supported (aggregate to weeks). Weekly needs about 1.5+ years too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.kpi import FREQ, KPI, _is_complete, format_value, get_kpi, kpi_series
from engine.verify import to_plain

OK, LOW_CONFIDENCE, UNKNOWN = "OK", "LOW_CONFIDENCE", "UNKNOWN"
SEASON = {"M": 12, "Q": 4, "W": 52}
DEFAULT_HORIZON = {"M": 6, "Q": 4, "W": 13}


class ForecastError(ValueError):
    """Raised for bad inputs (unsupported frequency, unknown KPI). Message is for the user."""


@dataclass(frozen=True)
class ForecastConfig:
    level: float = 0.8             # band level (0.8 = 80% band)
    min_calibration: int = 10      # past errors needed before a band can be built
    min_bucket: int = 8            # errors needed to use a per-distance bucket
    min_checks: int = 30           # walk-forward checks needed to call coverage reliable
    coverage_tolerance: float = 0.12   # |coverage - level| allowed for status OK
    growth_clip: tuple = (0.5, 2.0)


@dataclass
class Forecast:
    kpi: str
    label: str
    fmt: str
    freq: str
    level: float
    status: str                    # OK / LOW_CONFIDENCE / UNKNOWN
    reason: str
    model: str
    history: pd.DataFrame          # period, value (the evidence the forecast is built on)
    points: pd.DataFrame           # period, h, forecast, lo, hi
    validation: dict               # coverage, n_checks, holdout, skill, ... (see _validate)
    notes: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.points.empty:
            return f"{self.label} forecast: UNKNOWN. {self.reason}"
        v = self.validation
        first, last = self.points.iloc[0], self.points.iloc[-1]
        fmt = self.fmt
        head = (f"{self.label} forecast, next {len(self.points)} periods "
                f"({first['period']:%b %Y} to {last['period']:%b %Y}), status {self.status}.")
        rows = [f"  {r['period']:%b %Y}: {format_value(r['forecast'], fmt)} "
                f"({self.level:.0%} range {format_value(r['lo'], fmt)} to {format_value(r['hi'], fmt)})"
                for _, r in self.points.iterrows()]
        tail = (f"Check: in {v['n_checks']} past walk-forward forecasts the {self.level:.0%} band held "
                f"{v['coverage']:.0%} of the actual values." if v.get("n_checks") else "")
        return "\n".join([head, *rows, tail, self.reason]).strip()

    def facts(self) -> dict[str, str]:
        """Every number this forecast may be described with, as display strings (for the
        explanation validator). Keys look like 'forecast.2025-01', 'lo.2025-01', 'coverage'."""
        out: dict[str, str] = {}
        for _, r in self.points.iterrows():
            k = f"{r['period']:%Y-%m-%d}"
            out[f"forecast.{k}"] = format_value(r["forecast"], self.fmt)
            out[f"lo.{k}"] = format_value(r["lo"], self.fmt)
            out[f"hi.{k}"] = format_value(r["hi"], self.fmt)
        v = self.validation
        if v.get("n_checks"):
            out["coverage"] = f"{v['coverage']:.0%}"
            out["n_checks"] = f"{v['n_checks']}"
            out["level"] = f"{self.level:.0%}"
        if v.get("skill") is not None and np.isfinite(v["skill"]):
            out["skill"] = f"{v['skill']:.0%}"
        return out

    def to_dict(self) -> dict:
        d = {f: getattr(self, f) for f in ("kpi", "label", "fmt", "freq", "level", "status", "reason", "model",
                                           "validation", "notes", "filters")}
        d["history"] = self.history.to_dict("records")
        d["points"] = self.points.to_dict("records")
        return to_plain(d)


# ----------------------------------------------------------------------------
# The two models (pure numpy; y is a 1-D float array, oldest first)
# ----------------------------------------------------------------------------
def _k(m: int) -> int:
    return max(2, m // 2)


def _min_history(m: int) -> int:
    return m + _k(m)


def _seasonal_growth(y: np.ndarray, m: int, H: int, additive: bool, clip) -> np.ndarray:
    n, k = len(y), _k(m)
    if n < m + k:
        return np.full(H, np.nan)
    last, prev = y[n - k:], y[n - m - k:n - m]
    if additive:
        d = float(last.mean() - prev.mean())
    else:
        ps = float(prev.sum())
        g = float(np.clip(last.sum() / ps, *clip)) if ps > 0 else 1.0
    out = np.empty(H)
    for h in range(H):
        base = y[n + h - m * (h // m + 1)]
        out[h] = base + d if additive else base * g
    return out


def _seasonal_index(y: np.ndarray, m: int, additive: bool) -> np.ndarray | None:
    n = len(y)
    if n < 2 * m:
        return None
    w = np.r_[0.5, np.ones(m - 1), 0.5] / m if m % 2 == 0 else np.ones(m) / m
    ma = np.convolve(y, w, mode="valid")
    off = (len(w) - 1) // 2
    seg = y[off:off + len(ma)]
    ratio = seg - ma if additive else seg / ma
    pos = (np.arange(len(ma)) + off) % m
    idx = np.array([ratio[pos == s].mean() if np.any(pos == s) else np.nan for s in range(m)])
    if np.any(~np.isfinite(idx)):
        return None
    return idx - idx.mean() if additive else idx / idx.mean()


def _deseason_trend(y: np.ndarray, m: int, H: int, additive: bool) -> np.ndarray:
    n = len(y)
    idx = _seasonal_index(y, m, additive)
    if idx is None:
        return np.full(H, np.nan)
    pos = np.arange(n) % m
    d = y - idx[pos] if additive else y / idx[pos]
    w = min(n, m)
    x = np.arange(w, dtype=float)
    slope, icpt = np.polyfit(x, d[n - w:], 1)
    level_end = icpt + slope * (w - 1)
    phi = 0.9 ** (12.0 / m)                      # same yearly damping whatever the frequency
    out = np.empty(H)
    for h in range(1, H + 1):
        damp = sum(phi ** j for j in range(1, h + 1))
        dh = level_end + slope * damp
        s = idx[(n - 1 + h) % m]
        out[h - 1] = dh + s if additive else dh * s
    return out


def _predict(y: np.ndarray, m: int, H: int, additive: bool, clip) -> tuple[np.ndarray, str]:
    a = _seasonal_growth(y, m, H, additive, clip)
    if len(y) >= 2 * m:
        b = _deseason_trend(y, m, H, additive)
        both = np.vstack([a, b])
        with np.errstate(all="ignore"):
            f = np.where(np.isfinite(b), both.mean(axis=0), a)
        return f, "blend of seasonal growth and deseasonalised trend"
    return a, "seasonal growth (under 2 full years of history)"


def _naive(y: np.ndarray, m: int, target: int) -> float:
    return float(y[target - m]) if target - m >= 0 else float("nan")


def _scale(y: np.ndarray, t: int, m: int) -> float:
    s = float(np.mean(np.abs(y[max(0, t - m):t])))
    return s if s > 1e-12 else 1.0


# ----------------------------------------------------------------------------
# Calibration and walk-forward validation
# ----------------------------------------------------------------------------
@dataclass
class _Back:
    origin: np.ndarray
    h: np.ndarray
    target: np.ndarray
    fc: np.ndarray
    err: np.ndarray        # (actual - forecast) / scale
    scale: np.ndarray


def _backtest(y: np.ndarray, m: int, H: int, additive: bool, clip) -> _Back:
    n = len(y)
    o, hh, tg, fc, er, sc = [], [], [], [], [], []
    for t in range(_min_history(m), n):
        f, _ = _predict(y[:t], m, H, additive, clip)
        s = _scale(y, t, m)
        for h in range(1, H + 1):
            target = t + h - 1
            if target >= n or not np.isfinite(f[h - 1]):
                continue
            o.append(t); hh.append(h); tg.append(target); fc.append(f[h - 1])
            er.append((y[target] - f[h - 1]) / s); sc.append(s)
    return _Back(*(np.array(a) for a in (o, hh, tg, fc, er, sc)))


def _conformal_q(errs: np.ndarray, lo_p: float, hi_p: float) -> tuple[float, float]:
    e = np.sort(errs)
    n = len(e)
    i_lo = max(0, int(math.floor((n + 1) * lo_p)) - 1)
    i_hi = min(n - 1, int(math.ceil((n + 1) * hi_p)) - 1)
    return float(e[i_lo]), float(e[i_hi])


def _band_quantiles(back: _Back, known: np.ndarray, H: int, level: float, cfg: ForecastConfig):
    """(q_lo[h], q_hi[h]) for h=1..H from the past errors flagged True in `known`."""
    lo_p, hi_p = (1 - level) / 2, 1 - (1 - level) / 2
    e_all, h_all = back.err[known], back.h[known]
    if len(e_all) < cfg.min_calibration:
        return None
    width = max(1, math.ceil(H / 3))
    qlo, qhi = np.empty(H), np.empty(H)
    pooled = _conformal_q(e_all, lo_p, hi_p)
    for h in range(1, H + 1):
        b = min(2, (h - 1) // width)
        in_b = np.minimum(2, (h_all - 1) // width) == b
        qlo[h - 1], qhi[h - 1] = _conformal_q(e_all[in_b], lo_p, hi_p) if in_b.sum() >= cfg.min_bucket else pooled
    return qlo, qhi


def _validate(y: np.ndarray, m: int, H: int, additive: bool, nonneg: bool, back: _Back, cfg: ForecastConfig,
              detail: bool = False) -> dict:
    n = len(y)
    inside, width, aerr, nerr, origins, tgts, hs = [], [], [], [], [], [], []
    for t in np.unique(back.origin):
        known = back.target < t
        q = _band_quantiles(back, known, H, cfg.level, cfg)
        if q is None:
            continue
        rows = np.flatnonzero(back.origin == t)
        for i in rows:
            h = int(back.h[i])
            f, s = back.fc[i], back.scale[i]
            lo, hi = f + q[0][h - 1] * s, f + q[1][h - 1] * s
            if nonneg:
                lo = max(lo, 0.0)
            actual = y[int(back.target[i])]
            inside.append(lo <= actual <= hi)
            width.append((hi - lo) / s)
            aerr.append(abs(actual - f) / s)
            nv = _naive(y, m, int(back.target[i]))
            nerr.append(abs(actual - nv) / s if np.isfinite(nv) else np.nan)
            origins.append(int(t))
            tgts.append(int(back.target[i]))
            hs.append(h)
    inside, aerr, nerr, origins = map(np.array, (inside, aerr, nerr, origins))
    out = {"level": cfg.level, "n_checks": int(len(inside)), "coverage": None, "mean_band_width": None,
           "mae": None, "mae_naive": None, "skill": None, "holdout": None, "n_origins": int(len(set(origins)))}
    if detail:
        out["detail"] = {"origin": origins, "target": np.array(tgts), "h": np.array(hs),
                         "inside": inside, "width": np.array(width)}
    if not len(inside):
        return out
    out["coverage"] = float(inside.mean())
    out["mean_band_width"] = float(np.mean(width))
    both = np.isfinite(nerr)
    if both.any():
        out["mae"], out["mae_naive"] = float(aerr[both].mean()), float(nerr[both].mean())
        out["skill"] = float(1 - out["mae"] / out["mae_naive"]) if out["mae_naive"] > 0 else None
    ho = origins == n - H
    if ho.any():
        out["holdout"] = {"n": int(ho.sum()), "coverage": float(inside[ho].mean()),
                          "from": int(n - H)}
    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def forecast_series(history: pd.Series, freq: str = "M", horizon: int | None = None, is_ratio: bool = False,
                    config: ForecastConfig | None = None, label: str = "Value", fmt: str = "money",
                    kpi: str = "value") -> Forecast:
    """Forecast a complete period series (index = period start dates, no gaps, no partial periods)."""
    cfg = config or ForecastConfig()
    if freq not in SEASON:
        raise ForecastError("Forecasting supports monthly (M), quarterly (Q) and weekly (W) data. "
                            "Aggregate daily data to weeks.")
    m = SEASON[freq]
    H = DEFAULT_HORIZON[freq] if horizon is None else int(horizon)
    if H < 1 or H > m:
        raise ForecastError(f"horizon must be between 1 and {m} periods.")
    notes: list[str] = []
    s = history.astype(float).copy()
    hist_df = pd.DataFrame({"period": s.index, "value": s.to_numpy()})

    def unknown(reason):
        return Forecast(kpi, label, fmt, freq, cfg.level, UNKNOWN, reason, "none", hist_df,
                        pd.DataFrame(columns=["period", "h", "forecast", "lo", "hi"]), {"level": cfg.level,
                        "n_checks": 0}, notes)

    nan_share = float(s.isna().mean()) if len(s) else 1.0
    if len(s) == 0 or nan_share > 0.2:
        return unknown("Too many empty periods to forecast.")
    if s.isna().any():
        s = s.interpolate(limit_direction="both")
        notes.append(f"{int(hist_df['value'].isna().sum())} empty period(s) were filled by interpolation.")
    y = s.to_numpy()
    if len(y) < _min_history(m) + 4:
        return unknown(f"Only {len(y)} complete periods of history; at least {_min_history(m) + 4} are needed to "
                       f"measure how wrong the forecast tends to be, so no forecast is given.")
    additive = bool(is_ratio or np.any(y <= 0))
    nonneg = bool(np.all(y >= 0))
    if additive and not is_ratio:
        notes.append("The series has zero or negative values, so changes are modelled by addition, not by ratio.")

    back = _backtest(y, m, H, additive, cfg.growth_clip)
    q = _band_quantiles(back, back.target < len(y), H, cfg.level, cfg) if len(back.err) else None
    if q is None:
        return unknown("Not enough past forecasts to measure the typical error, so no band can be built.")
    f, model = _predict(y, m, H, additive, cfg.growth_clip)
    if not np.all(np.isfinite(f)):
        return unknown("The model could not produce a forecast for this series.")
    sc = _scale(y, len(y), m)
    lo, hi = f + q[0] * sc, f + q[1] * sc
    if nonneg:
        lo = np.maximum(lo, 0.0)
        f = np.maximum(f, 0.0)
    hi = np.maximum(hi, f)
    lo = np.minimum(lo, f)
    idx = pd.date_range(s.index[-1], periods=H + 1, freq=FREQ[freq][1])[1:]
    points = pd.DataFrame({"period": idx, "h": np.arange(1, H + 1), "forecast": f, "lo": lo, "hi": hi})

    val = _validate(y, m, H, additive, nonneg, back, cfg)
    cov = val["coverage"]
    if cov is None or val["n_checks"] < cfg.min_checks:
        status = LOW_CONFIDENCE
        reason = (f"Only {val['n_checks']} past checks are available, fewer than the {cfg.min_checks} needed to "
                  f"say whether the band is the right width. Treat the band as rough.")
    elif abs(cov - cfg.level) <= cfg.coverage_tolerance:
        status = OK
        reason = (f"The {cfg.level:.0%} band held {cov:.0%} of the actual values in {val['n_checks']} past "
                  f"walk-forward checks, within {cfg.coverage_tolerance * 100:.0f} points of its target. A forecast extends the past; it does not know "
                  f"about future promotions, price changes or shocks.")
    else:
        status = LOW_CONFIDENCE
        reason = (f"The {cfg.level:.0%} band held {cov:.0%} of the actual values in {val['n_checks']} past checks, "
                  f"which is {'too few' if cov < cfg.level else 'too many'} for its target: the band is "
                  f"{'too narrow' if cov < cfg.level else 'wider than needed'}. Treat it with care.")
    if val["skill"] is not None and val["skill"] <= 0:
        notes.append("In past checks this model was not better than simply repeating last year's value.")
    if len(y) < 2 * m:
        notes.append("Less than two full years of history: the yearly pattern is estimated from one year only.")
    return Forecast(kpi, label, fmt, freq, cfg.level, status, reason, model, hist_df, points, val, notes)


def history_for(ds, kpi_key: str, freq: str = "M", filters: dict | None = None):
    """Complete periods of a KPI: drops a partial first or last period. Returns (series, kpi, notes)."""
    if freq not in SEASON:
        raise ForecastError("Forecasting supports monthly (M), quarterly (Q) and weekly (W) data. "
                            "Aggregate daily data to weeks.")
    try:
        kpi = get_kpi(ds, kpi_key)
    except Exception as err:                                  # unknown / unavailable KPI
        raise ForecastError(str(err)) from err
    df = kpi_series(ds, kpi_key, freq, filters)
    notes: list[str] = []
    if df.empty:
        return pd.Series(dtype=float), kpi, notes
    s = pd.Series(df["value"].to_numpy(), index=pd.DatetimeIndex(df["period"]))
    if not _is_complete(s.index[-1], freq, pd.Timestamp(ds.max_date)):
        s = s.iloc[:-1]
        notes.append("The last period is incomplete, so it was left out.")
    first_start = pd.Timestamp(ds.min_date).normalize()
    if len(s) and first_start > s.index[0]:
        s = s.iloc[1:]
        notes.append("The first period is incomplete, so it was left out.")
    return s, kpi, notes


def forecast_kpi(ds, kpi_key: str, freq: str = "M", filters: dict | None = None, horizon: int | None = None,
                 config: ForecastConfig | None = None) -> Forecast:
    """Forecast a KPI of a loaded dataset (optionally for a slice, e.g. filters={"Region": "West"})."""
    s, kpi, notes = history_for(ds, kpi_key, freq, filters)
    f = forecast_series(s, freq, horizon, kpi.den is not None, config, kpi.label, kpi.fmt, kpi.key)
    f.notes = notes + f.notes
    f.filters = dict(filters or {})
    return f


if __name__ == "__main__":
    import sys
    from engine.ingest import load_dataset

    if len(sys.argv) < 2:
        print("Usage: python -m engine.forecast <csv or xlsx> [kpi] [M|Q|W]")
        sys.exit(1)
    data = load_dataset(sys.argv[1])
    key = sys.argv[2] if len(sys.argv) > 2 else "revenue"
    fq = sys.argv[3] if len(sys.argv) > 3 else "M"
    result = forecast_kpi(data, key, fq)
    print(f"{data.name}: {data.n_rows:,} rows, {data.min_date.date()} to {data.max_date.date()}\n")
    print(result.summary())
    for n in result.notes:
        print("Note:", n)
