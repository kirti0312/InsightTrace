"""Change detection: find KPI movements that are NOT just normal seasonality.

The idea, in plain words
------------------------
"Revenue fell 30% in January" is not news if January is always slow. So for
every period we ask: "what did we EXPECT, given the same period last year and
how the business has been growing lately?" Then we flag periods that are far
from that expectation.

How expectation is built (for monthly data, season length P = 12):
    expected(t) = value(t - P) + recent_growth
where "recent_growth" is the median of the last few year-over-year gaps that
were NORMAL. Unusual periods are left out of that median, so one bad quarter
does not hide the next one.

How "far" is measured: z = (actual - expected) / typical_noise, where
typical_noise is a robust spread (MAD) of past year-over-year gaps.
A period is flagged when the change is big enough to matter (min_change) AND
either |z| >= z_threshold on its own, or |z| >= z_soft for two periods in a row.
Neighbouring flagged periods are merged into one "episode"
(e.g. "revenue below expectation from Jun 2024 to Nov 2024").

Other rules worth knowing:
  - A pattern that repeats every year (e.g. a yearly promotion) is flagged the
    first time, then recognised as seasonal and not flagged again.
  - A permanent change is flagged at first, then (after about half a year)
    recorded as a level shift, so it is neither flagged forever nor echoed
    a year later.
  - Sums (revenue, profit, orders) are compared per day, so a 31-day month is
    not compared unfairly with a 28-day one.
  - Headline KPIs: revenue, profit, orders, profit margin. Units, average order
    value and average discount are "supporting clues".

Honest limits (also reported in `notes` when they apply)
--------------------------------------------------------
  - Needs about 16+ months of history for seasonal detection. With less, a
    simple trailing baseline is used and seasonality is NOT removed.
  - Small effects hidden in noisy data will not be flagged. That is intended:
    we would rather stay quiet than cry wolf.
  - Measured on 16 held-out synthetic datasets (3 years, monthly): the planted
    margin collapse was found 16/16 times, the customer churn 12/16, the
    unexplained September dip 14/16 and the 6-week stockout 12/16, with about
    0.56 false alarms per dataset. On real, volatile data (Superstore) it
    raises about 21 alerts. Treat alerts as LEADS to investigate (the next
    steps test them), not as proof.

No LLM is used here. All numbers come from this code.

Try it from the project folder:
    python -m engine.detect data/synthetic_demo.csv M [high|normal|low]
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.ingest import Dataset, load_dataset
from engine.kpi import FREQ, available_kpis, format_value, get_kpi, kpi_series, kpi_value

SEASON_LENGTH = {"W": 52, "M": 12, "Q": 4}
# KPIs that drive headline alerts; the others (margin, order value, discount) are
# supporting clues, because they are often just a side effect of the headline change.
PRIMARY_KPIS = {"revenue", "profit", "orders", "profit_margin"}
ADAPT_AFTER = {"W": 26, "M": 6, "Q": 3}


@dataclass(frozen=True)
class DetectConfig:
    z_threshold: float = 2.75      # a single period this unusual is flagged
    z_soft: float = 1.75           # two periods in a row this unusual are flagged
    min_sigma_points: int = 8      # normal-noise estimate needs this many gaps
    sigma_long_window: int = 24    # long-run spread of ALL periods (guards against shrinkage)
    sigma_long_share: float = 0.8  # noise is never taken below this share of the long-run spread
    sigma_inflate: float = 1.15    # caution multiplier (flagged periods are left out of the noise estimate)
    sigma_prior_weight: int = 6    # how many 'pretend' observations of prior noise to blend in
    two_year_baseline: bool = False  # optional: also project from 2 years back (tested: no clear gain)
    min_change: float = 0.05       # smallest relative change that matters (5%)
    min_change_points: float = 0.02  # for percentage KPIs: 2 percentage points
    min_change_ratio: float = 0.10   # for ratio KPIs like average order value: 10%
    pool_size: int = 6             # how many recent normal gaps set "growth"
    sigma_window: int = 12         # how many recent normal gaps set "noise"
    min_pool: int = 4              # need this many normal gaps before judging
    min_sigma: float = 0.03        # never assume noise below 3%
    prior_sigma: float = 0.10      # noise assumed before we have enough data
    merge_gap: int = 1             # flagged periods this close are one episode


# Sensitivity presets. Measured on 8 held-out synthetic datasets (monthly data,
# events found out of 8; "false" = alarms in the clean period, whole 8 datasets):
#   high   : A 8, B 6, U 7, C 6, false 4   (finds the most, a few more false alarms)
#   normal : A 8, B 6, U 7, C 5, false 3   (default)
#   low    : A 7, B 4, U 4, C 3, false 0   (very quiet, but misses many real changes)
# On real, volatile data (Superstore) the number of alerts barely changes between
# levels, because its swings are huge compared with its normal noise.
SENSITIVITY = {
    "high": (2.5, 1.75),
    "normal": (2.75, 1.75),
    "low": (3.25, 2.25),
}


def config_for(level: str = "normal") -> DetectConfig:
    """A DetectConfig for 'high', 'normal' or 'low' sensitivity."""
    if level not in SENSITIVITY:
        raise ValueError(f"sensitivity must be one of {list(SENSITIVITY)}")
    z, z_soft = SENSITIVITY[level]
    return DetectConfig(z_threshold=z, z_soft=z_soft)


@dataclass
class ChangeEvent:
    kpi: str
    label: str
    freq: str
    direction: str                 # "down" or "up"
    start: pd.Timestamp            # first flagged period (start date)
    end: pd.Timestamp              # last day of the last flagged period
    n_periods: int
    peak_period: pd.Timestamp
    peak_z: float
    observed: float                # KPI over the whole episode window
    expected: float                # what we expected over that window
    impact: float                  # observed - expected (KPI units)
    pct_vs_expected: float         # relative change; for % KPIs: point change
    is_points: bool                # True when pct_vs_expected is in points
    ongoing: bool                  # True if it runs to the end of the data
    severity: str                  # "high" / "medium" / "low"
    compare_start: pd.Timestamp | None = None   # same window one season earlier
    compare_end: pd.Timestamp | None = None
    fmt: str = "money"
    primary: bool = True            # False for supporting KPIs

    def headline(self) -> str:
        arrow = "fell" if self.direction == "down" else "rose"
        span = (self.start.strftime("%b %Y") if self.n_periods == 1 else
                f"{self.start.strftime('%b %Y')} to {self.end.strftime('%b %Y')}")
        change = (f"{abs(self.pct_vs_expected) * 100:.1f} pts" if self.is_points
                  else f"{abs(self.pct_vs_expected):.1%}")      # the word fell/rose already carries the sign
        tail = ", still ongoing" if self.ongoing else ""
        return (f"{self.label} {arrow} {change} vs expected ({span}{tail}): "
                f"{format_value(self.observed, self.fmt)} vs "
                f"{format_value(self.expected, self.fmt)} expected")


@dataclass
class DetectionResult:
    kpi: str
    freq: str
    frame: pd.DataFrame            # one row per period (see detect_series)
    events: list[ChangeEvent]
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Core algorithm on a plain series (no database needed, easy to test)
# ----------------------------------------------------------------------------
def _mad_sigma(values) -> float:
    arr = np.asarray(values, dtype=float)
    return 1.4826 * float(np.median(np.abs(arr - np.median(arr))))


def detect_series(series: pd.DataFrame, fmt: str, freq: str = "M",
                  config: DetectConfig | None = None, scale=None,
                  min_change: float | None = None):
    """Flag unusual periods. `series` needs columns `period` and `value`.
    `min_change` overrides the smallest relative change that matters.
    `scale` (optional, one number per period) divides each value first, e.g.
    the number of days in the period, so a 31-day month is compared fairly with
    a 28-day month. Returns (frame, notes). Frame columns: period, value, expected, lower,
    upper, deviation, pct_dev, z, assessed, flagged, direction, method."""
    cfg = config or DetectConfig()
    P = SEASON_LENGTH[freq]
    notes: list[str] = []

    period = pd.to_datetime(series["period"]).reset_index(drop=True)
    v_raw = pd.to_numeric(series["value"], errors="coerce").to_numpy(dtype=float)
    sc = np.ones(len(v_raw)) if scale is None else np.asarray(scale, dtype=float)
    v = v_raw / sc                       # per-day (or per-unit) values used for detection
    n = len(v)
    valid = np.isfinite(v)
    if valid.sum() < 6:
        raise ValueError("Need at least 6 periods of data to look for changes.")

    additive = fmt == "pct" or bool(np.nanmin(v) <= 0)
    ref = float(np.nanmedian(np.abs(v))) or 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = v.copy() if additive else np.log(v)

    if fmt == "pct":
        floor_sigma, prior_sigma, min_effect = 0.005, 0.02, cfg.min_change_points
    elif additive:
        floor_sigma, prior_sigma = cfg.min_sigma * ref, cfg.prior_sigma * ref
        min_effect = (min_change if min_change is not None else cfg.min_change) * ref
    else:
        floor_sigma, prior_sigma = cfg.min_sigma, cfg.prior_sigma
        min_effect = math.log1p(min_change if min_change is not None else cfg.min_change)

    seasonal = n >= P + cfg.min_pool + 4
    Ex = np.full(n, np.nan)
    sig = np.full(n, np.nan)
    z = np.full(n, np.nan)
    flagged = np.zeros(n, dtype=bool)
    assessed = np.zeros(n, dtype=bool)

    if seasonal:
        adapt_after = ADAPT_AFTER[freq]
        clean = x.copy()                  # history with anomalies replaced by expectation
        ctr = np.full(n, np.nan)          # the growth estimate used at each period
        res_arr = np.full(n, np.nan)      # actual - expected, for every judged period
        off_at = np.zeros(n)              # lasting level shift in force at each period
        offset = 0.0                      # accumulated lasting level shift
        pool: list[tuple[int, float]] = []  # (index, gap) of recent NORMAL periods
        gaps: list[tuple[int, float]] = []  # (index, gap) of NOT-flagged periods (noise estimate)
        all_gaps: list[float] = []        # gaps of EVERY judged period (long-run spread)
        run, run_sign = 0, 0
        prev_soft = None                  # previous period if it was "moderately unusual"
        for t in range(P, n):
            off_at[t] = offset
            if not valid[t]:
                prev_soft = None
                continue
            # Baselines (each corrected for any lasting shift that happened since):
            #  - last year's cleaned value;
            #  - last year's ACTUAL value if last year was flagged, so a pattern
            #    that REPEATS every year is seen as seasonal and not flagged again.
            bases = []
            if np.isfinite(clean[t - P]):
                adj1 = offset - off_at[t - P]
                one_year = clean[t - P] + adj1
                # Optional two-year baseline: also project from TWO years back.
                if (cfg.two_year_baseline and t - 2 * P >= 0
                        and np.isfinite(clean[t - 2 * P]) and np.isfinite(ctr[t - P])):
                    two_year = clean[t - 2 * P] + ctr[t - P] + (offset - off_at[t - 2 * P])
                    bases.append((one_year + two_year) / 2)
                else:
                    bases.append(one_year)
                if flagged[t - P] and np.isfinite(x[t - P]) and x[t - P] != clean[t - P]:
                    bases.append(x[t - P] + adj1)
            if not bases:
                continue
            if len(pool) < cfg.min_pool:       # warm-up: cannot judge yet
                d0 = x[t] - bases[0]
                pool.append((t, d0))
                gaps.append((t, d0))
                all_gaps.append(d0)
                continue

            center = float(np.median([g for _, g in pool[-cfg.pool_size:]]))
            ctr[t] = center
            if len(gaps) >= cfg.min_sigma_points:
                recent = [g for _, g in gaps[-cfg.sigma_window:]]
                k, m = len(recent), cfg.sigma_prior_weight
                s_data = _mad_sigma(recent)
                # blend the data's noise with a prior so a lucky-quiet stretch of
                # history cannot make us over-confident (matters when k is small)
                blended = math.sqrt((k * s_data ** 2 + m * prior_sigma ** 2) / (k + m))
                sigma = blended * math.sqrt(1 + 1 / k) * cfg.sigma_inflate
                if len(all_gaps) >= cfg.min_sigma_points:
                    long_run = _mad_sigma(all_gaps[-cfg.sigma_long_window:])
                    sigma = max(sigma, cfg.sigma_long_share * long_run)
                sigma = max(sigma, floor_sigma)
            else:
                sigma = max(prior_sigma, floor_sigma)
            base = min(bases, key=lambda b: abs(x[t] - (b + center)))
            Ex[t] = base + center
            sig[t] = sigma
            d_t = x[t] - base
            all_gaps.append(d_t)
            resid = x[t] - Ex[t]
            res_arr[t] = resid
            z[t] = resid / sigma
            assessed[t] = True

            big = abs(resid) >= min_effect
            strong = big and abs(z[t]) >= cfg.z_threshold
            soft = big and abs(z[t]) >= cfg.z_soft
            sign = 1 if resid > 0 else -1
            prev_same = prev_soft is not None and prev_soft == (t - 1, sign)
            continuing = bool(flagged[t - 1]) and run_sign == sign   # already in an episode

            if strong or (soft and (prev_same or continuing)):
                if prev_same and not flagged[t - 1]:       # include the first of the pair
                    flagged[t - 1] = True
                    clean[t - 1] = Ex[t - 1]
                    pool = [(i, g) for i, g in pool if i != t - 1]
                    gaps = [(i, g) for i, g in gaps if i != t - 1]
                    run, run_sign = 1, sign
                flagged[t] = True
                run = run + 1 if sign == run_sign else 1
                run_sign = sign
                clean[t] = Ex[t]
                if run >= adapt_after:
                    # Lasting change: record it as a level shift. From the next
                    # period on, expectations include it, and the offset is
                    # cancelled again once last year's base also contains it
                    # (so there is no false "echo" alert a year later).
                    offset += float(np.median(res_arr[t - adapt_after + 1:t + 1]))
                    run, run_sign = 0, 0
                prev_soft = None
            else:
                run, run_sign = 0, 0
                gaps.append((t, d_t))        # flagged periods are left out of the noise estimate
                if not soft:        # only genuinely normal periods update the growth estimate
                    pool.append((t, d_t))
                prev_soft = (t, sign) if soft else None
        method = "seasonal"
    else:
        notes.append("Less than about 16 months of data: seasonality could NOT be "
                     "removed, so ordinary seasonal swings may be flagged as changes.")
        W = 6
        for t in range(W, n):
            hist = x[max(0, t - W):t]
            hist = hist[np.isfinite(hist)]
            if not valid[t] or len(hist) < 4:
                continue
            Ex[t] = float(np.median(hist))
            sigma = max(_mad_sigma(np.diff(hist)) / math.sqrt(2) if len(hist) > 2 else prior_sigma,
                        floor_sigma)
            sig[t] = sigma
            resid = x[t] - Ex[t]
            z[t] = resid / sigma
            assessed[t] = True
            flagged[t] = abs(z[t]) >= cfg.z_threshold and abs(resid) >= min_effect
        method = "trailing"

    # back to KPI units
    def inv(a):
        return a if additive else np.exp(a)

    exp_u = inv(Ex)                      # expected, in per-day units
    if additive:
        d_u = v - exp_u
        pct_dev = d_u.copy() if fmt == "pct" else d_u / np.where(np.abs(exp_u) > 0, np.abs(exp_u), np.nan)
    else:
        pct_dev = v / exp_u - 1.0
    expected = exp_u * sc                # back to the real period totals
    lower = inv(Ex - cfg.z_threshold * sig) * sc
    upper = inv(Ex + cfg.z_threshold * sig) * sc
    deviation = v_raw - expected

    frame = pd.DataFrame({
        "period": period, "value": v_raw, "expected": expected, "lower": lower,
        "upper": upper, "deviation": deviation, "pct_dev": pct_dev, "z": z,
        "assessed": assessed, "flagged": flagged,
        "direction": np.where(flagged, np.where(deviation < 0, "down", "up"), ""),
        "method": method,
    })
    return frame, notes


# ----------------------------------------------------------------------------
# Turning flagged periods into episodes
# ----------------------------------------------------------------------------
def _period_end(start: pd.Timestamp, freq: str, cap: pd.Timestamp | None) -> pd.Timestamp:
    end = pd.Period(start, freq=FREQ[freq][2]).end_time.normalize()
    return min(end, cap) if cap is not None else end


def _severity(peak_z: float) -> str:
    a = abs(peak_z)
    return "high" if a >= 4 else "medium" if a >= 3 else "low"


def find_events(frame: pd.DataFrame, kpi, freq: str, ds: Dataset | None = None,
                filters: dict | None = None, config: DetectConfig | None = None):
    cfg = config or DetectConfig()
    idx = np.flatnonzero(frame["flagged"].to_numpy())
    if len(idx) == 0:
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        same_dir = frame["direction"].iloc[i] == frame["direction"].iloc[cur[-1]]
        if i - cur[-1] - 1 <= cfg.merge_gap and same_dir:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    P = SEASON_LENGTH[freq]
    cap = ds.max_date if ds is not None else None
    last_assessed = int(np.flatnonzero(frame["assessed"].to_numpy())[-1])
    is_points = kpi.fmt == "pct"
    is_ratio = kpi.den is not None
    events = []
    for g in groups:
        lo, hi = g[0], g[-1]
        rows = frame.iloc[lo:hi + 1]
        start = rows["period"].iloc[0]
        end = _period_end(rows["period"].iloc[-1], freq, cap)
        peak = rows.loc[rows["z"].abs().idxmax()]
        direction = "down" if rows["deviation"].sum() < 0 else "up"

        if ds is not None:                         # exact figure for the whole window
            observed = kpi_value(ds, kpi.key, start, end, filters)
        else:
            observed = float(rows["value"].mean() if is_ratio else rows["value"].sum())
        if is_ratio:
            # Ratios (margin, average order value, ...) cannot be summed over time:
            # use the average gap per period instead.
            gap = float(rows["deviation"].mean())
            expected, impact = observed - gap, gap
            pct = gap if is_points else (gap / abs(expected) if expected else float("nan"))
        else:
            impact = float(rows["deviation"].sum())
            expected = observed - impact
            pct = impact / abs(expected) if expected else float("nan")

        shift = pd.DateOffset(months=12 if freq != "Q" else 12) if freq in ("M", "Q") else pd.DateOffset(weeks=52)
        events.append(ChangeEvent(
            kpi=kpi.key, label=kpi.label, freq=freq, direction=direction,
            start=start, end=end, n_periods=hi - lo + 1,
            peak_period=peak["period"], peak_z=float(peak["z"]),
            observed=observed, expected=expected, impact=impact,
            pct_vs_expected=pct, is_points=is_points,
            ongoing=bool(hi >= last_assessed), severity=_severity(float(peak["z"])),
            compare_start=start - shift, compare_end=end - shift, fmt=kpi.fmt,
            primary=kpi.key in PRIMARY_KPIS,
        ))
    return events


# ----------------------------------------------------------------------------
# Public functions that work on a Dataset
# ----------------------------------------------------------------------------
def detect_kpi(ds: Dataset, key: str, freq: str = "M", filters: dict | None = None,
               config: DetectConfig | None = None) -> DetectionResult:
    if freq not in SEASON_LENGTH:
        raise ValueError("freq must be 'W', 'M' or 'Q' for change detection")
    kpi = get_kpi(ds, key)
    series = kpi_series(ds, key, freq, filters)
    # drop an unfinished last period so a half-month is not read as a collapse
    if len(series) > 1:
        last_end = pd.Period(series["period"].iloc[-1], freq=FREQ[freq][2]).end_time.normalize()
        if ds.max_date < last_end:
            series = series.iloc[:-1]
    scale = None
    if kpi.den is None and freq in ("M", "Q"):      # sums: compare per-day, not per-month
        def days(p):
            per = pd.Period(p, freq=FREQ[freq][2])
            return (per.end_time.normalize() - per.start_time.normalize()).days + 1
        scale = [days(p) for p in series["period"]]
    cfg = config or DetectConfig()
    ratio_money = kpi.den is not None and kpi.fmt != "pct"
    frame, notes = detect_series(series, kpi.fmt, freq, cfg, scale,
                                 min_change=cfg.min_change_ratio if ratio_money else None)
    return DetectionResult(key, freq, frame, find_events(frame, kpi, freq, ds, filters, config), notes)


def detect_all(ds: Dataset, freq: str = "M", kpis: list[str] | None = None,
               filters: dict | None = None, config: DetectConfig | None = None,
               primary_only: bool = False):
    """Run every available KPI. Returns (events, notes). Headline (primary) KPIs
    come first, then strongest signal first. With primary_only=True the
    supporting KPIs (units, order value, discount) are left out."""
    events, notes = [], []
    for k in available_kpis(ds):
        if kpis and k.key not in kpis:
            continue
        try:
            res = detect_kpi(ds, k.key, freq, filters, config)
        except ValueError as err:
            notes.append(f"{k.label}: {err}")
            continue
        events += res.events
        notes += [f"{k.label}: {n}" for n in res.notes if n not in notes]
    if primary_only:
        events = [e for e in events if e.primary]
    events.sort(key=lambda e: (not e.primary, -abs(e.peak_z), e.start))
    return events, list(dict.fromkeys(notes))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m engine.detect <csv or xlsx> [W|M|Q] [high|normal|low]")
        sys.exit(1)
    data = load_dataset(sys.argv[1])
    f = sys.argv[2] if len(sys.argv) > 2 else "M"
    level = sys.argv[3] if len(sys.argv) > 3 else "normal"
    evs, nts = detect_all(data, f, config=config_for(level))
    print(f"{data.name}: {data.n_rows:,} rows, {data.min_date.date()} to {data.max_date.date()}"
          f"   (sensitivity: {level})\n")
    if not evs:
        print("No unusual changes found.")
    for title, group in (("HEADLINE ALERTS (revenue, profit, orders, margin)", [e for e in evs if e.primary]),
                         ("SUPPORTING CLUES (units, order value, discount)", [e for e in evs if not e.primary])):
        if group:
            print(title)
            for e in sorted(group, key=lambda e: e.start):
                print(f"  [{e.severity:<6}] {e.headline()}   (z {e.peak_z:+.1f})")
            print()
    for n in nts:
        print("Note:", n)
