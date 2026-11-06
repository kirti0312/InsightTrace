"""Verification: is a big contributor a REAL cause, or just where the change shows up?

The idea, in plain words
------------------------
decompose.py says "West accounts for most of the margin drop". That is a LEAD.
Here we test the lead with deterministic checks and give one of three verdicts:

    CAUSE        every check passed
    NOT_A_CAUSE  there is evidence AGAINST it (moved with everything else, started
                 earlier, repeats every year, rebound from an odd year, too small)
    UNKNOWN      we cannot tell (this slice is normally this noisy, or too little data)

The checks (each is recorded with its numbers and pass / fail / unknown)
-----------------------------------------------------------------------
 1. Size and distinctiveness - the group explains a real share of the change AND is much
    bigger in the change than in the normal business ("lift"). If everything fell 18%,
    each region explains "its own size" of the fall: lift ~ 1, nothing stands out.
 2. Differs from the rest     - versus the REST of the business over the same dates (this
    removes overall growth and seasonality) the group moved clearly more.
 3. Started inside the window - the gap clearly GREW inside the window: in the equal-length
    period just before, the group differed from the rest by clearly less. (A steady,
    already-running loss is blamed on the window where it started, not on later ones.)
 4. Not a yearly repeat       - the same group did not do the same in the same window a year
    before, and last year's window was not abnormal in the OPPOSITE direction (then this
    year's "surge/drop" is a rebound from an odd base, not a new cause).
 5. Unusual for its history   - placebo: in ordinary past windows of the same length, take
    the same number of worst-moving members (of similar size) the same way. How big do
    their gaps get by luck? The real gap must be far outside that. This is fair to the
    "pick the biggest loser afterwards" bias: normal windows also have biggest losers.

Groups and drill-down
---------------------
For each dimension the "group" is the fewest top movers that explain `cover` of the change,
plus any member that moved at least `tail_ratio` as much as the biggest (so 5 churned
customers are not cut to 2). A verified group is drilled into (West -> Phones) with the same
checks inside that slice. The LEAF slices are the reported causes; overlapping descriptions
of the same thing are merged (`also_described_as`).

Honest limits
-------------
  - Statistics, not proof. A CAUSE may still be a side effect of something not in the file.
    We say "the evidence supports", never "proved".
  - Placebo needs >= ~1.5 years of history; with less the verdict is UNKNOWN.
  - Several causes in one long window dilute each other: use verify_subwindows /
    verify_by_month so each cause meets its own window.
  - Drill-down stops when the narrower slice is not clearly more distinctive than its parent
    (e.g. it reports "Phones", not "Phones Max in East+South", because Phones Max is already
    most of Phones).
  - For distinct counts (orders), "rest = total - group" is approximate when an order can
    belong to several members. Daily sums assume one order belongs to one date.
  - Thresholds were tuned on the synthetic data only.

No LLM is used. Each verdict carries the SQL behind its numbers (`queries`).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.decompose import (BLANK_LABEL, OTHER_LABEL, DecomposeError, _total, contributions,
                              monthly_windows, one_year_earlier)
from engine.ingest import TABLE, Dataset, q
from engine.kpi import KPI, build_where, get_kpi

CAUSE, NOT_A_CAUSE, UNKNOWN = "CAUSE", "NOT_A_CAUSE", "UNKNOWN"
PASS, FAIL, UNSURE = "pass", "fail", "unknown"


@dataclass(frozen=True)
class VerifyConfig:
    cover: float = 0.60            # group = fewest top movers explaining this share of the change
    max_group: int = 8             # ... but never more than this many members
    tail_ratio: float = 0.4        # ... then add members that moved >= this fraction of the biggest
    size_match: float = 0.5        # placebo only uses members >= this fraction as big as the group's smallest
    min_share: float = 0.25        # group must explain at least this share of the change
    min_lift: float = 2.0          # ... and be this many times bigger in the change than normally
    min_gap: float = 0.10          # group vs rest must differ by >= 10% (relative KPIs)
    min_gap_points: float = 0.02   # ... or 2 percentage points (% KPIs)
    pre_ratio: float = 0.7         # gap in the period BEFORE must be below this fraction of the gap inside
    recur_ratio: float = 0.5       # gap a year earlier above this fraction of the gap = repeat / rebound
    placebo_z: float = 3.0         # the gap must be this many robust spreads beyond placebo gaps
    min_placebo: int = 6           # need this many placebo windows, otherwise UNKNOWN
    min_change: float = 0.03       # the KPI must move >= 3% (0.5 pts for % KPIs) to be worth explaining
    max_depth: int = 3             # how many dimensions deep to drill
    auto_growth: bool = True       # sum KPIs without an expected value: remove last year's ordinary growth
    growth_clip: tuple = (0.5, 2.0)  # ... measured over the 12 months BEFORE the window, kept within this range


@dataclass
class Check:
    name: str
    status: str                    # "pass" / "fail" / "unknown"
    detail: str                    # plain-English sentence; every number in it comes from code
    values: dict = field(default_factory=dict)


@dataclass
class SliceVerdict:
    dimension: str
    members: list[str]
    filters: dict                  # the full slice, e.g. {"Region": ["West"], "Sub-Category": ["Phones"]}
    verdict: str
    reason: str
    checks: list[Check]
    contribution: float            # group's contribution to the change (within the parent slice)
    share_of_change: float         # ... as a share of the parent slice's change
    lift: float
    queries: list[dict] = field(default_factory=list)
    children: list["SliceVerdict"] = field(default_factory=list)


@dataclass
class Cause:
    filters: dict
    share_of_window_change: float  # how much of the WHOLE window's change this slice accounts for
    contribution: float
    chain: list[SliceVerdict]      # the drill-down path (top level first, leaf last)
    also_described_as: list[str] = field(default_factory=list)

    @property
    def leaf(self) -> SliceVerdict:
        return self.chain[-1]

    def describe(self) -> str:
        return " + ".join(f"{k} = {', '.join(v)}" for k, v in self.filters.items())


@dataclass
class WindowResult:
    kpi: str
    label: str
    fmt: str
    window: tuple[pd.Timestamp, pd.Timestamp]
    compare: tuple[pd.Timestamp, pd.Timestamp]
    change: float                  # KPI change vs the baseline (growth-adjusted for sum KPIs)
    conclusion: str                # CAUSE if at least one cause was verified, else UNKNOWN
    reason: str
    causes: list[Cause]
    candidates: list[SliceVerdict]  # every top-level candidate, with verdict and why
    notes: list[str] = field(default_factory=list)

    @property
    def explained_share(self) -> float:
        return float(sum(c.share_of_window_change for c in self.causes)) if self.causes else 0.0


# ----------------------------------------------------------------------------
# Daily panel: per-member daily sums, so any window can be summed instantly
# ----------------------------------------------------------------------------
class Unavailable(Exception):
    """A needed window lies outside the data."""


class Panel:
    def __init__(self, ds: Dataset, kpi: KPI, dimension: str, filters: dict | None = None):
        self.kpi, self.dimension, self.ratio = kpi, dimension, kpi.den is not None
        self.t0 = pd.Timestamp(ds.min_date).normalize()
        self.T = int((pd.Timestamp(ds.max_date).normalize() - self.t0).days) + 1
        where, params = build_where(ds, filters)
        den_sql = kpi.den if kpi.den else "NULL"
        sql_m = (f"SELECT order_date AS day, COALESCE(CAST({q(dimension)} AS VARCHAR), '{BLANK_LABEL}') "
                 f"AS member, {kpi.num} AS num, {den_sql} AS den FROM {TABLE}{where} GROUP BY 1, 2")
        sql_t = f"SELECT order_date AS day, {kpi.num} AS num, {den_sql} AS den FROM {TABLE}{where} GROUP BY 1"
        m, t = ds.query(sql_m, params), ds.query(sql_t, params)
        sp = [str(p) for p in params]
        self.queries = [{"sql": sql_m, "params": sp}, {"sql": sql_t, "params": sp}]

        self.members = np.array(sorted(m["member"].unique()), dtype=object)
        pos = {name: i for i, name in enumerate(self.members)}
        M = len(self.members)
        mi = m["member"].map(pos).to_numpy(dtype=int)
        mt = (pd.to_datetime(m["day"]).dt.normalize() - self.t0).dt.days.to_numpy(dtype=int)
        num = np.zeros((M, self.T))
        num[mi, mt] = pd.to_numeric(m["num"]).fillna(0.0).to_numpy()
        self._cn = np.concatenate([np.zeros((M, 1)), np.cumsum(num, axis=1)], axis=1)
        self._cd = None
        if self.ratio:
            den = np.zeros((M, self.T))
            den[mi, mt] = pd.to_numeric(m["den"]).fillna(0.0).to_numpy()
            self._cd = np.concatenate([np.zeros((M, 1)), np.cumsum(den, axis=1)], axis=1)
        tt = (pd.to_datetime(t["day"]).dt.normalize() - self.t0).dt.days.to_numpy(dtype=int)
        tn, td = np.zeros(self.T), np.zeros(self.T)
        tn[tt] = pd.to_numeric(t["num"]).fillna(0.0).to_numpy()
        if self.ratio:
            td[tt] = pd.to_numeric(t["den"]).fillna(0.0).to_numpy()
        self._tn = np.concatenate([[0.0], np.cumsum(tn)])
        self._td = np.concatenate([[0.0], np.cumsum(td)])

    def _span(self, w) -> tuple[int, int]:
        s = (pd.Timestamp(w[0]).normalize() - self.t0).days
        e = (pd.Timestamp(w[1]).normalize() - self.t0).days
        if s < 0 or e >= self.T or e < s:
            raise Unavailable(f"{pd.Timestamp(w[0]).date()} to {pd.Timestamp(w[1]).date()} is outside the data")
        return s, e + 1

    def sums(self, w) -> tuple[np.ndarray, np.ndarray]:
        s, e = self._span(w)
        n = self._cn[:, e] - self._cn[:, s]
        d = (self._cd[:, e] - self._cd[:, s]) if self.ratio else np.zeros(len(n))
        return n, d

    def total(self, w) -> tuple[float, float]:
        s, e = self._span(w)
        return float(self._tn[e] - self._tn[s]), (float(self._td[e] - self._td[s]) if self.ratio else float("nan"))

    def positions(self, names) -> np.ndarray:
        lookup = {n: i for i, n in enumerate(self.members)}
        return np.array([lookup[n] for n in names], dtype=int)


def _contrib(panel: Panel, w, c, scale: float):
    n1, d1 = panel.sums(w)
    n0, d0 = panel.sums(c)
    keep = (n1 != 0) | (n0 != 0) | (d1 != 0) | (d0 != 0)
    names = panel.members[keep]
    cur = pd.DataFrame({"member": names, "num": n1[keep], "den": d1[keep] if panel.ratio else np.nan})
    base = pd.DataFrame({"member": names, "num": n0[keep], "den": d0[keep] if panel.ratio else np.nan})
    return contributions(cur, base, panel.total(w), panel.total(c), panel.ratio,
                         1.0 if panel.ratio else scale)


def _moves(panel: Panel, idx: np.ndarray, w, c):
    """(group move, rest move) between baseline c and window w. Relative change for sums and
    money ratios, percentage-point change for % KPIs. None when undefined."""
    try:
        n1, d1 = panel.sums(w)
        n0, d0 = panel.sums(c)
        (TN1, TD1), (TN0, TD0) = panel.total(w), panel.total(c)
    except Unavailable:
        return None
    g1n, g0n = float(n1[idx].sum()), float(n0[idx].sum())
    r1n, r0n = TN1 - g1n, TN0 - g0n
    if not panel.ratio:
        if g0n <= 0 or r0n <= 0:
            return None
        return g1n / g0n - 1.0, r1n / r0n - 1.0
    g1d, g0d = float(d1[idx].sum()), float(d0[idx].sum())
    r1d, r0d = TD1 - g1d, TD0 - g0d
    if min(g1d, g0d, r1d, r0d) <= 0:
        return None
    a1, a0, b1, b0 = g1n / g1d, g0n / g0d, r1n / r1d, r0n / r0d
    if panel.kpi.fmt == "pct":
        return a1 - a0, b1 - b0
    if a0 == 0 or b0 == 0:
        return None
    return a1 / a0 - 1.0, b1 / b0 - 1.0


def _gap(panel, idx, w, c) -> float:
    mv = _moves(panel, idx, w, c)
    return float("nan") if mv is None else mv[0] - mv[1]


def _fmt_move(x: float, pct: bool) -> str:
    return f"{x * 100:+.1f} pts" if pct else f"{x:+.1%}"


def _shift_back(w, days: int):
    return pd.Timestamp(w[0]) - pd.Timedelta(days=days), pd.Timestamp(w[1]) - pd.Timedelta(days=days)


def _mad_sigma(x) -> float:
    a = np.asarray(x, dtype=float)
    return 1.4826 * float(np.median(np.abs(a - np.median(a))))


def _placebo_gaps(panel: Panel, window, n: int, sign: int, min_weight_share: float) -> list[float]:
    """Gaps of the n worst movers (direction `sign`) in ordinary windows of the same length
    lying entirely before `window`, each against the same dates a year earlier. Only members
    whose normal weight is >= `min_weight_share` are eligible (size-matched)."""
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    L = (end - start).days + 1
    step = max(7, L // 4)
    s = panel.t0 + pd.DateOffset(years=1)
    out: list[float] = []
    while s + pd.Timedelta(days=L - 1) < start:
        w = (s, s + pd.Timedelta(days=L - 1))
        c = one_year_earlier(*w)
        s = s + pd.Timedelta(days=step)
        try:
            tn1, _ = panel.total(w)
            tn0, td0 = panel.total(c)
            scale = tn1 / tn0 if (not panel.ratio and tn0 > 0 and tn1 > 0) else 1.0
            rows, _, _, _ = _contrib(panel, w, c, scale)
            n0, d0 = panel.sums(c)
        except (Unavailable, DecomposeError):
            continue
        tot = td0 if panel.ratio else tn0
        if not tot or tot <= 0:
            continue
        weight = dict(zip(panel.members, (d0 if panel.ratio else n0) / tot))
        r = rows[rows["member"] != OTHER_LABEL]
        r = r[[weight.get(m, 0.0) >= min_weight_share for m in r["member"]]]
        if len(r) < n:
            continue
        r = r.assign(_k=sign * r["contribution"]).sort_values("_k", ascending=False, kind="mergesort")
        g = _gap(panel, panel.positions(list(r["member"].iloc[:n])), w, c)
        if np.isfinite(g):
            out.append(g)
    return out


# ----------------------------------------------------------------------------
# Judging one dimension's group
# ----------------------------------------------------------------------------
def _pick_group(rows: pd.DataFrame, change: float, cfg: VerifyConfig) -> pd.DataFrame:
    sign = 1 if change > 0 else -1
    r = rows[(rows["member"] != OTHER_LABEL) & (np.sign(rows["contribution"]) == sign)]
    if r.empty:
        return r
    cum = r["contribution"].cumsum().to_numpy() / change
    hit = np.flatnonzero(cum >= cfg.cover)
    k = int(hit[0]) + 1 if len(hit) else len(r)
    big = np.abs(r["contribution"].to_numpy()) >= cfg.tail_ratio * abs(float(r["contribution"].iloc[0]))
    k = max(k, int(np.sum(np.cumprod(big))))
    return r.head(min(k, cfg.max_group))


def _judge(panel: Panel, window, compare, scale: float, cfg: VerifyConfig, filters: dict) -> SliceVerdict:
    kpi, dim = panel.kpi, panel.dimension
    pct = kpi.fmt == "pct"
    min_gap = cfg.min_gap_points if pct else cfg.min_gap

    def result(verdict, reason, checks, members=(), contribution=float("nan"), share=float("nan"),
               lift=float("nan")):
        fl = {**filters, dim: list(members)} if members else dict(filters)
        return SliceVerdict(dim, list(members), fl, verdict, reason, checks, contribution, share, lift,
                            list(panel.queries))

    try:
        rows, change, _, _ = _contrib(panel, window, compare, scale)
    except (Unavailable, DecomposeError) as err:
        return result(UNKNOWN, f"Cannot compare the windows: {err}", [])
    if not np.isfinite(change) or change == 0:
        return result(UNKNOWN, "There is no change in this slice to explain.", [])
    sign = 1 if change > 0 else -1
    group = _pick_group(rows, change, cfg)
    if group.empty:
        return result(NOT_A_CAUSE, f"No {dim} member moved in the direction of the change.", [])
    members = [str(m) for m in group["member"]]
    idx = panel.positions(members)
    contribution = float(group["contribution"].sum())
    share = contribution / change

    n0, d0 = panel.sums(compare)
    t0n, t0d = panel.total(compare)
    if panel.ratio:
        biz = float(d0[idx].sum()) / t0d if t0d > 0 else float("nan")
    else:
        biz = float(n0[idx].sum()) / t0n if t0n > 0 else float("nan")
    lift = share / biz if biz and np.isfinite(biz) and biz > 0 else float("nan")

    checks: list[Check] = []
    names = ", ".join(members[:5])
    # 1. size and distinctiveness
    if not np.isfinite(lift):
        checks.append(Check("Size and distinctiveness", UNSURE,
                            "The group's normal share of the business cannot be measured.",
                            {"share_of_change": share}))
    else:
        ok = share >= cfg.min_share and lift >= cfg.min_lift
        checks.append(Check(
            "Size and distinctiveness", PASS if ok else FAIL,
            f"{names} explain {share:.0%} of the change but are only {biz:.0%} of the normal business "
            f"({lift:.1f}x their normal weight)." +
            ("" if ok else " That is not distinctive enough: the change is spread across the business."),
            {"share_of_change": share, "normal_share": biz, "lift": lift,
             "min_share": cfg.min_share, "min_lift": cfg.min_lift}))

    # 2. differs from the rest
    mv = _moves(panel, idx, window, compare)
    gap = float("nan")
    if mv is None:
        checks.append(Check("Differs from the rest", UNSURE,
                            "The group or the rest has no usable baseline, so the two cannot be compared.", {}))
    else:
        gap = mv[0] - mv[1]
        ok = abs(gap) >= min_gap and np.sign(gap) == sign
        checks.append(Check(
            "Differs from the rest", PASS if ok else FAIL,
            f"The group moved {_fmt_move(mv[0], pct)} while the rest of the business moved "
            f"{_fmt_move(mv[1], pct)} over the same dates (gap {_fmt_move(gap, pct)}).",
            {"group_move": mv[0], "rest_move": mv[1], "gap": gap, "min_gap": min_gap}))

    # 3. timing
    L = (pd.Timestamp(window[1]) - pd.Timestamp(window[0])).days + 1
    pre_w = _shift_back(window, L)
    pre_gap = _gap(panel, idx, pre_w, one_year_earlier(*pre_w))
    if not np.isfinite(gap) or not np.isfinite(pre_gap):
        checks.append(Check("Started inside the window", UNSURE,
                            "There is no data to compare the period just before the window.", {}))
    else:
        early = np.sign(pre_gap) == sign and abs(pre_gap) > cfg.pre_ratio * abs(gap)
        checks.append(Check(
            "Started inside the window", FAIL if early else PASS,
            f"In the {L} days just before the window the gap was {_fmt_move(pre_gap, pct)} "
            f"(inside the window: {_fmt_move(gap, pct)})." +
            (" The difference had already started earlier." if early else ""),
            {"pre_gap": pre_gap, "gap": gap}))

    # 4. recurrence / rebound
    prev_w = one_year_earlier(*window)
    prev_gap = _gap(panel, idx, prev_w, one_year_earlier(*prev_w))
    if not np.isfinite(gap) or not np.isfinite(prev_gap):
        checks.append(Check("Not a yearly repeat", UNSURE,
                            "There is no earlier year to check whether this repeats.", {}))
    else:
        big_prev = abs(prev_gap) >= cfg.recur_ratio * abs(gap) and abs(prev_gap) >= min_gap / 2
        repeat = bool(big_prev and np.sign(prev_gap) == sign)
        rebound = bool(big_prev and np.sign(prev_gap) != sign)
        checks.append(Check(
            "Not a yearly repeat", FAIL if (repeat or rebound) else PASS,
            f"The same window a year earlier showed a gap of {_fmt_move(prev_gap, pct)}." +
            (" It repeats every year, so it looks seasonal." if repeat else "") +
            (" Last year's window was abnormal in the opposite direction, so this looks like a "
             "rebound from an odd base year, not a new cause." if rebound else ""),
            {"previous_year_gap": prev_gap, "gap": gap}))

    # 5. placebo
    placebo: list[float] = []
    if np.isfinite(gap):
        tot_w = t0d if panel.ratio else t0n
        wts = (d0 if panel.ratio else n0)[idx] / tot_w if tot_w > 0 else np.array([0.0])
        placebo = _placebo_gaps(panel, window, len(members), sign, cfg.size_match * float(np.min(wts)))
    if not np.isfinite(gap):
        checks.append(Check("Unusual for its own history", UNSURE, "No gap to test.", {}))
    elif len(placebo) < cfg.min_placebo:
        checks.append(Check("Unusual for its own history", UNSURE,
                            f"Only {len(placebo)} ordinary past windows are available (need {cfg.min_placebo}); "
                            f"cannot tell whether this gap is unusual.", {"n_placebo": len(placebo)}))
    else:
        center = float(np.median(placebo))
        sigma = max(_mad_sigma(placebo), min_gap / 5)
        z = (gap - center) / sigma
        ok = z * sign >= cfg.placebo_z
        checks.append(Check(
            "Unusual for its own history", PASS if ok else FAIL,
            f"In {len(placebo)} ordinary past windows the same kind of group had a typical gap of "
            f"{_fmt_move(center, pct)} (spread {_fmt_move(sigma, pct)}); this gap is {abs(z):.1f} spreads away." +
            ("" if ok else " Swings this size are normal for this slice, so it may be luck."),
            {"n_placebo": len(placebo), "placebo_median": center, "placebo_sigma": sigma, "z": z,
             "placebo_gaps": [float(x) for x in placebo]}))

    by = {c.name: c for c in checks}
    for name in ("Size and distinctiveness", "Differs from the rest", "Started inside the window",
                 "Not a yearly repeat"):
        if by[name].status == FAIL:
            return result(NOT_A_CAUSE, by[name].detail, checks, members, contribution, share, lift)
    unsure = [c for c in checks if c.status == UNSURE]
    if by["Unusual for its own history"].status == FAIL:
        return result(UNKNOWN, by["Unusual for its own history"].detail, checks, members, contribution, share, lift)
    if unsure:
        return result(UNKNOWN, unsure[0].detail, checks, members, contribution, share, lift)
    return result(CAUSE, f"All checks passed: {names} moved differently from the rest of the business, "
                  "starting in this window, and by far more than ordinary swings.",
                  checks, members, contribution, share, lift)


# ----------------------------------------------------------------------------
# Whole windows, with drill-down
# ----------------------------------------------------------------------------
def _dims_to_try(ds: Dataset, filters: dict) -> list[str]:
    return [d for d in ds.schema.dimensions if d not in filters and ds.schema.cardinality.get(d, 2) >= 2]


def _verify_level(ds, kpi, window, compare, scale, cfg, filters, depth) -> list[SliceVerdict]:
    out = []
    for dim in _dims_to_try(ds, filters):
        panel = Panel(ds, kpi, dim, filters)
        if len(panel.members) < 2:
            continue
        v = _judge(panel, window, compare, scale, cfg, filters)
        if v.verdict == CAUSE and depth < cfg.max_depth:
            v.children = [c for c in _verify_level(ds, kpi, window, compare, scale, cfg, v.filters, depth + 1)
                          if c.verdict == CAUSE or c.children]
        out.append(v)
    return out


def _leaves(v: SliceVerdict, path: list[SliceVerdict]):
    here = path + [v]
    kids = [c for c in v.children if c.verdict == CAUSE]
    if not kids:
        yield here
    for k in kids:
        yield from _leaves(k, here)


def _slice_signature(ds: Dataset, kpi: KPI, window, compare, filters: dict):
    """Fingerprint of the rows in a slice, so State=Texas and City=Texas Metro (same rows) are one cause."""
    (a, b), (c, d) = _total(ds, kpi, *window, filters)[0], _total(ds, kpi, *compare, filters)[0]
    return tuple(round(float(x), 6) if x == x else None for x in (a, b, c, d))


def _slice_contribution(ds, kpi, w, c, filters, scale) -> float:
    """Contribution of a whole slice (all filters AND-ed) to the KPI change, treating
    the slice and the rest as two members."""
    (sn1, sd1), (sn0, sd0) = _total(ds, kpi, *w, filters)[0], _total(ds, kpi, *c, filters)[0]
    (tn1, td1), (tn0, td0) = _total(ds, kpi, *w, None)[0], _total(ds, kpi, *c, None)[0]
    ratio = kpi.den is not None
    cur = pd.DataFrame({"member": ["slice", "rest"], "num": [sn1, tn1 - sn1],
                        "den": [sd1, td1 - sd1] if ratio else np.nan})
    base = pd.DataFrame({"member": ["slice", "rest"], "num": [sn0, tn0 - sn0],
                         "den": [sd0, td0 - sd0] if ratio else np.nan})
    rows, *_ = contributions(cur, base, (tn1, td1), (tn0, td0), ratio, 1.0 if ratio else scale)
    return float(rows.loc[rows["member"] == "slice", "contribution"].iloc[0])


def _intersect(f1: dict, f2: dict):
    out = {k: list(v) for k, v in f1.items()}
    for k, v in f2.items():
        if k in out:
            both = [m for m in out[k] if m in v]
            if not both:
                return None
            out[k] = both
        else:
            out[k] = list(v)
    return out


def _prune_overlaps(ds, kpi, w, c, causes: list[Cause], scale: float) -> list[Cause]:
    """Several drill-down paths often describe the same thing (West+Phones, West+Phones Max...).
    Keep the biggest; fold a smaller one into it when most of it lies inside the bigger one."""
    kept: list[Cause] = []
    for cause in sorted(causes, key=lambda x: -abs(x.contribution)):
        for k in kept:
            inter = _intersect(cause.filters, k.filters)
            if inter is None:
                continue
            ci = _slice_contribution(ds, kpi, w, c, inter, scale)
            if np.sign(ci) == np.sign(cause.contribution) and abs(ci) >= 0.7 * abs(cause.contribution):
                k.also_described_as.append(cause.describe())
                break
        else:
            kept.append(cause)
    return kept


def _prior_growth(top, w, cfg: VerifyConfig):
    """Growth of a sum KPI over the 12 months just BEFORE the window versus the same 12 months a year
    earlier (so the window itself, which may contain the problem, never influences it). None if the data
    does not reach back far enough or the baseline is empty."""
    try:
        pre = (w[0] - pd.DateOffset(years=1), w[0] - pd.Timedelta(days=1))
        pre0 = one_year_earlier(*pre)
        a, b = top.total(pre)[0], top.total(pre0)[0]
    except Unavailable:
        return None
    if not (b > 0 and a > 0):
        return None
    lo, hi = cfg.growth_clip
    return float(min(hi, max(lo, a / b)))


def verify_window(ds: Dataset, kpi_key: str, window: tuple, compare: tuple | None = None,
                  filters: dict | None = None, expected_total: float | None = None,
                  config: VerifyConfig | None = None) -> WindowResult:
    """Test every top contributor of a KPI change over `window` and report the verified causes
    (leaf slices). `compare` defaults to the same dates one year earlier. `expected_total`
    (sum KPIs) removes ordinary growth, as in decompose.decompose_event."""
    cfg = config or VerifyConfig()
    kpi = get_kpi(ds, kpi_key)
    w = (pd.Timestamp(window[0]), pd.Timestamp(window[1]))
    c = (pd.Timestamp(compare[0]), pd.Timestamp(compare[1])) if compare else one_year_earlier(*w)
    filters = dict(filters or {})
    notes: list[str] = []

    top = Panel(ds, kpi, ds.schema.dimensions[0], filters)
    try:
        t1, t0 = top.total(w), top.total(c)
    except Unavailable as err:
        raise DecomposeError(f"Cannot verify: {err}.") from err
    scale = 1.0
    if kpi.den is None and expected_total is not None and t0[0] > 0 and expected_total > 0:
        scale = float(expected_total) / t0[0]
    elif kpi.den is None and expected_total is not None:
        notes.append("Baseline or expected total is not positive; compared with last year directly.")
    if kpi.den:
        if not (t1[1] > 0 and t0[1] > 0):
            raise DecomposeError("The ratio is undefined in one of the two windows.")
        change = t1[0] / t1[1] - t0[0] / t0[1]
    else:
        change = t1[0] - t0[0] * scale
    if kpi.den is not None and kpi.fmt == "pct":
        tiny = abs(change) < 0.005
    elif kpi.den is not None:
        tiny = abs(change) < cfg.min_change * abs(t0[0] / t0[1])
    else:
        tiny = abs(change) < cfg.min_change * abs(t0[0] * scale)
    if tiny and kpi.den is None and expected_total is None and cfg.auto_growth and t0[0] > 0:
        # Flat versus last year can HIDE a loss: the business grew, and the loss cancelled the growth.
        # Only in that case, compare with last year's value grown by the ordinary growth of the
        # 12 months before the window. (Windows that already show a clear change are left as they are.)
        g = _prior_growth(top, w, cfg)
        if g is not None:
            adj = t1[0] - t0[0] * g
            if abs(adj) >= cfg.min_change * abs(t0[0] * g):
                scale, change, tiny = g, adj, False
                notes.append("This window looks flat against last year, but the business was growing, so "
                             "ordinary growth over the previous year was removed from the comparison.")
    if kpi.den is None and expected_total is None and scale == 1.0:
        notes.append("No expected value was given, so the window is compared with last year directly "
                     "(ordinary growth counts as change). Alerts from detect.py pass their expected value.")

    if tiny:
        return WindowResult(kpi.key, kpi.label, kpi.fmt, w, c, float(change), UNKNOWN,
                            "The KPI barely changed in this window, so there is nothing meaningful to explain.",
                            [], [], notes)

    cands = _verify_level(ds, kpi, w, c, scale, cfg, filters, 1)
    cands.sort(key=lambda v: (v.verdict != CAUSE,
                              -(abs(v.share_of_change) if np.isfinite(v.share_of_change) else 0)))

    causes: list[Cause] = []
    seen: set = set()
    for v in cands:
        if v.verdict != CAUSE:
            continue
        for chain in _leaves(v, []):
            leaf = chain[-1]
            sig = _slice_signature(ds, kpi, w, c, leaf.filters)
            if sig in seen:
                continue
            seen.add(sig)
            contrib = _slice_contribution(ds, kpi, w, c, leaf.filters, scale)
            causes.append(Cause(leaf.filters, contrib / change if change else float("nan"), contrib, chain))
    causes = _prune_overlaps(ds, kpi, w, c, causes, scale)
    causes.sort(key=lambda x: -(abs(x.share_of_window_change) if np.isfinite(x.share_of_window_change) else 0))

    if causes:
        conclusion = CAUSE
        total_share = min(1.0, sum(abs(x.share_of_window_change) for x in causes))
        reason = (f"{len(causes)} verified cause(s) explain about {total_share:.0%} of the change. "
                  f"The evidence supports them; it does not prove them.")
    else:
        conclusion = UNKNOWN
        reasons = [v.reason for v in cands if v.verdict in (UNKNOWN, NOT_A_CAUSE)]
        reason = ("No region, product, customer or other slice passed all the checks, so the cause of this "
                  "change is UNKNOWN from the data in this file." + (f" Example: {reasons[0]}" if reasons else ""))
    return WindowResult(kpi.key, kpi.label, kpi.fmt, w, c, float(change), conclusion, reason, causes, cands, notes)


def verify_event(ds: Dataset, event, filters: dict | None = None,
                 config: VerifyConfig | None = None) -> WindowResult:
    """Verify a detect.ChangeEvent over its own window and compare window."""
    kpi = get_kpi(ds, event.kpi)
    expected = event.expected if kpi.den is None else None
    return verify_window(ds, event.kpi, (event.start, event.end), (event.compare_start, event.compare_end),
                         filters, expected, config)


def verify_subwindows(ds: Dataset, kpi_key: str, windows: list[tuple], filters: dict | None = None,
                      config: VerifyConfig | None = None) -> list[WindowResult]:
    """Verify several sub-windows separately (each against the same dates a year earlier)."""
    return [verify_window(ds, kpi_key, w, None, filters, None, config) for w in windows]


def verify_by_month(ds: Dataset, kpi_key: str, start, end, filters: dict | None = None,
                    config: VerifyConfig | None = None) -> list[WindowResult]:
    """verify_subwindows over every calendar month of [start, end]."""
    return verify_subwindows(ds, kpi_key, monthly_windows(start, end), filters, config)


# ----------------------------------------------------------------------------
# JSON-safe export (for the API and the LLM evidence records)
# ----------------------------------------------------------------------------
def to_plain(obj):
    """Turn results (dataclasses, numpy numbers, timestamps) into plain JSON-able Python.
    Non-finite numbers become None so the JSON is valid."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
