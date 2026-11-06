"""Decomposition: WHO contributed to a KPI change?

The idea, in plain words
------------------------
detect.py tells us "revenue fell in Jun-Nov 2024". This module answers the next
question: "which region / product / customer is that fall made of?"

We compare two date windows:
    - the CURRENT window (e.g. the flagged episode), and
    - a BASELINE window (normally the same dates one year earlier).
For one dimension (say Region) we split the KPI into its members (East, West...)
and give each member a CONTRIBUTION. The contributions add up EXACTLY to the
total change. (An explicit "(overlap / other)" row soaks up any leftover, see below.)

Three kinds of KPI, three formulas
----------------------------------
1. Sums (revenue, profit, units):   contribution_i = current_i - baseline_i.
2. Ratios (profit margin, discount, order value), R = N / D:
       contribution_i = (n1_i - R0 * d1_i) / D1  -  (n0_i - R0 * d0_i) / D0
   where R0 is the OLD overall ratio, n/d are the member's numerator/denominator,
   and D1, D0 are the overall denominators. These add up exactly to R1 - R0.
   Plain meaning: a member pulls the margin down when its profit is low compared
   with what the OLD margin would give it, weighted by how big it is now; then we
   subtract the same thing for the old window.
3. Counts of distinct things (orders): an order can touch several members (an
   order with phones AND chairs), so member counts may not add up to the total.
   The difference is shown as a visible "(overlap / other)" row, never hidden.

Baseline choices
----------------
"last_year": baseline = the same dates one year earlier (always available).
"expected" (decompose_event, sum KPIs only): baseline = last year's values scaled
    so their total equals what detect.py EXPECTED for the window. This removes
    ordinary growth, so contributions add up to the event's `impact`
    (observed - expected) and uniform growth does not look like a "cause".

Honest limits
-------------
  - A big contribution means "this is where the change happened". It does NOT
    prove cause. verify.py (next step) tests whether it is a real cause.
  - The members are one dimension at a time. Use `filters` to drill down
    (e.g. Region=Central, then decompose by Customer Name).
  - Contributions are about the window you give. If several causes overlap in
    time (B, U, C), decompose the sub-windows separately (decompose_subwindows).

No LLM is used here. Every row carries the SQL that produced it (`queries`),
so the Root Cause screen can show the evidence behind each number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.ingest import TABLE, Dataset, q
from engine.kpi import KPI, build_where, format_value, get_kpi

OTHER_LABEL = "(overlap / other)"
BLANK_LABEL = "(blank)"


class DecomposeError(ValueError):
    """Raised when a decomposition is impossible. The message is for the user."""


@dataclass
class Decomposition:
    kpi: str
    label: str
    fmt: str
    dimension: str
    window: tuple[pd.Timestamp, pd.Timestamp]
    compare: tuple[pd.Timestamp, pd.Timestamp]
    is_ratio: bool
    baseline_kind: str            # "last_year" or "expected"
    cur_total: float              # KPI over the current window
    base_total: float             # KPI over the baseline window (scaled if "expected")
    change: float                 # cur_total - base_total  (points for % KPIs)
    rows: pd.DataFrame            # see `contributions` for the columns
    residual: float               # the leftover shown on the "(overlap / other)" row
    filters: dict = field(default_factory=dict)
    queries: list[dict] = field(default_factory=list)   # evidence: SQL + parameters
    notes: list[str] = field(default_factory=list)

    @property
    def members(self) -> pd.DataFrame:
        """Rows without the overlap/other row."""
        return self.rows[self.rows["member"] != OTHER_LABEL]

    def top(self, n: int = 5, direction: str | None = None) -> pd.DataFrame:
        """Biggest contributors. direction='down'/'up' keeps only members that
        pushed the KPI that way."""
        r = self.members
        if direction == "down":
            r = r[r["contribution"] < 0]
        elif direction == "up":
            r = r[r["contribution"] > 0]
        return r.head(n)

    def top_k_share(self, k: int = 3) -> float:
        """Share of the total change explained by the k biggest members that
        moved WITH the change. 1.0 = those k members are the whole story.
        Can exceed 1 if other members pushed the opposite way."""
        if not self.change or not np.isfinite(self.change):
            return float("nan")
        sign = 1.0 if self.change > 0 else -1.0
        r = self.members
        r = r[np.sign(r["contribution"]) == sign].head(k)
        return float(r["contribution"].sum() / self.change)

    def summary(self, n: int = 3) -> str:
        pts = self.fmt == "pct"
        chg = f"{self.change * 100:+.1f} pts" if pts else format_value(self.change, self.fmt)
        lines = [f"{self.label} by {self.dimension}: {chg} "
                 f"({format_value(self.base_total, self.fmt)} -> {format_value(self.cur_total, self.fmt)})"]
        for _, r in self.top(n).iterrows():
            c = f"{r['contribution'] * 100:+.1f} pts" if pts else format_value(r["contribution"], self.fmt)
            lines.append(f"  {r['member']}: {c}  ({r['share']:.0%} of change)"
                         if np.isfinite(r["share"]) else f"  {r['member']}: {c}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# The maths (pure pandas: no database, easy to test)
# ----------------------------------------------------------------------------
def contributions(cur: pd.DataFrame, base: pd.DataFrame, cur_total: tuple[float, float],
                  base_total: tuple[float, float], ratio: bool,
                  base_scale: float = 1.0) -> tuple[pd.DataFrame, float, float, float]:
    """Split a KPI change into member contributions.

    cur / base : columns `member`, `num`, `den` (one row per member; den may be NaN for sums)
    cur_total / base_total : (num, den) over the WHOLE window (not the sum of the members)
    ratio      : True for N/D KPIs
    base_scale : multiplies the baseline numerators (sum KPIs only; used for "expected")

    Returns (rows, total_change, cur_value, base_value). `rows` has columns:
    member, cur_num, base_num, cur_den, base_den, cur_value, base_value,
    contribution, share, direction. A "(overlap / other)" row is added only when
    the members do not add up to the total (e.g. distinct orders spanning members).
    """
    if ratio and base_scale != 1.0:
        raise DecomposeError("A scaled baseline is only defined for sum KPIs.")
    members = pd.Index(cur["member"]).union(pd.Index(base["member"]), sort=False)
    c = cur.set_index("member").reindex(members)
    b = base.set_index("member").reindex(members)
    n1, n0 = c["num"].fillna(0.0).to_numpy(float), b["num"].fillna(0.0).to_numpy(float) * base_scale
    d1, d0 = c["den"].fillna(0.0).to_numpy(float), b["den"].fillna(0.0).to_numpy(float)

    N1, D1 = cur_total
    N0, D0 = base_total
    N0 = (0.0 if N0 is None or pd.isna(N0) else float(N0)) * base_scale
    N1 = 0.0 if N1 is None or pd.isna(N1) else float(N1)

    if ratio:
        D1 = 0.0 if D1 is None or pd.isna(D1) else float(D1)
        D0 = 0.0 if D0 is None or pd.isna(D0) else float(D0)
        if D1 == 0 or D0 == 0:
            raise DecomposeError("The ratio is undefined in one of the two windows "
                                 "(no rows, or the denominator is zero).")
        R1, R0 = N1 / D1, N0 / D0
        total = R1 - R0
        contrib = (n1 - R0 * d1) / D1 - (n0 - R0 * d0) / D0
        cur_value_total, base_value_total = R1, R0
    else:
        total = N1 - N0
        contrib = n1 - n0
        cur_value_total, base_value_total = N1, N0

    with np.errstate(divide="ignore", invalid="ignore"):
        cur_v = np.where(d1 != 0, n1 / d1, np.nan) if ratio else n1
        base_v = np.where(d0 != 0, n0 / d0, np.nan) if ratio else n0

    rows = pd.DataFrame({
        "member": [BLANK_LABEL if (m is None or (isinstance(m, float) and math.isnan(m))) else str(m)
                   for m in members],
        "cur_num": n1, "base_num": n0, "cur_den": d1, "base_den": d0,
        "cur_value": cur_v, "base_value": base_v, "contribution": contrib,
    })

    residual = float(total - contrib.sum())
    tol = 1e-9 * max(1.0, abs(total), float(np.abs(contrib).sum()))
    if abs(residual) > tol:
        rows.loc[len(rows)] = {"member": OTHER_LABEL, "cur_num": np.nan, "base_num": np.nan,
                               "cur_den": np.nan, "base_den": np.nan, "cur_value": np.nan,
                               "base_value": np.nan, "contribution": residual}
    else:
        residual = 0.0

    rows["share"] = rows["contribution"] / total if total else np.nan
    rows["direction"] = np.where(rows["contribution"] < 0, "down",
                                 np.where(rows["contribution"] > 0, "up", "flat"))
    is_other = rows["member"] == OTHER_LABEL
    rows = pd.concat([rows[~is_other].sort_values("contribution", key=np.abs, ascending=False,
                                                  kind="mergesort"),
                      rows[is_other]], ignore_index=True)
    return rows, float(total), float(cur_value_total), float(base_value_total)


# ----------------------------------------------------------------------------
# The database layer
# ----------------------------------------------------------------------------
def _den_sql(kpi: KPI) -> str:
    return kpi.den if kpi.den else "NULL"


def _grouped(ds: Dataset, kpi: KPI, dim: str, start, end, filters):
    where, params = build_where(ds, filters, start, end)
    sql = (f"SELECT COALESCE(CAST({q(dim)} AS VARCHAR), '{BLANK_LABEL}') AS member, "
           f"{kpi.num} AS num, {_den_sql(kpi)} AS den FROM {TABLE}{where} GROUP BY 1")
    df = ds.query(sql, params)
    df["num"] = pd.to_numeric(df["num"])
    df["den"] = pd.to_numeric(df["den"]) if kpi.den else np.nan
    return df, {"sql": sql, "params": [str(p) for p in params]}


def _total(ds: Dataset, kpi: KPI, start, end, filters):
    where, params = build_where(ds, filters, start, end)
    sql = f"SELECT {kpi.num} AS num, {_den_sql(kpi)} AS den FROM {TABLE}{where}"
    row = ds.query(sql, params).iloc[0]
    num = float(row["num"]) if not pd.isna(row["num"]) else 0.0
    den = float(row["den"]) if (kpi.den and not pd.isna(row["den"])) else (0.0 if kpi.den else np.nan)
    return (num, den), {"sql": sql, "params": [str(p) for p in params]}


def one_year_earlier(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The same calendar dates one year earlier."""
    off = pd.DateOffset(years=1)
    return pd.Timestamp(start) - off, pd.Timestamp(end) - off


def decompose(ds: Dataset, kpi_key: str, dimension: str, window: tuple, compare: tuple | None = None,
              filters: dict | None = None, expected_total: float | None = None) -> Decomposition:
    """Split the change in a KPI between `compare` (baseline) and `window` into
    contributions by the members of `dimension`.

    window / compare : (start, end) dates, inclusive. compare defaults to the same
                       dates one year earlier.
    filters          : optional drill-down, e.g. {"Region": "Central"}.
    expected_total   : sum KPIs only. If given, the baseline is last year's values
                       scaled so their total equals this number (growth removed).
    """
    kpi = get_kpi(ds, kpi_key)
    if dimension not in ds.schema.dimensions:
        raise DecomposeError(f"'{dimension}' is not a known category column. "
                             f"Known: {ds.schema.dimensions}")
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    if end < start:
        raise DecomposeError("The window ends before it starts.")
    cstart, cend = (pd.Timestamp(compare[0]), pd.Timestamp(compare[1])) if compare else one_year_earlier(start, end)
    ratio = kpi.den is not None
    notes: list[str] = []

    cur_g, q1 = _grouped(ds, kpi, dimension, start, end, filters)
    base_g, q2 = _grouped(ds, kpi, dimension, cstart, cend, filters)
    cur_t, q3 = _total(ds, kpi, start, end, filters)
    base_t, q4 = _total(ds, kpi, cstart, cend, filters)
    if cur_g.empty and base_g.empty:
        raise DecomposeError("No rows in either window (check the dates and filters).")

    scale, kind = 1.0, "last_year"
    if expected_total is not None:
        if ratio:
            notes.append("A growth-adjusted baseline is only used for sum KPIs; "
                         "this ratio KPI is compared with last year directly.")
        elif base_t[0] > 0 and expected_total > 0:
            scale, kind = float(expected_total) / base_t[0], "expected"
        else:
            notes.append("Baseline or expected total is not positive, so the growth-adjusted "
                         "baseline was skipped; compared with last year directly.")

    rows, total, cur_value, base_value = contributions(cur_g, base_g, cur_t, base_t, ratio, scale)
    residual = float(rows.loc[rows["member"] == OTHER_LABEL, "contribution"].sum())
    if residual:
        notes.append("Members do not add up to the total (distinct orders can belong to several "
                     "members); the leftover is shown as the overlap/other row.")
    if len(cur_g) == 1 and len(base_g) == 1 and not rows.empty:
        notes.append("Only one member in this slice; nothing to compare.")
    return Decomposition(
        kpi=kpi.key, label=kpi.label, fmt=kpi.fmt, dimension=dimension,
        window=(start, end), compare=(cstart, cend), is_ratio=ratio, baseline_kind=kind,
        cur_total=cur_value, base_total=base_value, change=total, rows=rows, residual=residual,
        filters=dict(filters or {}), queries=[q1, q2, q3, q4], notes=notes)


def decompose_event(ds: Dataset, event, dimension: str, filters: dict | None = None) -> Decomposition:
    """Decompose a detect.ChangeEvent over its own window and compare window.
    For sum KPIs the baseline is growth-adjusted so contributions add up to
    event.impact (observed - expected)."""
    expected = event.expected if get_kpi(ds, event.kpi).den is None else None
    return decompose(ds, event.kpi, dimension, (event.start, event.end),
                     (event.compare_start, event.compare_end), filters, expected)


def decompose_subwindows(ds: Dataset, kpi_key: str, dimension: str, windows: list[tuple],
                         filters: dict | None = None) -> list[Decomposition]:
    """Decompose several sub-windows separately (each against the same dates one
    year earlier). Use when causes overlap in time, e.g. Jun-Aug, Sep, 15 Oct-30 Nov."""
    return [decompose(ds, kpi_key, dimension, w, None, filters) for w in windows]


def monthly_windows(start, end) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Calendar-month windows covering [start, end] (first/last month clipped)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for p in pd.period_range(start, end, freq="M"):
        out.append((max(p.start_time.normalize(), start), min(p.end_time.normalize(), end)))
    return out


def rank_dimensions(ds: Dataset, kpi_key: str, window: tuple, compare: tuple | None = None,
                    filters: dict | None = None, top_k: int = 3,
                    expected_total: float | None = None) -> pd.DataFrame:
    """Which dimension 'explains' the change best? For every category column we
    take the top_k members and measure:
        top_k_share   = their share of the total change
        concentration = top_k_share / (top_k / n_members)
    The second number is "how much more than an even spread". A dimension with only
    3 members always has top_k_share ~ 100% but concentration ~ 1 (nothing special);
    one customer out of 400 carrying 30% of the change has a very high concentration.
    Dimensions are sorted by concentration, highest first. Dimensions with a single
    member, or already fixed by `filters`, are skipped.
    This is a heuristic for where to LOOK first (many-member columns score higher
    for the same pattern); it is not evidence of cause."""
    out = []
    fixed = set((filters or {}).keys())
    for dim in ds.schema.dimensions:
        if dim in fixed or ds.schema.cardinality.get(dim, 2) < 2:
            continue
        try:
            d = decompose(ds, kpi_key, dim, window, compare, filters, expected_total)
        except DecomposeError:
            continue
        m = d.members
        n = int(len(m))
        share = d.top_k_share(top_k)
        even = min(1.0, top_k / n) if n else float("nan")
        out.append({"dimension": dim, "n_members": n, "top_k_share": share,
                    "concentration": share / even if n and np.isfinite(share) else float("nan"),
                    "top_member": m["member"].iloc[0] if n else None,
                    "top_contribution": float(m["contribution"].iloc[0]) if n else float("nan")})
    cols = ["dimension", "n_members", "top_k_share", "concentration", "top_member", "top_contribution"]
    df = pd.DataFrame(out, columns=cols)
    return df.sort_values("concentration", ascending=False, na_position="last").reset_index(drop=True)
