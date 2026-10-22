"""KPI definitions and time series.

A KPI is a number we track over time: revenue, profit, orders, profit margin...
Each KPI is described by two SQL pieces:
    value = numerator / denominator       (denominator is None for plain sums)
We keep numerator and denominator SEPARATE because later steps (the root-cause
engine) need them to split a ratio change (like margin) into its parts.

All math happens here in Python/SQL. No LLM ever calculates a number.

Try it from the project folder:
    python -m engine.kpi data/Sample_-_Superstore.csv
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.ingest import TABLE, Dataset, IngestError, load_dataset, q


@dataclass(frozen=True)
class KPI:
    key: str
    label: str
    num: str                 # SQL for the numerator (or the whole value)
    den: str | None          # SQL for the denominator; None means plain sum/count
    fmt: str                 # "money", "int", "pct"
    needs: tuple[str, ...]   # standard columns that must exist
    additive: bool           # True if parts add up to the total (sums)


_ALL_KPIS = [
    KPI("revenue", "Revenue", "SUM(revenue)", None, "money", ("revenue",), True),
    KPI("profit", "Profit", "SUM(profit)", None, "money", ("profit",), True),
    KPI("orders", "Orders", "COUNT(DISTINCT order_id)", None, "int", ("order_id",), False),
    KPI("quantity", "Units sold", "SUM(quantity)", None, "int", ("quantity",), True),
    KPI("profit_margin", "Profit margin", "SUM(profit)", "SUM(revenue)", "pct",
        ("profit", "revenue"), False),
    KPI("avg_order_value", "Average order value", "SUM(revenue)",
        "COUNT(DISTINCT order_id)", "money", ("revenue", "order_id"), False),
    KPI("avg_discount", "Average discount", "SUM(discount)", "COUNT(discount)",
        "pct", ("discount",), False),
]
# If a file has no order id column we still count rows so "orders" is available.
_ROWCOUNT_KPI = KPI("orders", "Transactions", "COUNT(*)", None, "int", (), True)

FREQ = {  # name -> (DuckDB date_trunc unit, pandas range frequency, pandas period)
    "D": ("day", "D", "D"),
    "W": ("week", "W-MON", "W-SUN"),
    "M": ("month", "MS", "M"),
    "Q": ("quarter", "QS", "Q"),
}


def _columns(ds: Dataset) -> set[str]:
    s = ds.schema
    cols = {"revenue"}
    for role in ("profit", "quantity", "discount", "order_id"):
        if getattr(s, role):
            cols.add(role)
    return cols


def available_kpis(ds: Dataset) -> list[KPI]:
    cols = _columns(ds)
    out = [k for k in _ALL_KPIS if set(k.needs) <= cols]
    if "order_id" not in cols:
        out.insert(0, _ROWCOUNT_KPI)
    return out


def get_kpi(ds: Dataset, key: str) -> KPI:
    for k in available_kpis(ds):
        if k.key == key:
            return k
    raise IngestError(f"KPI '{key}' is not available for this file. "
                      f"Available: {[k.key for k in available_kpis(ds)]}")


# ----------------------------------------------------------------------------
# Filters (safe: column names must be known dimensions, values are parameters)
# ----------------------------------------------------------------------------
def build_where(ds: Dataset, filters: dict | None = None,
                start=None, end=None) -> tuple[str, list]:
    clauses, params = [], []
    if start is not None:
        clauses.append("order_date >= ?")
        params.append(pd.Timestamp(start).date())
    if end is not None:
        clauses.append("order_date <= ?")
        params.append(pd.Timestamp(end).date())
    for col, val in (filters or {}).items():
        if col not in ds.schema.dimensions:
            raise IngestError(f"'{col}' is not a known category column. "
                              f"Known: {ds.schema.dimensions}")
        vals = list(val) if isinstance(val, (list, tuple, set)) else [val]
        clauses.append(f"{q(col)} IN ({', '.join('?' for _ in vals)})")
        params.extend(str(v) for v in vals)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


# ----------------------------------------------------------------------------
# Time series
# ----------------------------------------------------------------------------
def kpi_series(ds: Dataset, key: str, freq: str = "M", filters: dict | None = None,
               start=None, end=None) -> pd.DataFrame:
    """One row per period with columns: period, value, num, den.
    Missing periods are filled in (sums become 0, ratios become NaN)."""
    if freq not in FREQ:
        raise ValueError(f"freq must be one of {list(FREQ)}")
    kpi = get_kpi(ds, key)
    unit, range_freq, _ = FREQ[freq]
    where, params = build_where(ds, filters, start, end)
    den_sql = kpi.den if kpi.den else "NULL"
    sql = (f"SELECT date_trunc('{unit}', order_date) AS period, "
           f"{kpi.num} AS num, {den_sql} AS den FROM {TABLE}{where} "
           f"GROUP BY 1 ORDER BY 1")
    df = ds.query(sql, params)
    if df.empty:
        return pd.DataFrame(columns=["period", "value", "num", "den"])
    df["period"] = pd.to_datetime(df["period"])
    full = pd.date_range(df["period"].min(), df["period"].max(), freq=range_freq)
    df = df.set_index("period").reindex(full)
    df.index.name = "period"
    df["num"] = pd.to_numeric(df["num"])
    if kpi.den:
        df["den"] = pd.to_numeric(df["den"])
        df["value"] = np.where(df["den"].fillna(0) != 0, df["num"] / df["den"], np.nan)
    else:
        df["num"] = df["num"].fillna(0.0)
        df["den"] = np.nan
        df["value"] = df["num"]
    return df.reset_index()[["period", "value", "num", "den"]]


def kpi_value(ds: Dataset, key: str, start, end, filters: dict | None = None) -> float:
    """The KPI over an arbitrary date window [start, end] (inclusive)."""
    kpi = get_kpi(ds, key)
    where, params = build_where(ds, filters, start, end)
    den_sql = kpi.den if kpi.den else "NULL"
    row = ds.query(f"SELECT {kpi.num} AS num, {den_sql} AS den FROM {TABLE}{where}", params).iloc[0]
    num = 0.0 if pd.isna(row["num"]) else float(row["num"])
    if kpi.den:
        den = row["den"]
        return float("nan") if pd.isna(den) or den == 0 else num / float(den)
    return num


# ----------------------------------------------------------------------------
# Headline numbers for the dashboard cards
# ----------------------------------------------------------------------------
def _is_complete(period_start: pd.Timestamp, freq: str, max_date: pd.Timestamp) -> bool:
    _, _, pperiod = FREQ[freq]
    end = pd.Period(period_start, freq=pperiod).end_time.normalize()
    return max_date >= end


def headline(ds: Dataset, key: str, freq: str = "M") -> dict:
    """Latest COMPLETE period vs the one before it (and vs the same period
    last year, if the data goes back that far)."""
    kpi = get_kpi(ds, key)
    s = kpi_series(ds, key, freq)
    if s.empty:
        raise IngestError("No data for this KPI.")
    if not _is_complete(s["period"].iloc[-1], freq, ds.max_date) and len(s) > 1:
        s = s.iloc[:-1]
    s = s.set_index("period")
    latest_p = s.index[-1]
    latest = float(s["value"].iloc[-1])
    prev = float(s["value"].iloc[-2]) if len(s) > 1 else float("nan")

    yoy_p = latest_p - pd.DateOffset(years=1)
    yoy = float(s.loc[yoy_p, "value"]) if yoy_p in s.index else float("nan")

    def change(a, b):
        if kpi.fmt == "pct":  # for ratios, report the point change
            return a - b
        return (a - b) / abs(b) if b and not np.isnan(b) else float("nan")

    return {
        "key": key, "label": kpi.label, "fmt": kpi.fmt, "freq": freq,
        "period": latest_p, "value": latest,
        "previous": prev, "change_vs_previous": change(latest, prev),
        "year_ago": yoy, "change_vs_year_ago": change(latest, yoy),
        "change_is_points": kpi.fmt == "pct",
    }


def format_value(value: float, fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if fmt == "money":
        return f"{value:,.0f}"
    if fmt == "pct":
        return f"{value:.1%}"
    return f"{value:,.0f}"


def dimension_values(ds: Dataset, col: str, limit: int = 200) -> list[str]:
    if col not in ds.schema.dimensions:
        raise IngestError(f"'{col}' is not a known category column.")
    df = ds.query(f"SELECT {q(col)} AS v, SUM(revenue) AS r FROM {TABLE} "
                  f"GROUP BY 1 ORDER BY r DESC LIMIT {int(limit)}")
    return df["v"].tolist()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m engine.kpi <path to csv or xlsx>")
        sys.exit(1)
    data = load_dataset(sys.argv[1])
    print(f"Loaded {data.n_rows:,} rows, {data.min_date.date()} to {data.max_date.date()}\n")
    print(f"{'KPI':<22}{'Latest month':<14}{'Value':>14}{'vs prev':>10}{'vs year ago':>13}")
    for k in available_kpis(data):
        h = headline(data, k.key, "M")

        def fmt_change(c, pts):
            if np.isnan(c):
                return "n/a"
            return f"{c * 100:+.1f} pts" if pts else f"{c:+.1%}"

        print(f"{h['label']:<22}{h['period'].strftime('%Y-%m'):<14}"
              f"{format_value(h['value'], h['fmt']):>14}"
              f"{fmt_change(h['change_vs_previous'], h['change_is_points']):>10}"
              f"{fmt_change(h['change_vs_year_ago'], h['change_is_points']):>13}")
    print("\nLast 6 months of revenue:")
    print(kpi_series(data, "revenue", "M").tail(6).round({"value": 0, "num": 0}).to_string(index=False))
