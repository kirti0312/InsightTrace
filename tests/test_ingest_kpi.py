"""Tests for engine/ingest.py and engine/kpi.py.

Run from the project folder:
    pytest tests/test_ingest_kpi.py -v
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.ingest import IngestError, load_dataset  # noqa: E402
from engine.kpi import (available_kpis, build_where, dimension_values,  # noqa: E402
                        headline, kpi_series, kpi_value)

SUPERSTORE = ROOT / "data" / "Sample_-_Superstore.csv"
SYNTHETIC = ROOT / "data" / "synthetic_demo.csv"
needs_superstore = pytest.mark.skipif(not SUPERSTORE.exists(), reason="Superstore CSV missing")
needs_synthetic = pytest.mark.skipif(not SYNTHETIC.exists(), reason="synthetic CSV missing")


def csv_bytes(text: str) -> bytes:
    return text.encode("utf-8")


# ---------------------------------------------------------------- real files
@needs_superstore
def test_superstore_roles_and_values():
    ds = load_dataset(SUPERSTORE)
    s = ds.schema
    assert (s.date, s.revenue, s.profit) == ("Order Date", "Sales", "Profit")
    assert (s.quantity, s.discount, s.order_id) == ("Quantity", "Discount", "Order ID")
    assert ds.n_rows == 9994
    assert ds.min_date == pd.Timestamp("2014-01-03")
    assert ds.max_date == pd.Timestamp("2017-12-30")
    assert s.date_format == "format %m/%d/%Y"

    raw = pd.read_csv(SUPERSTORE, encoding="latin-1")
    assert ds.query("SELECT SUM(revenue) AS t FROM sales")["t"][0] == pytest.approx(raw["Sales"].sum())
    assert ds.query("SELECT SUM(profit) AS t FROM sales")["t"][0] == pytest.approx(raw["Profit"].sum())


@needs_superstore
def test_superstore_dimensions_are_sensible():
    ds = load_dataset(SUPERSTORE)
    dims = ds.schema.dimensions
    for wanted in ("Region", "Category", "Sub-Category", "Segment", "Customer Name", "Product Name"):
        assert wanted in dims
    for unwanted in ("Row ID", "Order ID", "Order Date", "Ship Date", "Postal Code", "Country"):
        assert unwanted not in dims
    assert ds.schema.aliases == {"Customer ID": "Customer Name"}
    assert any("not UTF-8" in w for w in ds.warnings)


@needs_synthetic
def test_synthetic_loads_and_orders_match():
    ds = load_dataset(SYNTHETIC)
    raw = pd.read_csv(SYNTHETIC)
    assert ds.schema.date_format == "format %Y-%m-%d"
    assert ds.n_rows == len(raw)
    orders = ds.query("SELECT COUNT(DISTINCT order_id) AS n FROM sales")["n"][0]
    assert orders == raw["Order ID"].nunique()


@needs_synthetic
def test_excel_upload_works():
    raw = pd.read_csv(SYNTHETIC).head(500)
    buf = io.BytesIO()
    raw.to_excel(buf, index=False)
    ds = load_dataset(buf.getvalue(), filename="upload.xlsx")
    assert ds.n_rows == 500
    assert ds.query("SELECT SUM(revenue) AS t FROM sales")["t"][0] == pytest.approx(raw["Sales"].sum())


# ------------------------------------------------------------- messy inputs
MESSY = (
    "Order_Date;Region;Product;Qty;Revenue (USD);Customer\n"
    "13/01/2024;North;Pen;2;\"$1,200.50\";Anna\n"
    "25/01/2024;South;Book;1;\"$300.00\";Ben\n"
    "02/02/2024;North;Pen;4;\"(50.25)\";Anna\n"
    "20/02/2024;South;Desk;1;\"$2,000.00\";Cara\n"
)


def test_messy_file_semicolons_currency_and_dayfirst():
    ds = load_dataset(csv_bytes(MESSY), filename="messy.csv")
    s = ds.schema
    assert (s.date, s.revenue, s.quantity) == ("Order_Date", "Revenue (USD)", "Qty")
    assert s.date_format == "format %d/%m/%Y"
    assert ds.min_date == pd.Timestamp("2024-01-13")
    assert ds.max_date == pd.Timestamp("2024-02-20")
    assert ds.query("SELECT SUM(revenue) AS t FROM sales")["t"][0] == pytest.approx(3450.25)
    assert {"Region", "Product", "Customer"} <= set(s.dimensions)
    assert any("Separator" in w for w in ds.warnings)


def test_ambiguous_dates_trigger_warning():
    text = "Date,Sales,Region\n03/04/2024,10,A\n05/06/2024,20,B\n01/02/2024,30,A\n"
    ds = load_dataset(csv_bytes(text), filename="amb.csv")
    assert any("ambiguous" in w for w in ds.warnings)
    assert ds.min_date == pd.Timestamp("2024-01-02")   # month/day assumed


def test_header_with_quote_characters_is_safe():
    text = 'Date,Sales,Cust "VIP"\n2024-01-01,10,a\n2024-01-02,20,b\n2024-01-03,5,a\n'
    ds = load_dataset(csv_bytes(text), filename="q.csv")
    assert 'Cust "VIP"' in ds.schema.dimensions
    assert kpi_value(ds, "revenue", "2024-01-01", "2024-01-31", {'Cust "VIP"': "a"}) == 15


def test_overrides_force_a_column():
    if not SUPERSTORE.exists():
        pytest.skip("Superstore CSV missing")
    ds = load_dataset(SUPERSTORE, overrides={"revenue": "Profit", "profit": None})
    raw = pd.read_csv(SUPERSTORE, encoding="latin-1")
    assert ds.schema.profit is None
    assert ds.query("SELECT SUM(revenue) AS t FROM sales")["t"][0] == pytest.approx(raw["Profit"].sum())


def test_a_few_bad_dates_are_dropped_and_reported():
    good = "".join(f"2024-01-{d:02d},10,A\n" for d in range(1, 20))
    text = "Date,Sales,Region\n" + good * 2 + "not a date,20,B\nalso bad,5,B\n"
    ds = load_dataset(csv_bytes(text), filename="bad.csv")      # 38 good rows, 2 bad (~5%)
    assert ds.n_rows == 38
    assert any("Skipped 2 rows" in w for w in ds.warnings)


def test_mostly_garbage_dates_fail_clearly():
    text = "Date,Sales\n" + "2024-01-01,10\nnope,20\nbad,5\nx,7\n" * 5
    with pytest.raises(IngestError, match="dates"):
        load_dataset(csv_bytes(text), filename="bad.csv")


# ------------------------------------------------------------------- errors
def test_no_date_column_gives_clear_error():
    with pytest.raises(IngestError, match="No date column"):
        load_dataset(csv_bytes("Region,Sales\nA,1\nB,2\n"), filename="x.csv")


def test_no_revenue_column_gives_clear_error():
    with pytest.raises(IngestError, match="sales/revenue"):
        load_dataset(csv_bytes("Date,Region,Notes\n2024-01-01,A,hello\n2024-01-02,B,bye\n"), filename="x.csv")


def test_empty_and_missing_and_xls_files():
    with pytest.raises(IngestError, match="empty"):
        load_dataset(b"", filename="x.csv")
    with pytest.raises(IngestError, match="not found"):
        load_dataset(ROOT / "nope.csv")
    with pytest.raises(IngestError, match=r"\.xls"):
        load_dataset(b"junk", filename="old.xls")


# ---------------------------------------------------------------------- KPIs
@needs_synthetic
def test_monthly_and_weekly_series_add_up_to_total():
    ds = load_dataset(SYNTHETIC)
    total = ds.query("SELECT SUM(revenue) AS t FROM sales")["t"][0]
    for freq in ("D", "W", "M", "Q"):
        assert kpi_series(ds, "revenue", freq)["value"].sum() == pytest.approx(total)
    m = kpi_series(ds, "revenue", "M")
    assert len(m) == 24
    assert m["period"].is_monotonic_increasing


def test_missing_months_are_filled_with_zero_for_sums_and_nan_for_ratios():
    text = ("Order Date,Sales,Profit,Region\n2024-01-05,100,20,A\n2024-01-20,100,10,A\n"
            "2024-03-05,50,5,A\n")
    ds = load_dataset(csv_bytes(text), filename="gap.csv")
    rev = kpi_series(ds, "revenue", "M")
    assert rev["value"].tolist() == [200, 0, 50]
    margin = kpi_series(ds, "profit_margin", "M")
    assert margin["value"][0] == pytest.approx(30 / 200)
    assert np.isnan(margin["value"][1])


@needs_synthetic
def test_filters_and_kpi_value_agree_with_pandas():
    ds = load_dataset(SYNTHETIC)
    raw = pd.read_csv(SYNTHETIC)
    sub = raw[(raw["Region"] == "West") & (raw["Order Date"] >= "2024-03-01") & (raw["Order Date"] <= "2024-04-30")]
    got = kpi_value(ds, "revenue", "2024-03-01", "2024-04-30", {"Region": "West"})
    assert got == pytest.approx(sub["Sales"].sum())
    margin = kpi_value(ds, "profit_margin", "2024-03-01", "2024-04-30", {"Region": "West"})
    assert margin == pytest.approx(sub["Profit"].sum() / sub["Sales"].sum())
    two = kpi_value(ds, "revenue", "2024-03-01", "2024-04-30", {"Region": ["West", "East"]})
    both = raw[raw["Region"].isin(["West", "East"]) & (raw["Order Date"] >= "2024-03-01")
               & (raw["Order Date"] <= "2024-04-30")]
    assert two == pytest.approx(both["Sales"].sum())


@needs_synthetic
def test_filters_reject_unknown_columns_and_injection():
    ds = load_dataset(SYNTHETIC)
    with pytest.raises(IngestError):
        build_where(ds, {'Region"; DROP TABLE sales; --': "x"})
    with pytest.raises(IngestError):
        build_where(ds, {"Sales": 5})              # not a dimension
    # hostile VALUES are just text, so they match nothing and nothing breaks
    assert kpi_value(ds, "revenue", "2023-01-01", "2024-12-31", {"Region": "x'; DROP TABLE sales; --"}) == 0
    assert ds.query("SELECT COUNT(*) AS n FROM sales")["n"][0] == ds.n_rows


@needs_synthetic
def test_headline_uses_last_complete_period():
    ds = load_dataset(SYNTHETIC)                      # data ends 31 Dec -> Dec complete
    h = headline(ds, "revenue", "M")
    assert h["period"] == pd.Timestamp("2024-12-01")
    dec23 = kpi_value(ds, "revenue", "2023-12-01", "2023-12-31")
    assert h["year_ago"] == pytest.approx(dec23)
    assert h["change_vs_year_ago"] == pytest.approx((h["value"] - dec23) / dec23)

    text = "Order Date,Sales\n" + "\n".join(f"2024-0{m}-{d:02d},10" for m in (1, 2, 3) for d in (1, 15)) \
           + "\n2024-04-02,10\n"                       # April only has 1 day -> incomplete
    ds2 = load_dataset(csv_bytes(text), filename="p.csv")
    assert headline(ds2, "revenue", "M")["period"] == pd.Timestamp("2024-03-01")


@needs_synthetic
def test_ratio_kpis_report_point_change():
    ds = load_dataset(SYNTHETIC)
    h = headline(ds, "profit_margin", "M")
    assert h["change_is_points"] is True
    assert abs(h["change_vs_previous"]) < 0.2        # a few points, not a huge number


def test_orders_fall_back_to_row_count_without_order_id():
    ds = load_dataset(csv_bytes("Date,Sales\n2024-01-01,5\n2024-01-02,6\n"), filename="n.csv")
    kpis = {k.key: k for k in available_kpis(ds)}
    assert kpis["orders"].label == "Transactions"
    assert kpi_value(ds, "orders", "2024-01-01", "2024-01-31") == 2
    assert "profit" not in kpis and "profit_margin" not in kpis


@needs_synthetic
def test_dimension_values_sorted_by_revenue():
    ds = load_dataset(SYNTHETIC)
    vals = dimension_values(ds, "Region")
    assert set(vals) == {"East", "West", "Central", "South"}
    with pytest.raises(IngestError):
        dimension_values(ds, "Sales")
