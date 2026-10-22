"""Checks that the synthetic data really contains the planted events.

Run from the project folder:
    pytest tests/test_synthetic.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.make_synthetic import generate  # noqa: E402

SUPERSTORE_COLUMNS = [
    "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID",
    "Customer Name", "Segment", "Country", "City", "State", "Postal Code",
    "Region", "Product ID", "Category", "Sub-Category", "Product Name",
    "Sales", "Quantity", "Discount", "Profit",
]


@pytest.fixture(scope="module")
def data():
    df, truth = generate()
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df, truth


def _between(df, start, end):
    return df[(df["Order Date"] >= start) & (df["Order Date"] <= end)]


def test_shape_and_columns(data):
    df, _ = data
    assert list(df.columns) == SUPERSTORE_COLUMNS
    assert len(df) > 15000
    assert df["Sales"].gt(0).all()


def test_same_seed_gives_same_data():
    a, _ = generate(seed=7)
    b, _ = generate(seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_ground_truth_lists_all_six_events(data):
    _, truth = data
    ids = {e["id"] for e in truth["events"]}
    assert ids == {
        "A_west_binders_discount", "B_central_top_customers_churn",
        "C_south_phonesmax_stockout", "D1_east_chairs_march_dip",
        "D2_january_seasonal_dip", "U_september_uniform_dip",
    }


def test_event_A_discount_spike_wrecks_margin(data):
    df, _ = data
    wb = df[(df["Region"] == "West") & (df["Sub-Category"] == "Binders")]
    inside = _between(wb, "2024-03-01", "2024-04-30")
    outside = wb.drop(inside.index)
    assert inside["Discount"].mean() > 0.30
    assert outside["Discount"].mean() < 0.15
    margin_in = inside["Profit"].sum() / inside["Sales"].sum()
    margin_out = outside["Profit"].sum() / outside["Sales"].sum()
    assert margin_in < 0.05
    assert margin_out > 0.20


def test_event_B_top_customers_vanish(data):
    df, truth = data
    event = next(e for e in truth["events"] if e["id"] == "B_central_top_customers_churn")
    ids = event["where"]["Customer ID"]
    assert len(ids) == 5
    mine = df[df["Customer ID"].isin(ids)]
    assert len(_between(mine, "2023-01-01", "2024-05-31")) > 50
    assert len(_between(mine, "2024-06-01", "2024-12-31")) == 0


def test_event_C_stockout_cuts_units(data):
    df, _ = data
    p = df[(df["Region"] == "South") & (df["Product Name"] == "Phones Max")]
    units_2023 = _between(p, "2023-10-15", "2023-11-30")["Quantity"].sum()
    units_2024 = _between(p, "2024-10-15", "2024-11-30")["Quantity"].sum()
    assert units_2023 > 20
    assert units_2024 < 0.3 * units_2023


def test_decoy_D1_east_chairs_dip_every_march(data):
    """East Chairs make up a smaller share of East sales in March than in
    other months, and this repeats in BOTH years (so it is recurring)."""
    df, _ = data
    east = df[df["Region"] == "East"]
    is_chair = east["Sub-Category"] == "Chairs"
    for year in (2023, 2024):
        y = east[east["Order Date"].dt.year == year]
        yc = is_chair[y.index]
        march = y["Order Date"].dt.month == 3
        share_march = yc[march].mean()
        share_other = yc[~march].mean()
        assert share_march < 0.75 * share_other, (
            f"{year}: march {share_march:.3f} vs other {share_other:.3f}")


def test_decoy_D2_january_is_slower_than_december(data):
    df, _ = data
    dec = _between(df, "2023-12-01", "2023-12-31")["Sales"].sum()
    jan = _between(df, "2024-01-01", "2024-01-31")["Sales"].sum()
    assert jan < 0.8 * dec


def test_event_U_september_dip_exists_and_is_not_one_region(data):
    """Unexplained dip: total Sept 2024 sales are weak vs Sept 2023 (adjusted
    for August's year-on-year change), and the weakness shows up in several
    regions rather than in just one."""
    df, _ = data

    def sales(frame, a, b):
        return _between(frame, a, b)["Sales"].sum()

    def ratio(frame):
        sept = sales(frame, "2024-09-01", "2024-09-30") / sales(frame, "2023-09-01", "2023-09-30")
        aug = sales(frame, "2024-08-01", "2024-08-31") / sales(frame, "2023-08-01", "2023-08-31")
        return sept / aug

    assert ratio(df) < 0.95
    weak_regions = [r for r, g in df.groupby("Region") if ratio(g) < 1.0]
    assert len(weak_regions) >= 2, weak_regions
