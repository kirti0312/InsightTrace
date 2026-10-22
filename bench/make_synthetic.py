"""Synthetic sales data with KNOWN answers, used to test InsightTrace.

Real datasets (like Superstore) have no "correct" explanation, so we cannot
prove the detective works on them. Here we plant specific events in fake data,
and write down the truth in a JSON file. Later, the benchmark checks whether
the engine finds the planted causes, rejects the decoys, and says UNKNOWN when
nothing explains a drop.

Run from the project folder:
    python bench/make_synthetic.py

Outputs (in data/):
    synthetic_demo.csv            the fake sales data (Superstore-style columns)
    synthetic_ground_truth.json   the planted events and the expected verdicts
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = date(2023, 1, 1)
END = date(2024, 12, 31)
OUT_DIR = Path(__file__).resolve().parent.parent / "data"

# ----------------------------------------------------------------------------
# Business setup
# ----------------------------------------------------------------------------
REGIONS = {"East": 0.28, "West": 0.32, "Central": 0.22, "South": 0.18}
STATES = {
    "East": ["New York", "Pennsylvania", "Virginia"],
    "West": ["California", "Washington", "Oregon"],
    "Central": ["Illinois", "Texas", "Ohio"],
    "South": ["Florida", "Georgia", "Tennessee"],
}
SEGMENTS = {"Consumer": 0.52, "Corporate": 0.30, "Home Office": 0.18}
SHIP_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]

# (category, sub-category, min price, max price, profit margin, popularity)
SUBCATS = [
    ("Furniture", "Chairs", 120, 450, 0.28, 1.0),
    ("Furniture", "Tables", 180, 700, 0.15, 0.6),
    ("Furniture", "Bookcases", 100, 500, 0.18, 0.6),
    ("Office Supplies", "Binders", 5, 40, 0.38, 2.5),
    ("Office Supplies", "Paper", 4, 30, 0.42, 2.0),
    ("Office Supplies", "Storage", 15, 120, 0.30, 1.2),
    ("Technology", "Phones", 80, 900, 0.25, 1.2),
    ("Technology", "Accessories", 10, 120, 0.32, 1.5),
    ("Technology", "Machines", 300, 2500, 0.20, 0.3),
]
VARIANTS = ["Basic", "Plus", "Pro", "Max"]
VARIANT_WEIGHTS = [1.0, 1.0, 0.8, 0.6]

# Seasonality: Jan/Feb are slow, Nov/Dec are busy
SEASON = [0.78, 0.80, 0.92, 0.95, 1.00, 1.00, 0.97, 1.00, 1.10, 1.15, 1.35, 1.45]
WEEKDAY = [1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.70]  # Mon..Sun
TREND_PER_DAY = 0.0003  # slow business growth
BASE_ORDERS_PER_DAY = 11.0

DISCOUNT_VALUES = [0.0, 0.1, 0.2, 0.3]
DISCOUNT_PROBS = [0.60, 0.22, 0.15, 0.03]

# ----------------------------------------------------------------------------
# Planted events (the "answers")
# ----------------------------------------------------------------------------
A_START, A_END = date(2024, 3, 1), date(2024, 4, 30)     # discount spike
B_START = date(2024, 6, 1)                               # customer churn begins
C_START, C_END = date(2024, 10, 15), date(2024, 11, 30)  # stockout
U_START, U_END = date(2024, 9, 1), date(2024, 9, 30)     # unexplained dip
STOCKOUT_PRODUCT_NAME = "Phones Max"


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _build_catalog(rng: np.random.Generator):
    """Products grouped by sub-category, with prices and selection weights."""
    catalog = {}
    pid = 1
    for cat, sub, lo, hi, margin, pop in SUBCATS:
        prods = []
        for i, variant in enumerate(VARIANTS):
            price = lo + (hi - lo) * (i / 3) * rng.uniform(0.85, 1.0)
            name = f"{sub} {variant}"
            weight = VARIANT_WEIGHTS[i]
            if name == STOCKOUT_PRODUCT_NAME:
                weight = 5.0  # make it a best seller so its loss is visible
            prods.append({
                "id": f"{cat[:3].upper()}-{sub[:2].upper()}-{pid:05d}",
                "name": name,
                "price": round(float(price), 2),
                "weight": weight,
            })
            pid += 1
        w = np.array([p["weight"] for p in prods])
        catalog[sub] = {
            "category": cat, "margin": margin, "products": prods,
            "probs": w / w.sum(),
        }
    return catalog


def _build_customers(rng: np.random.Generator):
    """~240 customers, each living in one region, with different buying power."""
    total = 240
    customers = []
    n = 1
    for region, share in REGIONS.items():
        count = round(total * share)
        for _ in range(count):
            seg = rng.choice(list(SEGMENTS), p=list(SEGMENTS.values()))
            customers.append({
                "id": f"CU-{n:04d}", "name": f"Customer {n:03d}",
                "region": region, "segment": str(seg),
                "weight": float(rng.lognormal(0, 0.6)),
            })
            n += 1
    # Event B: the 5 biggest Central customers get an extra boost (so that
    # losing them is clearly visible), and they later stop ordering.
    central = sorted((c for c in customers if c["region"] == "Central"),
                     key=lambda c: -c["weight"])
    churned = central[:5]
    for c in churned:
        c["weight"] *= 4.0
    by_region = {}
    for region in REGIONS:
        group = [c for c in customers if c["region"] == region]
        w = np.array([c["weight"] for c in group])
        by_region[region] = (group, w / w.sum())
    return customers, by_region, {c["id"] for c in churned}


def generate(seed: int = SEED):
    """Build the dataset. Returns (DataFrame, ground_truth_dict)."""
    rng = np.random.default_rng(seed)
    catalog = _build_catalog(rng)
    customers, by_region, churned_ids = _build_customers(rng)
    sub_names = [s[1] for s in SUBCATS]
    base_pop = np.array([s[5] for s in SUBCATS], dtype=float)
    idx = {name: i for i, name in enumerate(sub_names)}

    rows = []
    order_no = 0
    for t, d in enumerate(_daterange(START, END)):
        season = SEASON[d.month - 1]
        wk = WEEKDAY[d.weekday()]
        trend = 1 + TREND_PER_DAY * t

        for region, share in REGIONS.items():
            n_orders = rng.poisson(BASE_ORDERS_PER_DAY * share * season * wk * trend)
            group, probs = by_region[region]

            for _ in range(n_orders):
                cust = group[rng.choice(len(group), p=probs)]

                # Event B: top Central customers have left
                if cust["id"] in churned_ids and d >= B_START:
                    continue

                # Sub-category choice, with event A and decoy D1 adjustments
                w = base_pop.copy()
                if region == "West" and A_START <= d <= A_END:
                    w[idx["Binders"]] *= 1.5          # promo drives volume
                if region == "East" and d.month == 3:
                    w[idx["Chairs"]] *= 0.35          # recurring March dip (decoy)
                w = w / w.sum()

                n_items = min(5, 1 + rng.poisson(1.2))
                items = []
                for _ in range(n_items):
                    sub = sub_names[rng.choice(len(sub_names), p=w)]
                    info = catalog[sub]
                    prod = info["products"][rng.choice(4, p=info["probs"])]

                    # Event C: best-selling South phone is out of stock
                    if (region == "South" and prod["name"] == STOCKOUT_PRODUCT_NAME
                            and C_START <= d <= C_END and rng.random() < 0.92):
                        continue
                    # Event U: random, spread-out loss of sales across everything
                    if U_START <= d <= U_END and rng.random() < 0.18:
                        continue

                    # Discount (event A forces a deep discount on West Binders)
                    if region == "West" and sub == "Binders" and A_START <= d <= A_END:
                        discount = float(rng.choice([0.35, 0.40, 0.45]))
                    else:
                        discount = float(rng.choice(DISCOUNT_VALUES, p=DISCOUNT_PROBS))

                    qty = int(rng.choice([1, 2, 3, 4, 5], p=[0.30, 0.30, 0.20, 0.12, 0.08]))
                    gross = prod["price"] * qty
                    sales = gross * (1 - discount)
                    cost = gross * (1 - info["margin"]) * (1 + rng.normal(0, 0.03))
                    items.append((sub, info, prod, qty, discount, sales, sales - cost))

                if not items:
                    continue

                order_no += 1
                order_id = f"SY-{d.year}-{order_no:06d}"
                state = str(rng.choice(STATES[region]))
                ship = d + timedelta(days=int(rng.integers(2, 7)))
                mode = str(rng.choice(SHIP_MODES, p=[0.6, 0.2, 0.15, 0.05]))
                postal = int(rng.integers(10000, 99999))
                for sub, info, prod, qty, discount, sales, profit in items:
                    rows.append({
                        "Order ID": order_id,
                        "Order Date": d.isoformat(),
                        "Ship Date": ship.isoformat(),
                        "Ship Mode": mode,
                        "Customer ID": cust["id"],
                        "Customer Name": cust["name"],
                        "Segment": cust["segment"],
                        "Country": "United States",
                        "City": f"{state} Metro",
                        "State": state,
                        "Postal Code": postal,
                        "Region": region,
                        "Product ID": prod["id"],
                        "Category": info["category"],
                        "Sub-Category": sub,
                        "Product Name": prod["name"],
                        "Sales": round(sales, 2),
                        "Quantity": qty,
                        "Discount": discount,
                        "Profit": round(profit, 2),
                    })

    df = pd.DataFrame(rows)
    df.insert(0, "Row ID", range(1, len(df) + 1))
    ground_truth = _ground_truth(churned_ids, catalog)
    return df, ground_truth


def _ground_truth(churned_ids, catalog):
    stockout_id = next(p["id"] for p in catalog["Phones"]["products"]
                       if p["name"] == STOCKOUT_PRODUCT_NAME)
    return {
        "note": ("Planted events. 'expected_verdict' is what a correct "
                 "InsightTrace engine should conclude."),
        "date_range": [START.isoformat(), END.isoformat()],
        "events": [
            {
                "id": "A_west_binders_discount",
                "type": "TRUE_CAUSE",
                "kpi": "profit",
                "window": [A_START.isoformat(), A_END.isoformat()],
                "where": {"Region": "West", "Sub-Category": "Binders"},
                "what_happened": ("Discounts on Binders in the West jumped to "
                                  "35-45%. Volume rose, revenue held up, but "
                                  "profit collapsed."),
                "expected_verdict": "CAUSE",
                "expected_action": "Cap discounts on West Binders",
            },
            {
                "id": "B_central_top_customers_churn",
                "type": "TRUE_CAUSE",
                "kpi": "revenue",
                "window": [B_START.isoformat(), END.isoformat()],
                "where": {"Region": "Central", "Customer ID": sorted(churned_ids)},
                "what_happened": ("The 5 largest Central customers stopped "
                                  "ordering from June 2024 and never returned."),
                "expected_verdict": "CAUSE",
                "expected_action": "Win-back outreach to the churned accounts",
            },
            {
                "id": "C_south_phonesmax_stockout",
                "type": "TRUE_CAUSE",
                "kpi": "revenue",
                "window": [C_START.isoformat(), C_END.isoformat()],
                "where": {"Region": "South", "Product ID": stockout_id,
                          "Product Name": STOCKOUT_PRODUCT_NAME},
                "what_happened": ("The best-selling South phone was ~92% "
                                  "unavailable for 6 weeks."),
                "expected_verdict": "CAUSE",
                "expected_action": "Fix South inventory for this product",
            },
            {
                "id": "D1_east_chairs_march_dip",
                "type": "DECOY",
                "kpi": "revenue",
                "window": ["2023-03-01", "2023-03-31", "2024-03-01", "2024-03-31"],
                "where": {"Region": "East", "Sub-Category": "Chairs"},
                "what_happened": ("East Chairs dip every March, in both years. "
                                  "It overlaps event A but is recurring, not new."),
                "expected_verdict": "NOT_A_CAUSE",
                "expected_action": "None. Recurring pattern.",
            },
            {
                "id": "D2_january_seasonal_dip",
                "type": "DECOY",
                "kpi": "revenue",
                "window": ["2023-01-01", "2023-01-31", "2024-01-01", "2024-01-31"],
                "where": {},
                "what_happened": ("January is always much slower than December. "
                                  "A naive month-over-month alert would fire."),
                "expected_verdict": "NOT_A_CAUSE",
                "expected_action": "None. Normal seasonality.",
            },
            {
                "id": "U_september_uniform_dip",
                "type": "UNEXPLAINED",
                "kpi": "revenue",
                "window": [U_START.isoformat(), U_END.isoformat()],
                "where": {},
                "what_happened": ("About 18% of sales vanished at random across "
                                  "every region, product and customer."),
                "expected_verdict": "UNKNOWN",
                "expected_action": ("Say UNKNOWN; suggest checking outside "
                                    "factors (outage, holiday, data gap)."),
            },
        ],
    }


def main():
    df, truth = generate()
    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / "synthetic_demo.csv"
    json_path = OUT_DIR / "synthetic_ground_truth.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

    d = pd.to_datetime(df["Order Date"])
    print(f"Rows: {len(df):,}   Orders: {df['Order ID'].nunique():,}")
    print(f"Dates: {d.min().date()} to {d.max().date()}")
    print(f"Total sales: {df['Sales'].sum():,.0f}   Total profit: {df['Profit'].sum():,.0f}")
    m = df.assign(month=d.dt.to_period("M")).groupby("month")[["Sales", "Profit"]].sum().round(0)
    print("\nMonthly totals:")
    print(m.to_string())
    print(f"\nSaved: {csv_path}\nSaved: {json_path}")


if __name__ == "__main__":
    main()
