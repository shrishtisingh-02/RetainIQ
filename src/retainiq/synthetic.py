"""SYNTHETIC transaction generator.

!!! This data is artificial. !!!
It exists only so the pipeline, tests and dashboard can be exercised without the
real UCI dataset. Nothing derived from it is a real-world finding, and it must
never be reported as such. It deliberately contains dirty rows (missing customer
IDs, cancellations, zero prices, duplicates, fee codes) to exercise the cleaning step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COUNTRIES = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands"]
COUNTRY_WEIGHTS = [0.88, 0.04, 0.03, 0.02, 0.015, 0.015]


def generate_synthetic_transactions(n_customers: int = 800, start: str = "2010-12-01",
                                    end: str = "2011-12-09", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    n_days = (end_ts - start_ts).days

    n_products = 250
    products = pd.DataFrame({
        "StockCode": [f"P{i:04d}" for i in range(n_products)],
        "Description": [f"SYNTHETIC PRODUCT {i}" for i in range(n_products)],
        "UnitPrice": np.round(rng.lognormal(mean=1.0, sigma=0.6, size=n_products), 2).clip(0.2),
    })

    rows = []
    invoice_no = 500000
    for cid in range(12000, 12000 + n_customers):
        join_day = float(rng.integers(0, int(n_days * 0.85)))
        rate = rng.gamma(shape=1.6, scale=0.012)      # orders per day (customer-specific)
        lifetime = rng.exponential(scale=220.0)       # days until the customer drifts away
        basket_scale = rng.lognormal(mean=0.8, sigma=0.5)
        country = rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS)
        t = join_day
        first = True
        while True:
            t += 0.0 if first else rng.exponential(scale=1.0 / max(rate, 1e-3))
            first = False
            if t >= n_days or t > join_day + lifetime:
                break
            invoice_no += 1
            order_time = start_ts + pd.Timedelta(days=t) + pd.Timedelta(
                minutes=int(rng.integers(8 * 60, 18 * 60)))
            n_lines = int(rng.integers(1, 9))
            picks = rng.choice(n_products, size=n_lines, replace=False)
            for p in picks:
                qty = int(max(1, rng.poisson(3 * basket_scale)))
                rows.append((str(invoice_no), products.at[p, "StockCode"],
                             products.at[p, "Description"], qty, order_time,
                             float(products.at[p, "UnitPrice"]), float(cid), country))

    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    df = df[df["InvoiceDate"] <= end_ts].reset_index(drop=True)

    # ---- inject realistic dirt so the cleaning stage has work to do ----------
    n = len(df)
    dirty = df.sample(frac=0.02, random_state=seed).copy()
    dirty["CustomerID"] = np.nan                                  # guest checkouts
    cancel = df.sample(frac=0.015, random_state=seed + 1).copy()
    cancel["InvoiceNo"] = "C" + cancel["InvoiceNo"]
    cancel["Quantity"] = -cancel["Quantity"]                      # cancellations / returns
    zero_price = df.sample(frac=0.003, random_state=seed + 2).copy()
    zero_price["UnitPrice"] = 0.0
    fees = df.sample(n=min(30, n), random_state=seed + 3).copy()
    fees["StockCode"] = "POST"
    duplicates = df.sample(frac=0.005, random_state=seed + 4).copy()
    return pd.concat([df, dirty, cancel, zero_price, fees, duplicates],
                     ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
