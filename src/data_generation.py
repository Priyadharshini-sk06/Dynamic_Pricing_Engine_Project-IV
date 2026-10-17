"""
data_generation.py
-------------------
Generates a synthetic e-commerce pricing dataset since no real dataset was
supplied for this project. The goal is NOT to produce random noise -- it is
to simulate a plausible pricing environment with the kind of structure a
real dynamic-pricing model would need to learn:

  * price elasticity of demand (higher price -> lower demand, category-dependent)
  * competitive pressure (competitor price pulls the "right" price toward it)
  * inventory pressure (overstock -> discount to clear stock, scarcity -> raise price)
  * demand signals (recent clicks/orders correlate with willingness to raise price)
  * seasonality / day-of-week / holiday effects
  * customer segment sensitivity (premium shoppers tolerate higher prices)

SIMULATION LOGIC (documented step by step)
============================================
1. We create a fixed catalog of SKUs (100-300 products) across 7 categories.
   Each SKU gets a stable `base_price`, a `cost` (a fraction of base_price),
   and a `price_elasticity` coefficient (category-driven + per-SKU noise).

2. For each SKU we generate many observations across a ~12 month window
   (different dates / days of week / holidays), each with its own
   competitor price, inventory level, demand signal, and customer segment.
   This is what gives us 20k-30k rows from 100-300 distinct SKUs.

3. We derive the *target* `optimal_price` using the classic constant-
   elasticity monopoly markup rule as a starting point:

        optimal_price_theoretical = cost * elasticity / (elasticity - 1)

   This is the textbook profit-maximizing price when demand follows
   demand(price) ~ price^(-elasticity). We then adjust that theoretical
   price with realistic, explainable multipliers for competition,
   inventory, demand, seasonality/holiday, and customer segment, plus a
   little Gaussian noise to mimic real-world unexplained variance. This
   gives a learnable, non-trivial regression target that a tree ensemble
   can recover with ~75-85% R^2 without needing to be "perfect" (which
   matches the project's stated goal).

Run directly to (re)generate data/pricing_data.csv:
    python src/data_generation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
N_SKUS = 220                 # distinct products (within the 100-300 spec)
TARGET_ROWS = 26_000         # total observations (within the 20k-30k spec)
DATE_START = datetime(2025, 9, 1)
DATE_END = datetime(2026, 9, 1)  # ~1 year window ending "today" per project context

CATEGORIES = [
    "Electronics", "Apparel", "Home & Kitchen",
    "Beauty", "Sports & Outdoors", "Grocery", "Toys & Games",
]

# Category-specific base price ranges (min, max) in currency units.
CATEGORY_PRICE_RANGE = {
    "Electronics": (50, 800),
    "Apparel": (15, 120),
    "Home & Kitchen": (20, 200),
    "Beauty": (10, 60),
    "Sports & Outdoors": (20, 150),
    "Grocery": (5, 40),
    "Toys & Games": (10, 80),
}

# Category-specific base price elasticity of demand (how sharply demand
# falls as price rises). Higher = more price-sensitive shoppers.
CATEGORY_ELASTICITY = {
    "Electronics": 1.8,
    "Apparel": 2.2,
    "Home & Kitchen": 1.7,
    "Beauty": 1.6,
    "Sports & Outdoors": 1.9,
    "Grocery": 1.3,
    "Toys & Games": 2.0,
}

CUSTOMER_SEGMENTS = ["Budget", "Regular", "Premium"]
SEGMENT_WEIGHTS = [0.40, 0.40, 0.20]
# Premium shoppers push the "right" price up a bit, Budget shoppers push it down.
SEGMENT_PRICE_EFFECT = {"Budget": -0.04, "Regular": 0.0, "Premium": 0.04}

# A small fixed holiday calendar (month, day) used to flag is_holiday and to
# create demand/price spikes around major shopping events.
HOLIDAYS = {
    (11, 28),  # Black Friday-ish
    (11, 29),
    (12, 24), (12, 25), (12, 26),
    (1, 1),
    (7, 4),
    (10, 31),
    (2, 14),
    (5, 11),  # Mother's day (approx)
}


def build_sku_catalog(n_skus: int) -> pd.DataFrame:
    """Create the fixed per-SKU attributes: category, base_price, cost, elasticity."""
    categories = rng.choice(CATEGORIES, size=n_skus)
    rows = []
    for i, cat in enumerate(categories):
        lo, hi = CATEGORY_PRICE_RANGE[cat]
        base_price = float(rng.uniform(lo, hi))
        # Cost is a random fraction of base_price (retail markup varies by SKU).
        cost_ratio = float(rng.uniform(0.50, 0.75))
        cost = base_price * cost_ratio
        # Per-SKU elasticity = category baseline + small random noise, clipped > 1
        # (elasticity must be > 1 for the markup formula to yield a finite, positive price).
        elasticity = max(1.15, CATEGORY_ELASTICITY[cat] + rng.normal(0, 0.15))
        rows.append({
            "sku_id": f"SKU-{i+1:04d}",
            "product_category": cat,
            "base_price": round(base_price, 2),
            "cost": round(cost, 2),
            "price_elasticity": round(elasticity, 3),
        })
    return pd.DataFrame(rows)


def random_dates(n: int) -> pd.Series:
    span_days = (DATE_END - DATE_START).days
    offsets = rng.integers(0, span_days, size=n)
    return pd.Series([DATE_START + timedelta(days=int(o)) for o in offsets])


def seasonality_multiplier(date: pd.Timestamp) -> float:
    """Smooth yearly seasonality curve (e.g., Q4 shopping peak) via a sine wave,
    ranging roughly between 0.9x and 1.15x, plus a distinct holiday bump."""
    day_of_year = date.timetuple().tm_yday
    # Peak centered around day ~335 (early December) using a shifted cosine.
    seasonal = 1.0 + 0.12 * np.cos(2 * np.pi * (day_of_year - 335) / 365)
    return seasonal


def is_holiday(date: pd.Timestamp) -> bool:
    return (date.month, date.day) in HOLIDAYS


def generate_dataset() -> pd.DataFrame:
    catalog = build_sku_catalog(N_SKUS)

    # Distribute TARGET_ROWS observations across SKUs roughly evenly, with
    # some SKUs sampled more/less often (mimics popular vs niche products).
    sku_weights = rng.dirichlet(np.ones(N_SKUS) * 3.0)
    sku_choice_idx = rng.choice(N_SKUS, size=TARGET_ROWS, p=sku_weights)

    dates = random_dates(TARGET_ROWS)

    records = []
    for row_i in range(TARGET_ROWS):
        sku = catalog.iloc[sku_choice_idx[row_i]]
        date = dates.iloc[row_i]

        day_of_week = date.strftime("%A")
        holiday_flag = is_holiday(date)
        season_mult = seasonality_multiplier(date)
        # Weekend bump (Fri/Sat/Sun see slightly higher browsing/order volume).
        weekend_bump = 1.08 if day_of_week in ("Friday", "Saturday", "Sunday") else 1.0

        base_price = sku["base_price"]
        cost = sku["cost"]
        elasticity = sku["price_elasticity"]

        # --- competitor_price -------------------------------------------------
        # Usually within +-15% of our base price; ~12% chance competitor is
        # running an aggressive promo (down to 20-35% off).
        if rng.random() < 0.12:
            competitor_price = base_price * rng.uniform(0.65, 0.80)
        else:
            competitor_price = base_price * rng.uniform(0.85, 1.15)

        # --- inventory_level ----------------------------------------------------
        # Units in stock; skewed distribution (lognormal) so most SKUs sit in a
        # "normal" band with occasional very-low (scarce) or very-high (overstock).
        inventory_level = int(np.clip(rng.lognormal(mean=5.2, sigma=0.9), 0, 2000))

        # --- customer_segment -----------------------------------------------
        segment = rng.choice(CUSTOMER_SEGMENTS, p=SEGMENT_WEIGHTS)

        # --- demand_signal (observed recent clicks/orders volume) -----------
        # Higher when: our price is competitive vs competitor, during
        # holidays/seasonal peaks, on weekends, and for premium-leaning
        # popular SKUs. This is an INPUT feature (observed), independent of
        # (but correlated with) the target price derivation below.
        price_ratio = base_price / max(competitor_price, 1e-6)
        competitiveness_effect = np.clip(2.0 - price_ratio, 0.4, 1.6)
        holiday_demand_bump = 1.35 if holiday_flag else 1.0
        raw_demand = (
            120
            * competitiveness_effect
            * season_mult
            * weekend_bump
            * holiday_demand_bump
            * rng.lognormal(mean=0, sigma=0.35)
        )
        demand_signal = max(0, round(raw_demand))

        # =====================================================================
        # TARGET DERIVATION: optimal_price
        # =====================================================================
        # 1) Theoretical profit-maximizing price under constant-elasticity
        #    demand: price* = cost * elasticity / (elasticity - 1)
        opt_theoretical = cost * elasticity / (elasticity - 1)

        # 2) Blend toward the competitor price so we stay market-aware
        #    (70% our theoretical optimum, 30% competitor price).
        opt_price = 0.70 * opt_theoretical + 0.30 * competitor_price

        # 3) Inventory pressure: discount to clear overstock, raise price when scarce.
        if inventory_level < 60:
            opt_price *= 1.0 + rng.uniform(0.03, 0.08)       # scarcity -> raise
        elif inventory_level > 500:
            opt_price *= 1.0 - rng.uniform(0.04, 0.10)       # overstock -> discount

        # 4) Demand pressure: scale price slightly with normalized demand signal.
        demand_norm = np.clip((demand_signal - 100) / 200, -1, 1)
        opt_price *= 1.0 + 0.05 * demand_norm

        # 5) Seasonality / holiday effects.
        opt_price *= season_mult
        if holiday_flag:
            opt_price *= 1.0 + rng.uniform(0.02, 0.06)

        # 6) Customer segment sensitivity.
        opt_price *= 1.0 + SEGMENT_PRICE_EFFECT[segment]

        # 7) Unexplained real-world noise.
        # NOTE ON NOISE MAGNITUDE: the API/model only ever sees category,
        # base_price, competitor_price, inventory_level, demand_signal,
        # day_of_week, is_holiday and customer_segment (see src/features.py).
        # It never sees the per-SKU `cost` or exact `price_elasticity` used
        # above, nor the exact calendar date (only day-of-week/holiday). A
        # sizeable multiplicative noise term is added here to stand in for
        # everything a real pricing model can't observe either -- one-off
        # promotions, individual browsing behavior, competitor stockouts,
        # exact date-level seasonality, etc. This is what keeps a
        # well-built model's R^2 in a realistic ~75-85% band instead of a
        # trivial near-1.0 fit (see README "Model Performance" section).
        opt_price *= rng.normal(1.0, 0.33)

        # 8) Sane bounds: never below 1.05x cost, and never absurdly far
        #    from the catalog base_price even after noise.
        opt_price = max(opt_price, cost * 1.05)
        opt_price = np.clip(opt_price, base_price * 0.35, base_price * 2.3)

        records.append({
            "sku_id": sku["sku_id"],
            "order_date": date.strftime("%Y-%m-%d"),
            "product_category": sku["product_category"],
            "base_price": round(base_price, 2),
            "cost": round(cost, 2),
            "price_elasticity": round(elasticity, 3),
            "competitor_price": round(float(competitor_price), 2),
            "inventory_level": inventory_level,
            "demand_signal": demand_signal,
            "day_of_week": day_of_week,
            "is_holiday": int(holiday_flag),
            "customer_segment": segment,
            "optimal_price": round(float(opt_price), 2),
        })

    df = pd.DataFrame(records)
    return df


def main():
    df = generate_dataset()
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pricing_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows across {df['sku_id'].nunique()} SKUs -> {out_path}")
    print(df.head())
    print("\nSummary stats for optimal_price:")
    print(df["optimal_price"].describe())


if __name__ == "__main__":
    main()
