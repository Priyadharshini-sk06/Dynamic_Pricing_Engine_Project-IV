"""
features.py
------------
Shared feature-engineering logic used by BOTH the training pipeline
(src/train_model.py) and the live inference path (app/main.py via
src/predict.py). Keeping this in one place guarantees train/serve
consistency -- the exact same transformation is applied whether we're
fitting the model on historical data or scoring a single live API request.

IMPORTANT DESIGN NOTE ON WHAT THE MODEL SEES
=============================================
The model is only given features a caller can realistically supply at
request time (see app/main.py's PredictRequest): product_category,
base_price, competitor_price, inventory_level, demand_signal, a
seasonality proxy (day_of_week), is_holiday, and customer_segment.

It deliberately does NOT see the internal `cost` or exact per-SKU
`price_elasticity` used to generate the training labels (those are
business-internal figures, not something an API caller would provide),
nor the exact calendar date (only day-of-week/holiday are exposed). This
keeps training and serving features identical (no train/serve skew) and
is also *why* the model lands at a realistic ~75-85% R^2 rather than a
near-perfect fit: real dynamic pricing models never have full visibility
into the demand-generating process either.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

CATEGORICAL_COLS = ["product_category", "day_of_week", "customer_segment"]
NUMERIC_COLS = ["base_price", "competitor_price", "inventory_level", "demand_signal", "is_holiday"]

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CATEGORY_ORDER = [
    "Electronics", "Apparel", "Home & Kitchen",
    "Beauty", "Sports & Outdoors", "Grocery", "Toys & Games",
]
SEGMENT_ORDER = ["Budget", "Regular", "Premium"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived elasticity/price-relationship features to a raw records
    DataFrame. Input columns required:
      product_category, base_price, competitor_price, inventory_level,
      demand_signal, day_of_week, is_holiday, customer_segment
    """
    out = df.copy()

    # --- price-relationship / elasticity-proxy features ---------------------
    # How the product is priced relative to the competitor (>1 = we're pricier).
    out["competitor_price_ratio"] = out["base_price"] / out["competitor_price"].replace(0, np.nan)
    out["competitor_price_ratio"] = out["competitor_price_ratio"].fillna(1.0)

    # Absolute gap, useful for tree splits that aren't purely ratio-based.
    out["competitor_price_gap"] = out["base_price"] - out["competitor_price"]

    # Rough "demand per price unit" -- a cheap proxy for price elasticity
    # (how much observed demand there is relative to the going market price).
    out["demand_per_price"] = out["demand_signal"] / out["competitor_price"].replace(0, np.nan)
    out["demand_per_price"] = out["demand_per_price"].fillna(0.0)

    # --- inventory pressure buckets (mirrors the rule-based logic later) ----
    out["low_inventory_flag"] = (out["inventory_level"] < 60).astype(int)
    out["high_inventory_flag"] = (out["inventory_level"] > 500).astype(int)

    # --- seasonality proxy ----------------------------------------------------
    out["is_weekend"] = out["day_of_week"].isin(["Friday", "Saturday", "Sunday"]).astype(int)

    return out


def build_design_matrix(df: pd.DataFrame, fit_columns: list[str] | None = None):
    """One-hot encode categoricals and return (X, column_list).

    If `fit_columns` is given (the column layout learned at training time),
    the output is reindexed to match it exactly -- this is what makes a
    single-row inference request produce a DataFrame with the same columns
    (and column order) the model was trained on, filling any missing
    dummy columns with 0.
    """
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    feature_cols = NUMERIC_COLS + [
        "competitor_price_ratio", "competitor_price_gap", "demand_per_price",
        "low_inventory_flag", "high_inventory_flag", "is_weekend",
    ]
    dummy_cols = [c for c in encoded.columns if c not in df.columns]
    all_cols = feature_cols + dummy_cols

    X = encoded.reindex(columns=all_cols, fill_value=0)

    if fit_columns is not None:
        X = X.reindex(columns=fit_columns, fill_value=0)
        return X, fit_columns

    return X, list(X.columns)
