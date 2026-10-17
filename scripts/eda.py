"""
eda.py
------
Lightweight exploratory data analysis over data/pricing_data.csv.
Saves 3 key plots to outputs/ so the pricing dynamics can be sanity-checked
before modeling:

  1. price_distribution.png       -- distribution of optimal_price vs base_price
  2. demand_vs_price.png          -- demand_signal vs (optimal_price / base_price) ratio
  3. seasonality_competitor.png   -- avg optimal_price by day_of_week/holiday,
                                      and optimal_price vs competitor_price

Run: python scripts/eda.py   (from the project root)
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering (no GUI backend needed)
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "pricing_data.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    return df


def plot_price_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(df["base_price"], bins=50, color="#4C72B0", alpha=0.8)
    axes[0].set_title("Base Price Distribution")
    axes[0].set_xlabel("Base Price")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["optimal_price"], bins=50, color="#55A868", alpha=0.8)
    axes[1].set_title("Optimal (Target) Price Distribution")
    axes[1].set_xlabel("Optimal Price")

    fig.suptitle("Price Distributions")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "price_distribution.png", dpi=140)
    plt.close(fig)


def plot_demand_vs_price(df: pd.DataFrame):
    df = df.copy()
    df["price_ratio"] = df["optimal_price"] / df["base_price"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.sample(min(4000, len(df)), random_state=0)
    ax.scatter(sample["price_ratio"], sample["demand_signal"], s=6, alpha=0.35, color="#C44E52")
    ax.set_xlabel("optimal_price / base_price (relative price level)")
    ax.set_ylabel("demand_signal (recent clicks/orders)")
    ax.set_title("Demand vs. Relative Price")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "demand_vs_price.png", dpi=140)
    plt.close(fig)


def plot_seasonality_and_competitor(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_dow = df.groupby("day_of_week")["optimal_price"].mean().reindex(dow_order)
    axes[0].bar(by_dow.index, by_dow.values, color="#8172B2")
    axes[0].set_title("Avg Optimal Price by Day of Week")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Avg Optimal Price")

    sample = df.sample(min(4000, len(df)), random_state=0)
    axes[1].scatter(sample["competitor_price"], sample["optimal_price"], s=6, alpha=0.3, color="#64B5CD")
    lims = [0, max(sample["competitor_price"].max(), sample["optimal_price"].max())]
    axes[1].plot(lims, lims, "--", color="gray", linewidth=1, label="y = x")
    axes[1].set_xlabel("Competitor Price")
    axes[1].set_ylabel("Optimal Price")
    axes[1].set_title("Optimal Price vs Competitor Price")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "seasonality_competitor.png", dpi=140)
    plt.close(fig)


def print_summary(df: pd.DataFrame):
    print("Rows:", len(df), "| SKUs:", df["sku_id"].nunique())
    print("\nOptimal price by category:")
    print(df.groupby("product_category")["optimal_price"].mean().round(2).sort_values(ascending=False))
    print("\nOptimal price by holiday flag:")
    print(df.groupby("is_holiday")["optimal_price"].mean().round(2))
    print("\nCorrelation with optimal_price:")
    numeric = df.select_dtypes("number")
    print(numeric.corr()["optimal_price"].round(3).sort_values(ascending=False))


def main():
    df = load_data()
    print_summary(df)
    plot_price_distribution(df)
    plot_demand_vs_price(df)
    plot_seasonality_and_competitor(df)
    print(f"\nSaved 3 plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
