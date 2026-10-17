"""
pricing_logic.py
-----------------
Business rule layer applied ON TOP of the trained model's raw price
prediction. This keeps two concerns separate, which is standard practice
in real pricing systems:

  * the ML MODEL learns the complex, data-driven relationship between
    product/market signals and a good price target.
  * simple, auditable BUSINESS RULES then nudge that prediction for
    situations the business wants explicit, guaranteed control over
    (e.g. "never ignore the competitor entirely", "always discount when
    we're drowning in stock") -- rules a stakeholder can read and approve
    without needing to trust a black box for every decision.

Both the training pipeline and the FastAPI inference endpoint import this
module so the logic is identical wherever it's used.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict

# --------------------------------------------------------------------------
# Configuration (kept as simple module-level constants so they're easy to
# find and tune without digging through logic).
# --------------------------------------------------------------------------

# Weighted average blend between the model's raw prediction and the current
# competitor price. 0.70 means "70% trust the model, 30% track the competitor".
MODEL_WEIGHT = 0.70
COMPETITOR_WEIGHT = 1.0 - MODEL_WEIGHT

# Inventory-based adjustment thresholds + magnitudes.
LOW_INVENTORY_THRESHOLD = 60          # units; below this we risk stocking out
HIGH_INVENTORY_THRESHOLD = 500        # units; above this we're overstocked
LOW_INVENTORY_MARKUP = 0.05           # +5% when scarce
HIGH_INVENTORY_DISCOUNT = 0.07        # -7% when overstocked

# Demand-based adjustment. demand_signal is normalized around this "typical"
# value; being `DEMAND_NORM_SCALE` units above/below it moves price by
# `DEMAND_MAX_ADJUSTMENT` at most.
DEMAND_TYPICAL_VALUE = 100
DEMAND_NORM_SCALE = 200
DEMAND_MAX_ADJUSTMENT = 0.05          # +-5% max swing from demand alone


@dataclass
class PriceAdjustmentResult:
    raw_model_prediction: float
    competitor_blended_price: float
    inventory_adjustment_pct: float
    demand_adjustment_pct: float
    final_recommended_price: float

    def as_dict(self) -> dict:
        return asdict(self)


def apply_competitor_blend(model_prediction: float, competitor_price: float,
                            model_weight: float = MODEL_WEIGHT) -> float:
    """Weighted average of the model's raw price and the competitor's price."""
    competitor_weight = 1.0 - model_weight
    return model_weight * model_prediction + competitor_weight * competitor_price


def inventory_adjustment_factor(inventory_level: int) -> float:
    """Returns a multiplicative adjustment (e.g. 0.05 = +5%, -0.07 = -7%)."""
    if inventory_level < LOW_INVENTORY_THRESHOLD:
        return LOW_INVENTORY_MARKUP
    if inventory_level > HIGH_INVENTORY_THRESHOLD:
        return -HIGH_INVENTORY_DISCOUNT
    return 0.0


def demand_adjustment_factor(demand_signal: float) -> float:
    """Scales price up/down proportionally to how far demand_signal is from
    a 'typical' baseline, capped at +-DEMAND_MAX_ADJUSTMENT."""
    demand_norm = (demand_signal - DEMAND_TYPICAL_VALUE) / DEMAND_NORM_SCALE
    demand_norm = max(-1.0, min(1.0, demand_norm))
    return DEMAND_MAX_ADJUSTMENT * demand_norm


def recommend_price(
    model_prediction: float,
    competitor_price: float,
    inventory_level: int,
    demand_signal: float,
    model_weight: float = MODEL_WEIGHT,
) -> PriceAdjustmentResult:
    """Full pipeline: model prediction -> competitor blend -> inventory rule
    -> demand rule -> final recommended price. Returns every intermediate
    value so the API/frontend can show a transparent breakdown."""

    blended = apply_competitor_blend(model_prediction, competitor_price, model_weight)

    inv_adj = inventory_adjustment_factor(inventory_level)
    dem_adj = demand_adjustment_factor(demand_signal)

    final_price = blended * (1.0 + inv_adj) * (1.0 + dem_adj)

    return PriceAdjustmentResult(
        raw_model_prediction=round(float(model_prediction), 2),
        competitor_blended_price=round(float(blended), 2),
        inventory_adjustment_pct=round(inv_adj * 100, 2),
        demand_adjustment_pct=round(dem_adj * 100, 2),
        final_recommended_price=round(float(final_price), 2),
    )
