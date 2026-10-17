"""
predict.py
----------
Inference helper: loads the trained model + its feature-column layout
ONCE and exposes a single `predict_price()` function used by the FastAPI
app (app/main.py). Keeping this separate from app/main.py means the
inference logic can also be unit-tested or reused from a notebook/script
without spinning up the web server.
"""

from pathlib import Path
import json

import joblib
import pandas as pd

from features import engineer_features, build_design_matrix
from pricing_logic import recommend_price, PriceAdjustmentResult

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


class PricingModel:
    """Wraps the trained XGBoost model + the exact feature-column layout it
    was trained on, so a single incoming request can be transformed and
    scored consistently."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        model_path = models_dir / "xgb_pricing_model.joblib"
        columns_path = models_dir / "feature_columns.json"

        if not model_path.exists() or not columns_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {models_dir}. "
                "Run `python src/train_model.py` first."
            )

        self.model = joblib.load(model_path)
        with open(columns_path) as f:
            self.feature_columns = json.load(f)

    def predict_raw(self, request: dict) -> float:
        """Transform a single raw request dict into the model's expected
        feature layout and return the raw (pre-business-rules) prediction."""
        df = pd.DataFrame([request])
        df_feat = engineer_features(df)
        X, _ = build_design_matrix(df_feat, fit_columns=self.feature_columns)
        pred = self.model.predict(X)[0]
        return float(pred)

    def predict_full(self, request: dict) -> PriceAdjustmentResult:
        """Raw model prediction -> competitor blend -> inventory/demand
        adjustment rules -> final recommended price (see pricing_logic.py)."""
        raw_pred = self.predict_raw(request)
        return recommend_price(
            model_prediction=raw_pred,
            competitor_price=request["competitor_price"],
            inventory_level=request["inventory_level"],
            demand_signal=request["demand_signal"],
        )


# Convenience singleton-style loader used by app/main.py. Importing this
# module does NOT load the model automatically (avoids surprises when the
# module is imported before training has run) -- call `get_model()` once at
# app startup instead.
_model_instance: PricingModel | None = None


def get_model() -> PricingModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = PricingModel()
    return _model_instance
