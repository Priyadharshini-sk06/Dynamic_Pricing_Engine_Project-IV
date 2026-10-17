"""
app/main.py
-----------
FastAPI backend for the Dynamic Pricing Engine.

Endpoints:
  GET  /health          -- basic health check
  POST /predict-price   -- returns a recommended price for a product given
                            its current market/inventory/demand signals

Run locally:
    uvicorn app.main:app --reload --port 8000
    (run from the project root, dynamic-pricing-engine/)

The trained model is loaded ONCE at startup (not per-request) via the
FastAPI lifespan handler. CORS is enabled for local development so the
plain HTML/JS frontend (served separately, e.g. via `python -m http.server`)
can call this API from a different origin/port.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Make src/ importable regardless of the working directory this is launched from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import get_model  # noqa: E402

ProductCategory = Literal[
    "Electronics", "Apparel", "Home & Kitchen",
    "Beauty", "Sports & Outdoors", "Grocery", "Toys & Games",
]
DayOfWeek = Literal[
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]
CustomerSegment = Literal["Budget", "Regular", "Premium"]


class PredictRequest(BaseModel):
    product_category: ProductCategory
    base_price: float = Field(..., gt=0, description="The product's current/reference price.")
    competitor_price: float = Field(..., gt=0, description="Observed competitor price for the same/similar product.")
    inventory_level: int = Field(..., ge=0, description="Units currently in stock.")
    demand_signal: float = Field(..., ge=0, description="Recent relative demand, e.g. click/order volume over the last period.")
    day_of_week: DayOfWeek = Field(..., description="Used as a seasonality proxy.")
    is_holiday: bool = Field(default=False, description="Whether the pricing date falls on/near a major shopping holiday.")
    customer_segment: CustomerSegment

    model_config = {
        "json_schema_extra": {
            "example": {
                "product_category": "Electronics",
                "base_price": 249.99,
                "competitor_price": 239.50,
                "inventory_level": 340,
                "demand_signal": 145,
                "day_of_week": "Friday",
                "is_holiday": False,
                "customer_segment": "Regular",
            }
        }
    }


class AdjustmentBreakdown(BaseModel):
    raw_model_prediction: float
    competitor_blended_price: float
    inventory_adjustment_pct: float
    demand_adjustment_pct: float


class PredictResponse(BaseModel):
    recommended_price: float
    raw_model_prediction: float
    adjustments: AdjustmentBreakdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loads the model + feature-column layout once, at process startup.
    app.state.model = get_model()
    print("Model loaded successfully. API ready.")
    yield


app = FastAPI(
    title="Dynamic Pricing Engine API",
    description="Predicts a recommended product price from market, inventory and demand signals.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local/demo project -- open CORS so the static frontend can call this from any local port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.post("/predict-price", response_model=PredictResponse)
def predict_price(request: PredictRequest):
    try:
        result = app.state.model.predict_full(request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictResponse(
        recommended_price=result.final_recommended_price,
        raw_model_prediction=result.raw_model_prediction,
        adjustments=AdjustmentBreakdown(
            raw_model_prediction=result.raw_model_prediction,
            competitor_blended_price=result.competitor_blended_price,
            inventory_adjustment_pct=result.inventory_adjustment_pct,
            demand_adjustment_pct=result.demand_adjustment_pct,
        ),
    )


# Optionally serve the static frontend directly from FastAPI at "/" so the
# whole app can run from a single process. If the frontend/ folder doesn't
# exist (e.g. running the API standalone) this is skipped gracefully.
_frontend_dir = ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
