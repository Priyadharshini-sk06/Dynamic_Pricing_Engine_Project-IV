"""
train_model.py
---------------
End-to-end training pipeline for the Dynamic Pricing Engine:

  1. Load data/pricing_data.csv
  2. Feature engineering (src/features.py) -- categorical encoding +
     price/demand elasticity-proxy features
  3. Train/validation/test split
  4. Baseline model: RandomForestRegressor
  5. Main model: XGBoostRegressor, hyperparameter-tuned with Optuna
  6. Evaluate both (MAE, RMSE, R^2) on the same held-out test set
  7. Save the final model + supporting artifacts to models/

Run: python scripts/train_model.py   (from the project root)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # reusable feature-engineering code lives in src/
from features import engineer_features, build_design_matrix, CATEGORICAL_COLS  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep console output clean

DATA_PATH = ROOT / "data" / "pricing_data.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_OPTUNA_TRIALS = 30


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df_feat = engineer_features(df)
    X, feature_cols = build_design_matrix(df_feat)
    y = df["optimal_price"]
    return X, y, feature_cols


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAPE_%": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 2),
    }


def train_baseline_rf(X_train, y_train):
    print("\n--- Training baseline: RandomForestRegressor ---")
    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = N_OPTUNA_TRIALS):
    """Optuna hyperparameter search for XGBoost, minimizing validation MAE."""
    print(f"\n--- Tuning XGBoost with Optuna ({n_trials} trials) ---")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - start
    print(f"Optuna finished in {elapsed:.1f}s | best val MAE={study.best_value:.3f}")
    print("Best params:", study.best_params)
    return study.best_params


def main():
    X, y, feature_cols = load_and_prepare()
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")

    # 80% train+val / 20% test, then split train+val into train/val (80/20 of that)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ---------------- Baseline: Random Forest ----------------
    rf_model = train_baseline_rf(X_train, y_train)
    rf_test_pred = rf_model.predict(X_test)
    rf_metrics = metrics(y_test, rf_test_pred)
    print("Random Forest (baseline) test metrics:", rf_metrics)

    # ---------------- Main model: XGBoost + Optuna ----------------
    best_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=N_OPTUNA_TRIALS)

    # Refit best XGBoost on train+val combined for the final model, then
    # evaluate once, honestly, on the untouched test set.
    final_xgb = xgb.XGBRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    final_xgb.fit(X_trainval, y_trainval)
    xgb_test_pred = final_xgb.predict(X_test)
    xgb_metrics = metrics(y_test, xgb_test_pred)
    print("\nXGBoost (tuned) test metrics:", xgb_metrics)

    print("\n=== Final comparison (held-out test set) ===")
    print(f"{'Model':<20}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'MAPE%':>10}")
    print(f"{'RandomForest':<20}{rf_metrics['MAE']:>10}{rf_metrics['RMSE']:>10}{rf_metrics['R2']:>10}{rf_metrics['MAPE_%']:>10}")
    print(f"{'XGBoost (tuned)':<20}{xgb_metrics['MAE']:>10}{xgb_metrics['RMSE']:>10}{xgb_metrics['R2']:>10}{xgb_metrics['MAPE_%']:>10}")

    note = (
        "Note: MAE/RMSE are in raw currency units, and this dataset spans "
        "categories from ~$5 (Grocery) to ~$800+ (Electronics) base prices, "
        "so absolute error is naturally larger than a single-category "
        "dataset would produce -- MAPE and R^2 are the more meaningful "
        "cross-category metrics here. R^2 in the ~75-85% range is the "
        "explicit target for this project (see README), not a maximum "
        "achievable ceiling to push past."
    )
    print("\n" + note)

    # ---------------- Save artifacts ----------------
    joblib.dump(final_xgb, MODELS_DIR / "xgb_pricing_model.joblib")
    joblib.dump(rf_model, MODELS_DIR / "rf_baseline_model.joblib")

    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({
            "random_forest_baseline": rf_metrics,
            "xgboost_tuned": xgb_metrics,
            "best_xgb_params": best_params,
            "note": note,
        }, f, indent=2)

    print(f"\nSaved model artifacts to {MODELS_DIR}")
    print(" - xgb_pricing_model.joblib   (final model served by the API)")
    print(" - rf_baseline_model.joblib   (baseline, for comparison)")
    print(" - feature_columns.json       (exact column layout the model expects)")
    print(" - metrics.json               (evaluation summary)")


if __name__ == "__main__":
    main()
