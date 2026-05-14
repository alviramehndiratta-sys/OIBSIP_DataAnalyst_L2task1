"""
House Price Prediction — Linear Regression
==========================================
Train, evaluate, and save a Linear Regression model on the Housing dataset.

Usage:
    python train.py
    python train.py --data data/Housing.csv --output outputs/
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib


# ── Feature engineering ────────────────────────────────────────────────────────

BINARY_COLS = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea",
]

FURNISH_MAP = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}


def load_and_preprocess(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, encode categoricals, return (X, y)."""
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

    for col in BINARY_COLS:
        df[col] = (df[col] == "yes").astype(int)

    df["furnishingstatus"] = df["furnishingstatus"].map(FURNISH_MAP)

    X = df.drop("price", axis=1)
    y = df["price"]
    return X, y


# ── Training ───────────────────────────────────────────────────────────────────

def train(data_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/4] Loading data …")
    X, y = load_and_preprocess(data_path)

    print("[2/4] Splitting 80/20 …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("[3/4] Fitting LinearRegression …")
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("[4/4] Evaluating …")
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))

    metrics = dict(r2=round(r2, 4), rmse=round(rmse), mae=round(mae),
                   train_size=len(X_train), test_size=len(X_test),
                   features=X.shape[1])

    print(f"\n  R²   : {r2:.4f}")
    print(f"  RMSE : ₹{rmse:,.0f}")
    print(f"  MAE  : ₹{mae:,.0f}")

    # Coefficient table
    coef_df = (
        pd.DataFrame({"feature": X.columns, "coefficient": model.coef_})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
    print("\n  Feature coefficients (sorted by importance):")
    print(coef_df.to_string(index=False))

    # Persist artefacts
    model_path   = os.path.join(output_dir, "model.pkl")
    metrics_path = os.path.join(output_dir, "metrics.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Model   → {model_path}")
    print(f"  Metrics → {metrics_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train house-price regressor")
    parser.add_argument("--data",   default="data/Housing.csv", help="Path to CSV")
    parser.add_argument("--output", default="outputs/",         help="Output directory")
    args = parser.parse_args()

    train(args.data, args.output)
