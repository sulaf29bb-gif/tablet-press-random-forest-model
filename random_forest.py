#!/usr/bin/env python3
"""
Random Forest regression: tablet press process parameters -> tablet quality attributes.

Inputs  (process parameters): press force, motor speed, particle size
Outputs (tablet attributes):  hardness, height, friability

One multi-output Random Forest is trained on rows 1-80 of the dataset, tested on
rows 81-90 and validated on rows 91-100 (the same split the whole group used).
For each split the script prints R2 / MAE / MSE per output, saves them to
results/metrics.csv and writes predicted-vs-actual plots to results/.

Usage:  python random_forest.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files; no display needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- Configuration -----------------------------------------------------------
DATA_FILE = Path("data/tablet_examination.xlsx")
OUT_DIR = Path("results")

FEATURES = ["Press Force (kg)", "Motor Speed (tablets/min)", "Particle Size (μm)"]
TARGETS = {  # column name -> short name used in file names
    "Tablet Hardness (N)": "Hardness",
    "Tablet Height (mm)": "Height",
    "Tablet Friability (%)": "Friability",
}

N_TRAIN = 80   # rows 1-80   -> training
N_TEST = 10    # rows 81-90  -> test; rows 91-100 -> validation
N_TREES = 100
SEED = 42
MODEL_NAME = "Random Forest Regressor"
# -----------------------------------------------------------------------------


def load_data():
    df = pd.read_excel(DATA_FILE)
    X = df[FEATURES]
    y = df[list(TARGETS)]
    a, b = N_TRAIN, N_TRAIN + N_TEST
    return {
        "train": (X.iloc[:a], y.iloc[:a]),
        "test": (X.iloc[a:b], y.iloc[a:b]),
        "val": (X.iloc[b:], y.iloc[b:]),
    }


def evaluate(split, y_true, y_pred):
    """Return one row of R2 / MAE / MSE per target for this split."""
    rows = []
    for i, col in enumerate(TARGETS):
        rows.append({
            "Split": split,
            "Target": col,
            "R2": r2_score(y_true.iloc[:, i], y_pred[:, i]),
            "MAE": mean_absolute_error(y_true.iloc[:, i], y_pred[:, i]),
            "MSE": mean_squared_error(y_true.iloc[:, i], y_pred[:, i]),
        })
    return rows


def plot_predictions(split_label, file_tag, y_true, y_pred):
    """Predicted-vs-actual scatter plot for every target (ideal = diagonal line)."""
    for i, (col, short) in enumerate(TARGETS.items()):
        actual, predicted = y_true.iloc[:, i], y_pred[:, i]
        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())

        plt.figure()
        plt.scatter(actual, predicted)
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel(f"Actual {col}")
        plt.ylabel(f"Predicted {col}")
        plt.title(f"Expected vs Actual - {split_label} {col}, {MODEL_NAME}")
        plt.savefig(OUT_DIR / f"RandomForest{file_tag}{short}.png", bbox_inches="tight")
        plt.close()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    X_train, y_train = data["train"]
    print({k: len(v[0]) for k, v in data.items()}, "rows per split")

    # The forest is ONE multi-output model: its splits minimise the squared error
    # summed over all three targets. Hardness (tens of N) would dominate height and
    # friability, so the targets are standardised for training (fitted on the
    # training rows only) and the predictions are converted back afterwards.
    scaler_y = StandardScaler().fit(y_train)
    model = RandomForestRegressor(n_estimators=N_TREES, random_state=SEED)
    model.fit(X_train, scaler_y.transform(y_train))

    rows = []
    for split, label, tag in (("test", "TEST", "Test"), ("val", "VALIDATION", "Val")):
        X_split, y_split = data[split]
        y_pred = scaler_y.inverse_transform(model.predict(X_split))
        rows += evaluate(split, y_split, y_pred)
        plot_predictions(label, tag, y_split, y_pred)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "metrics.csv", index=False, float_format="%.4f")
    print("\n", results.round(4).to_string(index=False))

    print("\nFeature importance (joint model, all three targets):")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name}: {imp:.3f}")

    print(f"\nSaved plots and metrics.csv to {OUT_DIR}/")


if __name__ == "__main__":
    main()
