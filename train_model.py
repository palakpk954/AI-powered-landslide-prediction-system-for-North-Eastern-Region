"""
Trains an XGBoost classifier on grid-cell features to predict landslide risk.

Key choices explained inline:
  - time-based train/test split (not random) to avoid leaking future rainfall
    patterns into training
  - scale_pos_weight to handle severe class imbalance
  - evaluated on precision-recall, not accuracy, since positives are rare
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

FEATURES = [
    "slope_deg",
    "elevation_m",
    "aspect_deg",
    "dist_to_road_m",
    "dist_to_past_incident_m",
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "soil_moisture",
    "ndvi",
]
TARGET = "landslide"


def time_based_split(df: pd.DataFrame, split_day: int):
    train = df[df["day"] < split_day]
    test = df[df["day"] >= split_day]
    return train, test


def train(csv_path: str = "grid_data.csv", model_out: str = "landslide_model.joblib"):
    df = pd.read_csv(csv_path)
    train_df, test_df = time_based_split(df, split_day=int(df["day"].max() * 0.8))

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, probs)
    print(f"Test set: {len(X_test)} rows, {y_test.sum()} positive")
    print(f"Average precision (PR-AUC): {ap:.3f}")

    # Show precision/recall at a few candidate thresholds so you can pick
    # an operating point based on how many false alarms are tolerable
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    print("\nThreshold  Precision  Recall")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (probs >= t).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        print(f"  {t:.2f}      {p:.3f}      {r:.3f}")

    print("\nFeature importance:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:28s} {imp:.3f}")

    joblib.dump(model, model_out)
    print(f"\nModel saved to {model_out}")
    return model


if __name__ == "__main__":
    train()
