"""
Simple rule-based baseline: flag a cell as high-risk if slope and recent
rainfall both exceed a threshold. Any ML model you build should beat this
on precision-recall before you trust it over the simple rule.
"""

import pandas as pd


def baseline_predict(df: pd.DataFrame, slope_threshold: float = 30.0, rain_3d_threshold: float = 60.0) -> pd.Series:
    return ((df["slope_deg"] > slope_threshold) & (df["rain_3d"] > rain_3d_threshold)).astype(int)


if __name__ == "__main__":
    from sklearn.metrics import precision_score, recall_score, f1_score

    df = pd.read_csv("grid_data.csv")
    preds = baseline_predict(df)

    print("Baseline rule performance (slope > 30deg AND 3-day rain > 60mm):")
    print(f"  Precision: {precision_score(df['landslide'], preds):.3f}")
    print(f"  Recall:    {recall_score(df['landslide'], preds):.3f}")
    print(f"  F1:        {f1_score(df['landslide'], preds):.3f}")
