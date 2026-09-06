"""
Minimal scoring API. Run with:  uvicorn api:app --reload

POST /predict with the feature values for a grid cell and current conditions,
get back a risk probability and severity band. This is the endpoint your
GIS dashboard and alert engine would call.
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Landslide Risk Scoring API")
model = joblib.load("landslide_model.joblib")

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


class CellFeatures(BaseModel):
    cell_id: int
    slope_deg: float
    elevation_m: float
    aspect_deg: float
    dist_to_road_m: float
    dist_to_past_incident_m: float
    rain_1d: float
    rain_3d: float
    rain_7d: float
    soil_moisture: float
    ndvi: float


def severity_band(prob: float) -> str:
    # Placeholder thresholds -- calibrate these against real precision/recall
    # trade-offs once you have real evaluation data, not synthetic data.
    if prob >= 0.6:
        return "severe"
    if prob >= 0.4:
        return "high"
    if prob >= 0.2:
        return "moderate"
    return "low"


@app.post("/predict")
def predict(cell: CellFeatures):
    row = pd.DataFrame([cell.dict()])[FEATURES]
    prob = float(model.predict_proba(row)[0, 1])
    return {
        "cell_id": cell.cell_id,
        "risk_probability": round(prob, 4),
        "severity": severity_band(prob),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
