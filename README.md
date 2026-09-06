# Landslide risk model prototype

A working, runnable skeleton of the risk-prediction pipeline: synthetic data
generator -> baseline rule -> XGBoost model -> scoring API. Swap the synthetic
data for real sources when you have access (see below).

## Run it

```bash
pip install xgboost scikit-learn pandas numpy fastapi uvicorn

python generate_synthetic_data.py   # writes grid_data.csv
python baseline_model.py            # prints baseline precision/recall
python train_model.py               # trains XGBoost, prints PR curve + feature importance, saves landslide_model.joblib
uvicorn api:app --reload            # serves POST /predict on localhost:8000
```

## Files

- `generate_synthetic_data.py` — creates a realistic-shaped dataset (grid cells x days) so you can build and test the whole pipeline before real data is wired in. Comments show exactly what to replace with a real source.
- `baseline_model.py` — the simple slope+rainfall rule your ML model needs to beat.
- `train_model.py` — XGBoost with a **time-based split** (train on older dates, test on newer — a random split would leak future rainfall into training) and **precision-recall evaluation** (accuracy is meaningless here since positives are ~1-3% of rows).
- `api.py` — FastAPI endpoint that takes a cell's features and returns a risk probability + severity band. This is what your GIS dashboard and alert engine would call.

## Note on the numbers you'll see

Because the synthetic data has a strong seasonal pattern (heavy rain concentrated in a "monsoon" window) and the time-based test split falls in a lower-rainfall period, the reported precision/recall will look weak — this is realistic in one useful way: it demonstrates why time-based splits matter and why seasonal distribution shift is a real problem you'll hit with actual data too. Don't read the specific numbers as a performance target; read the *shapes* (the code structure, the evaluation approach, the threshold trade-off table) as the reusable part.

## Swapping in real data

| Feature | Synthetic source (here) | Real source |
|---|---|---|
| slope, aspect, elevation | random per cell | SRTM or Cartosat DEM, processed once with `rasterio` + `richdem` or QGIS |
| rainfall (1d/3d/7d) | gamma-distributed random | IMD API (apply for access), or gridded rainfall products (IMD gridded data, CHIRPS) |
| soil moisture | noisy function of rainfall | field IoT sensors where deployed; SMAP satellite soil moisture as a fallback for ungauged cells |
| NDVI | random | Sentinel-2 via Google Earth Engine (free tier) or ISRO Bhuvan |
| landslide labels | logistic function of features | Geological Survey of India (GSI) landslide inventory; state disaster management department incident logs |

### Getting access
- **IMD**: rainfall/weather API access typically requires registering on the IMD data portal (mausam.imd.gov.in) or going through NDMA/state DM department channels — start this early, it's not instant.
- **GSI landslide inventory**: available through GSI's Bhukosh portal and periodic published reports; state DM departments often hold more granular local incident logs than the national inventory.
- **DEM/Sentinel**: SRTM (30m) is free via USGS EarthExplorer; Sentinel-2 imagery is free via Copernicus Open Access Hub or Google Earth Engine — no registration friction here, so these are the easiest to start with.

## Next steps once real data is in

1. Recompute feature importance — soil moisture and rainfall should dominate; if terrain features dominate instead, your rainfall data is probably too coarse (e.g. district-level instead of cell-level).
2. Re-tune the severity thresholds in `api.py` against real precision/recall trade-offs, in consultation with the district officials who'll act on the alerts.
3. Add the confirmed/false-alarm feedback field to your alert schema (see main conversation) so you can retrain on real outcomes.
