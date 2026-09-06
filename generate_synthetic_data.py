"""
Generates a synthetic grid-cell dataset shaped like the real thing, so you can
build and test the full pipeline before real IMD/GSI/DEM data is wired in.

Swap this out for real loaders later:
  - slope/aspect/elevation  -> derived once from an SRTM/Cartosat DEM (rasterio + richdem)
  - rainfall_1d/3d/7d       -> IMD API or gridded rainfall product, aggregated per cell/day
  - ndvi                    -> Sentinel-2 via Google Earth Engine or Bhuvan
  - label                   -> GSI landslide inventory / state DM incident records,
                               joined to the same grid + date

Each row = one grid cell on one day.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def generate(n_cells: int = 400, n_days: int = 365, out_path: str = "grid_data.csv") -> pd.DataFrame:
    cell_ids = np.arange(n_cells)

    # Static per-cell terrain features (generated once per cell, repeated across days)
    slope = RNG.uniform(5, 60, n_cells)          # degrees
    elevation = RNG.uniform(200, 2500, n_cells)  # meters
    aspect = RNG.uniform(0, 360, n_cells)        # degrees
    dist_to_road_m = RNG.uniform(0, 3000, n_cells)
    dist_to_past_incident_m = RNG.uniform(0, 20000, n_cells)

    rows = []
    for day in range(n_days):
        # Rainfall is seasonal: heavier and more variable in monsoon months (day 90-240)
        monsoon = 1.5 if 90 <= day <= 240 else 0.4
        daily_rain = RNG.gamma(shape=2.0, scale=8.0 * monsoon, size=n_cells)

        for i in range(n_cells):
            rows.append({
                "cell_id": cell_ids[i],
                "day": day,
                "slope_deg": slope[i],
                "elevation_m": elevation[i],
                "aspect_deg": aspect[i],
                "dist_to_road_m": dist_to_road_m[i],
                "dist_to_past_incident_m": dist_to_past_incident_m[i],
                "rain_today_mm": daily_rain[i],
            })

    df = pd.DataFrame(rows)

    # Rolling rainfall windows, computed per cell over time
    df = df.sort_values(["cell_id", "day"])
    df["rain_1d"] = df.groupby("cell_id")["rain_today_mm"].shift(0)
    df["rain_3d"] = df.groupby("cell_id")["rain_today_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["rain_7d"] = df.groupby("cell_id")["rain_today_mm"].transform(lambda s: s.rolling(7, min_periods=1).sum())

    # Soil moisture and NDVI as noisy proxies (real versions come from sensors / Sentinel-2)
    df["soil_moisture"] = np.clip(0.15 + 0.01 * df["rain_7d"] + RNG.normal(0, 0.05, len(df)), 0, 1)
    df["ndvi"] = np.clip(RNG.normal(0.5, 0.15, len(df)), -1, 1)  # lower = deforested/bare slope

    # Synthetic ground truth: risk rises with slope, rainfall, soil saturation, low NDVI, road proximity
    logit = (
        -8
        + 0.05 * df["slope_deg"]
        + 0.015 * df["rain_3d"]
        + 3.0 * df["soil_moisture"]
        - 1.0 * df["ndvi"]
        - 0.0004 * df["dist_to_road_m"]
    )
    prob = 1 / (1 + np.exp(-logit))
    df["landslide"] = (RNG.uniform(0, 1, len(df)) < prob).astype(int)

    df = df.drop(columns=["rain_today_mm"])
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({df['landslide'].sum()} positive) to {out_path}")
    return df


if __name__ == "__main__":
    generate()
