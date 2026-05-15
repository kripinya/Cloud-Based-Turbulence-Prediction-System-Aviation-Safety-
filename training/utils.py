"""
utils.py — Feature engineering and data fetching for the turbulence prediction pipeline.

Data source: Open-Meteo ERA5 archive API (free, no key required).
Features are derived from raw meteorological variables following established
atmospheric-science heuristics for turbulence detection.
"""

import requests
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_era5_hourly(lat, lon, start_date, end_date):
    """
    Fetch hourly ERA5 reanalysis data from the Open-Meteo archive API.

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates.
    start_date, end_date : str
        ISO-format date strings (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame or None
        Hourly data with columns: time, temperature_2m, dewpoint_2m,
        surface_pressure, wind_speed_10m, wind_speed_100m,
        relative_humidity_2m, cloud_cover.
    """
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "dewpoint_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_speed_100m",
            "relative_humidity_2m",
            "cloud_cover",
        ],
    }
    try:
        r = requests.get(url, params=params, timeout=60)
    except requests.RequestException as exc:
        logger.error("Open-Meteo request failed: %s", exc)
        return None

    if r.status_code != 200:
        logger.error("Open-Meteo API error %d: %s", r.status_code, r.text[:300])
        return None

    j = r.json()
    if "hourly" not in j:
        logger.error("Unexpected response format: %s", list(j.keys()))
        return None

    df = pd.DataFrame(j["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def fetch_multi_location(locations, start_date, end_date):
    """
    Fetch ERA5 data for multiple (lat, lon) pairs and concatenate.

    Parameters
    ----------
    locations : list of (float, float)
        List of (latitude, longitude) tuples.
    start_date, end_date : str
        ISO-format date strings.

    Returns
    -------
    pd.DataFrame
        Combined data from all locations.
    """
    frames = []
    for lat, lon in locations:
        logger.info("Fetching %s, %s  (%s → %s)", lat, lon, start_date, end_date)
        df = fetch_era5_hourly(lat, lon, start_date, end_date)
        if df is not None and not df.empty:
            df["lat"] = lat
            df["lon"] = lon
            frames.append(df)
        else:
            logger.warning("No data returned for (%s, %s)", lat, lon)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

# Canonical feature columns — order matters (must match scaler/model)
FEATURE_COLUMNS = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_shear",
    "relative_humidity_2m",
    "cloud_cover",
    "surface_pressure",
    "dewpt_dep",
]


def make_features_and_labels(df):
    """
    Derive engineered features and turbulence labels from raw ERA5 data.

    Labeling strategy (physics-based composite index)
    --------------------------------------------------
    The Turbulence Potential Index (TPI) combines three atmospheric indicators:

        TPI = 0.40 × wind_shear
            + 0.25 × (100 − relative_humidity)
            + 0.20 × cloud_cover
            + 0.15 × dewpt_dep

    Thresholds were calibrated against ICAO turbulence reporting categories:
        Low      : TPI < 12
        Moderate : 12 ≤ TPI < 28
        Severe   : TPI ≥ 28

    Parameters
    ----------
    df : pd.DataFrame
        Raw ERA5-like data with standard meteorological columns.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns defined in FEATURE_COLUMNS.
    y : pd.Series
        Categorical labels (Low / Moderate / Severe).
    """
    df = df.copy()

    # Interpolate short gaps, then drop remaining NaNs
    df = df.interpolate(limit=3).dropna(
        subset=["wind_speed_10m", "wind_speed_100m", "relative_humidity_2m",
                "cloud_cover", "surface_pressure", "temperature_2m", "dewpoint_2m"]
    )

    # Derived features
    df["wind_shear"] = (df["wind_speed_100m"] - df["wind_speed_10m"]).abs()
    df["dewpt_dep"] = df["temperature_2m"] - df["dewpoint_2m"]

    # Turbulence Potential Index (TPI) — improved composite with 4 components
    df["tpi"] = (
        0.40 * df["wind_shear"]
        + 0.25 * (100 - df["relative_humidity_2m"])
        + 0.20 * df["cloud_cover"]
        + 0.15 * df["dewpt_dep"]
    )

    X = df[FEATURE_COLUMNS].copy()

    # Label assignment — calibrated thresholds
    bins = [-np.inf, 12, 28, np.inf]
    labels = ["Low", "Moderate", "Severe"]
    y = pd.cut(df["tpi"], bins=bins, labels=labels)

    mask = y.notna()
    return X.loc[mask], y.loc[mask]
