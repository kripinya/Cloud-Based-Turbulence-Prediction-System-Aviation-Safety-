# utils.py
import requests
import pandas as pd
import numpy as np

def fetch_era5_hourly(lat, lon, start_date, end_date):
    """
    Use open-meteo's ERA5 archive endpoint (free) to pull hourly variables.
    Returns DataFrame or None on error.
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
            "cloud_cover"
        ],
        # optionally add "timezone": "UTC"
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print("Open-Meteo API error:", r.status_code, r.text[:200])
        return None
    j = r.json()
    if "hourly" not in j:
        print("Unexpected response format:", j)
        return None
    df = pd.DataFrame(j["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

def make_features_and_labels(df):
    """
    Input: raw ERA5-like df with columns:
      temperature_2m, dewpoint_2m, surface_pressure,
      wind_speed_10m, wind_speed_100m, relative_humidity_2m, cloud_cover
    Output: X (DataFrame), y (Series labels Low/Moderate/Severe)
    """
    # fill short gaps
    df = df.copy()
    df = df.interpolate(limit=3).dropna()

    # features
    df["wind_shear"] = (df["wind_speed_100m"] - df["wind_speed_10m"]).abs()
    # simple instability proxy (you can refine later)
    df["instability"] = 0.5 * df["wind_shear"] + 0.3 * (100 - df["relative_humidity_2m"]) + 0.2 * df["cloud_cover"]
    # lapse proxy: approximate from small diffs (if vertical temps available use that)
    # use dewpoint depression
    df["dewpt_dep"] = df["temperature_2m"] - df["dewpoint_2m"]

    features = ["wind_speed_10m", "wind_speed_100m", "wind_shear", "relative_humidity_2m", "cloud_cover", "surface_pressure", "dewpt_dep"]
    X = df[features].copy()

    # simple thresholding for labels (adjust thresholds after inspection)
    bins = [-999, 10, 25, 999]
    labels = ["Low", "Moderate", "Severe"]
    y = pd.cut(df["instability"], bins=bins, labels=labels)
    # drop any rows with NA labels
    mask = y.notna()
    return X.loc[mask], y.loc[mask]
