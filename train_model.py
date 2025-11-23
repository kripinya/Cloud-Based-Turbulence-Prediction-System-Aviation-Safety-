"""
train_model.py
Simple pipeline:
 - fetch ERA5-like hourly data via Open-Meteo archive API (example city)
 - feature engineering (wind_shear, TPI proxy)
 - label by simple thresholds into Low/Moderate/Severe
 - train RandomForest, evaluate, save model artifact and scaler
"""

import os
import joblib
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from utils import fetch_era5_hourly, make_features_and_labels

MODEL_DIR = "model_artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_for_location(lat, lon, start_date, end_date, save_name="rf_model.joblib"):
    print(f"Fetching data for {lat},{lon} from {start_date} to {end_date}")
    df = fetch_era5_hourly(lat, lon, start_date, end_date)
    if df is None or df.empty:
        raise RuntimeError("No data returned from fetch_era5_hourly")

    print("Preparing features and labels")
    X, y = make_features_and_labels(df)

    print("Train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Scaling features")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training RandomForest (quick default)")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)

    print("Evaluating")
    y_pred = rf.predict(X_test_s)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save artifacts
    model_path = os.path.join(MODEL_DIR, save_name)
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")

    return model_path, scaler_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=28.6139, help="Latitude (default Delhi)")
    parser.add_argument("--lon", type=float, default=77.2090, help="Longitude (default Delhi)")
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--out", type=str, default="rf_model.joblib", help="Saved model name")
    args = parser.parse_args()

    # default: last 30 days if not provided
    if args.end is None:
        end = datetime.utcnow().date()
    else:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if args.start is None:
        start = end - timedelta(days=30)
    else:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()

    train_for_location(args.lat, args.lon, start.isoformat(), end.isoformat(), save_name=args.out)
