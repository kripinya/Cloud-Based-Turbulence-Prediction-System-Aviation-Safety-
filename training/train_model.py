"""
train_model.py — Multi-location RF training pipeline with cross-validation.

Pipeline:
  1. Fetch ERA5 hourly data for one or more locations via Open-Meteo.
  2. Derive meteorological features and physics-based turbulence labels.
  3. Scale features → train RandomForest with stratified k-fold CV.
  4. Evaluate on held-out test set → save model + scaler artifacts.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

try:
    from training.utils import (
        fetch_era5_hourly,
        fetch_multi_location,
        make_features_and_labels,
        FEATURE_COLUMNS,
    )
except ImportError:
    from utils import (
        fetch_era5_hourly,
        fetch_multi_location,
        make_features_and_labels,
        FEATURE_COLUMNS,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve model_artifacts relative to the project root (one level up from training/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_PROJECT_ROOT, "model_artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

# Default multi-location set — major Indian airports / turbulence-prone regions
DEFAULT_LOCATIONS = [
    (28.6139, 77.2090),   # Delhi (IGI)
    (19.0896, 72.8656),   # Mumbai (CSIA)
    (12.9716, 77.5946),   # Bangalore (KIA)
    (22.5726, 88.3639),   # Kolkata (NSCBI)
    (13.0827, 80.2707),   # Chennai (MAA)
]


def train_pipeline(
    locations,
    start_date,
    end_date,
    save_name="rf_model.joblib",
    n_estimators=300,
    cv_folds=5,
):
    """
    End-to-end training pipeline with cross-validation.

    Parameters
    ----------
    locations : list of (lat, lon)
    start_date, end_date : str  (YYYY-MM-DD)
    save_name : str             Output model filename
    n_estimators : int          RF tree count
    cv_folds : int              Stratified k-fold splits
    """
    # ── 1. Data collection ───────────────────────────────────────────────
    if len(locations) == 1:
        lat, lon = locations[0]
        logger.info("Single-location mode: (%.4f, %.4f)", lat, lon)
        raw_df = fetch_era5_hourly(lat, lon, start_date, end_date)
        if raw_df is None or raw_df.empty:
            raise RuntimeError("No data returned for the given location/date range.")
    else:
        logger.info("Multi-location mode: %d locations", len(locations))
        raw_df = fetch_multi_location(locations, start_date, end_date)
        if raw_df.empty:
            raise RuntimeError("No data returned for any location.")

    logger.info("Total raw records: %d", len(raw_df))

    # ── 2. Feature engineering & labeling ─────────────────────────────────
    X, y = make_features_and_labels(raw_df)
    logger.info("Feature matrix shape: %s  |  Label distribution:", X.shape)
    logger.info("\n%s", y.value_counts().to_string())

    # ── 3. Train / test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ── 4. Scaling ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── 5. Cross-validation ──────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Running %d-fold stratified cross-validation...", cv_folds)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_s, y_train, cv=cv, scoring="f1_macro")
    logger.info(
        "CV F1-macro: %.4f ± %.4f  (per-fold: %s)",
        cv_scores.mean(),
        cv_scores.std(),
        np.round(cv_scores, 4).tolist(),
    )

    # ── 6. Final training on full training set ───────────────────────────
    logger.info("Training final model on full training set...")
    rf.fit(X_train_s, y_train)

    # ── 7. Evaluation ────────────────────────────────────────────────────
    y_pred = rf.predict(X_test_s)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Test set classification report:\n%s", report)
    logger.info("Confusion matrix:\n%s", cm)

    # Feature importances
    importances = dict(zip(FEATURE_COLUMNS, rf.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("Feature importances:")
    for feat, imp in sorted_imp:
        logger.info("  %-25s %.4f", feat, imp)

    # ── 8. Save artifacts ────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, save_name)
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Saved model  → %s", model_path)
    logger.info("Saved scaler → %s", scaler_path)

    # Save training metadata
    meta = {
        "locations": locations,
        "date_range": [start_date, end_date],
        "total_samples": len(X),
        "features": FEATURE_COLUMNS,
        "cv_folds": cv_folds,
        "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_macro_std": round(float(cv_scores.std()), 4),
        "n_estimators": n_estimators,
        "feature_importances": {k: round(v, 4) for k, v in sorted_imp},
    }
    meta_path = os.path.join(MODEL_DIR, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata → %s", meta_path)

    return model_path, scaler_path


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train turbulence prediction model on ERA5 data."
    )
    parser.add_argument(
        "--lat", type=float, default=None,
        help="Latitude for single-location mode (default: multi-location).",
    )
    parser.add_argument(
        "--lon", type=float, default=None,
        help="Longitude for single-location mode.",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--multi", action="store_true",
        help="Train on 5 major Indian airports (default if no --lat/--lon given).",
    )
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds (default: 5)")
    parser.add_argument("--trees", type=int, default=300, help="Number of RF estimators")
    parser.add_argument("--out", type=str, default="rf_model.joblib", help="Model filename")
    args = parser.parse_args()

    # Date defaults: last 180 days
    if args.end is None:
        end = datetime.utcnow().date()
    else:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if args.start is None:
        start = end - timedelta(days=180)
    else:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()

    # Location selection
    if args.lat is not None and args.lon is not None:
        locs = [(args.lat, args.lon)]
    else:
        locs = DEFAULT_LOCATIONS
        logger.info("No --lat/--lon provided; using %d default locations.", len(locs))

    train_pipeline(
        locations=locs,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        save_name=args.out,
        n_estimators=args.trees,
        cv_folds=args.cv,
    )
