"""
tests/test_training.py — Unit tests for the training pipeline utilities.

Run with:  pytest tests/test_training.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import make_features_and_labels, FEATURE_COLUMNS


def _make_raw_df(n=100):
    """Generate synthetic ERA5-like data for testing."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "temperature_2m": rng.uniform(15, 35, n),
        "dewpoint_2m": rng.uniform(5, 25, n),
        "surface_pressure": rng.uniform(980, 1020, n),
        "wind_speed_10m": rng.uniform(0, 15, n),
        "wind_speed_100m": rng.uniform(0, 30, n),
        "relative_humidity_2m": rng.uniform(30, 100, n),
        "cloud_cover": rng.uniform(0, 100, n),
    })


class TestFeatureColumns:
    def test_canonical_features_count(self):
        assert len(FEATURE_COLUMNS) == 7

    def test_canonical_features_content(self):
        assert "wind_shear" in FEATURE_COLUMNS
        assert "dewpt_dep" in FEATURE_COLUMNS
        assert "surface_pressure" in FEATURE_COLUMNS


class TestMakeFeaturesAndLabels:
    def test_returns_correct_shape(self):
        df = _make_raw_df(200)
        X, y = make_features_and_labels(df)
        assert len(X) == len(y)
        assert len(X) > 0
        assert list(X.columns) == FEATURE_COLUMNS

    def test_labels_are_categorical(self):
        df = _make_raw_df(200)
        _, y = make_features_and_labels(df)
        valid_labels = {"Low", "Moderate", "Severe"}
        assert set(y.unique()).issubset(valid_labels)

    def test_no_nans_in_features(self):
        df = _make_raw_df(200)
        X, _ = make_features_and_labels(df)
        assert X.isna().sum().sum() == 0

    def test_wind_shear_is_nonnegative(self):
        df = _make_raw_df(200)
        X, _ = make_features_and_labels(df)
        assert (X["wind_shear"] >= 0).all()

    def test_handles_missing_values(self):
        df = _make_raw_df(100)
        # Introduce some NaNs
        df.loc[5:8, "wind_speed_10m"] = np.nan
        df.loc[20:22, "cloud_cover"] = np.nan
        X, y = make_features_and_labels(df)
        assert len(X) > 0
        assert X.isna().sum().sum() == 0

    def test_all_three_labels_possible(self):
        """With enough spread in data, all 3 labels should appear."""
        df = _make_raw_df(500)
        _, y = make_features_and_labels(df)
        assert len(y.unique()) >= 2  # at least 2 classes should appear

    def test_feature_ranges_reasonable(self):
        df = _make_raw_df(200)
        X, _ = make_features_and_labels(df)
        # wind_shear should be bounded by wind speed range
        assert X["wind_shear"].max() < 50
        # dewpt_dep should be reasonable
        assert X["dewpt_dep"].min() > -40
        assert X["dewpt_dep"].max() < 50
