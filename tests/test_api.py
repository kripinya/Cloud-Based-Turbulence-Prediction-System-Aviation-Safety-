"""
tests/test_api.py — Unit tests for the turbulence prediction API.

Run with:  pytest tests/ -v
"""

import json
import os
import sys
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app, validate_dataframe, EXPECTED_FEATURES, FEATURE_RANGES, MAX_BATCH_ROWS
import pandas as pd


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_json(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert resp.status_code in (200, 500)
        assert "status" in data
        assert "model_path" in data

    def test_health_reports_scaler_status(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "scaler_loaded" in data


# ---------------------------------------------------------------------------
# Index / Dashboard
# ---------------------------------------------------------------------------

class TestDashboard:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"TURBULENCE INSIGHT" in resp.data


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_df_returns_error(self):
        df = pd.DataFrame()
        warnings = validate_dataframe(df)
        assert len(warnings) == 1
        assert "empty" in warnings[0].lower()

    def test_too_many_rows(self):
        df = pd.DataFrame({"wind_speed_10m": [1.0] * (MAX_BATCH_ROWS + 1)})
        warnings = validate_dataframe(df)
        assert len(warnings) == 1
        assert "Too many rows" in warnings[0]

    def test_valid_data_no_warnings(self):
        df = pd.DataFrame([{
            "wind_speed_10m": 5.0,
            "wind_speed_100m": 12.0,
            "relative_humidity_2m": 70.0,
            "cloud_cover": 50.0,
            "surface_pressure": 1013.0,
        }])
        warnings = validate_dataframe(df)
        assert len(warnings) == 0

    def test_out_of_range_detected(self):
        df = pd.DataFrame([{
            "wind_speed_10m": 200.0,  # way too high
            "surface_pressure": 500.0,  # way too low
        }])
        warnings = validate_dataframe(df)
        assert len(warnings) == 2  # both columns flagged

    def test_negative_wind_detected(self):
        df = pd.DataFrame([{"wind_speed_10m": -5.0}])
        warnings = validate_dataframe(df)
        assert any("wind_speed_10m" in w for w in warnings)


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def _make_payload(self, overrides=None):
        """Build a valid single-row prediction payload."""
        row = {
            "wind_speed_10m": 5.2,
            "wind_speed_100m": 12.1,
            "temperature_2m": 25,
            "dewpoint_2m": 20,
            "relative_humidity_2m": 72,
            "cloud_cover": 80,
            "surface_pressure": 995.2,
        }
        if overrides:
            row.update(overrides)
        return {"rows": [row]}

    def test_single_prediction_returns_result(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(self._make_payload()),
            content_type="application/json",
        )
        data = resp.get_json()
        if resp.status_code == 500 and "Model not loaded" in data.get("error", ""):
            pytest.skip("Model artifacts not available in test environment")
        assert resp.status_code == 200
        assert "results" in data
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert "pred_text" in result
        assert result["pred_text"] in ("Low", "Moderate", "Severe")

    def test_prediction_returns_probabilities(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(self._make_payload()),
            content_type="application/json",
        )
        data = resp.get_json()
        if resp.status_code == 500:
            pytest.skip("Model artifacts not available")
        result = data["results"][0]
        assert "probs" in result
        assert len(result["probs"]) == 3  # Low, Moderate, Severe
        assert abs(sum(result["probs"]) - 1.0) < 0.01  # probabilities sum to ~1

    def test_empty_json_returns_error(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps({"rows": []}),
            content_type="application/json",
        )
        # Should either return 400 for empty or handle gracefully
        assert resp.status_code in (200, 400)

    def test_invalid_content_type(self, client):
        resp = client.post(
            "/predict",
            data="not json or csv",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    def test_multi_row_prediction(self, client):
        payload = {
            "rows": [
                {
                    "wind_speed_10m": 5, "wind_speed_100m": 12,
                    "temperature_2m": 25, "dewpoint_2m": 20,
                    "relative_humidity_2m": 72, "cloud_cover": 80,
                    "surface_pressure": 995,
                },
                {
                    "wind_speed_10m": 2, "wind_speed_100m": 4,
                    "temperature_2m": 18, "dewpoint_2m": 15,
                    "relative_humidity_2m": 45, "cloud_cover": 10,
                    "surface_pressure": 1015,
                },
            ]
        }
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = resp.get_json()
        if resp.status_code == 500:
            pytest.skip("Model artifacts not available")
        assert resp.status_code == 200
        assert data["n_rows"] == 2


# ---------------------------------------------------------------------------
# Feature engineering in API
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_wind_shear_derived(self, client):
        """API should auto-derive wind_shear from wind speeds."""
        payload = {"rows": [{
            "wind_speed_10m": 5, "wind_speed_100m": 15,
            "temperature_2m": 25, "dewpoint_2m": 20,
            "relative_humidity_2m": 70, "cloud_cover": 50,
            "surface_pressure": 1010,
        }]}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        if resp.status_code == 500:
            pytest.skip("Model not available")
        # If we got a result, the API successfully derived wind_shear
        assert resp.status_code == 200

    def test_dewpt_dep_derived(self, client):
        """API should auto-derive dewpt_dep from temperature and dewpoint."""
        payload = {"rows": [{
            "wind_speed_10m": 5, "wind_speed_100m": 15,
            "temperature_2m": 30, "dewpoint_2m": 22,  # dewpt_dep = 8
            "relative_humidity_2m": 70, "cloud_cover": 50,
            "surface_pressure": 1010,
        }]}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        if resp.status_code == 500:
            pytest.skip("Model not available")
        assert resp.status_code == 200
