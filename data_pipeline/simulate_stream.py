# simulate_stream.py
import time
import pandas as pd
import numpy as np
import requests
import json

# Point at your running API
API_URL = "http://127.0.0.1:8080/predict"
DELAY_SECONDS = 0.5   # small delay between batches (adjust)

def make_json_safe(value):
    """Convert pandas/numpy types and NaN -> JSON-serializable Python types."""
    # handle pandas NaT / numpy NaN / pd.NA
    if value is None:
        return None
    # pandas NA / numpy nan
    try:
        if pd.isna(value):
            return None
    except Exception:
        # pd.isna may raise for some objects; ignore
        pass

    # numpy scalar -> native python
    if isinstance(value, np.generic):
        return value.item()

    # pandas Timestamp -> ISO string
    if isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype, pd.DatetimeIndex)):
        try:
            return pd.Timestamp(value).isoformat()
        except Exception:
            return str(value)

    # plain python int/float/str/bool are OK
    if isinstance(value, (int, float, str, bool)):
        return value

    # fallback to string
    return str(value)

def send_row_to_api(row):
    # convert pandas Series -> dict, make JSON-safe
    raw = row.to_dict()
    payload = {k: make_json_safe(v) for k, v in raw.items()}

    # your API accepts either a single object or an array; we'll send a single object
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
    except Exception as e:
        print("Request error:", e)
        return

    if resp.status_code != 200:
        print("Error:", resp.status_code, resp.text)
        return

    try:
        rj = resp.json()
    except Exception as e:
        print("Invalid JSON response:", e, resp.text[:300])
        return

    # adapt to your API's response shape: {"n_rows":.., "results": [{...}, ...]}
    results = rj.get("results") or []
    if not results:
        print("No results in response:", rj)
        return

    # take the first predicted record (this script posts one row at a time)
    first = results[0]
    pred_text = first.get("pred_text") or first.get("pred") or str(first.get("pred_label"))
    probs = first.get("probs") or []
    conf = max(probs) if probs else None

    if conf is None:
        print(f"▶ Prediction: {pred_text}")
    else:
        print(f"▶ Prediction: {pred_text}, Confidence: {conf:.3f}")

    if pred_text and str(pred_text).lower().startswith("severe"):
        print("⚠ ALERT! Severe turbulence detected!")

def simulate(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Streaming {len(df)} samples...")

    for idx, row in df.iterrows():
        print(f"\n--- Sending sample {idx+1} ---")
        send_row_to_api(row)
        time.sleep(DELAY_SECONDS)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python simulate_stream.py <csv_file>")
        exit(1)
    simulate(sys.argv[1])