#!/usr/bin/env python3
"""
read_mosdac.py
Read MOSDAC .h5 L2B products and dump a flattened CSV for ML preprocessing.

Output: mosdac_flat.csv.gz
"""
import os, glob
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = "mosdac_data"        # change if your download path differs
OUT_CSV = "mosdac_flat.csv.gz"  # gzipped CSV

# Choose variables to extract - adjust as needed
VARS = [
    "CTP",   # cloud top pressure
    "CTT",   # cloud top temperature
    "Latitude",
    "Longitude",
    "time"
]
# Optionally include radiances / other arrays you saw (CLRFR_TIR1, CSBT_TIR1, etc.)

# Optional bounding box filter (lon_min, lat_min, lon_max, lat_max) or None
BBOX = None
# BBOX = (68.0, 6.0, 98.0, 37.0)  # example you used earlier

def h5_time_to_iso(h5_time_array):
    # Many MOSDAC products store time as seconds since epoch or similar.
    # Try common conversions - adapt if different in your files.
    # We'll attempt: if single scalar, convert; if array, try elementwise.
    try:
        # often the /time is seconds since 1970-01-01 or days since...
        t0 = datetime(1970,1,1)
        arr = np.array(h5_time_array).ravel()
        # if values look like large ints -> seconds
        if np.nanmax(arr) > 1e9:
            return [(t0 + timedelta(seconds=float(x))).isoformat() for x in arr]
        # if values are small (~2000) maybe it's year-day -> fallback:
        return [str(x) for x in arr]
    except Exception:
        return [str(x) for x in np.array(h5_time_array).ravel()]

def process_file(path, bbox=None):
    rows = []
    with h5py.File(path, "r") as f:
        # check keys
        keys = list(f.keys())
        # ensure required variables exist
        for v in VARS:
            if v not in f:
                # try some common alternatives
                pass

        # read arrays
        try:
            lat = f["Latitude"][:]
            lon = f["Longitude"][:]
        except KeyError:
            # try Geo arrays if present
            lat = f.get("CSBT_Latitude")[:] if "CSBT_Latitude" in f else None
            lon = f.get("CSBT_Longitude")[:] if "CSBT_Longitude" in f else None

        # read CTP/CTT if present
        ctp = f["CTP"][:] if "CTP" in f else None
        ctt = f["CTT"][:] if "CTT" in f else None

        # get time array
        t_raw = f["time"][:] if "time" in f else None
        time_iso_list = h5_time_to_iso(t_raw) if t_raw is not None else None

        # flatten arrays to 1D rows (assume 2D grids with same shape)
        def flat(x):
            return np.array(x).ravel() if x is not None else np.full(lat.size, np.nan)

        lat_flat = flat(lat)
        lon_flat = flat(lon)
        ctp_flat = flat(ctp)
        ctt_flat = flat(ctt)

        n = lat_flat.size
        # time handling: some files have single timestamp, some per-pixel
        if time_iso_list is None:
            times = [None]*n
        else:
            if len(time_iso_list) == n:
                times = time_iso_list
            elif len(time_iso_list) == 1:
                times = [time_iso_list[0]]*n
            else:
                # fallback: broadcast first element
                times = [time_iso_list[0]]*n

        # optional bbox filter
        mask = np.ones(n, dtype=bool)
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            mask = (lon_flat >= lon_min) & (lon_flat <= lon_max) & (lat_flat >= lat_min) & (lat_flat <= lat_max)

        # build rows
        for i in np.where(mask)[0]:
            rows.append((os.path.basename(path), times[i], float(lat_flat[i]), float(lon_flat[i]),
                         float(ctp_flat[i]) if not np.isnan(ctp_flat[i]) else None,
                         float(ctt_flat[i]) if not np.isnan(ctt_flat[i]) else None))
    df = pd.DataFrame(rows, columns=["source_file","time","lat","lon","CTP","CTT"])
    return df

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    if not files:
        print("No .h5 files found in", DATA_DIR)
        return
    out_dfs = []
    for p in files:
        try:
            print("Processing", p)
            df = process_file(p, bbox=BBOX)
            out_dfs.append(df)
        except Exception as e:
            print("ERROR processing", p, e)
    if out_dfs:
        big = pd.concat(out_dfs, ignore_index=True)
        print("Total rows:", len(big))
        # small cleanup - drop rows with missing lat/lon
        big = big.dropna(subset=["lat","lon"])
        # write compressed CSV
        big.to_csv(OUT_CSV, index=False, compression="gzip")
        print("Wrote", OUT_CSV)
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()
