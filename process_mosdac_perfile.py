#!/usr/bin/env python3
"""
process_mosdac_perfile.py
Read MOSDAC .h5 files (mosdac_data/*.h5) and write one cleaned gzipped CSV per file
into processed_csv/<basename>.csv.gz. Filters out fill-value lat=32767.
"""
import os, glob
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = "mosdac_data"
OUT_DIR = "processed_csv"
VARS = ["Latitude","Longitude","time","CTP","CTT"]

os.makedirs(OUT_DIR, exist_ok=True)

def h5_time_to_iso(h5_time_array):
    try:
        t0 = datetime(2000,1,1)  # these files use "minutes since 2000-01-01 00:00:00"
        arr = np.array(h5_time_array).ravel()
        # if values look large (seconds) handle; otherwise treat as minutes since 2000
        if arr.size == 0:
            return []
        if np.nanmax(arr) > 1e9:
            return [(datetime(1970,1,1) + timedelta(seconds=float(x))).isoformat() for x in arr]
        # MOSDAC time unit seen in your files: minutes since 2000-01-01
        return [(t0 + timedelta(minutes=float(x))).isoformat() for x in arr]
    except Exception:
        return [str(x) for x in np.array(h5_time_array).ravel()]

def flat_or_fill(ds, n):
    if ds is None:
        return np.full(n, np.nan)
    arr = np.array(ds)
    return arr.ravel() if arr.size else np.full(n, np.nan)

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
if not files:
    print("No .h5 files found in", DATA_DIR)
    raise SystemExit(1)

for p in files:
    base = os.path.basename(p)
    out_path = os.path.join(OUT_DIR, base.replace(".h5", ".csv.gz"))
    if os.path.exists(out_path):
        print("SKIP (exists):", base)
        continue
    try:
        with h5py.File(p, "r") as f:
            # prefer Latitude/Longitude, fall back to CSBT_*
            lat_ds = f.get("Latitude") or f.get("CSBT_Latitude")
            lon_ds = f.get("Longitude") or f.get("CSBT_Longitude")
            if lat_ds is None or lon_ds is None:
                print("SKIP (no geo):", base)
                continue

            lat = np.array(lat_ds)
            lon = np.array(lon_ds)
            n = lat.size

            # read other vars if present
            ctp = f.get("CTP")
            ctt = f.get("CTT")
            time_ds = f.get("time")

            lat_flat = lat.ravel()
            lon_flat = lon.ravel()
            ctp_flat = flat_or_fill(ctp, n)
            ctt_flat = flat_or_fill(ctt, n)
            # convert time to iso strings (handles single timestamp or per-pixel)
            if time_ds is None:
                times = [None]*n
            else:
                tlist = h5_time_to_iso(time_ds[:])
                if len(tlist) == n:
                    times = tlist
                elif len(tlist) == 1:
                    times = [tlist[0]] * n
                else:
                    # fallback: broadcast first
                    times = [tlist[0]] * n

            # mask out fill values (latitude fill in these products is 32767)
            valid_mask = (lat_flat != 32767) & (~np.isnan(lat_flat)) & (~np.isnan(lon_flat))
            valid_idx = np.where(valid_mask)[0]
            if valid_idx.size == 0:
                print("NO VALID PIXELS:", base)
                # still create an empty small CSV with header for bookkeeping
                df_empty = pd.DataFrame(columns=["source_file","time","lat","lon","CTP","CTT"])
                df_empty.to_csv(out_path, index=False, compression="gzip")
                continue

            # build DataFrame in chunks to avoid memory pressure
            chunk_size = 500000  # adjust if needed
            written = False
            for start in range(0, valid_idx.size, chunk_size):
                sel = valid_idx[start:start+chunk_size]
                rows = {
                    "source_file": [base]*sel.size,
                    "time": [times[i] for i in sel],
                    "lat": lat_flat[sel].astype(float),
                    "lon": lon_flat[sel].astype(float),
                    "CTP": [float(x) if (not np.isnan(x)) else None for x in ctp_flat[sel]],
                    "CTT": [float(x) if (not np.isnan(x)) else None for x in ctt_flat[sel]],
                }
                df = pd.DataFrame(rows)
                # append or write
                if not written:
                    df.to_csv(out_path, index=False, compression="gzip")
                    written = True
                else:
                    # append without header: write to temp and concatenate; pandas doesn't append gzip easily
                    # so use mode='ab' and header=False with to_csv and compression=None for subsequent chunks
                    df.to_csv(out_path, index=False, compression="gzip", mode="ab", header=False)
            print("✔ Processed:", base, "→ valid rows:", valid_idx.size, "out:", out_path)
    except Exception as e:
        print("ERROR processing", base, e)
