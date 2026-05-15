#!/usr/bin/env python3
import os, glob
import h5py
import numpy as np
import pandas as pd

DATA_DIR = "mosdac_data"
OUT_CSV = "mosdac_flat.csv"

# Write header once
HEADER_WRITTEN = False

def flat(arr, size):
    return np.array(arr).ravel() if arr is not None else np.full(size, np.nan)

def process_file(path, out_file):
    global HEADER_WRITTEN

    with h5py.File(path, "r") as f:
        # read variables
        lat = f.get("Latitude")
        lon = f.get("Longitude")
        ctp = f.get("CTP")
        ctt = f.get("CTT")
        time = f.get("time")

        # basic sanity
        if lat is None or lon is None:
            print(f"Skipping {path} — no lat/lon")
            return

        lat = lat[:]
        lon = lon[:]
        size = lat.size

        # flatten
        lat_f = flat(lat, size)
        lon_f = flat(lon, size)
        ctp_f = flat(ctp, size)
        ctt_f = flat(ctt, size)

        # time handling
        if time is None:
            t_list = [""] * size
        else:
            t_raw = np.array(time).ravel()
            if len(t_raw) == 1:
                t_list = [float(t_raw[0])] * size
            else:
                t_list = [float(x) for x in t_raw]

        # build DataFrame
        df = pd.DataFrame({
            "source_file": os.path.basename(path),
            "time": t_list,
            "lat": lat_f,
            "lon": lon_f,
            "CTP": ctp_f,
            "CTT": ctt_f
        })

        # append to CSV
        df.to_csv(out_file, mode="a",
                  index=False, header=not HEADER_WRITTEN)

        HEADER_WRITTEN = True
        print(f"✔ Processed: {os.path.basename(path)}  → rows: {len(df)}")


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))

    if not files:
        print("No .h5 files found.")
        return

    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    print(f"Found {len(files)} .h5 files. Starting streaming conversion...\n")

    for fpath in files:
        process_file(fpath, OUT_CSV)

    print("\n🎉 DONE! Output CSV:", OUT_CSV)

if __name__ == "__main__":
    main()
