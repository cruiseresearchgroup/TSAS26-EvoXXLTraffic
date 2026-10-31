"""Unthresholded distances and kernels for one EvoXXLTraffic district-year, from coordinates_<DISTRICT>.csv.gz.

    python coordinates/distances.py PEMS04 2014 --out pems04_2014.npz                # historical coordinates
    python coordinates/distances.py PEMS04 2014 --dated --out pems04_2014_dated.npz  # no-look-ahead coordinates
    python coordinates/distances.py PEMS04 2014 --check graph.npz                    # rebuild the stored graph

The output holds the ordered sensor_ids (= flow-matrix columns), the full Haversine distance matrix in km
(NaN where a dated coordinate is missing) and, with --kernel, the Gaussian kernel exp(-d^2/sigma^2) with
sigma = population SD of positive pairwise distances. --threshold applies the benchmark's sparsification
(the stored graphs use 0.1); without it the kernel is not thresholded.
"""
import argparse, csv, gzip, os
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("district"); ap.add_argument("year", type=int)
ap.add_argument("--dated", action="store_true", help="use dated_lat/dated_lon (missing -> NaN distances)")
ap.add_argument("--kernel", action="store_true"); ap.add_argument("--threshold", type=float)
ap.add_argument("--out"); ap.add_argument("--check", help="graph.npz to compare with (historical coordinates)")
a = ap.parse_args()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coordinates_%s.csv.gz" % a.district.upper())
rows = [r for r in csv.DictReader(gzip.open(path, "rt")) if int(r["year"]) == a.year]
rows.sort(key=lambda r: int(r["column"]))
ids = np.array([int(r["sensor_id"]) for r in rows])
la_k, lo_k = ("dated_lat", "dated_lon") if a.dated else ("lat", "lon")
lat = np.array([float(r[la_k]) if r[la_k] else np.nan for r in rows])
lon = np.array([float(r[lo_k]) if r[lo_k] else np.nan for r in rows])
la, lo = np.radians(lat), np.radians(lon)
h = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
     + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
out = {"sensor_ids": ids, "distance_km": d.astype(np.float32)}
if a.kernel or a.threshold is not None or a.check:
    pos = d[np.isfinite(d) & (d > 0)]
    sigma = pos.std()
    w = np.exp(-(d ** 2) / sigma ** 2)
    if a.threshold is not None:
        w[w < a.threshold] = 0.0
    np.fill_diagonal(w, 0.0)
    out.update(kernel=np.nan_to_num(w).astype(np.float32), sigma_km=np.float64(sigma))
if a.check:
    g = np.load(a.check, allow_pickle=False)
    assert np.array_equal(g["sensor_ids"], ids), "sensor order differs"
    w = np.exp(-(d ** 2) / sigma ** 2); w[w < 0.1] = 0.0; np.fill_diagonal(w, 0.0)
    flips = int(((w > 0) != (g["x"] > 0)).sum())
    print("edges stored %d, edge flips %d, max |diff| %.2e" % ((g["x"] > 0).sum(), flips, np.abs(w - g["x"]).max()))
if a.out:
    np.savez_compressed(a.out, **out); print("wrote", a.out)
print("%s %d: %d sensors, %d without coordinates" % (a.district.upper(), a.year, len(ids), int(np.isnan(lat).sum())))
