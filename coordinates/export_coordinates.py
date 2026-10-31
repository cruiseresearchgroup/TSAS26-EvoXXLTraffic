"""Export EvoXXLTraffic sensor coordinates per district-year and verify them against the stored graphs.

historical coordinates (used by every stored graph and the main comparison)
    The original notebooks read the district's metadata snapshots meta/dXX_text_meta_*.txt in filename
    (= date) order; a station takes the first snapshot listing it with non-zero latitude and longitude.
    Stations found in none get the mean latitude/longitude of the stations that were found
    (centroid fallback, flagged).
dated coordinates (no look-ahead)
    Only snapshots dated on or before the catalog anchor are eligible (1 January of the year; for a
    district's first cached year, its first cached day). All snapshots in the archive are used,
    including the 213 PEMS04 snapshots in meta/pems04/. A station takes its earliest eligible valid
    coordinate; stations with none are flagged and get no coordinate.

The stored graph of each year is rebuilt from the historical coordinates with the published kernel
(Haversine km, R = 6371; sigma = population SD of positive pairwise distances; w = exp(-d^2/sigma^2);
w < 0.1 -> 0; zero diagonal) and compared entry-wise with graph.npz.

Historical snapshot sets follow graph.npz "coordinate_source": meta/d04_text_meta_*.txt (4 files) for
PEMS04 and claude/dXX/dXX_text_meta_*.txt for the other districts.

usage: export_coordinates.py <code_dir> <out_dir> <district>=<cache_root> [...]
"""
import csv, glob, json, os, re, sys
from datetime import date, datetime, timezone
import numpy as np

code_dir, out_dir, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
R = 6371.0


def snap_date(path):
    y, m, d = re.search(r"_(\d{4})_(\d{2})_(\d{2})\.txt$", path).groups()
    return date(int(y), int(m), int(d))


def read_meta(path):
    out = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        head = f.readline().rstrip("\n").split("\t")
        if not {"ID", "Latitude", "Longitude"} <= set(head):    # e.g. an empty download
            print("skip unreadable snapshot", os.path.basename(path), flush=True)
            return out
        i_id, i_lat, i_lon = head.index("ID"), head.index("Latitude"), head.index("Longitude")
        for line in f:
            p = line.rstrip("\n").split("\t")
            try:
                sid, lat, lon = int(p[i_id]), float(p[i_lat]), float(p[i_lon])
            except (ValueError, IndexError):
                continue
            if lat != 0 and lon != 0 and np.isfinite(lat) and np.isfinite(lon) and sid not in out:
                out[sid] = (lat, lon)
    return out


def kernel(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    a = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    d = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    sigma = d[d > 0].std()
    w = np.exp(-(d ** 2) / sigma ** 2)
    w[w < 0.1] = 0.0
    np.fill_diagonal(w, 0.0)
    return w, sigma


os.makedirs(out_dir, exist_ok=True)
report = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "districts": {}}
for spec in specs:
    dist, root = spec.split("=", 1)
    dd = dist.lower().replace("pems", "d")                       # PEMS04 -> d04
    pat = "%s_text_meta_[12][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9].txt" % dd
    top = sorted(glob.glob(os.path.join(code_dir, "meta", pat)))
    dense = sorted(glob.glob(os.path.join(code_dir, "meta", dist.lower(), pat)))
    claude = sorted(glob.glob(os.path.join(code_dir, "claude", dd, pat)))
    hist_files = top if dist == "PEMS04" else claude
    all_files = sorted({snap_date(p): p for p in top + dense + claude}.items())
    hist_meta = [read_meta(p) for p in hist_files]
    all_meta = [(d, read_meta(p)) for d, p in all_files]
    years = sorted(int(os.path.basename(p)) for p in glob.glob(os.path.join(root, "[12][0-9][0-9][0-9]")))
    rows, rep = [], {"historical_snapshots": [os.path.basename(p) for p in hist_files],
                     "all_snapshots": len(all_files), "years": {}}
    for k, y in enumerate(years):
        g = np.load(os.path.join(root, str(y), "graph.npz"), allow_pickle=False)
        ids = g["sensor_ids"].astype(np.int64)
        if k == 0:
            ts = np.load(os.path.join(root, str(y), "data.npz"), allow_pickle=False)["timestamps"]
            t0 = int(ts[0]); t0 = t0 // 10 ** 9 if t0 > 10 ** 12 else t0
            anchor = datetime.fromtimestamp(t0, timezone.utc).date()
        else:
            anchor = date(y, 1, 1)
        hist, src, fb = {}, {}, []
        for sid in ids:
            for p, m in zip(hist_files, hist_meta):
                if sid in m:
                    hist[sid] = m[sid]; src[sid] = os.path.basename(p)[-14:-4]; break
        found = list(hist.values())
        cen = (float(np.mean([c[0] for c in found])), float(np.mean([c[1] for c in found]))) if found else (np.nan, np.nan)
        for sid in ids:
            if sid not in hist:
                hist[sid] = cen; src[sid] = "centroid"; fb.append(int(sid))
        dated, dsrc = {}, {}
        for sid in ids:
            for d, m in all_meta:
                if d > anchor:
                    break
                if sid in m:
                    dated[sid] = m[sid]; dsrc[sid] = d.isoformat(); break
        lat = np.array([hist[s][0] for s in ids]); lon = np.array([hist[s][1] for s in ids])
        w, sigma = kernel(lat, lon)
        stored = g["x"].astype(np.float64)
        flips = int(((w > 0) != (stored > 0)).sum())
        maxdiff = float(np.abs(w - stored).max())
        n_hist_only = sum(1 for s in ids if dsrc.get(s) is None)
        rep["years"][y] = dict(n=int(len(ids)), anchor=anchor.isoformat(), centroid_fallback=len(fb),
                               dated_missing=n_hist_only, sigma_km=round(float(sigma), 4),
                               edges_stored=int((stored > 0).sum()), edge_flips=flips, max_abs_diff=maxdiff,
                               reproduced=bool(flips == 0 and maxdiff < 1e-5))
        for c, s in enumerate(ids):
            dl = dated.get(s, (None, None))
            rows.append([dist, y, c, int(s), "%.6f" % hist[s][0], "%.6f" % hist[s][1], src[s],
                         "" if dl[0] is None else "%.6f" % dl[0], "" if dl[1] is None else "%.6f" % dl[1],
                         dsrc.get(s, "")])
        print(dist, y, rep["years"][y], flush=True)
    with open(os.path.join(out_dir, "coordinates_%s.csv" % dist), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["district", "year", "column", "sensor_id", "lat", "lon", "lat_lon_source",
                     "dated_lat", "dated_lon", "dated_snapshot"])
        wr.writerows(rows)
    report["districts"][dist] = rep
with open(os.path.join(out_dir, "VERIFICATION_%s.json" % "_".join(s.split("=")[0] for s in specs)), "w") as f:
    json.dump(report, f, indent=1)
