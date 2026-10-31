"""Naive reference forecasts on the corrected yearly benchmark caches (no training).

  constant : every target = scalar mean of the training normalisation (what a model that outputs 0
             in z-score space predicts)
  last     : every lead = last input value, de-normalised (persistence)

Metrics follow the paper: complete 12-in/12-out windows inside the test split; horizon h pools
leads 1..h; accepted targets only (`observed`); MAPE excludes |y| <= 1e-6 and is in percent;
cohorts 'all' and 'new' (catalog_y minus catalog_{y-1}, incl. returning; none in the first year);
each year weighted equally over years with >= 1 accepted target in the cohort.

usage: naive_refs_full.py <inputs_root> <district_dir> <first_year> <last_year> <out_csv>
"""
import csv, os, sys
import numpy as np

root, dd, y0, y1, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
H = (3, 6, 12)
rows, prev_ids = [], None
for y in range(y0, y1 + 1):
    z = np.load(os.path.join(root, dd, str(y), "data.npz"), allow_pickle=False)
    yv, ob, x, av, ids = z["y"], z["observed"], z["x"], z["available"], z["sensor_ids"]
    b0, b1, b2, b3 = (int(v) for v in z["bounds"])
    # recover the cache's own normalisation: x = (y - mu) / sd where the input is an accepted value
    m = ob[b0:b1] & av[b0:b1]
    xs, ys = x[b0:b1][m].astype(np.float64), yv[b0:b1][m].astype(np.float64)
    sd, mu = np.polyfit(xs, ys, 1)
    origins = np.arange(b2 + 11, b3 - 12)
    last = (mu + sd * x[origins].astype(np.float64))
    cohorts = {"all": np.ones(len(ids), bool)}
    if prev_ids is not None:
        cohorts["new"] = ~np.isin(ids, prev_ids)
    acc = {(p, c): np.zeros((12, 5)) for p in ("constant", "last") for c in cohorts}
    for k in range(1, 13):
        s = origins + k
        t = yv[s].astype(np.float64); o = ob[s]
        for p, pred in (("constant", mu), ("last", last)):
            e = t - pred
            ok_m = o & (np.abs(t) > 1e-6)
            ape = np.zeros_like(t); np.divide(np.abs(e), np.abs(t), out=ape, where=ok_m)
            for c, cols in cohorts.items():
                oc, mc = o[:, cols], ok_m[:, cols]
                ec = e[:, cols]
                acc[(p, c)][k - 1] += (np.abs(ec)[oc].sum(), (ec ** 2)[oc].sum(), oc.sum(),
                                       ape[:, cols][mc].sum(), mc.sum())
    for (p, c), a in acc.items():
        cum = np.cumsum(a, axis=0)
        for h in H:
            sae, sse, n, sape, nm = cum[h - 1]
            if n == 0:
                continue
            rows.append(dict(district=dd.upper(), year=y, predictor=p, cohort=c, horizon=h,
                             MAE=sae / n, RMSE=np.sqrt(sse / n), MAPE=100 * sape / nm if nm else np.nan,
                             n_targets=int(n), mu=mu, sd=sd))
    prev_ids = ids
    print(y, "done", flush=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# equal-year summary
import collections
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["predictor"], r["cohort"], r["horizon"])].append(r)
for key in sorted(agg):
    rs = agg[key]
    print("%s %-8s %-3s h%-2d years=%2d MAE=%.3f RMSE=%.3f MAPE=%.2f" % (
        dd.upper(), *key, len(rs), np.mean([r["MAE"] for r in rs]), np.mean([r["RMSE"] for r in rs]),
        np.nanmean([r["MAPE"] for r in rs])))
