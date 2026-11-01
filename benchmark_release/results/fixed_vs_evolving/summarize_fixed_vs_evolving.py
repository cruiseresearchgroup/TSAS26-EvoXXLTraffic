"""Recompute summary_fixed_vs_evolving.csv from the annual rows (standard library only).
Each seed weights its eight 2018-2025 TEST values equally; the table reports the mean and sample SD (n-1) over seeds
47-49. E-F rows are paired within each seed before the cross-seed statistics.
usage: python summarize_fixed_vs_evolving.py [results_fixed_vs_evolving.csv] [summary_fixed_vs_evolving.csv]"""
import csv, os, statistics as st, sys

here = os.path.dirname(os.path.abspath(__file__))
annual = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results_fixed_vs_evolving.csv")
summary = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "summary_fixed_vs_evolving.csv")
A = {}
for r in csv.DictReader(open(annual)):
    key = (r["district"], r["method"], r["train_population"], r["eval_population"], r["metric"])
    A.setdefault(key, {}).setdefault(int(r["seed"]), {})[int(r["year"])] = float(r["value"])
assert all(len(ys) == 8 for seeds in A.values() for ys in seeds.values())

def seed_means(key):
    return {s: st.mean(ys.values()) for s, ys in sorted(A[key].items())}

worst, n = 0.0, 0
for r in csv.DictReader(open(summary)):
    d, m, p, e, mt = r["district"], r["method"], r["train_population"], r["eval_population"], r["metric"]
    if p == "E-F":
        E, F = seed_means((d, m, "E", e, mt)), seed_means((d, m, "F", e, mt))
        v = [E[s] - F[s] for s in E]
    else:
        v = list(seed_means((d, m, p, e, mt)).values())
    worst = max(worst, abs(st.mean(v) - float(r["mean"])), abs(st.stdev(v) - float(r["sd"])))
    n += 1
print("%d summary rows recomputed from the annual rows; max abs difference %.2e" % (n, worst))
