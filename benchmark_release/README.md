# Corrected benchmark reproduction

This directory contains the revision evidence: 251 main method–seed chains and three additional coordinate-policy chains. `index.json` is the complete allowlist. The 120 chains added in `tsas26-r1.3` run seven methods on PEMS03, PEMS06, PEMS07, PEMS08, PEMS10, PEMS11 and PEMS12 (see below). The six repaired methods are not extended beyond PEMS04 and PEMS05. The fixed- versus evolving-catalog control (24 further chains on PEMS05/PEMS06) is documented separately in `results/fixed_vs_evolving/`; its code is in `../reviewer_fixed_vs_evolving/`.

## Inspect results without training

Only Python's standard library is needed:

```bash
python benchmark_release/summarize.py --output /tmp/evo-recomputed.csv
```

This recomputes equal-year means and sample standard deviations from all 268,110 exported metric records (including explicit empty cohorts). `results/runs.csv` identifies 254 receipts by SHA-256, environment, seed, and method. `results/year_checks.csv` and `results/replacement_year_checks.csv` record 5,958 yearly completion/checkpoint entries. No scheduler credentials, model weights, raw traffic, or personal cluster paths are needed for this step.

## Prepare one exact experiment

Original GPU runs used H200, PyTorch 2.2.1 and CUDA 12.1. The 120 runs added in `tsas26-r1.3` used H100 GPUs with the same PyTorch and CUDA versions. `eac/environment.yaml` is the checked-in environment specification, not a claim that a fresh installation was tested on every platform. The frozen tests ran as part of each receipt.

Download the corrected caches from <https://huggingface.co/datasets/yind7/EvoXXLTraffic> (`caches/<district>/<year>/data.npz`, `graph.npz`) and use that `caches/` folder as `DATA_ROOT`. Their SHA-256 values equal the `data_sha256`/`graph_sha256` in the manifests. The coordinate variant additionally uses `graph_asof.npz`. Do not substitute the legacy filled matrices of the original cloud-disk archive. Every input must match the hashes in the selected manifest.

```bash
python benchmark_release/prepare_run.py \
  --label pems04-s47-Online-AN \
  --data-root /path/to/corrected-caches \
  --output /path/to/new-experiment
cd /path/to/new-experiment
# Run only on an allocated GPU, using the experiment environment:
python -m benchmark_v2.formal_run --manifest manifest.json
```

Preparation verifies all frozen source hashes and data/graph hashes before creating a fresh directory. It selects the original PEMS04 runner overlay when needed. Only paths and output locations change; model settings, seeds, budgets, chronological tasks, protocol digests and input bytes stay fixed. It does not submit to a scheduler or start training. To prepare a coordinate-policy run, select a label ending in `-coordinates` and provide the audited alternate graph files; initialization starts again from 2002.

`protocols/` preserves the frozen protocols. `pems04_overlay/` preserves the distinct PEMS04 entry point. Do not launch PEMS04 using an arbitrarily overwritten common runner: the source/protocol checks deliberately reject that mismatch. Historical preparation/audit modules contain original cluster assumptions; only `prepare_run.py` is advertised here as a portable launcher preparation interface. The verified corrected NPZs are on Hugging Face; porting raw-cache construction remains a release task.

## Runs added in `tsas26-r1.3`

The 120 chains of Pretrain, Retrain, Online-NN, Online-AN, TrafficStream, EAC and ST-TTC on PEMS03, PEMS06, PEMS07, PEMS08, PEMS10, PEMS11 and PEMS12 (seeds 47–49) ran on another GPU cluster with the frozen code of `tsas26-r1.2`. Before each run, all 176 frozen source files were checked against its protocol and every yearly input against the frozen catalog by SHA-256, using the caches of the Hugging Face dataset at revision `4cbd8b0`. All 120 completed on their first attempt and passed the 43 frozen regression tests. Their manifests are in `manifests/` and their frozen protocols in `protocols/` (PEMS07, PEMS08, PEMS11 and PEMS12 are new, the others are unchanged). The manifests differ from the batch manifests the runs used only in where the protocol file is located.

The frozen protocols pin each yearly input by its original cluster path and SHA-256, and `benchmark_v2/formal_run.py` compares those path strings. These runs were therefore started with `formal_run_relocated.py`, which is `formal_run.py` with that one check adapted: each local input must have the catalog SHA-256, and every other field must equal the catalog entry. The protocol digest, budget, frozen-source and regression-test checks and the receipt format are unchanged, and each receipt names this launcher. To repeat one of these runs, prepare it with `prepare_run.py` as above and start it on an allocated GPU with:

```bash
python benchmark_release/formal_run_relocated.py --manifest /path/to/new-experiment/manifest.json
```

## Graphs and provenance

`ordered_catalogs.json` preserves audited IDs and graph provenance. `make_distance_blocks.py` is a prepared utility for real coordinates and ordered IDs (`--help`, `--self-test`); it leaves missing-coordinate distances unavailable. The coordinates and a tool that builds unthresholded distances are in `../coordinates/`. `historical_build_asof.py` preserves the construction implementation and its three synthetic checks; its original paths are historical, not a portable data distribution.

The coordinate experiment uses the retained snapshots dated 2001-08-01, 2004-03-25, 2014-05-09 and 2024-03-20. Missing eligible coordinates isolate spatial edges before model-specific self loops; Gaussian bandwidth is recomputed from positive known-coordinate distances, threshold 0.1. This jointly changes timing and missing-location handling at a fixed retrospective catalog. It is not a threshold sweep, an all-method ranking check, or proof of operational metadata availability.

## Final results (accepted replacement chains)

The six repaired implementations (DCRNN, ASTGCN, TGCN, PECPM, STRAP, STKEC) are in `benchmark_v3/`. Their 48 PEMS04/PEMS05
chains replace the earlier simplified versions; `results/replacement_chains.csv` lists each replacement receipt and the
receipt it replaces. `manifests_repaired/` holds the manifests these 48 chains ran with, byte for byte. `manifests/` keeps their base
versions, from which `benchmark_v3/manifests.py` derives them, and `prepare_run.py` prepares only those base versions: for these six
methods it gives the earlier configuration, not the accepted run. `results/yearly_metrics.csv.gz` and `results/runs.csv` contain the accepted chains; yearly training checks are in `results/year_checks.csv` (unchanged chains) and `results/replacement_year_checks.csv` (the 48 replacements), and
`summarize.py` reproduces `results/summary.csv`, which covers the full period, 2018–2025 and the years before 2018.
The PEMS04/PEMS05 rows of `results/summary.csv` equal the accepted table `results/final_metrics.csv` exactly.

`training_free_references.py` computes the two training-free references of the article from the caches: each year's
training mean predicted everywhere, and the last input value repeated for all leads. Their results are in
`results/training_free_references.csv`.

## Remaining release boundaries

Code, configuration, caches and metric evidence are public. The licenses are MIT for author-owned code and CC BY-NC 4.0 for data products, and adapted baseline code keeps its original license (`../NOTICE.txt`). `release_manifest.json` is a dated snapshot of the first release; this section supersedes its notes on the cache upload and licensing. Measurement-quality sensitivity and alternative-threshold robustness remain outside the validated experiment scope. The inherited gap/hourly/daily results are not part of this release; see the article's Appendix B.
