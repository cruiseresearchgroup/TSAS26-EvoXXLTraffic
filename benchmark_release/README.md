# Corrected benchmark reproduction

This directory contains the bounded revision evidence: 131 main method–seed chains and three additional coordinate-policy chains. `index.json` is the complete allowlist. No automatic extension to other districts/methods is included.

## Inspect results without training

Only Python's standard library is needed:

```bash
python benchmark_release/summarize.py --output /tmp/evo-recomputed.csv
```

This recomputes equal-year means and sample standard deviations from all 137,025 exported metric records (including explicit empty cohorts). `results/runs.csv` identifies 134 original receipts by SHA-256, environment, seed, and method. `results/year_checks.csv` records 3,045 yearly completion/checkpoint entries. No scheduler credentials, model weights, raw traffic, or personal cluster paths are needed for this step.

## Prepare one exact experiment

Original GPU runs used H200, PyTorch 2.2.1 and CUDA 12.1. `eac/environment.yaml` is the checked-in environment specification, not a claim that a fresh installation was tested on every platform. The original frozen tests ran as part of each receipt; this release adds no new GPU experiment.

Obtain the corrected caches and lay them out as `DATA_ROOT/pems04/2002/data.npz`, `graph.npz`, etc. The coordinate variant additionally uses `graph_asof.npz`. **The legacy cloud-disk archive has not yet been verified to contain these corrected files.** Every input must match the hashes in the selected manifest.

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

`protocols/` preserves the frozen protocols. `pems04_overlay/` preserves the distinct PEMS04 entry point. Do not launch PEMS04 using an arbitrarily overwritten common runner: the source/protocol checks deliberately reject that mismatch. Historical preparation/audit modules contain original cluster assumptions; only `prepare_run.py` is advertised here as a portable launcher preparation interface. Porting raw-cache construction and distributing verified corrected NPZs remain release tasks.

## Graphs and provenance

`ordered_catalogs.json` preserves audited IDs and graph provenance. `make_distance_blocks.py` is a prepared utility for real coordinates and ordered IDs (`--help`, `--self-test`); it leaves missing-coordinate distances unavailable. Full coordinate/distance publication has not been verified. `historical_build_asof.py` preserves the construction implementation and its three synthetic checks; its original paths are historical, not a portable data distribution.

The coordinate experiment uses the retained snapshots dated 2001-08-01, 2004-03-25, 2014-05-09 and 2024-03-20. Missing eligible coordinates isolate spatial edges before model-specific self loops; Gaussian bandwidth is recomputed from positive known-coordinate distances, threshold 0.1. This jointly changes timing and missing-location handling at a fixed retrospective catalog. It is not a threshold sweep, an all-method ranking check, or proof of operational metadata availability.

## Remaining release boundaries

The code/configuration/metric evidence can be inspected here. A complete data-based rerun still requires corrected NPZ distribution and the documented environment. No completed artifact-specific Evo license is asserted; see `../NOTICE.txt`. Measurement-quality sensitivity, alternative-threshold robustness, and inherited gap/hourly/daily numerical provenance are outside the validated experiment scope.
