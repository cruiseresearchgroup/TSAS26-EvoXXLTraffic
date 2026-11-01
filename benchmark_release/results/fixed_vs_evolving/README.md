# Fixed- versus evolving-catalog training control

This folder holds the matched control of Section 6.3.5 and Table 17 of the article. It asks whether training on the
evolving sensor catalog changes accuracy on long-lived sensors (reviewer question R2.5).

## Design

* **Scope:** PEMS05 and PEMS06, yearly tasks 2018–2025, Retrain and Online-AN, seeds 47–49: 24 chains of 8 years.
* **Fixed set S:** the sensors present in every 2018–2025 catalog, 240 on PEMS05 (of 572 in 2025) and 476 on PEMS06
  (of 746). `receipt.json` (`fixed_sensor_sets`) gives the sha256 of the sorted IDs, one per line.
* **E (evolving):** each year's full catalog and archived graph, as in the main benchmark.
* **F (fixed):** S only. Input z-scores use the accepted training values of S, the graph is the induced subgraph of the
  archived yearly graph with unchanged weights, and early stopping uses validation targets on S.
* **Training:** both populations start new models in 2018; Online-AN then fine-tunes within its own chain. The rest
  follows the frozen protocol: 31 stored days, 60/20/20 chronological split, 12 inputs and 12 outputs,
  `finite_reported` targets, AdamW (learning rate 0.03, weight decay 0.01), batch 128, at most 100 epochs, patience 5.
* **Evaluation (TEST):** E and F on the same accepted targets of S. E is also scored on all sensors and on
  new-to-previous-year sensors; for 2018 the previous catalog is the 2017 catalog, and no 2017 data or model is used.

## Files

* `results_fixed_vs_evolving.csv`: 1,152 annual rows (`district, method, train_population, eval_population, seed,
  year, metric, value, n_targets`), full 12-lead horizon.
* `summary_fixed_vs_evolving.csv`: mean and sample SD over seeds of equal-year means; `E-F` rows are paired within seed.
* `summarize_fixed_vs_evolving.py`: recomputes the summary from the annual rows with the standard library only.
* `receipt.json`: protocol, input checksums, the definition of S, and per-chain status, epochs, optimizer steps and
  checkpoint checksums. Run-directory paths are relative; `source_receipt_sha256` identifies the original receipt.
* `RESULTS_EN.txt`: a short description of the results on S.
* `provenance/`: `CODE_RECEIPT.json` (sha256 of the 566 source files sealed for the runs), `ENVIRONMENT_FREEZE.txt`
  (installed packages), `CONTRACT_TESTS.txt` (the wrapper's unit tests as run), the two lists of S, and `fetch_inputs.py`
  (downloads and checks the 32 input files). All are unchanged from the run.
* `make_code_receipt.py`: seals a fresh checkout for a rerun.

## Code

The wrapper is in [`reviewer_fixed_vs_evolving/`](../../../reviewer_fixed_vs_evolving/):

* `PROTOCOL.json`, `plan.json`: the protocol, S, the 2017 reference catalogs and the input checksums.
* `audit_inputs.py`, `common.py`: check the frozen inputs and build the F inputs.
* `run_chain.py`: runs one chain through the unchanged frozen runner and adds the TEST cohorts.
* `aggregate.py`: builds the CSV files and the receipt from the annual results.
* `test_contracts.py`: unit tests of the scaler, the induced graph, the metrics and the statistics.
* `job.py`, `job.sh`, `profile.py`: how the original Slurm jobs profiled resources and then ran the E and F chains of a
  method–seed pair on one GPU. A manual rerun does not need them.

## Provenance

* The 32 input files (data and graph, 2018–2025) have the same sha256 as the corrected caches in the run manifests.
* Runs used NVIDIA H100 GPUs with PyTorch 2.2.1 and CUDA 12.1. The 24 chains took about 5 hours in total; the E and F
  chains of a method–seed pair shared one GPU, so the jobs used about 2.8 GPU-hours.
* The frozen runner of `tsas26-r1` (commit cd95d84) was called through the wrapper, which changes only the training
  population and adds the TEST cohorts. It only adds files; no frozen source changes.
* `receipt.json` names the source of the runs as commit 88bb373, two commits on `tsas26-r1`. Those commits are not in
  this repository's history, because they include `submit.py`, the scheduler script of the original cluster, which holds
  site-specific account details and plays no part in the computation.
* Every file listed in `provenance/CODE_RECEIPT.json` is byte-identical in this release, except `submit.py` (left out),
  `README.md` and `benchmark_release/README.md` (documentation).
* Run from this release on a CPU, `audit_inputs.py` reproduces the sha256 of all 32 F input files in `receipt.json`.
* As a consistency check, E-trained Retrain, which starts a new model every year, agrees with the accepted 2018–2025
  Retrain results within the seed spread (all-sensor MAE 10.717 ± 0.041 against 10.730 ± 0.037 on PEMS05).

## Rerunning

A chain needs one NVIDIA GPU, PyTorch 2.2.1 and NumPy 1.26.4; `run_chain.py` stops on other versions. The steps before the
training call were checked from this release on a CPU-only machine; the GPU runs themselves were not repeated.

```bash
ROOT=/path/to/new-run; mkdir -p $ROOT
git clone --branch tsas26-r1.1 https://github.com/cruiseresearchgroup/TSAS26-EvoXXLTraffic $ROOT/source
cd $ROOT/source
cp reviewer_fixed_vs_evolving/plan.json $ROOT/INPUT_PLAN.json
cp benchmark_release/results/fixed_vs_evolving/provenance/fetch_inputs.py $ROOT/
python $ROOT/fetch_inputs.py                                       # 32 files from Hugging Face, sha256 checked
python benchmark_release/results/fixed_vs_evolving/make_code_receipt.py $ROOT
python -m reviewer_fixed_vs_evolving.audit_inputs --root $ROOT     # writes INPUT_AUDIT.json and the F inputs
# One chain; outside Slurm, SLURM_JOB_ID can be any label.
SLURM_JOB_ID=manual CUDA_VISIBLE_DEVICES=0 python -m reviewer_fixed_vs_evolving.run_chain \
  --root $ROOT --district PEMS05 --method Retrain --population F --seed 47
# After all 24 chains:
python -m reviewer_fixed_vs_evolving.aggregate --root $ROOT
```

## Result

Paired E − F MAE on S: PEMS05 Retrain −0.096 ± 0.041 and Online-AN −0.087 ± 0.180; PEMS06 Retrain +0.025 ± 0.063 and
Online-AN +0.110 ± 0.115. Each difference is below 1% of the error level and the sign differs between districts, so
training on the evolving catalog shows no consistent accuracy change on the long-lived sensors. Three seeds and two
districts do not support a significance claim.
