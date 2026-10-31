# EvoXXLTraffic: dataset and sensor-evolving benchmark

EvoXXLTraffic extends the XXLTraffic dataset family with yearly sensor catalogs, persistent station IDs, flow matrices and
geographic graph snapshots for nine California PeMS districts. It covers up to 27 years, from 1999 to 2025.

This repository holds the benchmark code, frozen protocols, run manifests, per-year results and sensor coordinates for
the TSAS revision of *From XXLTraffic to EvoXXLTraffic*. The article is a dataset and benchmark contribution; it does not
introduce a new forecasting model.

**Release `tsas26-r1`.** The data live on Hugging Face: <https://huggingface.co/datasets/yind7/EvoXXLTraffic>. That repository has the corrected yearly caches with explicit target/input masks, the coordinates, and 3,523 dated PeMS metadata snapshots.

## Corrected experiment scope

| District | Evaluated years | Methods | Seeds | Complete sequential runs |
|---|---|---:|---|---:|
| PEMS04 | 2002–2025 | 13 | 47–51 | 65 |
| PEMS05 | 2005–2025 | 13 | 47–49 | 39 |
| PEMS03 | 2001–2025 | Retrain, Online-AN, ST-TTC | 47–49 | 9 |
| PEMS06 | 2005–2025 | Retrain, Online-AN, ST-TTC | 47–49 | 9 |
| PEMS10 | 2006–2025 | Retrain, Online-AN, ST-TTC | 47–49 | 9 |

There are 131 main runs and 3 PEMS04 Online-AN coordinate-policy runs.

* Each district is trained and tested on its own; these are not transfer experiments.
* Each yearly task covers the first 31 stored days of the year.
* Six implementations were repaired after a source audit: DCRNN, ASTGCN, TGCN, PECPM, STRAP and STKEC. The repaired code is in `benchmark_v3/`.
* The repairs' 48 PEMS04/PEMS05 chains replace the earlier results (see `benchmark_release/results/replacement_chains.csv`).

**Results.** All numbers are full-12-lead MAE, mean ± sample SD over seeds after equal-year aggregation. Full tables are in `benchmark_release/results/final_metrics.csv` and `summary.csv`.

* **PEMS04:** PECPM has the lowest MAE on all sensors (20.760 ± 0.081) and on new-to-previous-year sensors (17.315 ± 0.093).
* **PEMS05:** Online-AN has the lowest MAE on all sensors (11.515 ± 0.037) and on new-to-previous-year sensors (9.984 ± 0.034).
* **Other districts:** Online-AN also has the lowest MAE among the three methods run on PEMS03, PEMS06 and PEMS10.
* **Pretrain:** worse than the training-free last-value reference in both districts.
* **TGCN:** under its frozen configuration, TGCN's error barely changes with the horizon and stays close to that of predicting the training mean everywhere. It is reported as a degenerate fit, not as evidence about the method.

## Audited archive statistics

Counts are columns in the stored yearly sensor catalogs, not counts of valid observations or of physical road growth.

| District | Stored years | First catalog | Last catalog | Growth |
|---|---|---:|---:|---:|
| PEMS03 | 2001–2025 (25) | 174 | 1,859 | +968% |
| PEMS04 | 2001–2025 (25) | 1,266 | 4,110 | +225% |
| PEMS05 | 2005–2025 (21) | 6 | 572 | +9,433% |
| PEMS06 | 2005–2025 (21) | 13 | 746 | +5,638% |
| PEMS07 | 2001–2025 (25) | 3,215 | 4,888 | +52% |
| PEMS08 | 2001–2025 (25) | 262 | 2,074 | +692% |
| PEMS10 | 2006–2025 (20) | 45 | 1,396 | +3,002% |
| PEMS11 | 1999–2025 (27) | 424 | 1,440 | +240% |
| PEMS12 | 2002–2025 (24) | 1,783 | 2,587 | +45% |

PEMS04/2001 remains in the archive but is excluded from the benchmark, because its first 31 days have no accepted targets.

## Reproduce and inspect

* [`benchmark_release/README.md`](benchmark_release/README.md): how to recompute every reported mean and SD from the per-year export (standard library only), and how to prepare one exact run from its manifest.
* [`benchmark_release/index.json`](benchmark_release/index.json) and `benchmark_release/manifests/`: all 134 run manifests, with source, protocol, data and graph checksums.
* `benchmark_release/results/`: per-year scores and denominators (`yearly_metrics.csv.gz`), run receipts (`runs.csv`), recomputed summaries for the full period and 2018–2025 (`summary.csv`), the accepted table (`final_metrics.csv`), training-free references, the coordinate-policy sensitivity and the catalog statistics.
* [`coordinates/`](coordinates/README.md): sensor coordinates in matrix-column order for every stored district-year, with source snapshots, no-look-ahead (dated) coordinates and a tool that builds unthresholded distances or any kernel. Rebuilding the stored graphs from these coordinates reproduces all 212 of them entry for entry.

**Metric conventions.**

* Each year receives equal weight within a seed. The seed mean and sample SD are then reported.
* Horizons 3/6/12 pool the first 3/6/12 leads.
* Missing targets are masked and valid zeros are kept. MAPE excludes |y| ≤ 1e-6.
* New-to-previous-year sensors include returning sensors and receive current-year training. This is supervised adaptation, not zero-shot cold start.

**Graph caveat.** Graphs describe retrospective geographic proximity, not road connectivity. The coordinate-policy runs restrict coordinate dates and isolate sensors without a dated coordinate. They used four PEMS04 metadata snapshots, so they isolate more sensors than the full snapshot set would require.

## Data access

* **Hugging Face dataset:** <https://huggingface.co/datasets/yind7/EvoXXLTraffic>. It has four parts:
  * the corrected benchmark caches;
  * the full-year 5-minute flow matrices and graphs of all 213 district-years, with `full_year/catalog.json` giving sensor IDs and start times;
  * the coordinates;
  * the metadata snapshots.

  Check the files with `SHA256SUMS` and `SHA256SUMS.full_year`. The full-year matrices are stored filled (forward, then backward, then zero). The dataset card describes the processing pipeline and what this filling means.
* **Original processed archive and raw records:** [processed](https://pan.baidu.com/s/1llz16kYY33TrWlKENNHC5A?pwd=xxtf) and [raw](https://pan.baidu.com/s/1BPuxL96npWlfRXDv38duww?pwd=xxtf), access code `xxtf`. The processed matrices there are forward- and backward-filled. Do not use them in place of the corrected caches.
* **Preprocessing:** the original notebooks are in `xxltrafficdata/`. Baseline code is in `eac/` and is run through `benchmark_v2/` or `benchmark_v3/`.

## License

* **Code:** author-written code is under the MIT License ([`LICENSE`](LICENSE)).
* **Data products:** catalogs, caches, graphs, coordinates, manifests and results are under CC BY-NC 4.0 ([`LICENSE-DATA`](LICENSE-DATA)).
* **Third-party code:** adapted baseline code keeps the license of its original repository.
* **Upstream records:** PeMS terms apply.

See [`NOTICE.txt`](NOTICE.txt).

## Citation

Paper: [From XXLTraffic to EvoXXLTraffic](https://arxiv.org/abs/2605.29768). The inherited XXLTraffic work is
[SIGSPATIAL 2025](https://dl.acm.org/doi/10.1145/3748636.3762762).
