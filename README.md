# EvoXXLTraffic: dataset and sensor-evolving benchmark

This revision separates the nine-district dataset archive from the **five-district corrected evaluation**. It accompanies the TSAS dataset/benchmark revision; it does not introduce a new forecasting model.

## Corrected experiment scope

| District | Evaluated years | Methods | Seeds | Complete sequential runs |
|---|---|---:|---|---:|
| PEMS04 | 2002–2025 | 13 | 47–51 | 65 |
| PEMS05 | 2005–2025 | 13 | 47–49 | 39 |
| PEMS03 | 2001–2025 | Retrain, Online-AN, ST-TTC | 47–49 | 9 |
| PEMS06 | 2005–2025 | Retrain, Online-AN, ST-TTC | 47–49 | 9 |
| PEMS10 | 2006–2025 | Retrain, Online-AN, ST-TTC | 47–49 | 9 |

There are **131 main runs**, plus **3 PEMS04 Online-AN coordinate-policy sensitivity runs**, totaling 3,045 model–year tasks. Each district is trained and tested separately; these are not district-transfer experiments. Each yearly task covers 31 stored days, not a full calendar year. Gap/hourly/daily experiments are inherited work and were not revalidated by these runs.

## Audited archive statistics

Counts below are columns in the stored yearly sensor catalogs, not counts of valid observations or physical road growth. The original PEMS04 “~25 sensors” statement was erroneous.

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

PEMS04/2001 remains in archive visualizations but is excluded from corrected forecasting evaluation. See [audited statistics](benchmark_release/results/district_stats.csv).

## Reproduce and inspect the revised results

- [Reproduction instructions and limitations](benchmark_release/README.md)
- [All 134 portable experiment manifests](benchmark_release/index.json), with original configuration/input checksums
- [Completed-run evidence](benchmark_release/results/runs.csv) and [yearly checks](benchmark_release/results/year_checks.csv)
- [Per-year scores and denominators](benchmark_release/results/yearly_metrics.csv.gz)
- [Coordinate-policy sensitivity](benchmark_release/results/coordinate_sensitivity.json)
- [Artifact licensing status](NOTICE.txt)

The main metric averages each eligible year equally within a seed, then reports seed mean and sample standard deviation. Horizons 3/6/12 pool the first 3/6/12 leads; “Avg” is the 12-lead score. Missing targets are masked, valid zeros are retained, and MAPE excludes targets within 1e-6 of zero. New-to-previous-year nodes include returning sensors and receive current-year training; this is supervised adaptation, not zero-shot cold start.

Main graphs describe retrospective geographic proximity. Restricting coordinate dates and isolating missing-location nodes jointly raises Online-AN MAE by 2.72% for all nodes and 11.40% for new-to-previous-year nodes. This experiment neither isolates the two interventions nor checks all-method ranking stability. The catalog remains retrospective; no strictly prospective deployment claim is made.

## Data access

[Original processed archive](https://pan.baidu.com/s/1llz16kYY33TrWlKENNHC5A?pwd=xxtf) and [original raw records](https://pan.baidu.com/s/1BPuxL96npWlfRXDv38duww?pwd=xxtf) use access code `xxtf`. These are the existing author-provided links. Their contents have **not** been verified as a distribution of the newly corrected observation-aware NPZ caches. The manifests identify the exact corrected cache hashes; download access alone does not establish a match.

The corrected-cache upload and artifact-specific licensing notices remain release tasks. Legacy filled NPZ matrices must not be substituted for the explicit masks used in the revised benchmark. Original preprocessing notebooks remain under `xxltrafficdata/`; original baseline commands under `eac/` describe the earlier implementation unless invoked through the frozen revised runner below.

Paper: [From XXLTraffic to EvoXXLTraffic](https://arxiv.org/abs/2605.29768). The inherited XXLTraffic work is [SIGSPATIAL 2025](https://dl.acm.org/doi/10.1145/3748636.3762762).
