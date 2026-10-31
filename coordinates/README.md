# Sensor coordinates and unthresholded distances

`coordinates_<DISTRICT>.csv.gz` lists, for every stored district-year of EvoXXLTraffic, the sensors in matrix-column
order with their coordinates, so that any distance, kernel or threshold can be built instead of only the stored
thresholded graph. There are 212 district-years, which are the years with corrected benchmark caches. The PEMS04/2001
archive year is not cached.

| column | meaning |
|---|---|
| `district`, `year`, `column` | district, calendar year, column index in that year's flow matrix and graph |
| `sensor_id` | PeMS station ID |
| `lat`, `lon` | coordinates used by the stored graphs and the main comparison |
| `lat_lon_source` | date of the metadata snapshot that supplied `lat`/`lon`, or `centroid` for the missing-coordinate fallback |
| `dated_lat`, `dated_lon`, `dated_snapshot` | the earliest valid coordinate in a snapshot dated on or before the catalog anchor (no look-ahead); empty if none exists |

## How the coordinates are built

**Historical coordinates (`lat`, `lon`)** follow the original preprocessing notebooks:

* The district's metadata snapshots are read in date order.
* Each station takes the first snapshot that lists it with non-zero latitude and longitude.
* Stations listed in no snapshot get the mean coordinates of the stations that were found.
* PEMS04 uses four snapshots. The other districts use 77 to 1,474 snapshots each.

**Dated coordinates** use every snapshot in the archive, including 211 non-empty PEMS04 snapshots from 2001 to 2026. Only snapshots dated on or before the catalog anchor are eligible. The anchor is 1 January of the year, or the first cached day for a district's first year.

## Verification

`VERIFICATION.json` reports the result of rebuilding each stored graph from `lat`/`lon` with the published kernel:

* Haversine distance in km, R = 6371;
* sigma = population SD of positive pairwise distances;
* w = exp(-d²/sigma²);
* w < 0.1 set to 0, with a zero diagonal.

All 212 stored graphs are reproduced. No edge appears or disappears, and the largest difference is about 3e-8, which is float32 rounding.

## Use

```bash
python coordinates/distances.py PEMS04 2014 --out pems04_2014_distances.npz
python coordinates/distances.py PEMS04 2014 --dated --kernel --out pems04_2014_dated_kernel.npz
python coordinates/distances.py PEMS04 2014 --check /path/to/pems04/2014/graph.npz
```

`export_coordinates.py` regenerates the CSV files from the metadata snapshots and the corrected caches.

These coordinates support geographic-proximity graphs only. They do not describe directed road connectivity or travel time.
