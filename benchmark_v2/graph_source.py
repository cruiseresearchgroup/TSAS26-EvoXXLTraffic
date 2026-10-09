#!/usr/bin/env python3
"""Build sensor metadata aligned with every XXLTraffic sensor axis.

The PEMS ``RawData`` and ``FastData`` archives only contain numeric arrays; the
station IDs that formed their columns were not persisted.  The preprocessing
notebooks did, however, preserve a deterministic witness: each yearly graph was
built in the same (ascending station-ID) order from PeMS metadata coordinates.

This script recovers the yearly PEMS station-ID lists by matching historical
metadata snapshots against those graphs.  A mapping is written only after a
blockwise comparison of every reconstructed graph entry succeeds.  Truly
indistinguishable co-located IDs are retained as explicit alternatives rather
than silently guessed.

TFNSW does not require recovery: its yearly ``*_nodes.csv`` files already store
the exact array index.  They are normalized into the same output layout.

Default output layout::

    visualization_exports/
      mapping_audit.csv
      sensor_metadata_manifest.json
      pems03/
        sensor_catalog.csv
        sensor_catalog.npz
        2001/
          sensor_metadata.csv
          sensor_metadata.npz
      ...
      tfnsw/
        ...

Run with the environment used by the project, for example::

    /srv/scratch/cruise/du/conda/envs/basicts058/bin/python \
      xxltrafficdata/build_visualization_sensor_metadata.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


PEMS_DISTRICTS = ("03", "04", "05", "06", "07", "08", "10", "11", "12")
FIRST_DATES = {
    "03": date(2001, 3, 4),
    "04": date(2001, 9, 1),
    "05": date(2005, 3, 24),
    "06": date(2005, 9, 1),
    "07": date(2001, 1, 1),
    "08": date(2001, 3, 1),
    "10": date(2006, 7, 1),
    "11": date(1999, 7, 1),
    "12": date(2002, 1, 1),
}
META_DATE_RE = re.compile(r"_(\d{4})_(\d{2})_(\d{2})\.txt$")
FULL_MAX_ABS_TOLERANCE = 1e-5
SAMPLE_MAX_ABS_TOLERANCE = 1e-5


@dataclass(frozen=True)
class MetaRow:
    station_id: int
    latitude: float
    longitude: float
    freeway: str
    direction: str
    station_type: str
    lanes: str


@dataclass
class StationRecord:
    latitude: float = math.nan
    longitude: float = math.nan
    freeway: str = ""
    direction: str = ""
    station_type: str = ""
    lanes: str = ""
    coordinate_source: str = ""


@dataclass(frozen=True)
class CandidateSet:
    station_ids: tuple[int, ...]
    snapshots: tuple[Path, ...]


@dataclass(frozen=True)
class GraphValidation:
    sigma_km: float
    topology_mismatches: int
    max_abs_error: float
    mean_abs_error: float
    compared_entries: int

    @property
    def passed(self) -> bool:
        return self.topology_mismatches == 0 and self.max_abs_error <= FULL_MAX_ABS_TOLERANCE


@dataclass
class CandidateValidation:
    candidate: CandidateSet
    graph_coordinates: np.ndarray
    graph_missing: np.ndarray
    validation: GraphValidation


@dataclass
class RecoveredYear:
    year: int
    canonical: CandidateValidation
    valid_candidates: list[CandidateValidation]
    source_snapshot: Path
    mapping_status: str
    id_status: np.ndarray
    alternatives: np.ndarray


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_default = script_path.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <repo-root>/visualization_exports",
    )
    parser.add_argument(
        "--pems-districts",
        nargs="*",
        default=list(PEMS_DISTRICTS),
        choices=PEMS_DISTRICTS,
        help="Subset of PEMS district numbers to process.",
    )
    parser.add_argument("--skip-tfnsw", action="store_true")
    parser.add_argument(
        "--validation",
        choices=("full", "sample"),
        default="full",
        help="Full compares all N^2 graph entries; sample is intended for development only.",
    )
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


def clean_header(value: str) -> str:
    return value.lstrip("\ufeff").strip().strip('"')


def parse_float(value: str) -> float:
    try:
        return float(value.strip().strip('"'))
    except (TypeError, ValueError):
        return math.nan


def iter_meta_rows(path: Path) -> Iterator[MetaRow]:
    """Parse needed leading TSV fields, tolerating malformed quotes in Name.

    Several historical D04 files contain an unmatched quote in a late text
    column.  A normal CSV parser rejects the complete file even though ID and
    coordinates occur before that column.  Literal tab splitting recovers the
    fields used by the preprocessing notebooks.
    """

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        first = handle.readline()
        if not first:
            return
        headers = [clean_header(item) for item in first.rstrip("\r\n").split("\t")]
        indices = {name: idx for idx, name in enumerate(headers)}
        if "ID" not in indices:
            return

        def value(parts: list[str], field: str) -> str:
            idx = indices.get(field)
            return parts[idx].strip().strip('"') if idx is not None and idx < len(parts) else ""

        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            raw_id = value(parts, "ID")
            try:
                station_id = int(float(raw_id))
            except (TypeError, ValueError):
                continue
            yield MetaRow(
                station_id=station_id,
                latitude=parse_float(value(parts, "Latitude")),
                longitude=parse_float(value(parts, "Longitude")),
                freeway=value(parts, "Fwy"),
                direction=value(parts, "Dir"),
                station_type=value(parts, "Type"),
                lanes=value(parts, "Lanes"),
            )


def valid_coordinate(latitude: float, longitude: float) -> bool:
    return (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and latitude != 0.0
        and longitude != 0.0
    )


def snapshot_date(path: Path) -> date:
    match = META_DATE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse metadata date from {path}")
    return date(*(int(part) for part in match.groups()))


def relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def merge_record(catalog: dict[int, StationRecord], row: MetaRow, source: Path) -> None:
    record = catalog.setdefault(row.station_id, StationRecord())
    if not valid_coordinate(record.latitude, record.longitude) and valid_coordinate(
        row.latitude, row.longitude
    ):
        record.latitude = row.latitude
        record.longitude = row.longitude
        record.coordinate_source = str(source)
    if not record.freeway and row.freeway:
        record.freeway = row.freeway
    if not record.direction and row.direction:
        record.direction = row.direction
    if not record.station_type and row.station_type:
        record.station_type = row.station_type
    if not record.lanes and row.lanes:
        record.lanes = row.lanes


def merge_catalogs(
    destination: dict[int, StationRecord], source: dict[int, StationRecord]
) -> None:
    """Fill gaps in a chronological catalog without replacing earlier values."""

    for station_id, incoming in source.items():
        record = destination.setdefault(station_id, StationRecord())
        if not valid_coordinate(record.latitude, record.longitude) and valid_coordinate(
            incoming.latitude, incoming.longitude
        ):
            record.latitude = incoming.latitude
            record.longitude = incoming.longitude
            record.coordinate_source = incoming.coordinate_source
        if not record.freeway:
            record.freeway = incoming.freeway
        if not record.direction:
            record.direction = incoming.direction
        if not record.station_type:
            record.station_type = incoming.station_type
        if not record.lanes:
            record.lanes = incoming.lanes


def load_snapshot_candidates(
    files: Sequence[Path], required_sizes: set[int]
) -> tuple[dict[int, list[CandidateSet]], dict[int, StationRecord]]:
    grouped: dict[int, dict[tuple[int, ...], list[Path]]] = {}
    catalog: dict[int, StationRecord] = {}

    for path in sorted(files, key=lambda item: (snapshot_date(item), item.name)):
        rows = list(iter_meta_rows(path))
        if not rows:
            continue
        by_id: dict[int, MetaRow] = {}
        for row in rows:
            by_id.setdefault(row.station_id, row)
            merge_record(catalog, row, path)
        station_ids = tuple(sorted(by_id))
        if len(station_ids) in required_sizes:
            grouped.setdefault(len(station_ids), {}).setdefault(station_ids, []).append(path)

    candidates = {
        size: [
            CandidateSet(station_ids=ids, snapshots=tuple(paths))
            for ids, paths in by_ids.items()
        ]
        for size, by_ids in grouped.items()
    }
    return candidates, catalog


def build_station_catalog(files: Iterable[Path]) -> dict[int, StationRecord]:
    catalog: dict[int, StationRecord] = {}
    valid_files = [path for path in files if META_DATE_RE.search(path.name)]
    for path in sorted(valid_files, key=lambda item: (snapshot_date(item), item.name)):
        for row in iter_meta_rows(path):
            merge_record(catalog, row, path)
    return catalog


def npz_array_shape(path: Path, key: str) -> tuple[int, ...]:
    """Read an NPY header inside an NPZ without materializing the array."""

    with zipfile.ZipFile(path) as archive, archive.open(f"{key}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, _, _ = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY version {version} in {path}:{key}")
    return tuple(int(item) for item in shape)


def graph_files_and_sizes(repo_root: Path, district: str) -> list[tuple[int, Path, int]]:
    graph_dir = repo_root / "xxltrafficdata" / "preprocessed" / f"pems{district}" / "graph"
    result = []
    for path in sorted(graph_dir.glob("*_adj.npz")):
        year = int(path.stem.split("_")[0])
        shape = npz_array_shape(path, "x")
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"Expected square graph in {path}, got {shape}")
        result.append((year, path, shape[0]))
    if not result:
        raise FileNotFoundError(f"No graph files found in {graph_dir}")
    return result


def catalog_coordinates(
    station_ids: Sequence[int], catalog: dict[int, StationRecord], *, fill_missing: bool
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.full((len(station_ids), 2), np.nan, dtype=np.float64)
    for idx, station_id in enumerate(station_ids):
        record = catalog.get(station_id)
        if record and valid_coordinate(record.latitude, record.longitude):
            coordinates[idx] = (record.latitude, record.longitude)
    missing = ~np.isfinite(coordinates).all(axis=1)
    if fill_missing and missing.any():
        present = ~missing
        if not present.any():
            raise ValueError("Cannot apply coordinate fallback: no station has valid coordinates")
        coordinates[missing] = coordinates[present].mean(axis=0)
    return coordinates, missing


def haversine_cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_lat = np.radians(left[:, 0])[:, None]
    left_lon = np.radians(left[:, 1])[:, None]
    right_lat = np.radians(right[:, 0])[None, :]
    right_lon = np.radians(right[:, 1])[None, :]
    delta_lat = left_lat - right_lat
    delta_lon = left_lon - right_lon
    value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(left_lat) * np.cos(right_lat) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.maximum(value, 0.0)))


def sample_axes(size: int, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if size <= limit:
        axis = np.arange(size, dtype=np.int64)
        return axis, axis
    left = np.unique(np.linspace(0, size - 1, limit, dtype=np.int64))
    right = np.unique(np.linspace(1, size - 1, limit + 3, dtype=np.int64))
    return left, right


def estimate_sigma_and_sample_error(
    graph: np.ndarray, coordinates: np.ndarray, limit: int = 96
) -> GraphValidation:
    left_indices, right_indices = sample_axes(len(coordinates), limit)
    distances = haversine_cross(coordinates[left_indices], coordinates[right_indices])
    weights = graph[np.ix_(left_indices, right_indices)].astype(np.float64)
    off_diagonal = left_indices[:, None] != right_indices[None, :]
    usable = (
        off_diagonal
        & (weights > 0.0)
        & (weights < 0.9999999)
        & (distances > 1e-12)
    )
    if not usable.any():
        raise ValueError("Could not estimate graph sigma from sampled positive edges")
    sigma_values = distances[usable] / np.sqrt(-np.log(weights[usable]))
    sigma = float(np.median(sigma_values))

    predicted = np.exp(-(distances**2) / (sigma**2))
    predicted[predicted < 0.1] = 0.0
    predicted[~off_diagonal] = 0.0
    errors = np.abs(predicted[off_diagonal] - weights[off_diagonal])
    topology = int(
        np.count_nonzero(((predicted > 0.0) != (weights > 0.0)) & off_diagonal)
    )
    return GraphValidation(
        sigma_km=sigma,
        topology_mismatches=topology,
        max_abs_error=float(errors.max(initial=0.0)),
        mean_abs_error=float(errors.mean()) if errors.size else 0.0,
        compared_entries=int(errors.size),
    )


def validate_full_graph(
    graph: np.ndarray, coordinates: np.ndarray, sigma: float, block_size: int
) -> GraphValidation:
    topology_mismatches = 0
    max_abs_error = 0.0
    absolute_error_sum = 0.0
    compared_entries = 0
    node_count = len(coordinates)

    for start in range(0, node_count, block_size):
        end = min(start + block_size, node_count)
        distances = haversine_cross(coordinates[start:end], coordinates)
        predicted = np.exp(-(distances**2) / (sigma**2))
        predicted[predicted < 0.1] = 0.0
        predicted[np.arange(end - start), np.arange(start, end)] = 0.0
        observed = graph[start:end].astype(np.float64)
        topology_mismatches += int(np.count_nonzero((predicted > 0.0) != (observed > 0.0)))
        errors = np.abs(predicted - observed)
        max_abs_error = max(max_abs_error, float(errors.max(initial=0.0)))
        absolute_error_sum += float(errors.sum(dtype=np.float64))
        compared_entries += int(errors.size)

    return GraphValidation(
        sigma_km=sigma,
        topology_mismatches=topology_mismatches,
        max_abs_error=max_abs_error,
        mean_abs_error=absolute_error_sum / compared_entries if compared_entries else 0.0,
        compared_entries=compared_entries,
    )


def preferred_snapshot(candidate: CandidateSet, anchor: date) -> Path:
    def rank(path: Path) -> tuple[int, int, date, str]:
        current = snapshot_date(path)
        delta = (current - anchor).days
        return abs(delta), 0 if delta <= 0 else 1, current, path.name

    return min(candidate.snapshots, key=rank)


def recover_one_year(
    *,
    year: int,
    graph_path: Path,
    candidates: Sequence[CandidateSet],
    graph_catalog: dict[int, StationRecord],
    anchor: date,
    validation_mode: str,
    block_size: int,
) -> RecoveredYear:
    with np.load(graph_path) as archive:
        graph = archive["x"]

    sampled: list[tuple[GraphValidation, CandidateSet, np.ndarray, np.ndarray]] = []
    for candidate in candidates:
        coordinates, missing = catalog_coordinates(
            candidate.station_ids, graph_catalog, fill_missing=True
        )
        validation = estimate_sigma_and_sample_error(graph, coordinates)
        sampled.append((validation, candidate, coordinates, missing))

    sampled.sort(
        key=lambda item: (
            item[0].topology_mismatches,
            item[0].max_abs_error,
            item[0].mean_abs_error,
        )
    )
    promising = [
        item
        for item in sampled
        if item[0].topology_mismatches == 0
        and item[0].max_abs_error <= SAMPLE_MAX_ABS_TOLERANCE
    ]
    if not promising:
        promising = sampled[: min(3, len(sampled))]

    valid: list[CandidateValidation] = []
    for sample_validation, candidate, coordinates, missing in promising:
        if validation_mode == "full":
            validation = validate_full_graph(
                graph, coordinates, sample_validation.sigma_km, block_size
            )
        else:
            validation = sample_validation
        if validation.passed:
            valid.append(
                CandidateValidation(
                    candidate=candidate,
                    graph_coordinates=coordinates,
                    graph_missing=missing,
                    validation=validation,
                )
            )

    if not valid:
        best = sampled[0][0] if sampled else None
        raise RuntimeError(
            f"No verified metadata mapping for {graph_path}; "
            f"candidate_count={len(candidates)}, best_sample={best}"
        )

    canonical = min(
        valid,
        key=lambda item: (
            abs((snapshot_date(preferred_snapshot(item.candidate, anchor)) - anchor).days),
            0 if snapshot_date(preferred_snapshot(item.candidate, anchor)) <= anchor else 1,
            preferred_snapshot(item.candidate, anchor).name,
        ),
    )
    station_count = len(canonical.candidate.station_ids)
    alternatives = np.empty(station_count, dtype=object)
    id_status = np.full(station_count, "exact", dtype=object)
    mapping_status = "exact"

    for index in range(station_count):
        ids = sorted({item.candidate.station_ids[index] for item in valid})
        alternatives[index] = "|".join(str(item) for item in ids)
        if len(ids) <= 1:
            continue
        points = np.asarray([item.graph_coordinates[index] for item in valid])
        co_located = bool(np.allclose(points, points[0], rtol=0.0, atol=1e-10))
        id_status[index] = "co_located_ambiguous" if co_located else "graph_ambiguous"
        mapping_status = "co_located_ambiguous" if co_located else "graph_ambiguous"

    return RecoveredYear(
        year=year,
        canonical=canonical,
        valid_candidates=valid,
        source_snapshot=preferred_snapshot(canonical.candidate, anchor),
        mapping_status=mapping_status,
        id_status=np.asarray(id_status, dtype=str),
        alternatives=np.asarray(alternatives, dtype=str),
    )


def pems_meta_sources(repo_root: Path, district: str) -> tuple[list[Path], list[Path], str]:
    downloaded = sorted((repo_root / "claude" / f"d{district}").glob(f"d{district}_text_meta_*.txt"))
    if not downloaded and district == "04":
        downloaded = sorted((repo_root / "meta" / "pems04").glob("d04_text_meta_*.txt"))
    if not downloaded:
        downloaded = sorted((repo_root / "meta").glob(f"d{district}_text_meta_*.txt"))
    if not downloaded:
        raise FileNotFoundError(f"No metadata snapshots found for PEMS{district}")

    # PEMS04 graph warnings and weights reproduce exactly with these four
    # selected snapshots.  The malformed full D04 history was used to identify
    # yearly station sets, but was not the coordinate directory used to build
    # the saved graphs.  Other districts reproduce with the downloaded history.
    if district == "04":
        graph_sources = sorted((repo_root / "meta").glob("d04_text_meta_*.txt"))
        source_label = "meta/d04_text_meta_*.txt"
    else:
        graph_sources = downloaded
        source_label = f"claude/d{district}"
    if not graph_sources:
        raise FileNotFoundError(f"No graph-coordinate snapshots found for PEMS{district}")
    return downloaded, graph_sources, source_label


def string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=str)


def write_pems_year(
    *,
    repo_root: Path,
    output_root: Path,
    district: str,
    recovered: RecoveredYear,
    actual_catalog: dict[int, StationRecord],
    first_year: dict[int, int],
    last_year: dict[int, int],
    validation_mode: str,
    graph_source_label: str,
) -> dict[str, object]:
    dataset = f"pems{district}"
    station_ids = np.asarray(recovered.canonical.candidate.station_ids, dtype=np.int64)
    actual_coordinates, actual_missing = catalog_coordinates(
        station_ids, actual_catalog, fill_missing=False
    )
    graph_coordinates = recovered.canonical.graph_coordinates
    graph_missing = recovered.canonical.graph_missing
    records = [actual_catalog.get(int(item), StationRecord()) for item in station_ids]
    year_dir = output_root / dataset / str(recovered.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = relative_path(recovered.source_snapshot, repo_root)
    validation = recovered.canonical.validation

    payload = {
        "dataset": np.asarray(dataset),
        "year": np.asarray(recovered.year, dtype=np.int16),
        "sensor_index": np.arange(len(station_ids), dtype=np.int32),
        "station_id": station_ids,
        "latitude": actual_coordinates[:, 0],
        "longitude": actual_coordinates[:, 1],
        "coordinate_status": string_array(
            ["missing" if flag else "metadata" for flag in actual_missing]
        ),
        "graph_latitude": graph_coordinates[:, 0],
        "graph_longitude": graph_coordinates[:, 1],
        "graph_coordinate_status": string_array(
            ["mean_fallback" if flag else "metadata" for flag in graph_missing]
        ),
        "first_year": np.asarray([first_year[int(item)] for item in station_ids], dtype=np.int16),
        "last_year": np.asarray([last_year[int(item)] for item in station_ids], dtype=np.int16),
        "freeway": string_array([item.freeway for item in records]),
        "direction": string_array([item.direction for item in records]),
        "station_type": string_array([item.station_type for item in records]),
        "lanes": string_array([item.lanes for item in records]),
        "id_status": recovered.id_status,
        "station_id_alternatives": recovered.alternatives,
        "mapping_status": np.asarray(recovered.mapping_status),
        "source_snapshot": np.asarray(source_snapshot),
        "candidate_snapshots": string_array(
            sorted(
                {
                    relative_path(path, repo_root)
                    for item in recovered.valid_candidates
                    for path in item.candidate.snapshots
                }
            )
        ),
        "graph_coordinate_source": np.asarray(graph_source_label),
        "graph_validation_mode": np.asarray(validation_mode),
        "graph_sigma_km": np.asarray(validation.sigma_km, dtype=np.float64),
        "graph_topology_mismatches": np.asarray(
            validation.topology_mismatches, dtype=np.int64
        ),
        "graph_max_abs_error": np.asarray(validation.max_abs_error, dtype=np.float64),
        "graph_mean_abs_error": np.asarray(validation.mean_abs_error, dtype=np.float64),
        "graph_compared_entries": np.asarray(validation.compared_entries, dtype=np.int64),
    }
    np.savez_compressed(year_dir / "sensor_metadata.npz", **payload)

    csv_path = year_dir / "sensor_metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sensor_index",
            "station_id",
            "latitude",
            "longitude",
            "coordinate_status",
            "graph_latitude",
            "graph_longitude",
            "graph_coordinate_status",
            "first_year",
            "last_year",
            "freeway",
            "direction",
            "station_type",
            "lanes",
            "id_status",
            "station_id_alternatives",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, station_id in enumerate(station_ids):
            writer.writerow(
                {
                    "sensor_index": index,
                    "station_id": int(station_id),
                    "latitude": actual_coordinates[index, 0],
                    "longitude": actual_coordinates[index, 1],
                    "coordinate_status": "missing" if actual_missing[index] else "metadata",
                    "graph_latitude": graph_coordinates[index, 0],
                    "graph_longitude": graph_coordinates[index, 1],
                    "graph_coordinate_status": (
                        "mean_fallback" if graph_missing[index] else "metadata"
                    ),
                    "first_year": first_year[int(station_id)],
                    "last_year": last_year[int(station_id)],
                    "freeway": records[index].freeway,
                    "direction": records[index].direction,
                    "station_type": records[index].station_type,
                    "lanes": records[index].lanes,
                    "id_status": recovered.id_status[index],
                    "station_id_alternatives": recovered.alternatives[index],
                }
            )

    return {
        "dataset": dataset,
        "year": recovered.year,
        "sensor_count": int(len(station_ids)),
        "mapping_status": recovered.mapping_status,
        "valid_candidate_sets": len(recovered.valid_candidates),
        "source_snapshot": source_snapshot,
        "missing_coordinates": int(actual_missing.sum()),
        "graph_fallback_coordinates": int(graph_missing.sum()),
        "graph_validation_mode": validation_mode,
        "graph_sigma_km": validation.sigma_km,
        "graph_topology_mismatches": validation.topology_mismatches,
        "graph_max_abs_error": validation.max_abs_error,
        "graph_mean_abs_error": validation.mean_abs_error,
        "graph_compared_entries": validation.compared_entries,
    }


def write_pems_catalog(
    *,
    output_root: Path,
    district: str,
    recovered_years: Sequence[RecoveredYear],
    actual_catalog: dict[int, StationRecord],
    first_year: dict[int, int],
    last_year: dict[int, int],
) -> None:
    dataset = f"pems{district}"
    years_by_station: dict[int, list[int]] = {}
    ambiguity_by_station: dict[int, set[str]] = {}
    alternatives_by_station: dict[int, set[int]] = {}
    for recovered in recovered_years:
        for index, station_id in enumerate(recovered.canonical.candidate.station_ids):
            years_by_station.setdefault(station_id, []).append(recovered.year)
            ambiguity_by_station.setdefault(station_id, set()).add(recovered.id_status[index])
            alternatives_by_station.setdefault(station_id, set()).update(
                int(item) for item in recovered.alternatives[index].split("|") if item
            )

    station_ids = sorted(years_by_station)
    records = [actual_catalog.get(item, StationRecord()) for item in station_ids]
    latitude = np.asarray([item.latitude for item in records], dtype=np.float64)
    longitude = np.asarray([item.longitude for item in records], dtype=np.float64)
    coordinate_status = string_array(
        [
            "metadata" if valid_coordinate(item.latitude, item.longitude) else "missing"
            for item in records
        ]
    )
    id_status = string_array(
        [
            "co_located_ambiguous"
            if "co_located_ambiguous" in ambiguity_by_station[item]
            else "graph_ambiguous"
            if "graph_ambiguous" in ambiguity_by_station[item]
            else "exact"
            for item in station_ids
        ]
    )
    alternatives = string_array(
        ["|".join(str(value) for value in sorted(alternatives_by_station[item])) for item in station_ids]
    )
    years_present = string_array(
        ["|".join(str(year) for year in years_by_station[item]) for item in station_ids]
    )

    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_dir / "sensor_catalog.npz",
        dataset=np.asarray(dataset),
        station_id=np.asarray(station_ids, dtype=np.int64),
        latitude=latitude,
        longitude=longitude,
        coordinate_status=coordinate_status,
        first_year=np.asarray([first_year[item] for item in station_ids], dtype=np.int16),
        last_year=np.asarray([last_year[item] for item in station_ids], dtype=np.int16),
        years_present=years_present,
        freeway=string_array([item.freeway for item in records]),
        direction=string_array([item.direction for item in records]),
        station_type=string_array([item.station_type for item in records]),
        lanes=string_array([item.lanes for item in records]),
        id_status=id_status,
        station_id_alternatives=alternatives,
    )

    with (dataset_dir / "sensor_catalog.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "station_id",
            "latitude",
            "longitude",
            "coordinate_status",
            "first_year",
            "last_year",
            "years_present",
            "freeway",
            "direction",
            "station_type",
            "lanes",
            "id_status",
            "station_id_alternatives",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, station_id in enumerate(station_ids):
            writer.writerow(
                {
                    "station_id": station_id,
                    "latitude": latitude[index],
                    "longitude": longitude[index],
                    "coordinate_status": coordinate_status[index],
                    "first_year": first_year[station_id],
                    "last_year": last_year[station_id],
                    "years_present": years_present[index],
                    "freeway": records[index].freeway,
                    "direction": records[index].direction,
                    "station_type": records[index].station_type,
                    "lanes": records[index].lanes,
                    "id_status": id_status[index],
                    "station_id_alternatives": alternatives[index],
                }
            )


def recover_pems_district(
    repo_root: Path,
    output_root: Path,
    district: str,
    validation_mode: str,
    block_size: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    graph_entries = graph_files_and_sizes(repo_root, district)
    required_sizes = {size for _, _, size in graph_entries}
    candidate_files, graph_source_files, graph_source_label = pems_meta_sources(
        repo_root, district
    )
    candidates_by_size, actual_catalog = load_snapshot_candidates(
        candidate_files, required_sizes
    )
    # Add any selected root snapshots to the display-coordinate catalog.
    merge_catalogs(
        actual_catalog,
        build_station_catalog(
            list((repo_root / "meta").glob(f"d{district}_text_meta_*.txt"))
        ),
    )
    graph_catalog = build_station_catalog(graph_source_files)

    recovered_years: list[RecoveredYear] = []
    print(
        f"[PEMS{district}] years={len(graph_entries)} candidates={len(candidate_files)} "
        f"graph_coords={graph_source_label}",
        flush=True,
    )
    first_dataset_year = graph_entries[0][0]
    for year, graph_path, station_count in graph_entries:
        candidates = candidates_by_size.get(station_count, [])
        if not candidates:
            raise RuntimeError(
                f"PEMS{district} {year}: no metadata snapshot has {station_count} unique IDs"
            )
        anchor = FIRST_DATES[district] if year == first_dataset_year else date(year, 1, 1)
        recovered = recover_one_year(
            year=year,
            graph_path=graph_path,
            candidates=candidates,
            graph_catalog=graph_catalog,
            anchor=anchor,
            validation_mode=validation_mode,
            block_size=block_size,
        )
        recovered_years.append(recovered)
        validation = recovered.canonical.validation
        print(
            f"  {year}: N={station_count} status={recovered.mapping_status} "
            f"valid_sets={len(recovered.valid_candidates)} topo={validation.topology_mismatches} "
            f"max_abs={validation.max_abs_error:.3g} "
            f"source={recovered.source_snapshot.name}",
            flush=True,
        )

    first_year: dict[int, int] = {}
    last_year: dict[int, int] = {}
    for recovered in recovered_years:
        for station_id in recovered.canonical.candidate.station_ids:
            first_year.setdefault(station_id, recovered.year)
            last_year[station_id] = recovered.year

    audit_rows = [
        write_pems_year(
            repo_root=repo_root,
            output_root=output_root,
            district=district,
            recovered=recovered,
            actual_catalog=actual_catalog,
            first_year=first_year,
            last_year=last_year,
            validation_mode=validation_mode,
            graph_source_label=graph_source_label,
        )
        for recovered in recovered_years
    ]
    write_pems_catalog(
        output_root=output_root,
        district=district,
        recovered_years=recovered_years,
        actual_catalog=actual_catalog,
        first_year=first_year,
        last_year=last_year,
    )

    unique_missing = sum(
        1
        for station_id in first_year
        if station_id not in actual_catalog
        or not valid_coordinate(
            actual_catalog[station_id].latitude, actual_catalog[station_id].longitude
        )
    )
    summary = {
        "dataset": f"pems{district}",
        "years": [item.year for item in recovered_years],
        "year_count": len(recovered_years),
        "unique_station_count": len(first_year),
        "unique_missing_coordinates": unique_missing,
        "ambiguous_years": [
            item.year for item in recovered_years if item.mapping_status != "exact"
        ],
        "candidate_metadata_source": relative_path(candidate_files[0].parent, repo_root),
        "graph_coordinate_source": graph_source_label,
        "validation_mode": validation_mode,
    }
    return audit_rows, summary


def read_tfnsw_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda item: int(item["idx"]))
    observed = [int(item["idx"]) for item in rows]
    if observed != list(range(len(rows))):
        raise ValueError(f"Non-contiguous TFNSW idx values in {path}")
    return rows


def build_tfnsw(
    repo_root: Path, output_root: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    meta_dir = repo_root / "tfnsw" / "meta"
    yearly: dict[int, list[dict[str, str]]] = {}
    for path in sorted(meta_dir.glob("*_nodes.csv")):
        year = int(path.stem.split("_")[0])
        rows = read_tfnsw_rows(path)
        graph_path = repo_root / "tfnsw" / "graph" / f"{year}_adj.npz"
        graph_shape = npz_array_shape(graph_path, "x")
        if graph_shape != (len(rows), len(rows)):
            raise ValueError(
                f"TFNSW {year}: metadata N={len(rows)} but graph shape={graph_shape}"
            )
        yearly[year] = rows
    if not yearly:
        raise FileNotFoundError(f"No TFNSW node metadata found in {meta_dir}")

    first_year: dict[str, int] = {}
    last_year: dict[str, int] = {}
    years_present: dict[str, list[int]] = {}
    latest_record: dict[str, dict[str, str]] = {}
    for year, rows in yearly.items():
        for row in rows:
            node = row["node"]
            first_year.setdefault(node, year)
            last_year[node] = year
            years_present.setdefault(node, []).append(year)
            latest_record[node] = row

    audit_rows: list[dict[str, object]] = []
    dataset_dir = output_root / "tfnsw"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for year, rows in yearly.items():
        year_dir = dataset_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        nodes = string_array([row["node"] for row in rows])
        station_ids = string_array([row["station_id"] for row in rows])
        directions = string_array([row["direction"] for row in rows])
        latitude = np.asarray([float(row["lat"]) for row in rows], dtype=np.float64)
        longitude = np.asarray([float(row["lon"]) for row in rows], dtype=np.float64)
        first = np.asarray([first_year[row["node"]] for row in rows], dtype=np.int16)
        last = np.asarray([last_year[row["node"]] for row in rows], dtype=np.int16)
        source = relative_path(meta_dir / f"{year}_nodes.csv", repo_root)
        np.savez_compressed(
            year_dir / "sensor_metadata.npz",
            dataset=np.asarray("tfnsw"),
            year=np.asarray(year, dtype=np.int16),
            sensor_index=np.arange(len(rows), dtype=np.int32),
            node_id=nodes,
            station_id=station_ids,
            direction=directions,
            latitude=latitude,
            longitude=longitude,
            coordinate_status=string_array(["metadata"] * len(rows)),
            first_year=first,
            last_year=last,
            id_status=string_array(["exact_source"] * len(rows)),
            station_id_alternatives=station_ids,
            mapping_status=np.asarray("exact_source"),
            source_snapshot=np.asarray(source),
            graph_validation_mode=np.asarray("direct_idx_and_shape"),
            graph_topology_mismatches=np.asarray(0, dtype=np.int64),
        )
        with (year_dir / "sensor_metadata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = [
                "sensor_index",
                "node_id",
                "station_id",
                "direction",
                "latitude",
                "longitude",
                "first_year",
                "last_year",
                "id_status",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows):
                writer.writerow(
                    {
                        "sensor_index": index,
                        "node_id": row["node"],
                        "station_id": row["station_id"],
                        "direction": row["direction"],
                        "latitude": row["lat"],
                        "longitude": row["lon"],
                        "first_year": first_year[row["node"]],
                        "last_year": last_year[row["node"]],
                        "id_status": "exact_source",
                    }
                )
        audit_rows.append(
            {
                "dataset": "tfnsw",
                "year": year,
                "sensor_count": len(rows),
                "mapping_status": "exact_source",
                "valid_candidate_sets": 1,
                "source_snapshot": source,
                "missing_coordinates": 0,
                "graph_fallback_coordinates": 0,
                "graph_validation_mode": "direct_idx_and_shape",
                "graph_sigma_km": "",
                "graph_topology_mismatches": 0,
                "graph_max_abs_error": "",
                "graph_mean_abs_error": "",
                "graph_compared_entries": len(rows),
            }
        )
        print(f"  [TFNSW] {year}: N={len(rows)} exact idx mapping", flush=True)

    nodes = sorted(first_year)
    records = [latest_record[node] for node in nodes]
    np.savez_compressed(
        dataset_dir / "sensor_catalog.npz",
        dataset=np.asarray("tfnsw"),
        node_id=string_array(nodes),
        station_id=string_array([row["station_id"] for row in records]),
        direction=string_array([row["direction"] for row in records]),
        latitude=np.asarray([float(row["lat"]) for row in records], dtype=np.float64),
        longitude=np.asarray([float(row["lon"]) for row in records], dtype=np.float64),
        coordinate_status=string_array(["metadata"] * len(nodes)),
        first_year=np.asarray([first_year[node] for node in nodes], dtype=np.int16),
        last_year=np.asarray([last_year[node] for node in nodes], dtype=np.int16),
        years_present=string_array(
            ["|".join(str(year) for year in years_present[node]) for node in nodes]
        ),
        id_status=string_array(["exact_source"] * len(nodes)),
    )
    with (dataset_dir / "sensor_catalog.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "node_id",
            "station_id",
            "direction",
            "latitude",
            "longitude",
            "first_year",
            "last_year",
            "years_present",
            "id_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for node, row in zip(nodes, records):
            writer.writerow(
                {
                    "node_id": node,
                    "station_id": row["station_id"],
                    "direction": row["direction"],
                    "latitude": row["lat"],
                    "longitude": row["lon"],
                    "first_year": first_year[node],
                    "last_year": last_year[node],
                    "years_present": "|".join(
                        str(year) for year in years_present[node]
                    ),
                    "id_status": "exact_source",
                }
            )

    summary = {
        "dataset": "tfnsw",
        "years": sorted(yearly),
        "year_count": len(yearly),
        "unique_station_count": len(nodes),
        "unique_missing_coordinates": 0,
        "ambiguous_years": [],
        "candidate_metadata_source": relative_path(meta_dir, repo_root),
        "graph_coordinate_source": "tfnsw/meta annual idx",
        "validation_mode": "direct_idx_and_shape",
    }
    return audit_rows, summary


def write_audit(output_root: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with (output_root / "mapping_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "visualization_exports"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    all_audit: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}

    for district in args.pems_districts:
        audit, summary = recover_pems_district(
            repo_root=repo_root,
            output_root=output_root,
            district=district,
            validation_mode=args.validation,
            block_size=args.block_size,
        )
        all_audit.extend(audit)
        summaries[summary["dataset"]] = summary

    if not args.skip_tfnsw:
        audit, summary = build_tfnsw(repo_root, output_root)
        all_audit.extend(audit)
        summaries[summary["dataset"]] = summary

    write_audit(output_root, all_audit)
    manifest = {
        "format_version": 1,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "pems_mapping_method": (
            "historical metadata station-ID set, ascending ID axis, validated against "
            "the saved thresholded-Gaussian adjacency matrix"
        ),
        "full_validation_tolerance": FULL_MAX_ABS_TOLERANCE,
        "datasets": summaries,
        "total_dataset_years": len(all_audit),
        "all_mappings_verified": all(
            row["mapping_status"]
            in {"exact", "exact_source", "co_located_ambiguous"}
            and int(row["graph_topology_mismatches"]) == 0
            for row in all_audit
        ),
    }
    with (output_root / "sensor_metadata_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"Done: {len(all_audit)} dataset-years -> {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
