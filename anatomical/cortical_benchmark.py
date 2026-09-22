"""Strict contract for cortical-only forward-model benchmark artifacts."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping
import zipfile

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ARRAY_NAMES = (
    "cortical_leadfield_native",
    "electrode_positions_m",
    "electrode_normals",
    "cortical_positions_m",
    "cortical_directions",
    "cortical_area_weights_m2",
    "cortical_triangles",
    "source_indices_75k",
)

REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "benchmark_kind",
    "subject_id",
    "coordinate_frame",
    "coordinate_unit",
    "leadfield_reference",
    "leadfield_unit",
    "leadfield_physical_scale_status",
    "amplitude_claims_allowed",
    "contains_hippocampal_operator",
    "sensor_names",
    "source_resolution",
    "solver",
    "provenance",
    "arrays_file",
    "arrays_sha256",
)


class CorticalBenchmarkValidationError(ValueError):
    """Raised when a cortical-only benchmark violates its contract."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("invalid cortical benchmark: " + "; ".join(self.errors))


@dataclass(frozen=True)
class CorticalBenchmark:
    manifest: dict[str, object]
    cortical_leadfield_native: FloatArray
    electrode_positions_m: FloatArray
    electrode_normals: FloatArray
    cortical_positions_m: FloatArray
    cortical_directions: FloatArray
    cortical_area_weights_m2: FloatArray
    cortical_triangles: IntArray
    source_indices_75k: IntArray
    directory: Path | None = None

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.manifest["sensor_names"])

    @property
    def number_of_sensors(self) -> int:
        return int(self.cortical_leadfield_native.shape[0])

    @property
    def number_of_sources(self) -> int:
        return int(self.cortical_leadfield_native.shape[1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def vertex_area_weights(
    positions_m: np.ndarray, triangles: np.ndarray
) -> FloatArray:
    """Return barycentric vertex areas for a triangular surface mesh."""

    positions = np.asarray(positions_m, dtype=np.float64)
    faces = np.asarray(triangles)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_m must have shape (vertices, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangles must have shape (faces, 3)")
    if not np.issubdtype(faces.dtype, np.integer):
        if not np.all(np.isfinite(faces)) or not np.allclose(
            faces, np.rint(faces), rtol=0.0, atol=0.0
        ):
            raise ValueError("triangle indices must be finite integers")
        faces = np.rint(faces).astype(np.int64)
    else:
        faces = faces.astype(np.int64, copy=False)
    if faces.size == 0 or int(faces.min()) < 0 or int(faces.max()) >= len(positions):
        raise ValueError("triangle indices are outside the vertex range")
    vertices = positions[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=1,
    )
    if not np.all(np.isfinite(face_areas)) or np.any(face_areas <= 0.0):
        raise ValueError("surface contains non-finite or degenerate triangles")
    weights = np.zeros(len(positions), dtype=np.float64)
    share = face_areas / 3.0
    for column in range(3):
        np.add.at(weights, faces[:, column], share)
    if np.any(weights <= 0.0):
        raise ValueError("surface contains vertices with zero incident area")
    return weights


def _array_mapping(benchmark: CorticalBenchmark) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(benchmark, name)) for name in ARRAY_NAMES}


def _deterministic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(
                filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


def _shape_error(
    errors: list[str], name: str, actual: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    if actual != expected:
        errors.append(f"{name} has shape {actual}, expected {expected}")


def validate_cortical_benchmark(
    benchmark: CorticalBenchmark, verify_checksum: bool = True
) -> dict[str, object]:
    """Validate geometry, referencing, provenance, and the no-claim boundary."""

    errors: list[str] = []
    manifest = benchmark.manifest
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            errors.append(f"manifest missing {field}")
    if errors:
        raise CorticalBenchmarkValidationError(errors)

    expected = {
        "schema_version": 1,
        "benchmark_kind": "cortical_only",
        "coordinate_frame": "MNI",
        "coordinate_unit": "m",
        "leadfield_reference": "common_average",
        "leadfield_unit": "native_new_york_head_undocumented",
        "leadfield_physical_scale_status": "not_established",
        "amplitude_claims_allowed": False,
        "contains_hippocampal_operator": False,
        "arrays_file": "arrays.npz",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            errors.append(f"{field} must be {value!r}")
    for field in ("subject_id", "source_resolution", "provenance"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            errors.append(f"{field} must be a nonempty string")
    solver = manifest.get("solver")
    if not isinstance(solver, dict):
        errors.append("solver must be an object")
    else:
        for field in ("name", "version", "method", "conductivity_model"):
            if not isinstance(solver.get(field), str) or not solver[field]:
                errors.append(f"solver.{field} must be a nonempty string")

    arrays = _array_mapping(benchmark)
    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number):
            errors.append(f"{name} must be numeric")
        elif not np.all(np.isfinite(array)):
            errors.append(f"{name} contains non-finite values")
    leadfield = np.asarray(benchmark.cortical_leadfield_native)
    if leadfield.ndim != 2:
        errors.append("cortical_leadfield_native must be two-dimensional")
    if errors:
        raise CorticalBenchmarkValidationError(errors)

    sensors, sources = leadfield.shape
    if sensors < 2 or sources < 3:
        errors.append("benchmark must have at least two sensors and three sources")
    _shape_error(
        errors, "electrode_positions_m", benchmark.electrode_positions_m.shape, (sensors, 3)
    )
    _shape_error(
        errors, "electrode_normals", benchmark.electrode_normals.shape, (sensors, 3)
    )
    _shape_error(
        errors, "cortical_positions_m", benchmark.cortical_positions_m.shape, (sources, 3)
    )
    _shape_error(
        errors, "cortical_directions", benchmark.cortical_directions.shape, (sources, 3)
    )
    _shape_error(
        errors,
        "cortical_area_weights_m2",
        benchmark.cortical_area_weights_m2.shape,
        (sources,),
    )
    _shape_error(
        errors, "source_indices_75k", benchmark.source_indices_75k.shape, (sources,)
    )
    if benchmark.cortical_triangles.ndim != 2 or benchmark.cortical_triangles.shape[1:] != (3,):
        errors.append("cortical_triangles must have shape (faces, 3)")

    names = manifest.get("sensor_names")
    if not isinstance(names, list) or len(names) != sensors:
        errors.append("sensor_names must match the sensor count")
    elif any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        errors.append("sensor_names must contain unique nonempty strings")

    for name, directions in (
        ("electrode_normals", benchmark.electrode_normals),
        ("cortical_directions", benchmark.cortical_directions),
    ):
        if directions.shape[1:] == (3,):
            norms = np.linalg.norm(directions, axis=1)
            if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-6):
                errors.append(f"{name} rows must have unit norm")

    source_indices = np.asarray(benchmark.source_indices_75k)
    if not np.issubdtype(source_indices.dtype, np.integer):
        errors.append("source_indices_75k must be integer-valued")
    elif (
        np.any(source_indices < 0)
        or np.any(source_indices >= 74382)
        or len(np.unique(source_indices)) != sources
    ):
        errors.append("source_indices_75k must be unique indices in [0, 74382)")

    triangles = np.asarray(benchmark.cortical_triangles)
    if triangles.ndim == 2 and triangles.shape[1:] == (3,):
        if not np.issubdtype(triangles.dtype, np.integer):
            errors.append("cortical_triangles must be integer-valued")
        elif triangles.size == 0 or int(triangles.min()) < 0 or int(triangles.max()) >= sources:
            errors.append("cortical_triangles contain out-of-range indices")
        else:
            try:
                recomputed_weights = vertex_area_weights(
                    benchmark.cortical_positions_m, triangles
                )
            except ValueError as error:
                errors.append(str(error))
            else:
                if not np.allclose(
                    benchmark.cortical_area_weights_m2,
                    recomputed_weights,
                    rtol=1e-10,
                    atol=1e-16,
                ):
                    errors.append("cortical area weights do not match mesh quadrature")

    weights = np.asarray(benchmark.cortical_area_weights_m2)
    if np.any(weights <= 0.0):
        errors.append("cortical_area_weights_m2 must be strictly positive")
    leadfield_scale = max(float(np.max(np.abs(leadfield))), np.finfo(float).tiny)
    car_residual = float(np.max(np.abs(np.mean(leadfield, axis=0))) / leadfield_scale)
    if car_residual > 1e-10:
        errors.append(
            f"lead field is not common-average referenced; relative residual={car_residual:g}"
        )

    digest = str(manifest.get("arrays_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        errors.append("arrays_sha256 must be a lowercase SHA-256 digest")
    if verify_checksum:
        if benchmark.directory is None:
            errors.append("cannot verify checksum without a benchmark directory")
        else:
            arrays_path = benchmark.directory / "arrays.npz"
            if not arrays_path.is_file():
                errors.append("arrays.npz is missing")
            elif sha256_file(arrays_path) != digest:
                errors.append("arrays SHA-256 mismatch")
    if errors:
        raise CorticalBenchmarkValidationError(errors)
    return {
        "valid": True,
        "subject_id": manifest["subject_id"],
        "source_resolution": manifest["source_resolution"],
        "number_of_sensors": sensors,
        "number_of_sources": sources,
        "number_of_triangles": int(triangles.shape[0]),
        "surface_area_m2": float(np.sum(weights)),
        "common_average_relative_residual": car_residual,
        "amplitude_claims_allowed": False,
        "contains_hippocampal_operator": False,
        "arrays_sha256": digest,
    }


def write_cortical_benchmark(
    directory: Path, benchmark: CorticalBenchmark
) -> CorticalBenchmark:
    """Write a deterministic artifact and reload it through the hard validator."""

    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"refusing nonempty benchmark directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    arrays_path = directory / "arrays.npz"
    _deterministic_savez(arrays_path, _array_mapping(benchmark))
    manifest = dict(benchmark.manifest)
    manifest["arrays_file"] = "arrays.npz"
    manifest["arrays_sha256"] = sha256_file(arrays_path)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return load_cortical_benchmark(directory)


def load_cortical_benchmark(
    directory: Path, verify_checksum: bool = True
) -> CorticalBenchmark:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    arrays_path = directory / "arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise CorticalBenchmarkValidationError(
            ["benchmark requires manifest.json and arrays.npz"]
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise CorticalBenchmarkValidationError(["manifest must contain an object"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        missing = [name for name in ARRAY_NAMES if name not in archive.files]
        if missing:
            raise CorticalBenchmarkValidationError(
                [f"arrays archive missing {name}" for name in missing]
            )
        arrays = {
            "cortical_leadfield_native": np.asarray(
                archive["cortical_leadfield_native"], dtype=np.float64
            ),
            "electrode_positions_m": np.asarray(
                archive["electrode_positions_m"], dtype=np.float64
            ),
            "electrode_normals": np.asarray(
                archive["electrode_normals"], dtype=np.float64
            ),
            "cortical_positions_m": np.asarray(
                archive["cortical_positions_m"], dtype=np.float64
            ),
            "cortical_directions": np.asarray(
                archive["cortical_directions"], dtype=np.float64
            ),
            "cortical_area_weights_m2": np.asarray(
                archive["cortical_area_weights_m2"], dtype=np.float64
            ),
            "cortical_triangles": np.asarray(
                archive["cortical_triangles"], dtype=np.int64
            ),
            "source_indices_75k": np.asarray(
                archive["source_indices_75k"], dtype=np.int64
            ),
        }
    benchmark = CorticalBenchmark(
        manifest=dict(manifest), directory=directory, **arrays
    )
    validate_cortical_benchmark(benchmark, verify_checksum=verify_checksum)
    return benchmark
