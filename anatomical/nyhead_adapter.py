"""Import the released New York Head cortical operator without inventing depth sources."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .cortical_benchmark import (
    CorticalBenchmark,
    sha256_file,
    vertex_area_weights,
    write_cortical_benchmark,
)


NYHEAD_SHA256 = "904fc0baef1783b7bfba9fc73a7f0837265fa303d59c338d14518ed87176b30e"
Resolution = Literal["75K", "10K", "5K", "2K"]


def _integer_vector(array: np.ndarray, label: str, one_based: bool) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64).reshape(-1, order="F")
    if not np.all(np.isfinite(values)) or not np.allclose(
        values, np.rint(values), rtol=0.0, atol=0.0
    ):
        raise ValueError(f"{label} must contain finite integer indices")
    indices = np.rint(values).astype(np.int64)
    if one_based:
        indices -= 1
    return indices


def _triangle_matrix(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError(f"{label} must have on-disk shape (3, faces)")
    if not np.all(np.isfinite(values)) or not np.allclose(
        values, np.rint(values), rtol=0.0, atol=0.0
    ):
        raise ValueError(f"{label} must contain finite integer indices")
    return np.rint(values).astype(np.int64).T - 1


def _decode_matlab_cellstr(handle: object, path: str) -> list[str]:
    import h5py

    dataset = handle[path]
    references = np.asarray(dataset).reshape(-1, order="F")
    labels: list[str] = []
    for reference in references:
        if not isinstance(reference, h5py.Reference) or not reference:
            raise ValueError(f"{path} contains an invalid MATLAB object reference")
        codes = np.asarray(handle[reference]).reshape(-1, order="F")
        if not np.issubdtype(codes.dtype, np.integer):
            raise ValueError(f"{path} references a non-character dataset")
        label = "".join(chr(int(code)) for code in codes if int(code) != 0).strip()
        if not label:
            raise ValueError(f"{path} contains an empty electrode label")
        labels.append(label)
    if len(labels) != len(set(labels)):
        raise ValueError("New York Head electrode labels are not unique")
    return labels


def _resolution_geometry(
    handle: object, resolution: Resolution
) -> tuple[np.ndarray, np.ndarray]:
    if resolution == "75K":
        source_indices = np.arange(74382, dtype=np.int64)
        triangles = _triangle_matrix(
            np.asarray(handle["sa/cortex75K/tri"]), "cortex75K triangles"
        )
    else:
        group = f"sa/cortex{resolution}"
        source_indices = _integer_vector(
            np.asarray(handle[f"{group}/in_from_cortex75K"]),
            f"cortex{resolution} source map",
            True,
        )
        triangles = _triangle_matrix(
            np.asarray(handle[f"{group}/tri"]),
            f"cortex{resolution} triangles",
        )
    if (
        np.any(source_indices < 0)
        or np.any(source_indices >= 74382)
        or len(np.unique(source_indices)) != len(source_indices)
    ):
        raise ValueError(f"cortex{resolution} contains invalid 75K source indices")
    if triangles.size == 0 or int(triangles.min()) < 0 or int(triangles.max()) >= len(source_indices):
        raise ValueError(f"cortex{resolution} triangles do not index the local mesh")
    return source_indices, triangles


def _normal_projection_error(
    handle: object, source_indices: np.ndarray, sample_count: int = 64
) -> float:
    count = min(sample_count, len(source_indices))
    offsets = np.unique(np.linspace(0, len(source_indices) - 1, count, dtype=int))
    samples = np.sort(source_indices[offsets])
    vector = np.asarray(
        handle["sa/cortex75K/V_fem"][:, samples, :], dtype=np.float64
    )
    normals = np.asarray(
        handle["sa/cortex75K/normals"][:, samples], dtype=np.float64
    )
    released = np.asarray(
        handle["sa/cortex75K/V_fem_normal"][samples, :], dtype=np.float64
    )
    projected = np.einsum("dsn,ds->sn", vector, normals)
    scale = max(float(np.max(np.abs(released))), np.finfo(float).tiny)
    return float(np.max(np.abs(projected - released)) / scale)


def build_nyhead_cortical_benchmark(
    model_path: Path,
    resolution: Resolution = "10K",
    expected_sha256: str = NYHEAD_SHA256,
) -> CorticalBenchmark:
    """Read one exact cortical subset from the audited MATLAB v7.3 model."""

    import h5py

    model_path = Path(model_path).resolve(strict=True)
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"New York Head SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    if resolution not in ("75K", "10K", "5K", "2K"):
        raise ValueError("resolution must be one of 75K, 10K, 5K, or 2K")

    with h5py.File(model_path, "r") as handle:
        sensor_names = _decode_matlab_cellstr(handle, "sa/clab_electrodes")
        locations = np.asarray(handle["sa/locs_3D"], dtype=np.float64)
        if locations.shape != (6, 231):
            raise ValueError("sa/locs_3D must have shape (6, 231)")
        source_indices, triangles = _resolution_geometry(handle, resolution)
        base_positions = np.asarray(handle["sa/cortex75K/vc"], dtype=np.float64)
        base_normals = np.asarray(handle["sa/cortex75K/normals"], dtype=np.float64)
        positions_m = base_positions[:, source_indices].T * 1e-3
        directions = base_normals[:, source_indices].T
        all_normal_leadfields = np.asarray(
            handle["sa/cortex75K/V_fem_normal"], dtype=np.float64
        )
        leadfield = all_normal_leadfields[source_indices, :].T
        projection_error = _normal_projection_error(handle, source_indices)

    if projection_error > 1e-10:
        raise ValueError(
            "released normal lead field disagrees with vector-field projection; "
            f"relative error={projection_error:g}"
        )
    electrode_positions_m = locations[:3, :].T * 1e-3
    electrode_normals = locations[3:, :].T
    area_weights = vertex_area_weights(positions_m, triangles)
    manifest = {
        "schema_version": 1,
        "benchmark_kind": "cortical_only",
        "subject_id": "ICBM-NY",
        "coordinate_frame": "MNI",
        "coordinate_unit": "m",
        "leadfield_reference": "common_average",
        "leadfield_unit": "native_new_york_head_undocumented",
        "leadfield_physical_scale_status": "not_established",
        "amplitude_claims_allowed": False,
        "contains_hippocampal_operator": False,
        "sensor_names": sensor_names,
        "source_resolution": resolution,
        "source_orientation": "cortical_surface_normal",
        "source_index_base_in_artifact": 0,
        "source_index_parent": "sa/cortex75K",
        "coordinate_conversion": "released MNI millimetres multiplied by 1e-3",
        "normal_projection_sample_count": min(64, len(source_indices)),
        "normal_projection_relative_error": projection_error,
        "solver": {
            "name": "New York Head",
            "version": "ICBM-NY data description 2017-02-28",
            "method": "finite element method",
            "conductivity_model": "released ICBM-NY model; see source publication",
        },
        "source_publication_doi": "10.1016/j.neuroimage.2015.12.019",
        "provenance": (
            "Imported from the audited sa_nyhead.mat release. The source "
            "documentation states MNI coordinates and common-average reference, "
            "but does not establish the numerical lead-field unit; therefore this "
            "artifact permits geometry/subspace tests only."
        ),
        "input_sha256": {"sa_nyhead.mat": actual_sha256},
        "arrays_file": "arrays.npz",
        "arrays_sha256": "0" * 64,
    }
    return CorticalBenchmark(
        manifest=manifest,
        cortical_leadfield_native=leadfield,
        electrode_positions_m=electrode_positions_m,
        electrode_normals=electrode_normals,
        cortical_positions_m=positions_m,
        cortical_directions=directions,
        cortical_area_weights_m2=area_weights,
        cortical_triangles=triangles,
        source_indices_75k=source_indices,
    )


def import_nyhead_cortical_benchmark(
    model_path: Path,
    output_directory: Path,
    resolution: Resolution = "10K",
    expected_sha256: str = NYHEAD_SHA256,
) -> CorticalBenchmark:
    benchmark = build_nyhead_cortical_benchmark(
        model_path, resolution=resolution, expected_sha256=expected_sha256
    )
    return write_cortical_benchmark(output_directory, benchmark)
