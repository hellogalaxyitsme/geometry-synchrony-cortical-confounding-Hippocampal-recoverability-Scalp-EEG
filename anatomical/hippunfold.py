"""Strict HippUnfold surface ingestion for hippocampal EEG source models.

The primary source direction is the subject-specific inner-to-outer ribbon
vector between corresponding HippUnfold vertices.  This follows the intrinsic
laminar (IO) construction and is deliberately distinct from a segmentation
boundary normal.  Dentate-gyrus surfaces are audited but excluded from the
CA/subiculum source sheet; they must be modelled as a separate source family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

from anatomical.hcp_forward import _apply_affine, _load_gifti_surface, sha256_file
from anatomical.nested_source import nested_patch_hierarchy


HEMISPHERES = (("L", -1), ("R", 1))
SURFACES = ("inner", "midthickness", "outer")


def _entity_match(
    directory: Path,
    suffix: str,
    required_entities: Iterable[str],
) -> Path:
    """Return exactly one BIDS-like file containing all declared entities."""

    directory = Path(directory)
    candidates = []
    for path in directory.glob(f"*_{suffix}"):
        name = f"_{path.name}"
        if all(f"_{entity}_" in name for entity in required_entities):
            candidates.append(path)
    if len(candidates) != 1:
        rendered = [path.name for path in sorted(candidates)]
        raise FileNotFoundError(
            f"expected exactly one {suffix} with {list(required_entities)}, found {rendered}"
        )
    return candidates[0]


def _load_metric(path: Path, expected_vertices: int) -> np.ndarray:
    image = nib.load(str(path))
    arrays = [np.asarray(array.data) for array in image.darrays]
    vectors = [value.reshape(-1) for value in arrays if value.size == expected_vertices]
    if len(vectors) != 1:
        raise ValueError(f"{path} must contain exactly one {expected_vertices}-value metric")
    result = np.asarray(vectors[0], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{path} contains non-finite values")
    return result


def _vector_transform(linear: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    transformed = np.asarray(vectors, dtype=np.float64) @ np.asarray(
        linear, dtype=np.float64
    ).T
    norms = np.linalg.norm(transformed, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("vector transform produced a non-finite or zero direction")
    return transformed / norms[:, None]


def _vertex_geometry(
    vertices_mm: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return quadrature geometry without assigning mass to collapsed elements.

    A zero-area triangle has exactly zero surface measure and therefore makes no
    contribution to the surface integral.  Vertices incident only on such
    triangles are excluded from the source quadrature.  This is deliberately
    different from repairing or perturbing the official HippUnfold geometry.
    """

    triangles = np.asarray(vertices_mm, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    doubled_area = np.linalg.norm(cross, axis=1)
    if not np.all(np.isfinite(doubled_area)):
        raise ValueError("midthickness surface contains non-finite triangle geometry")
    positive_faces = doubled_area > 0.0
    if not np.any(positive_faces):
        raise ValueError("midthickness surface has no positive-area triangles")
    face_area = 0.5 * doubled_area
    vertex_area = np.zeros(len(vertices_mm), dtype=np.float64)
    vertex_normal_sum = np.zeros_like(vertices_mm, dtype=np.float64)
    positive_face_indices = np.flatnonzero(positive_faces)
    for corner in range(3):
        indices = faces[positive_face_indices, corner]
        np.add.at(vertex_area, indices, face_area[positive_faces] / 3.0)
        np.add.at(vertex_normal_sum, indices, cross[positive_faces])
    source_mask = vertex_area > 0.0
    if np.count_nonzero(source_mask) < 3:
        raise ValueError("midthickness surface has fewer than three positive-measure vertices")
    normal_norm = np.linalg.norm(vertex_normal_sum, axis=1)
    normal_mask = normal_norm > 0.0
    vertex_normals = np.zeros_like(vertex_normal_sum)
    vertex_normals[normal_mask] = (
        vertex_normal_sum[normal_mask] / normal_norm[normal_mask, None]
    )
    excluded = np.flatnonzero(~source_mask)
    qc = {
        "policy": (
            "zero-area faces contribute zero quadrature weight; vertices with no "
            "positive-area incident face are excluded without coordinate repair"
        ),
        "positive_area_triangles": int(np.count_nonzero(positive_faces)),
        "zero_area_triangles": int(np.count_nonzero(~positive_faces)),
        "zero_area_triangle_fraction": float(np.mean(~positive_faces)),
        "positive_measure_vertices": int(np.count_nonzero(source_mask)),
        "excluded_zero_measure_vertices": int(len(excluded)),
        "excluded_zero_measure_vertex_indices": excluded.tolist(),
        "vertices_with_defined_surface_normal": int(np.count_nonzero(normal_mask)),
    }
    return vertex_area, vertex_normals, face_area, source_mask, normal_mask, qc


def _coordinate_qc(values: np.ndarray, label: str) -> dict[str, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if minimum < -0.05 or maximum > 1.05 or span < 0.75:
        raise ValueError(
            f"{label} intrinsic coordinate violates the expected approximately [0, 1] range"
        )
    return {"minimum": minimum, "maximum": maximum, "span": span}


def _surface_files(
    hippunfold_root: Path,
    subject: str,
    hemisphere: str,
    density: str,
    label: str,
) -> dict[str, Path]:
    surface_directory = Path(hippunfold_root) / f"sub-{subject}" / "surf"
    common = (
        f"sub-{subject}",
        f"hemi-{hemisphere}",
        "space-T1w",
        f"den-{density}",
        f"label-{label}",
    )
    return {
        surface: _entity_match(
            surface_directory, f"{surface}.surf.gii", common
        )
        for surface in SURFACES
    }


def _metric_file(
    hippunfold_root: Path,
    subject: str,
    hemisphere: str,
    density: str,
    direction: str,
) -> Path:
    metric_directory = Path(hippunfold_root) / f"sub-{subject}" / "metric"
    return _entity_match(
        metric_directory,
        "coords.shape.gii",
        (
            f"sub-{subject}",
            f"hemi-{hemisphere}",
            f"den-{density}",
            "label-hipp",
            f"dir-{direction}",
            "desc-laplace",
        ),
    )


def _dentate_audit(
    hippunfold_root: Path, subject: str, density: str
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for hemisphere, _ in HEMISPHERES:
        paths = _surface_files(
            hippunfold_root, subject, hemisphere, density, "dentate"
        )
        loaded = {name: _load_gifti_surface(path) for name, path in paths.items()}
        vertices, faces = loaded["midthickness"]
        for name, (candidate_vertices, candidate_faces) in loaded.items():
            if len(candidate_vertices) != len(vertices) or not np.array_equal(
                candidate_faces, faces
            ):
                raise ValueError(
                    f"HippUnfold {hemisphere} dentate {name} does not share "
                    "midthickness topology"
                )
        ribbon_thickness = np.linalg.norm(
            loaded["outer"][0] - loaded["inner"][0], axis=1
        )
        if (
            np.any(~np.isfinite(ribbon_thickness))
            or np.any(ribbon_thickness <= 0.0)
        ):
            raise ValueError("dentate ribbon contains a non-finite or zero thickness")
        records.append(
            {
                "hemisphere": hemisphere,
                "vertices": int(len(vertices)),
                "triangles": int(len(faces)),
                "topology_correspondence_validated": True,
                "ribbon_thickness_mm": {
                    "minimum": float(np.min(ribbon_thickness)),
                    "median": float(np.median(ribbon_thickness)),
                    "maximum": float(np.max(ribbon_thickness)),
                },
                "surface_sha256": {
                    name: sha256_file(path) for name, path in paths.items()
                },
            }
        )
    return {
        "present": True,
        "included_in_primary_source_model": False,
        "policy": "separate topology; excluded from CA/subiculum forward operator",
        "hemispheres": records,
    }


def hippunfold_source_hierarchy(
    hippunfold_root: Path,
    subject: str,
    scanner_to_bem: np.ndarray,
    levels: list[dict[str, object]],
    *,
    density: str = "8k",
    normal_bin_width: float = 0.5,
    audit_dentate: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    """Build exactly nested CA/subiculum sources from HippUnfold surfaces.

    The input midthickness, inner, and outer surfaces must have corresponding
    vertices and identical topology.  Source positions are on midthickness;
    orientations point from inner to outer at the same intrinsic vertex.
    """

    transform = np.asarray(scanner_to_bem, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("scanner_to_bem must be a finite 4x4 transform")
    combined = {
        str(level["name"]): {
            "positions_m": [],
            "directions": [],
            "area_weights_m2": [],
            "longitudinal_coordinate": [],
            "proximal_distal_coordinate": [],
            "hemisphere_code": [],
        }
        for level in levels
    }
    hemisphere_reports: list[dict[str, object]] = []
    for hemisphere, hemisphere_code in HEMISPHERES:
        paths = _surface_files(
            hippunfold_root, subject, hemisphere, density, "hipp"
        )
        loaded = {name: _load_gifti_surface(path) for name, path in paths.items()}
        faces = loaded["midthickness"][1]
        vertex_count = len(loaded["midthickness"][0])
        for name, (vertices, candidate_faces) in loaded.items():
            if len(vertices) != vertex_count or not np.array_equal(candidate_faces, faces):
                raise ValueError(
                    f"HippUnfold {hemisphere} {name} does not share midthickness topology"
                )
        inner = loaded["inner"][0]
        middle = loaded["midthickness"][0]
        outer = loaded["outer"][0]
        ribbon = outer - inner
        thickness_mm = np.linalg.norm(ribbon, axis=1)
        if (
            not np.all(np.isfinite(thickness_mm))
            or np.any(thickness_mm <= 0.05)
            or np.any(thickness_mm >= 10.0)
        ):
            raise ValueError("HippUnfold ribbon thickness lies outside the declared safety range")
        laminar = ribbon / thickness_mm[:, None]
        middle_from_inner = middle - inner
        fraction = np.sum(middle_from_inner * ribbon, axis=1) / (thickness_mm**2)
        perpendicular_mm = np.linalg.norm(
            middle_from_inner - fraction[:, None] * ribbon, axis=1
        )
        (
            area_mm2,
            surface_normals,
            face_area,
            source_mask,
            normal_mask,
            mesh_measure_qc,
        ) = _vertex_geometry(middle, faces)
        normal_audit_mask = source_mask & normal_mask
        if not np.any(normal_audit_mask):
            raise ValueError("midthickness surface has no auditable surface normals")
        surface_cosine = np.sum(
            surface_normals[normal_audit_mask] * laminar[normal_audit_mask], axis=1
        )
        aligned_surface_cosine = np.abs(surface_cosine)
        ap_path = _metric_file(
            hippunfold_root, subject, hemisphere, density, "AP"
        )
        pd_path = _metric_file(
            hippunfold_root, subject, hemisphere, density, "PD"
        )
        anterior_posterior = _load_metric(ap_path, vertex_count)
        proximal_distal = _load_metric(pd_path, vertex_count)
        ap_qc = _coordinate_qc(anterior_posterior, "AP")
        pd_qc = _coordinate_qc(proximal_distal, "PD")

        registered_middle = _apply_affine(transform, middle)[source_mask]
        registered_laminar = _vector_transform(
            transform[:3, :3], laminar[source_mask]
        )
        patch_levels, patch_report = nested_patch_hierarchy(
            registered_middle,
            registered_laminar,
            area_mm2[source_mask],
            levels,
            normal_bin_width,
            {
                "longitudinal_coordinate": anterior_posterior[source_mask],
                "proximal_distal_coordinate": proximal_distal[source_mask],
            },
        )
        for level in levels:
            name = str(level["name"])
            patch = patch_levels[name]
            if float(np.min(patch["coherence"])) < 0.70:
                raise ValueError("HippUnfold laminar aggregation is orientation-incoherent")
            combined[name]["positions_m"].append(patch["positions_mm"] / 1000.0)
            combined[name]["directions"].append(patch["directions"])
            combined[name]["area_weights_m2"].append(
                patch["areas_mm2"] / 1_000_000.0
            )
            combined[name]["longitudinal_coordinate"].append(
                patch["longitudinal_coordinate"]
            )
            combined[name]["proximal_distal_coordinate"].append(
                patch["proximal_distal_coordinate"]
            )
            combined[name]["hemisphere_code"].append(
                np.full(len(patch["areas_mm2"]), hemisphere_code, dtype=np.int8)
            )
        patch_report.update(
            {
                "hemisphere": hemisphere,
                "density": density,
                "raw_vertices": int(vertex_count),
                "source_eligible_vertices": int(np.count_nonzero(source_mask)),
                "raw_triangles": int(len(faces)),
                "raw_area_mm2": float(np.sum(face_area)),
                "mesh_measure_qc": mesh_measure_qc,
                "thickness_mm": {
                    "minimum": float(np.min(thickness_mm)),
                    "p01": float(np.quantile(thickness_mm, 0.01)),
                    "median": float(np.median(thickness_mm)),
                    "p99": float(np.quantile(thickness_mm, 0.99)),
                    "maximum": float(np.max(thickness_mm)),
                },
                "midthickness_segment_projection_audit": {
                    "policy": (
                        "diagnostic only: the official HippUnfold midthickness is an "
                        "independently generated surface and need not lie on each "
                        "vertexwise inner-to-outer line segment"
                    ),
                    "minimum": float(np.min(fraction)),
                    "p001": float(np.quantile(fraction, 0.001)),
                    "median": float(np.median(fraction)),
                    "p999": float(np.quantile(fraction, 0.999)),
                    "maximum": float(np.max(fraction)),
                    "outside_0_1_vertices": int(
                        np.count_nonzero((fraction < 0.0) | (fraction > 1.0))
                    ),
                    "outside_minus01_11_vertices": int(
                        np.count_nonzero((fraction < -0.1) | (fraction > 1.1))
                    ),
                    "perpendicular_distance_mm": {
                        "minimum": float(np.min(perpendicular_mm)),
                        "median": float(np.median(perpendicular_mm)),
                        "maximum": float(np.max(perpendicular_mm)),
                    },
                },
                "surface_normal_vs_intrinsic_laminar_absolute_cosine": {
                    "p01": float(np.quantile(aligned_surface_cosine, 0.01)),
                    "median": float(np.median(aligned_surface_cosine)),
                    "p99": float(np.quantile(aligned_surface_cosine, 0.99)),
                },
                "surface_normal_vs_intrinsic_laminar_signed_cosine_median": float(
                    np.median(surface_cosine)
                ),
                "AP_coordinate": ap_qc,
                "PD_coordinate": pd_qc,
                "input_sha256": {
                    **{name: sha256_file(path) for name, path in paths.items()},
                    "AP": sha256_file(ap_path),
                    "PD": sha256_file(pd_path),
                },
            }
        )
        hemisphere_reports.append(patch_report)

    result: dict[str, dict[str, np.ndarray]] = {}
    for level in levels:
        name = str(level["name"])
        result[name] = {
            key: np.concatenate(values) for key, values in combined[name].items()
        }
    report: dict[str, object] = {
        "schema_version": 1,
        "source_family": "HippUnfold label-hipp CA/subiculum midthickness",
        "orientation_definition": (
            "normalized outer-minus-inner vector at corresponding intrinsic vertices"
        ),
        "orientation_is_segmentation_boundary_normal": False,
        "orientation_is_intrinsic_laminar_ribbon_vector": True,
        "midthickness_segment_projection_is_audit_only": True,
        "zero_measure_vertex_policy": (
            "exclude vertices with zero midthickness quadrature area; retain and "
            "report the official surface geometry without repair"
        ),
        "exact_nested": all(bool(row["exact_nested"]) for row in hemisphere_reports),
        "density": density,
        "hemispheres": hemisphere_reports,
    }
    if audit_dentate:
        report["dentate"] = _dentate_audit(hippunfold_root, subject, density)
    else:
        report["dentate"] = {
            "included_in_primary_source_model": False,
            "policy": "not requested; never merged into label-hipp source family",
        }
    return result, report
