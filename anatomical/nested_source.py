"""Exactly nested HCP source hierarchies constructed from common raw elements."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from anatomical.hcp_forward import (
    HIPPOCAMPAL_LABELS,
    _aggregate_patches,
    _apply_affine,
    _boundary_faces,
    _load_gifti_surface,
    _normal_transform,
)


def _array_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(value, dtype="<i8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_levels(levels: list[dict[str, object]]) -> None:
    if [str(level["name"]) for level in levels] != ["coarse", "medium", "reference"]:
        raise ValueError("nested levels must be ordered coarse, medium, reference")
    bins = [float(level["spatial_bin_mm"]) for level in levels]
    if not (bins[0] > bins[1] > bins[2] > 0.0):
        raise ValueError("spatial bins must refine strictly")
    for parent, child in zip(bins[:-1], bins[1:]):
        ratio = parent / child
        if abs(ratio - round(ratio)) > 1e-12 or round(ratio) < 2:
            raise ValueError("each coarser bin must be an integer multiple of its child")


def nested_patch_hierarchy(
    positions_mm: np.ndarray,
    normals: np.ndarray,
    areas_mm2: np.ndarray,
    levels: list[dict[str, object]],
    normal_bin_width: float,
    scalar_fields: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    """Aggregate common raw elements and prove adjacent partitions are nested."""

    _validate_levels(levels)
    positions = np.asarray(positions_mm, dtype=np.float64)
    directions = np.asarray(normals, dtype=np.float64)
    areas = np.asarray(areas_mm2, dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape != directions.shape:
        raise ValueError("raw patch positions and normals must have shape (n, 3)")
    if len(positions) != len(areas) or len(areas) == 0:
        raise ValueError("raw patch arrays are empty or misaligned")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(directions)):
        raise ValueError("raw patch geometry must be finite")
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("raw patch areas must be finite and positive")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("raw patch normals must be nonzero")
    unit_normals = directions / norms[:, None]
    if normal_bin_width <= 0.0 or not np.isfinite(normal_bin_width):
        raise ValueError("normal bin width must be finite and positive")
    fields = {
        str(name): np.asarray(value, dtype=np.float64).reshape(-1)
        for name, value in (scalar_fields or {}).items()
    }
    for name, value in fields.items():
        if len(value) != len(areas) or not np.all(np.isfinite(value)):
            raise ValueError(f"scalar field {name} is non-finite or misaligned")

    normal_key = np.floor((unit_normals + 1.0) / normal_bin_width).astype(np.int64)
    output: dict[str, dict[str, np.ndarray]] = {}
    memberships: dict[str, np.ndarray] = {}
    level_reports: list[dict[str, object]] = []
    raw_area = float(areas.sum())
    for level in levels:
        name = str(level["name"])
        spatial_bin = float(level["spatial_bin_mm"])
        spatial_key = np.floor(positions / spatial_bin).astype(np.int64)
        keys = np.column_stack((spatial_key, normal_key))
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        aggregate_positions, aggregate_directions, aggregate_areas, coherence = (
            _aggregate_patches(
                positions, directions, areas, spatial_bin, normal_bin_width
            )
        )
        if len(unique_keys) != len(aggregate_areas):
            raise RuntimeError("aggregation membership and geometry disagree")
        current: dict[str, np.ndarray] = {
            "positions_mm": aggregate_positions,
            "directions": aggregate_directions,
            "areas_mm2": aggregate_areas,
            "coherence": coherence,
        }
        for field_name, field in fields.items():
            weighted = np.bincount(
                inverse, weights=areas * field, minlength=len(aggregate_areas)
            )
            current[field_name] = weighted / aggregate_areas
        output[name] = current
        memberships[name] = inverse.astype(np.int64, copy=False)
        level_reports.append(
            {
                "name": name,
                "spatial_bin_mm": spatial_bin,
                "groups": int(len(aggregate_areas)),
                "raw_elements": int(len(areas)),
                "minimum_orientation_coherence": float(coherence.min()),
                "area_relative_error": abs(float(aggregate_areas.sum()) / raw_area - 1.0),
                "raw_to_group_sha256": _array_sha256(inverse),
            }
        )

    relations: list[dict[str, object]] = []
    for parent_level, child_level in zip(levels[:-1], levels[1:]):
        parent_name = str(parent_level["name"])
        child_name = str(child_level["name"])
        parent = memberships[parent_name]
        child = memberships[child_name]
        pairs = np.unique(np.column_stack((child, parent)), axis=0)
        child_groups = len(output[child_name]["areas_mm2"])
        parent_counts = np.bincount(pairs[:, 0], minlength=child_groups)
        violations = int(np.count_nonzero(parent_counts != 1))
        if violations:
            raise ValueError(
                f"{violations} {child_name} groups do not have exactly one {parent_name} parent"
            )
        child_to_parent = np.empty(child_groups, dtype=np.int64)
        child_to_parent[pairs[:, 0]] = pairs[:, 1]
        ratio = float(parent_level["spatial_bin_mm"]) / float(
            child_level["spatial_bin_mm"]
        )
        relations.append(
            {
                "child": child_name,
                "parent": parent_name,
                "integer_spatial_ratio": int(round(ratio)),
                "child_groups": child_groups,
                "parent_groups": int(len(output[parent_name]["areas_mm2"])),
                "violating_child_groups": violations,
                "child_to_parent_sha256": _array_sha256(child_to_parent),
            }
        )
    return output, {
        "exact_nested": True,
        "raw_elements": int(len(areas)),
        "raw_area_mm2": raw_area,
        "levels": level_reports,
        "relations": relations,
    }


def hippocampal_source_hierarchy(
    segmentation_path: Path,
    scanner_to_bem: np.ndarray,
    levels: list[dict[str, object]],
    normal_bin_width: float = 0.5,
    smoothing_sigma_voxels: float = 1.0,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    """Build nested hippocampal boundary-normal surrogates from voxel faces."""

    _validate_levels(levels)
    image = nib.load(str(segmentation_path))
    affine = np.asarray(image.affine, dtype=np.float64)
    linear = affine[:3, :3]
    segmentation = np.asanyarray(image.dataobj)
    combined = {
        str(level["name"]): {
            "positions_m": [], "directions": [], "area_weights_m2": [],
            "longitudinal_coordinate": [], "longitudinal_cosine": [],
            "longitudinal_sine": [], "hemisphere_code": [],
        }
        for level in levels
    }
    hemisphere_reports: list[dict[str, object]] = []
    raw_areas: list[float] = []
    for hemisphere, label, hemisphere_code in HIPPOCAMPAL_LABELS:
        mask = segmentation == label
        if not np.any(mask):
            raise ValueError(f"hippocampal label {label} is absent")
        voxel_centers, fallback_normals = _boundary_faces(mask)
        smooth = gaussian_filter(
            mask.astype(np.float64), sigma=smoothing_sigma_voxels, mode="constant"
        )
        gradients = np.gradient(smooth)
        sampled_gradient = np.column_stack(
            [
                map_coordinates(
                    gradient, voxel_centers.T, order=1, mode="nearest", prefilter=False
                )
                for gradient in gradients
            ]
        )
        sampled_gradient *= -1.0
        weak = np.linalg.norm(sampled_gradient, axis=1) < 1e-10
        sampled_gradient[weak] = fallback_normals[weak]
        scanner_positions = _apply_affine(affine, voxel_centers)
        scanner_normals = _normal_transform(linear, sampled_gradient)
        positions = _apply_affine(scanner_to_bem, scanner_positions)
        normals = _normal_transform(np.asarray(scanner_to_bem)[:3, :3], scanner_normals)
        face_area_by_axis = []
        for axis in range(3):
            other = [value for value in range(3) if value != axis]
            face_area_by_axis.append(
                float(np.linalg.norm(np.cross(linear[:, other[0]], linear[:, other[1]])))
            )
        fallback_axis = np.argmax(np.abs(fallback_normals), axis=1)
        areas = np.asarray(face_area_by_axis)[fallback_axis]
        center = np.average(positions, axis=0, weights=areas)
        centered = positions - center
        covariance = (centered * areas[:, None]).T @ centered / areas.sum()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if long_axis[1] < 0.0:
            long_axis *= -1.0
        coordinate = centered @ long_axis
        span = float(coordinate.max() - coordinate.min())
        if span <= 0.0:
            raise ValueError("raw hippocampal longitudinal coordinate has zero span")
        coordinate = (coordinate - coordinate.min()) / span
        phase = 2.0 * np.pi * coordinate
        patch_levels, patch_report = nested_patch_hierarchy(
            positions,
            normals,
            areas,
            levels,
            normal_bin_width,
            {
                "longitudinal_coordinate": coordinate,
                "longitudinal_cosine": np.cos(phase),
                "longitudinal_sine": np.sin(phase),
            },
        )
        for level in levels:
            name = str(level["name"])
            patch = patch_levels[name]
            if float(patch["coherence"].min()) < 0.75:
                raise ValueError("hippocampal nested aggregation is orientation-incoherent")
            combined[name]["positions_m"].append(patch["positions_mm"] / 1000.0)
            combined[name]["directions"].append(patch["directions"])
            combined[name]["area_weights_m2"].append(patch["areas_mm2"] / 1_000_000.0)
            combined[name]["longitudinal_coordinate"].append(patch["longitudinal_coordinate"])
            combined[name]["longitudinal_cosine"].append(patch["longitudinal_cosine"])
            combined[name]["longitudinal_sine"].append(patch["longitudinal_sine"])
            combined[name]["hemisphere_code"].append(
                np.full(len(patch["areas_mm2"]), hemisphere_code, dtype=np.int8)
            )
        patch_report["hemisphere"] = hemisphere
        patch_report["label"] = label
        hemisphere_reports.append(patch_report)
        raw_areas.append(float(areas.sum()))
    result: dict[str, dict[str, np.ndarray]] = {}
    for level in levels:
        name = str(level["name"])
        result[name] = {
            key: np.concatenate(values)
            for key, values in combined[name].items()
        }
        result[name]["raw_surface_area_mm2"] = np.asarray(raw_areas)
    return result, {
        "exact_nested": all(report["exact_nested"] for report in hemisphere_reports),
        "longitudinal_basis": "area-weighted raw-face Fourier mode shared across all levels",
        "hemispheres": hemisphere_reports,
    }


def cortical_source_hierarchy(
    subject: str,
    input_root: Path,
    scanner_to_bem: np.ndarray,
    levels: list[dict[str, object]],
    normal_bin_width: float = 0.5,
    surface_name: str = "white",
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    """Build nested cortical fixed-normal surrogates from native triangles."""

    _validate_levels(levels)
    combined = {
        str(level["name"]): {
            "positions_m": [], "directions": [], "area_weights_m2": [],
            "hemisphere_code": [],
        }
        for level in levels
    }
    hemisphere_reports: list[dict[str, object]] = []
    native_areas: list[float] = []
    for hemisphere, hemisphere_code in (("L", -1), ("R", 1)):
        path = (
            input_root / subject / "T1w" / "Native"
            / f"{subject}.{hemisphere}.{surface_name}.native.surf.gii"
        )
        vertices, triangles = _load_gifti_surface(path)
        vertices = _apply_affine(scanner_to_bem, vertices)
        tri = vertices[triangles]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        doubled_area = np.linalg.norm(cross, axis=1)
        if np.any(doubled_area <= 0.0) or not np.all(np.isfinite(doubled_area)):
            raise ValueError(f"{path} contains degenerate triangles")
        normals = cross / doubled_area[:, None]
        areas = 0.5 * doubled_area
        centroids = tri.mean(axis=1)
        center = np.average(centroids, axis=0, weights=areas)
        if np.median(np.sum(normals * (centroids - center), axis=1)) < 0.0:
            normals *= -1.0
        patch_levels, patch_report = nested_patch_hierarchy(
            centroids, normals, areas, levels, normal_bin_width
        )
        for level in levels:
            name = str(level["name"])
            patch = patch_levels[name]
            if float(patch["coherence"].min()) < 0.70:
                raise ValueError("cortical nested aggregation is orientation-incoherent")
            combined[name]["positions_m"].append(patch["positions_mm"] / 1000.0)
            combined[name]["directions"].append(patch["directions"])
            combined[name]["area_weights_m2"].append(patch["areas_mm2"] / 1_000_000.0)
            combined[name]["hemisphere_code"].append(
                np.full(len(patch["areas_mm2"]), hemisphere_code, dtype=np.int8)
            )
        patch_report["hemisphere"] = hemisphere
        hemisphere_reports.append(patch_report)
        native_areas.append(float(areas.sum()))
    result: dict[str, dict[str, np.ndarray]] = {}
    for level in levels:
        name = str(level["name"])
        result[name] = {
            key: np.concatenate(values)
            for key, values in combined[name].items()
        }
        result[name]["native_surface_area_mm2"] = np.asarray(native_areas)
    return result, {
        "exact_nested": all(report["exact_nested"] for report in hemisphere_reports),
        "hemispheres": hemisphere_reports,
    }
