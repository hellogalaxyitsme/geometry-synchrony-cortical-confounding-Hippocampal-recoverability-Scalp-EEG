"""Subject-specific HCP source geometry, montage registration, and EEG forwards.

The hippocampal source sheet constructed here is deliberately an anatomical
boundary-normal *surrogate*.  HCP-YA Structural Recommended contains whole-
hippocampus labels, not histological layers or subfields, so this module never
labels the resulting normals as pyramidal-cell or laminar ground truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import mne
from mne.io.constants import FIFF
from mne.surface import _points_outside_surface
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


HIPPOCAMPAL_LABELS = (("left", 17, -1), ("right", 53, 1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (np.asarray(affine, dtype=np.float64) @ homogeneous.T).T[:, :3]


def _normal_transform(linear: np.ndarray, normals: np.ndarray) -> np.ndarray:
    transformed = np.asarray(normals, dtype=np.float64) @ np.linalg.inv(linear)
    norms = np.linalg.norm(transformed, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("normal transform produced a non-finite or zero vector")
    return transformed / norms[:, None]


def _load_gifti_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    point_intent = int(nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"])
    triangle_intent = int(nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"])
    points = [array for array in image.darrays if int(array.intent) == point_intent]
    triangles = [
        array for array in image.darrays if int(array.intent) == triangle_intent
    ]
    if len(points) != 1 or len(triangles) != 1:
        raise ValueError(f"{path} must have one pointset and one triangle array")
    vertices = np.asarray(points[0].data, dtype=np.float64)
    faces = np.asarray(triangles[0].data, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"invalid GIFTI points in {path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"invalid GIFTI triangles in {path}")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"non-finite GIFTI points in {path}")
    if faces.size == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"invalid GIFTI indices in {path}")
    return vertices, faces


def scanner_ras_to_bem_transform(brain_surface_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Derive scanner-RAS -> BEM surface-RAS from watershed metadata.

    ``mri_watershed -useSRAS`` writes vertices relative to the conformed
    volume's center RAS.  HCP native GIFTI and NIfTI coordinates are scanner
    RAS.  The required mapping is consequently a metadata-derived translation
    by ``-c_ras``.  No axis permutation is guessed from the separately stored
    non-conformed T1 header.
    """

    _, _, volume_info = mne.read_surface(
        brain_surface_path, read_metadata=True, verbose=False
    )
    if not volume_info or "cras" not in volume_info:
        raise ValueError("watershed brain surface has no valid center-RAS metadata")
    center_ras_mm = np.asarray(volume_info["cras"], dtype=np.float64)
    if center_ras_mm.shape != (3,) or not np.all(np.isfinite(center_ras_mm)):
        raise ValueError("watershed center-RAS metadata is invalid")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = -center_ras_mm
    return transform, center_ras_mm


def _aggregate_patches(
    positions_mm: np.ndarray,
    normals: np.ndarray,
    areas_mm2: np.ndarray,
    spatial_bin_mm: float,
    normal_bin_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if spatial_bin_mm <= 0.0 or normal_bin_width <= 0.0:
        raise ValueError("aggregation bins must be positive")
    positions_mm = np.asarray(positions_mm, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    areas_mm2 = np.asarray(areas_mm2, dtype=np.float64).reshape(-1)
    if positions_mm.shape != normals.shape or positions_mm.shape[1:] != (3,):
        raise ValueError("patch positions/normals have incompatible shapes")
    if len(positions_mm) != len(areas_mm2) or np.any(areas_mm2 <= 0.0):
        raise ValueError("patch areas must be positive and aligned")
    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(normal_norms <= 0.0) or not np.all(np.isfinite(normal_norms)):
        raise ValueError("patch normals must be finite and nonzero")
    normals = normals / normal_norms[:, None]
    spatial_key = np.floor(positions_mm / spatial_bin_mm).astype(np.int64)
    normal_key = np.floor((normals + 1.0) / normal_bin_width).astype(np.int64)
    keys = np.column_stack((spatial_key, normal_key))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    area = np.bincount(inverse, weights=areas_mm2, minlength=count)
    weighted_position = np.zeros((count, 3), dtype=np.float64)
    normal_sum = np.zeros((count, 3), dtype=np.float64)
    for axis in range(3):
        np.add.at(weighted_position[:, axis], inverse, areas_mm2 * positions_mm[:, axis])
        np.add.at(normal_sum[:, axis], inverse, areas_mm2 * normals[:, axis])
    positions = weighted_position / area[:, None]
    resultant = np.linalg.norm(normal_sum, axis=1)
    coherence = resultant / area
    if np.any(resultant <= 0.0) or not np.all(np.isfinite(coherence)):
        raise ValueError("patch aggregation cancelled one or more source directions")
    directions = normal_sum / resultant[:, None]
    return positions, directions, area, coherence


def _boundary_faces(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    voxel_normals: list[np.ndarray] = []
    for axis in range(3):
        for sign in (-1, 1):
            neighbor = np.zeros_like(mask, dtype=bool)
            destination = [slice(None)] * 3
            source = [slice(None)] * 3
            if sign > 0:
                destination[axis] = slice(0, -1)
                source[axis] = slice(1, None)
            else:
                destination[axis] = slice(1, None)
                source[axis] = slice(0, -1)
            neighbor[tuple(destination)] = mask[tuple(source)]
            indices = np.argwhere(mask & ~neighbor).astype(np.float64)
            indices[:, axis] += 0.5 * sign
            directions = np.zeros((len(indices), 3), dtype=np.float64)
            directions[:, axis] = sign
            centers.append(indices)
            voxel_normals.append(directions)
    return np.concatenate(centers), np.concatenate(voxel_normals)


def hippocampal_boundary_geometry(
    segmentation_path: Path,
    scanner_to_bem: np.ndarray,
    spatial_bin_mm: float = 2.0,
    normal_bin_width: float = 0.5,
    smoothing_sigma_voxels: float = 1.0,
) -> dict[str, np.ndarray]:
    image = nib.load(str(segmentation_path))
    affine = np.asarray(image.affine, dtype=np.float64)
    linear = affine[:3, :3]
    segmentation = np.asanyarray(image.dataobj)
    all_positions: list[np.ndarray] = []
    all_directions: list[np.ndarray] = []
    all_areas: list[np.ndarray] = []
    all_longitudinal: list[np.ndarray] = []
    all_hemisphere: list[np.ndarray] = []
    raw_surface_area: list[float] = []

    for _, label, hemisphere_code in HIPPOCAMPAL_LABELS:
        mask = segmentation == label
        if not np.any(mask):
            raise ValueError(f"hippocampal label {label} is absent")
        voxel_centers, fallback_voxel_normals = _boundary_faces(mask)
        smooth = gaussian_filter(
            mask.astype(np.float64), sigma=smoothing_sigma_voxels, mode="constant"
        )
        gradients = np.gradient(smooth)
        sampled_gradient = np.column_stack(
            [
                map_coordinates(
                    gradient,
                    voxel_centers.T,
                    order=1,
                    mode="nearest",
                    prefilter=False,
                )
                for gradient in gradients
            ]
        )
        # The mask is high inside; negative gradient points outward.
        sampled_gradient *= -1.0
        weak = np.linalg.norm(sampled_gradient, axis=1) < 1e-10
        sampled_gradient[weak] = fallback_voxel_normals[weak]
        scanner_positions = _apply_affine(affine, voxel_centers)
        scanner_normals = _normal_transform(linear, sampled_gradient)
        registered_positions = _apply_affine(scanner_to_bem, scanner_positions)
        registered_normals = _normal_transform(
            np.asarray(scanner_to_bem)[:3, :3], scanner_normals
        )

        # Exact area of each exposed voxel face under the NIfTI affine.
        face_areas_by_axis = []
        for axis in range(3):
            other = [value for value in range(3) if value != axis]
            face_areas_by_axis.append(
                float(np.linalg.norm(np.cross(linear[:, other[0]], linear[:, other[1]])))
            )
        fallback_axis = np.argmax(np.abs(fallback_voxel_normals), axis=1)
        face_areas = np.asarray(face_areas_by_axis)[fallback_axis]
        positions, directions, areas, coherence = _aggregate_patches(
            registered_positions,
            registered_normals,
            face_areas,
            spatial_bin_mm,
            normal_bin_width,
        )
        if float(coherence.min()) < 0.75:
            raise ValueError("hippocampal boundary aggregation is too orientation-incoherent")

        center = np.average(positions, axis=0, weights=areas)
        centered = positions - center
        covariance = (centered * areas[:, None]).T @ centered / areas.sum()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if long_axis[1] < 0.0:  # positive is anterior in scanner/surface RAS
            long_axis *= -1.0
        coordinate = centered @ long_axis
        span = float(coordinate.max() - coordinate.min())
        if span <= 0.0:
            raise ValueError("hippocampal longitudinal coordinate has zero span")
        coordinate = (coordinate - coordinate.min()) / span

        all_positions.append(positions)
        all_directions.append(directions)
        all_areas.append(areas)
        all_longitudinal.append(coordinate)
        all_hemisphere.append(np.full(len(positions), hemisphere_code, dtype=np.int8))
        raw_surface_area.append(float(face_areas.sum()))

    return {
        "positions_m": np.concatenate(all_positions) / 1000.0,
        "directions": np.concatenate(all_directions),
        "area_weights_m2": np.concatenate(all_areas) / 1_000_000.0,
        "longitudinal_coordinate": np.concatenate(all_longitudinal),
        "hemisphere_code": np.concatenate(all_hemisphere),
        "raw_surface_area_mm2": np.asarray(raw_surface_area),
    }


def cortical_surface_geometry(
    subject: str,
    input_root: Path,
    scanner_to_bem: np.ndarray,
    surface_name: str = "white",
    spatial_bin_mm: float = 8.0,
    normal_bin_width: float = 0.5,
) -> dict[str, np.ndarray]:
    all_positions: list[np.ndarray] = []
    all_directions: list[np.ndarray] = []
    all_areas: list[np.ndarray] = []
    all_hemisphere: list[np.ndarray] = []
    native_area: list[float] = []
    for hemisphere, hemisphere_code in (("L", -1), ("R", 1)):
        path = (
            input_root
            / subject
            / "T1w"
            / "Native"
            / f"{subject}.{hemisphere}.{surface_name}.native.surf.gii"
        )
        vertices, triangles = _load_gifti_surface(path)
        vertices = _apply_affine(scanner_to_bem, vertices)
        tri = vertices[triangles]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        doubled_area = np.linalg.norm(cross, axis=1)
        if np.any(doubled_area <= 0.0) or not np.all(np.isfinite(doubled_area)):
            raise ValueError(f"{path} contains degenerate geometric triangles")
        normals = cross / doubled_area[:, None]
        areas = 0.5 * doubled_area
        centroids = tri.mean(axis=1)
        hemisphere_center = np.average(centroids, axis=0, weights=areas)
        if np.median(np.sum(normals * (centroids - hemisphere_center), axis=1)) < 0.0:
            normals *= -1.0
        positions, directions, weights, coherence = _aggregate_patches(
            centroids, normals, areas, spatial_bin_mm, normal_bin_width
        )
        if float(coherence.min()) < 0.70:
            raise ValueError("cortical aggregation is too orientation-incoherent")
        all_positions.append(positions)
        all_directions.append(directions)
        all_areas.append(weights)
        all_hemisphere.append(np.full(len(positions), hemisphere_code, dtype=np.int8))
        native_area.append(float(areas.sum()))
    return {
        "positions_m": np.concatenate(all_positions) / 1000.0,
        "directions": np.concatenate(all_directions),
        "area_weights_m2": np.concatenate(all_areas) / 1_000_000.0,
        "hemisphere_code": np.concatenate(all_hemisphere),
        "native_surface_area_mm2": np.asarray(native_area),
    }


def cortical_surface_vertices(
    subject: str,
    input_root: Path,
    scanner_to_bem: np.ndarray,
    surface_name: str = "pial",
) -> dict[str, np.ndarray]:
    """Load every native cortical-surface vertex in BEM surface RAS."""

    all_positions: list[np.ndarray] = []
    all_hemisphere: list[np.ndarray] = []
    counts: list[int] = []
    for hemisphere, hemisphere_code in (("L", -1), ("R", 1)):
        path = (
            input_root
            / subject
            / "T1w"
            / "Native"
            / f"{subject}.{hemisphere}.{surface_name}.native.surf.gii"
        )
        vertices, _ = _load_gifti_surface(path)
        registered = _apply_affine(scanner_to_bem, vertices)
        all_positions.append(registered)
        all_hemisphere.append(
            np.full(len(registered), hemisphere_code, dtype=np.int8)
        )
        counts.append(len(registered))
    return {
        "positions_m": np.concatenate(all_positions) / 1000.0,
        "hemisphere_code": np.concatenate(all_hemisphere),
        "vertices_per_hemisphere": np.asarray(counts, dtype=np.int64),
    }


def _closed_surface_centroid(surface: dict[str, object]) -> np.ndarray:
    vertices = np.asarray(surface["rr"], dtype=np.float64)
    triangles = np.asarray(surface["tris"], dtype=np.int64)
    tri = vertices[triangles]
    signed_six_volume = np.einsum(
        "ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])
    )
    total = float(signed_six_volume.sum())
    if not np.isfinite(total) or abs(total) < 1e-12:
        raise ValueError("outer-skin mesh has zero signed volume")
    return np.sum(
        signed_six_volume[:, None] * tri.sum(axis=1) / 4.0, axis=0
    ) / total


def _fit_sphere_center(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    design = np.column_stack((2.0 * points, np.ones(len(points))))
    solution, _, rank, _ = np.linalg.lstsq(
        design, np.sum(points * points, axis=1), rcond=None
    )
    if rank != 4 or not np.all(np.isfinite(solution[:3])):
        raise ValueError("standard montage sphere fit is rank deficient")
    return solution[:3]


def _ray_surface_intersections(
    origin: np.ndarray, directions: np.ndarray, surface: dict[str, object]
) -> np.ndarray:
    vertices = np.asarray(surface["rr"], dtype=np.float64)
    triangles = np.asarray(surface["tris"], dtype=np.int64)
    tri = vertices[triangles]
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    relative_origin = np.asarray(origin, dtype=np.float64) - tri[:, 0]
    output = np.empty_like(directions, dtype=np.float64)
    epsilon = 1e-12
    for index, direction in enumerate(np.asarray(directions, dtype=np.float64)):
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.einsum("ij,ij->i", edge1, h)
        usable = np.abs(determinant) > epsilon
        inverse = np.zeros_like(determinant)
        inverse[usable] = 1.0 / determinant[usable]
        u = inverse * np.einsum("ij,ij->i", relative_origin, h)
        q = np.cross(relative_origin, edge1)
        v = inverse * (q @ direction)
        distance = inverse * np.einsum("ij,ij->i", edge2, q)
        valid = (
            usable
            & (u >= -epsilon)
            & (v >= -epsilon)
            & (u + v <= 1.0 + epsilon)
            & (distance > epsilon)
        )
        if not np.any(valid):
            raise ValueError(f"montage ray {index} did not intersect the scalp")
        output[index] = origin + float(distance[valid].min()) * direction
    return output


def register_standard_montage(
    outer_skin: dict[str, object], montage_name: str = "standard_1005"
) -> tuple[list[str], np.ndarray, dict[str, object]]:
    standard = mne.channels.make_standard_montage(montage_name, head_size=0.095)
    positions = standard.get_positions()["ch_pos"]
    catalogue_names = list(positions)
    names: list[str] = []
    standard_rows: list[np.ndarray] = []
    removed_aliases: dict[str, str] = {}
    for name in catalogue_names:
        point = np.asarray(positions[name], dtype=np.float64)
        duplicate = next(
            (
                kept_name
                for kept_name, kept_point in zip(names, standard_rows)
                if np.linalg.norm(point - kept_point) <= 1e-12
            ),
            None,
        )
        if duplicate is None:
            names.append(name)
            standard_rows.append(point)
        else:
            removed_aliases[name] = duplicate
    standard_points = np.asarray(standard_rows, dtype=np.float64)
    standard_center = _fit_sphere_center(standard_points)
    directions = standard_points - standard_center
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    subject_center = _closed_surface_centroid(outer_skin)
    if bool(_points_outside_surface(subject_center[None], outer_skin, n_jobs=1)[0]):
        raise ValueError("computed scalp ray origin is outside the outer-skin mesh")
    registered = _ray_surface_intersections(subject_center, directions, outer_skin)
    pairwise = np.linalg.norm(
        registered[:, None, :] - registered[None, :, :], axis=2
    )
    np.fill_diagonal(pairwise, np.inf)
    minimum_spacing = float(np.min(pairwise))
    if not np.all(np.isfinite(registered)) or minimum_spacing < 0.001:
        raise ValueError("registered montage has invalid or colliding electrodes")
    metadata = {
        "method": "standard_spherical_directions_ray_intersection",
        "montage": montage_name,
        "standard_head_radius_m": 0.095,
        "catalogue_channels": len(catalogue_names),
        "removed_collocated_aliases": removed_aliases,
        "standard_fitted_center_m": standard_center.tolist(),
        "subject_outer_skin_volume_centroid_m": subject_center.tolist(),
        "minimum_electrode_spacing_m": minimum_spacing,
        "empirical_digitization": False,
    }
    return names, registered, metadata


def _source_containment(
    positions_m: np.ndarray, inner_skull: dict[str, object], label: str
) -> dict[str, object]:
    outside = _points_outside_surface(positions_m, inner_skull, n_jobs=1)
    count = int(np.count_nonzero(outside))
    if count:
        raise ValueError(f"{count} {label} sources lie outside the inner skull")
    return {"sources": int(len(positions_m)), "outside_inner_skull": count}


def _make_info(names: list[str], positions_m: np.ndarray) -> mne.Info:
    montage = mne.channels.make_dig_montage(
        ch_pos={name: position for name, position in zip(names, positions_m)},
        coord_frame="head",
    )
    info = mne.create_info(names, sfreq=1000.0, ch_types="eeg", verbose=False)
    info.set_montage(montage, on_missing="raise", verbose=False)
    return info


def _compute_fixed_forward(
    info: mne.Info,
    positions_m: np.ndarray,
    directions: np.ndarray,
    bem_solution: object,
) -> mne.Forward:
    source_space = mne.setup_volume_source_space(
        pos={"rr": positions_m, "nn": directions},
        mri=None,
        sphere=None,
        bem=None,
        surface=None,
        verbose=False,
    )
    transform = mne.transforms.Transform("head", "mri", np.eye(4))
    forward = mne.make_forward_solution(
        info,
        transform,
        source_space,
        bem_solution,
        meg=False,
        eeg=True,
        mindist=0.0,
        n_jobs=1,
        verbose=False,
    )
    fixed = mne.convert_forward_solution(
        forward,
        surf_ori=True,
        force_fixed=True,
        use_cps=False,
        copy=False,
        verbose=False,
    )
    if fixed["source_ori"] != FIFF.FIFFV_MNE_FIXED_ORI:
        raise ValueError("MNE did not create a fixed-orientation forward")
    if fixed["nsource"] != len(positions_m):
        raise ValueError("MNE changed the declared source count")
    position_error = float(
        np.max(np.linalg.norm(fixed["source_rr"] - positions_m, axis=1))
    )
    direction_cosine = np.sum(fixed["source_nn"] * directions, axis=1)
    if position_error > 1e-9 or float(direction_cosine.min()) < 1.0 - 1e-9:
        raise ValueError("MNE forward source positions/orientations changed unexpectedly")
    leadfield = np.asarray(fixed["sol"]["data"], dtype=np.float64)
    if leadfield.shape != (len(info["ch_names"]), len(positions_m)):
        raise ValueError("fixed forward lead field has the wrong shape")
    if not np.all(np.isfinite(leadfield)):
        raise ValueError("fixed forward lead field contains non-finite values")
    return fixed


def compute_combined_fixed_leadfields(
    info: mne.Info,
    hippocampal_positions_m: np.ndarray,
    hippocampal_directions: np.ndarray,
    cortical_positions_m: np.ndarray,
    cortical_directions: np.ndarray,
    bem_solution: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both fixed source families in one MNE forward evaluation."""

    hippocampal_positions_m = np.asarray(hippocampal_positions_m, dtype=np.float64)
    hippocampal_directions = np.asarray(hippocampal_directions, dtype=np.float64)
    cortical_positions_m = np.asarray(cortical_positions_m, dtype=np.float64)
    cortical_directions = np.asarray(cortical_directions, dtype=np.float64)
    positions = np.vstack((hippocampal_positions_m, cortical_positions_m))
    directions = np.vstack((hippocampal_directions, cortical_directions))
    forward = _compute_fixed_forward(info, positions, directions, bem_solution)
    leadfield = np.asarray(forward["sol"]["data"], dtype=np.float64)
    split = len(hippocampal_positions_m)
    return leadfield[:, :split].copy(), leadfield[:, split:].copy()


def _referenced_rank(
    leadfield: np.ndarray, relative_tolerance: float = 1e-10
) -> tuple[int, np.ndarray]:
    centered = leadfield - leadfield.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    tolerance = relative_tolerance * singular[0]
    return int(np.count_nonzero(singular > tolerance)), singular


def _validate_forward_roundtrip(
    path: Path,
    expected_leadfield: np.ndarray,
    expected_positions: np.ndarray,
    expected_directions: np.ndarray,
) -> dict[str, float]:
    restored = mne.read_forward_solution(path, verbose=False)
    restored = mne.convert_forward_solution(
        restored,
        surf_ori=True,
        force_fixed=True,
        use_cps=False,
        copy=False,
        verbose=False,
    )
    actual = np.asarray(restored["sol"]["data"], dtype=np.float64)
    scale = max(float(np.max(np.abs(expected_leadfield))), np.finfo(float).tiny)
    relative_error = float(np.max(np.abs(actual - expected_leadfield)) / scale)
    position_error = float(
        np.max(np.linalg.norm(restored["source_rr"] - expected_positions, axis=1))
    )
    direction_cosine = np.sum(restored["source_nn"] * expected_directions, axis=1)
    minimum_cosine = float(direction_cosine.min())
    if relative_error > 1e-6 or position_error > 1e-6 or minimum_cosine < 1.0 - 1e-6:
        raise ValueError(
            "serialized forward did not reproduce the declared fixed projection"
        )
    return {
        "maximum_relative_leadfield_error": relative_error,
        "maximum_position_error_m": position_error,
        "minimum_direction_cosine": minimum_cosine,
    }


def build_hcp_forward_smoke(
    subject: str,
    input_root: Path,
    bem_root: Path,
    output_directory: Path,
    config: dict[str, object],
) -> dict[str, object]:
    subject_input = input_root / subject
    subject_bem = bem_root / subject
    segmentation_path = subject_input / "T1w" / "aparc+aseg.nii.gz"
    brain_surface_path = subject_bem / "bem" / "brain.surf"
    bem_surfaces_path = subject_bem / "bem" / f"{subject}-ico4-bem.fif"
    bem_solution_path = subject_bem / "bem" / f"{subject}-ico4-bem-sol.fif"
    surface_name = str(config["cortical_surface"])
    gifti_paths = [
        subject_input
        / "T1w"
        / "Native"
        / f"{subject}.{hemisphere}.{surface_name}.native.surf.gii"
        for hemisphere in ("L", "R")
    ]
    required = [
        segmentation_path,
        brain_surface_path,
        bem_surfaces_path,
        bem_solution_path,
        *gifti_paths,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing HCP forward inputs: {missing}")
    output_directory.mkdir(parents=True, exist_ok=False)

    scanner_to_bem, center_ras_mm = scanner_ras_to_bem_transform(
        brain_surface_path
    )
    hippocampal = hippocampal_boundary_geometry(
        segmentation_path,
        scanner_to_bem,
        spatial_bin_mm=float(config["hippocampal_spatial_bin_mm"]),
        normal_bin_width=float(config["normal_bin_width"]),
        smoothing_sigma_voxels=float(config["hippocampal_smoothing_sigma_voxels"]),
    )
    cortical = cortical_surface_geometry(
        subject,
        input_root,
        scanner_to_bem,
        surface_name=surface_name,
        spatial_bin_mm=float(config["cortical_spatial_bin_mm"]),
        normal_bin_width=float(config["normal_bin_width"]),
    )

    surfaces = mne.read_bem_surfaces(bem_surfaces_path, verbose=False)
    by_id = {int(surface["id"]): surface for surface in surfaces}
    inner_skull = by_id[int(FIFF.FIFFV_BEM_SURF_ID_BRAIN)]
    outer_skin = by_id[int(FIFF.FIFFV_BEM_SURF_ID_HEAD)]
    containment = {
        "hippocampal": _source_containment(
            hippocampal["positions_m"], inner_skull, "hippocampal"
        ),
        "cortical": _source_containment(
            cortical["positions_m"], inner_skull, "cortical"
        ),
    }
    sensor_names, electrode_positions_m, montage_metadata = register_standard_montage(
        outer_skin, str(config["montage"])
    )
    info = _make_info(sensor_names, electrode_positions_m)
    bem_solution = mne.read_bem_solution(bem_solution_path, verbose=False)
    hippocampal_forward = _compute_fixed_forward(
        info,
        hippocampal["positions_m"],
        hippocampal["directions"],
        bem_solution,
    )
    cortical_forward = _compute_fixed_forward(
        info,
        cortical["positions_m"],
        cortical["directions"],
        bem_solution,
    )

    hippocampal_forward_path = output_directory / "hippocampal-fwd.fif"
    cortical_forward_path = output_directory / "cortical-fwd.fif"
    hippocampal_metadata_path = output_directory / "hippocampal-source-metadata.npz"
    cortical_metadata_path = output_directory / "cortical-source-metadata.npz"
    montage_path = output_directory / "registered-montage.tsv"
    covariance_path = output_directory / "engineering-noise-cov.fif"
    hippocampal_matrix_path = output_directory / "hippocampal-fixed-leadfield.npy"
    cortical_matrix_path = output_directory / "cortical-fixed-leadfield.npy"
    mne.write_forward_solution(
        hippocampal_forward_path, hippocampal_forward, overwrite=True, verbose=False
    )
    mne.write_forward_solution(
        cortical_forward_path, cortical_forward, overwrite=True, verbose=False
    )
    h_leadfield = np.asarray(hippocampal_forward["sol"]["data"], dtype=np.float64)
    c_leadfield = np.asarray(cortical_forward["sol"]["data"], dtype=np.float64)
    np.save(hippocampal_matrix_path, h_leadfield, allow_pickle=False)
    np.save(cortical_matrix_path, c_leadfield, allow_pickle=False)
    roundtrip = {
        "hippocampal": _validate_forward_roundtrip(
            hippocampal_forward_path,
            h_leadfield,
            hippocampal["positions_m"],
            hippocampal["directions"],
        ),
        "cortical": _validate_forward_roundtrip(
            cortical_forward_path,
            c_leadfield,
            cortical["positions_m"],
            cortical["directions"],
        ),
    }
    np.savez_compressed(
        hippocampal_metadata_path,
        positions_m=hippocampal["positions_m"],
        directions=hippocampal["directions"],
        area_weights_m2=hippocampal["area_weights_m2"],
        longitudinal_coordinate=hippocampal["longitudinal_coordinate"],
        hemisphere_code=hippocampal["hemisphere_code"],
    )
    np.savez_compressed(
        cortical_metadata_path,
        positions_m=cortical["positions_m"],
        directions=cortical["directions"],
        area_weights_m2=cortical["area_weights_m2"],
        hemisphere_code=cortical["hemisphere_code"],
    )
    with montage_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("name\tx_m\ty_m\tz_m\n")
        for name, position in zip(sensor_names, electrode_positions_m):
            stream.write(f"{name}\t{position[0]:.12g}\t{position[1]:.12g}\t{position[2]:.12g}\n")
    variance = float(config["engineering_noise_rms_v"]) ** 2
    covariance = mne.Covariance(
        data=np.full(len(sensor_names), variance),
        names=sensor_names,
        bads=[],
        projs=[],
        nfree=1,
    )
    mne.write_cov(covariance_path, covariance, overwrite=True, verbose=False)

    h_rank, h_singular = _referenced_rank(h_leadfield)
    c_rank, c_singular = _referenced_rank(c_leadfield)
    expected_sensor_rank = len(sensor_names) - 1
    if c_rank != expected_sensor_rank:
        raise ValueError(
            f"cortical forward referenced rank {c_rank} != {expected_sensor_rank}"
        )

    output_paths = [
        hippocampal_forward_path,
        cortical_forward_path,
        hippocampal_metadata_path,
        cortical_metadata_path,
        montage_path,
        covariance_path,
        hippocampal_matrix_path,
        cortical_matrix_path,
    ]
    report = {
        "schema_version": 1,
        "ok": True,
        "protocol": str(config["protocol"]),
        "subject": subject,
        "python": platform.python_version(),
        "mne": mne.__version__,
        "nibabel": nib.__version__,
        "numpy": np.__version__,
        "coordinate_frame": "mne-head-identical-to-bem-surface-ras",
        "scanner_ras_to_bem_surface_ras": scanner_to_bem.tolist(),
        "watershed_center_ras_mm": center_ras_mm.tolist(),
        "source_model": {
            "hippocampal": "whole-label boundary-normal surrogate",
            "hippocampal_histological_laminar_ground_truth": False,
            "hippocampal_labels": {"left": 17, "right": 53},
            "hippocampal_sources": int(len(hippocampal["positions_m"])),
            "hippocampal_area_m2": float(hippocampal["area_weights_m2"].sum()),
            "cortical_surface": surface_name,
            "cortical_sources": int(len(cortical["positions_m"])),
            "cortical_area_m2": float(cortical["area_weights_m2"].sum()),
            "aggregation": {
                "hippocampal_spatial_bin_mm": float(
                    config["hippocampal_spatial_bin_mm"]
                ),
                "cortical_spatial_bin_mm": float(config["cortical_spatial_bin_mm"]),
                "normal_bin_width": float(config["normal_bin_width"]),
            },
        },
        "containment": containment,
        "montage": {**montage_metadata, "sensors": len(sensor_names)},
        "forward": {
            "hippocampal_shape": list(h_leadfield.shape),
            "cortical_shape": list(c_leadfield.shape),
            "hippocampal_referenced_rank": h_rank,
            "cortical_referenced_rank": c_rank,
            "expected_sensor_rank": expected_sensor_rank,
            "hippocampal_largest_singular_value": float(h_singular[0]),
            "cortical_largest_singular_value": float(c_singular[0]),
            "fixed_orientation": True,
            "fixed_projection_roundtrip": roundtrip,
            "referenced_rank_relative_tolerance": 1e-10,
            "cortical_null_to_largest_singular_ratio": float(
                c_singular[-1] / c_singular[0]
            ),
            "leadfield_unit": "V/(A m)",
            "reference": "common_gauge; Helmert contrasts applied downstream",
        },
        "engineering_noise_covariance": {
            "kind": "diagonal placeholder for pipeline validation only",
            "rms_v": float(config["engineering_noise_rms_v"]),
            "physiologically_calibrated": False,
        },
        "input_sha256": {str(path): sha256_file(path) for path in required},
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_paths
        },
    }
    report_path = output_directory / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
