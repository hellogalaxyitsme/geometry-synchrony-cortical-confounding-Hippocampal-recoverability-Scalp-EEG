"""Deterministic source-ensemble source ensembles for HippUnfold lead fields.

This module separates cancellation within a declared support from the loss of
integrated source moment caused by activating a smaller area.  All stochastic
objects use explicit, hash-derived seeds and all low-rank recoverability
statistics use the same standardized cortical nuisance convention as analysis
1.  No function in this module performs physiological amplitude calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from theory.recoverability import helmert_reference


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SourceSupport:
    """One deterministic anatomical support."""

    identifier: str
    hemisphere: str
    location: str
    target_extent: float
    indices: IntArray
    conditional_weights: FloatArray
    full_sheet_area_fraction: float
    realized_extent_within_selected_hemispheres: float
    realized_extent_by_hemisphere: Mapping[str, float]
    maximum_selected_hemisphere_patch_fraction: float


def _finite_vector(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def _weighted_mean(values: FloatArray, weights: FloatArray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weighted mean has non-positive total weight")
    return float(np.sum(values * weights) / total)


def _weighted_quantile(
    values: FloatArray, weights: FloatArray, probability: float
) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("weighted quantile probability must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = probability * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(order) - 1)
    return float(sorted_values[index])


def oriented_intrinsic_coordinates(
    positions_m: ArrayLike,
    longitudinal_coordinate: ArrayLike,
    proximal_distal_coordinate: ArrayLike,
    hemisphere_code: ArrayLike,
    area_weights_m2: ArrayLike,
    *,
    endpoint_decile: float,
    minimum_endpoint_separation_m: float,
) -> tuple[FloatArray, FloatArray, dict[str, object]]:
    """Orient HippUnfold AP so larger values are physically anterior.

    Positive head/surface-RAS ``y`` is anterior.  Polarity is inferred from
    area-weighted endpoint deciles independently in each hemisphere.
    """

    positions = np.asarray(positions_m, dtype=np.float64)
    ap = _finite_vector(longitudinal_coordinate, "longitudinal_coordinate")
    pd = _finite_vector(proximal_distal_coordinate, "proximal_distal_coordinate")
    hemispheres = np.asarray(hemisphere_code, dtype=np.int8).reshape(-1)
    areas = _finite_vector(area_weights_m2, "area_weights_m2")
    count = len(ap)
    if (
        positions.shape != (count, 3)
        or len(pd) != count
        or len(hemispheres) != count
        or len(areas) != count
        or np.any(areas <= 0.0)
        or set(np.unique(hemispheres)) != {-1, 1}
    ):
        raise ValueError("HippUnfold coordinate arrays are misaligned")
    if not 0.0 < endpoint_decile < 0.5:
        raise ValueError("endpoint_decile must lie in (0, 0.5)")
    if minimum_endpoint_separation_m <= 0.0:
        raise ValueError("minimum endpoint separation must be positive")

    anterior = np.empty(count, dtype=np.float64)
    pd_normalized = np.empty(count, dtype=np.float64)
    records: list[dict[str, object]] = []
    for code, label in ((-1, "left"), (1, "right")):
        indices = np.flatnonzero(hemispheres == code)
        a = areas[indices]
        raw_ap = ap[indices]
        raw_pd = pd[indices]
        ap_span = float(np.max(raw_ap) - np.min(raw_ap))
        pd_span = float(np.max(raw_pd) - np.min(raw_pd))
        if ap_span < 0.5 or pd_span < 0.5:
            raise ValueError(f"{label} intrinsic coordinate span is too small")
        ap01 = (raw_ap - float(np.min(raw_ap))) / ap_span
        pd01 = (raw_pd - float(np.min(raw_pd))) / pd_span
        low_cut = _weighted_quantile(ap01, a, endpoint_decile)
        high_cut = _weighted_quantile(ap01, a, 1.0 - endpoint_decile)
        low = ap01 <= low_cut
        high = ap01 >= high_cut
        if not np.any(low) or not np.any(high):
            raise ValueError(f"{label} AP endpoint audit has an empty endpoint")
        low_y = _weighted_mean(positions[indices[low], 1], a[low])
        high_y = _weighted_mean(positions[indices[high], 1], a[high])
        separation = high_y - low_y
        if abs(separation) < minimum_endpoint_separation_m:
            raise ValueError(
                f"{label} AP endpoints are separated by only {abs(separation):g} m"
            )
        high_is_anterior = separation > 0.0
        anterior[indices] = ap01 if high_is_anterior else 1.0 - ap01
        pd_normalized[indices] = pd01
        records.append(
            {
                "hemisphere": label,
                "raw_high_coordinate_is_anterior": bool(high_is_anterior),
                "low_endpoint_y_m": low_y,
                "high_endpoint_y_m": high_y,
                "endpoint_separation_m": abs(separation),
                "raw_AP_minimum": float(np.min(raw_ap)),
                "raw_AP_maximum": float(np.max(raw_ap)),
                "raw_PD_minimum": float(np.min(raw_pd)),
                "raw_PD_maximum": float(np.max(raw_pd)),
            }
        )
    return anterior, pd_normalized, {
        "definition": "HippUnfold AP polarity oriented toward positive head-RAS y",
        "endpoint_decile": endpoint_decile,
        "minimum_endpoint_separation_m": minimum_endpoint_separation_m,
        "hemispheres": records,
    }


def _prefix_for_extent(
    candidates: IntArray,
    order_key: tuple[FloatArray, FloatArray, IntArray],
    areas: FloatArray,
    target: float,
) -> IntArray:
    primary, secondary, original = order_key
    order = np.lexsort((original, secondary, primary))
    ordered = candidates[order]
    cumulative = np.cumsum(areas[ordered])
    threshold = target * float(cumulative[-1])
    stop = min(int(np.searchsorted(cumulative, threshold, side="left")) + 1, len(ordered))
    return np.sort(ordered[:stop])


def support_identifier(hemisphere: str, location: str, extent: float) -> str:
    return f"{hemisphere}__{location}__p{int(round(1000.0 * extent)):04d}"


def build_source_supports(
    anterior_coordinate: ArrayLike,
    proximal_distal_coordinate: ArrayLike,
    hemisphere_code: ArrayLike,
    area_weights_m2: ArrayLike,
    *,
    hemisphere_families: Iterable[str],
    focal_locations: Iterable[str],
    focal_extents: Iterable[float],
    include_whole_extent: bool,
) -> tuple[dict[str, SourceSupport], dict[str, object]]:
    """Create exact nested area-based support ladders."""

    anterior = _finite_vector(anterior_coordinate, "anterior_coordinate")
    pd = _finite_vector(proximal_distal_coordinate, "proximal_distal_coordinate")
    hemispheres = np.asarray(hemisphere_code, dtype=np.int8).reshape(-1)
    areas = _finite_vector(area_weights_m2, "area_weights_m2")
    if not (len(anterior) == len(pd) == len(hemispheres) == len(areas)):
        raise ValueError("support arrays differ in length")
    if np.any(areas <= 0.0) or np.any((anterior < 0.0) | (anterior > 1.0)):
        raise ValueError("invalid support geometry")
    families = tuple(hemisphere_families)
    locations = tuple(focal_locations)
    extents = tuple(float(value) for value in focal_extents)
    if families != ("left", "right", "bilateral"):
        raise ValueError("hemisphere families must be left, right, bilateral")
    if locations != ("anterior", "middle", "posterior"):
        raise ValueError("focal locations must be anterior, middle, posterior")
    if any(not 0.0 < value < 1.0 for value in extents) or tuple(sorted(extents)) != extents:
        raise ValueError("focal extents must be increasing values in (0, 1)")

    total_area = float(np.sum(areas))
    family_codes = {"left": (-1,), "right": (1,), "bilateral": (-1, 1)}
    supports: dict[str, SourceSupport] = {}
    nesting_checks: list[dict[str, object]] = []
    for family in families:
        codes = family_codes[family]
        family_indices = np.flatnonzero(np.isin(hemispheres, codes))
        family_area = float(np.sum(areas[family_indices]))
        if include_whole_extent:
            identifier = support_identifier(family, "whole", 1.0)
            weights = areas[family_indices] / family_area
            by_hemi = {
                "left" if code == -1 else "right": 1.0 for code in codes
            }
            maximum_patch = max(
                float(np.max(areas[hemispheres == code]) / np.sum(areas[hemispheres == code]))
                for code in codes
            )
            supports[identifier] = SourceSupport(
                identifier=identifier,
                hemisphere=family,
                location="whole",
                target_extent=1.0,
                indices=family_indices,
                conditional_weights=weights,
                full_sheet_area_fraction=family_area / total_area,
                realized_extent_within_selected_hemispheres=1.0,
                realized_extent_by_hemisphere=by_hemi,
                maximum_selected_hemisphere_patch_fraction=maximum_patch,
            )

        for location in locations:
            previous: set[int] = set()
            for extent in extents:
                selected_parts: list[IntArray] = []
                by_hemi: dict[str, float] = {}
                maximum_patch = 0.0
                for code in codes:
                    candidate = np.flatnonzero(hemispheres == code)
                    hemi_area = float(np.sum(areas[candidate]))
                    maximum_patch = max(
                        maximum_patch, float(np.max(areas[candidate]) / hemi_area)
                    )
                    if location == "anterior":
                        primary = -anterior[candidate]
                    elif location == "posterior":
                        primary = anterior[candidate]
                    else:
                        primary = np.abs(anterior[candidate] - 0.5)
                    selected = _prefix_for_extent(
                        candidate,
                        (primary, pd[candidate], candidate),
                        areas,
                        extent,
                    )
                    selected_parts.append(selected)
                    label = "left" if code == -1 else "right"
                    by_hemi[label] = float(np.sum(areas[selected]) / hemi_area)
                indices = np.sort(np.concatenate(selected_parts))
                current = set(int(value) for value in indices)
                nested = previous.issubset(current)
                nesting_checks.append(
                    {
                        "hemisphere": family,
                        "location": location,
                        "extent": extent,
                        "previous_is_subset": nested,
                    }
                )
                if not nested:
                    raise ValueError("support ladder is not exactly nested")
                previous = current
                selected_area = float(np.sum(areas[indices]))
                identifier = support_identifier(family, location, extent)
                supports[identifier] = SourceSupport(
                    identifier=identifier,
                    hemisphere=family,
                    location=location,
                    target_extent=extent,
                    indices=indices,
                    conditional_weights=areas[indices] / selected_area,
                    full_sheet_area_fraction=selected_area / total_area,
                    realized_extent_within_selected_hemispheres=selected_area
                    / family_area,
                    realized_extent_by_hemisphere=by_hemi,
                    maximum_selected_hemisphere_patch_fraction=maximum_patch,
                )
    expected = len(families) * (len(locations) * len(extents) + int(include_whole_extent))
    if len(supports) != expected:
        raise AssertionError("support count disagrees with the frozen factorial design")
    return supports, {
        "support_count": len(supports),
        "expected_support_count": expected,
        "all_exactly_nested": all(row["previous_is_subset"] for row in nesting_checks),
        "nesting_checks": nesting_checks,
    }


def referenced_leadfield(leadfield: ArrayLike) -> FloatArray:
    matrix = np.asarray(leadfield, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("leadfield must be a finite sensor-by-source matrix")
    return helmert_reference(matrix.shape[0]) @ matrix


def standardized_nuisance(
    hippocampal_referenced: ArrayLike,
    hippocampal_area_weights_m2: ArrayLike,
    cortical_leadfield: ArrayLike,
    cortical_area_weights_m2: ArrayLike,
    noise_fraction: float,
) -> tuple[FloatArray, float, float, dict[str, float]]:
    """Return nuisance Cholesky and whole-sheet reference traces."""

    hippocampal = np.asarray(hippocampal_referenced, dtype=np.float64)
    cortical = referenced_leadfield(cortical_leadfield)
    h_area = _finite_vector(hippocampal_area_weights_m2, "hippocampal areas")
    c_area = _finite_vector(cortical_area_weights_m2, "cortical areas")
    if hippocampal.shape[1] != len(h_area) or cortical.shape[1] != len(c_area):
        raise ValueError("lead fields and quadrature areas are misaligned")
    if noise_fraction <= 0.0 or not np.isfinite(noise_fraction):
        raise ValueError("noise_fraction must be positive and finite")
    h_weights = h_area / float(np.sum(h_area))
    c_weights = c_area / float(np.sum(c_area))
    h_trace = float(np.sum((hippocampal * hippocampal) @ h_weights))
    c_covariance = (cortical * np.sqrt(c_weights)[None, :]) @ (
        cortical * np.sqrt(c_weights)[None, :]
    ).T
    c_trace = float(np.trace(c_covariance))
    if h_trace <= 0.0 or c_trace <= 0.0:
        raise ValueError("reference covariance trace is non-positive")
    nuisance = c_covariance / c_trace
    nuisance += noise_fraction * np.eye(nuisance.shape[0]) / nuisance.shape[0]
    nuisance = 0.5 * (nuisance + nuisance.T)
    cholesky = np.linalg.cholesky(nuisance)
    return cholesky, h_trace, c_trace, {
        "hippocampal_incoherent_trace": h_trace,
        "cortical_incoherent_trace": c_trace,
        "nuisance_trace": float(np.trace(nuisance)),
    }


def deterministic_phase(anterior_coordinate: ArrayLike, cycles: float) -> FloatArray:
    coordinate = _finite_vector(anterior_coordinate, "anterior_coordinate")
    if not np.isfinite(cycles):
        raise ValueError("cycles must be finite")
    return 2.0 * math.pi * float(cycles) * coordinate


def _factor_eigenvalues(factor: FloatArray, cholesky: FloatArray, scale: float) -> FloatArray:
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("factor scale must be positive")
    factor = np.asarray(factor, dtype=np.float64)
    if factor.ndim != 2 or factor.shape[0] != cholesky.shape[0]:
        raise ValueError("sensor factor has incompatible dimensions")
    nonzero = np.linalg.norm(factor, axis=0) > np.finfo(float).tiny
    if not np.any(nonzero):
        return np.zeros(0, dtype=np.float64)
    whitened = np.linalg.solve(cholesky, factor[:, nonzero] / math.sqrt(scale))
    gram = whitened.T @ whitened
    values = np.linalg.eigvalsh(0.5 * (gram + gram.T))[::-1]
    tolerance = 1e-12 * max(float(values[0]), 1.0)
    if float(values[-1]) < -tolerance:
        raise FloatingPointError("low-rank recoverability factor is not PSD")
    return np.maximum(values, 0.0)


def _spectrum_record(values: FloatArray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "recoverability_index_bits": 0.0,
            "largest_eigenvalue": 0.0,
            "whitened_signal_power": 0.0,
            "participation_ratio": 0.0,
            "numerical_rank": 0.0,
        }
    total = float(np.sum(values))
    squared = float(np.sum(values * values))
    return {
        "recoverability_index_bits": float(0.5 * np.sum(np.log2(1.0 + values))),
        "largest_eigenvalue": float(values[0]),
        "whitened_signal_power": total,
        "participation_ratio": total * total / squared if squared > 0.0 else 0.0,
        "numerical_rank": float(np.count_nonzero(values > 1e-10 * max(values[0], 1.0))),
    }


def phase_locked_metrics(
    referenced_hippocampal_leadfield: ArrayLike,
    support: SourceSupport,
    phases_rad: ArrayLike,
    full_sheet_incoherent_trace: float,
    nuisance_cholesky: ArrayLike,
) -> tuple[dict[str, float], FloatArray]:
    """Evaluate one unit-magnitude phase-locked support."""

    leadfield = np.asarray(referenced_hippocampal_leadfield, dtype=np.float64)
    cholesky = np.asarray(nuisance_cholesky, dtype=np.float64)
    phase = _finite_vector(phases_rad, "phases_rad")
    if len(phase) != len(support.indices):
        raise ValueError("phase vector does not match support")
    weights = support.conditional_weights
    selected = leadfield[:, support.indices]
    baseline = float(np.sum((selected * selected) @ weights))
    real = selected @ (weights * np.cos(phase))
    imaginary = selected @ (weights * np.sin(phase))
    factor = np.column_stack((real, imaginary))
    power = float(np.sum(factor * factor))
    retained = power / baseline
    bound_error = max(0.0, retained - 1.0)
    full_fraction = support.full_sheet_area_fraction
    whole_values = _factor_eigenvalues(
        full_fraction * factor, cholesky, full_sheet_incoherent_trace
    )
    equal_values = _factor_eigenvalues(factor, cholesky, baseline)
    record = {
        "support_incoherent_trace": baseline,
        "support_normalized_retained_power_ratio": retained,
        "support_normalized_retained_rms_ratio": math.sqrt(max(retained, 0.0)),
        "whole_sheet_incoherent_power_ratio": full_fraction**2
        * power
        / full_sheet_incoherent_trace,
        "full_sheet_area_fraction": full_fraction,
        "analytical_bound_excess": bound_error,
        **{f"whole_sheet_{key}": value for key, value in _spectrum_record(whole_values).items()},
        **{f"equal_support_{key}": value for key, value in _spectrum_record(equal_values).items()},
    }
    return record, factor


def patch_coherence_metrics(
    referenced_hippocampal_leadfield: ArrayLike,
    support: SourceSupport,
    anterior_coordinate: ArrayLike,
    proximal_distal_coordinate: ArrayLike,
    hemisphere_code: ArrayLike,
    intrinsic_bin_width: float,
    full_sheet_incoherent_trace: float,
    nuisance_cholesky: ArrayLike,
) -> tuple[dict[str, float], FloatArray]:
    """Evaluate independent, internally coherent AP/PD patches."""

    if not 0.0 < intrinsic_bin_width <= 1.0:
        raise ValueError("intrinsic bin width must lie in (0, 1]")
    leadfield = np.asarray(referenced_hippocampal_leadfield, dtype=np.float64)
    anterior = _finite_vector(anterior_coordinate, "anterior_coordinate")
    pd = _finite_vector(proximal_distal_coordinate, "proximal_distal_coordinate")
    hemispheres = np.asarray(hemisphere_code, dtype=np.int8).reshape(-1)
    selected_indices = support.indices
    selected = leadfield[:, selected_indices]
    weights = support.conditional_weights
    baseline = float(np.sum((selected * selected) @ weights))
    ap_bin = np.minimum(
        np.floor(anterior[selected_indices] / intrinsic_bin_width).astype(np.int64),
        int(math.ceil(1.0 / intrinsic_bin_width)) - 1,
    )
    pd_bin = np.minimum(
        np.floor(pd[selected_indices] / intrinsic_bin_width).astype(np.int64),
        int(math.ceil(1.0 / intrinsic_bin_width)) - 1,
    )
    keys = np.column_stack((hemispheres[selected_indices], ap_bin, pd_bin))
    _, group = np.unique(keys, axis=0, return_inverse=True)
    groups = int(np.max(group)) + 1
    factor = np.zeros((leadfield.shape[0], groups), dtype=np.float64)
    np.add.at(factor.T, group, (selected * weights[None, :]).T)
    group_area = np.bincount(group, weights=weights, minlength=groups)
    if not np.isclose(float(np.sum(group_area)), 1.0, rtol=0.0, atol=1e-12):
        raise FloatingPointError("patch areas do not sum to one")
    power = float(np.sum(factor * factor))
    retained = power / baseline
    full_fraction = support.full_sheet_area_fraction
    whitened_whole = np.linalg.solve(
        np.asarray(nuisance_cholesky, dtype=np.float64),
        full_fraction * factor / math.sqrt(full_sheet_incoherent_trace),
    )
    whitened_equal = np.linalg.solve(
        np.asarray(nuisance_cholesky, dtype=np.float64),
        factor / math.sqrt(baseline),
    )
    record = {
        "coherent_patch_count": groups,
        "coherent_patch_area_minimum": float(np.min(group_area)),
        "coherent_patch_area_maximum": float(np.max(group_area)),
        "coherent_patch_area_participation_ratio": float(
            1.0 / np.sum(group_area * group_area)
        ),
        "support_incoherent_trace": baseline,
        "support_normalized_retained_power_ratio": retained,
        "support_normalized_retained_rms_ratio": math.sqrt(max(retained, 0.0)),
        "whole_sheet_incoherent_power_ratio": full_fraction**2
        * power
        / full_sheet_incoherent_trace,
        "whole_sheet_whitened_signal_power": float(np.sum(whitened_whole**2)),
        "equal_support_whitened_signal_power": float(np.sum(whitened_equal**2)),
        "full_sheet_area_fraction": full_fraction,
        "analytical_bound_excess": max(0.0, retained - 1.0),
    }
    return record, factor


def derive_seed(master_seed: int, *components: object) -> int:
    text = "|".join((str(int(master_seed)), *(str(value) for value in components)))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def random_correlated_phases(
    anterior_coordinate: ArrayLike,
    proximal_distal_coordinate: ArrayLike,
    hemisphere_code: ArrayLike,
    conditional_weights: ArrayLike,
    *,
    correlation_length: float,
    features: int,
    phase_standard_deviation_rad: float,
    master_seed: int,
    seed_components: Iterable[object],
) -> tuple[FloatArray, dict[str, object]]:
    """Sample a deterministic RFF approximation to an intrinsic RBF field."""

    anterior = _finite_vector(anterior_coordinate, "anterior_coordinate")
    pd = _finite_vector(proximal_distal_coordinate, "proximal_distal_coordinate")
    hemispheres = np.asarray(hemisphere_code, dtype=np.int8).reshape(-1)
    weights = _finite_vector(conditional_weights, "conditional_weights")
    if not (len(anterior) == len(pd) == len(hemispheres) == len(weights)):
        raise ValueError("random phase arrays differ in length")
    if not 0.0 < correlation_length <= 1.0:
        raise ValueError("correlation_length must lie in (0, 1]")
    if features < 16:
        raise ValueError("at least 16 random Fourier features are required")
    if phase_standard_deviation_rad <= 0.0:
        raise ValueError("phase standard deviation must be positive")
    phases = np.empty(len(anterior), dtype=np.float64)
    records: list[dict[str, object]] = []
    base_components = tuple(seed_components)
    for code, label in ((-1, "left"), (1, "right")):
        local = np.flatnonzero(hemispheres == code)
        if len(local) == 0:
            continue
        seed = derive_seed(master_seed, *base_components, label)
        rng = np.random.default_rng(seed)
        coordinates = np.column_stack((anterior[local], pd[local]))
        omega = rng.normal(
            loc=0.0, scale=1.0 / correlation_length, size=(features, 2)
        )
        offsets = rng.uniform(0.0, 2.0 * math.pi, size=features)
        coefficients = rng.normal(size=features)
        field = math.sqrt(2.0 / features) * (
            np.cos(coordinates @ omega.T + offsets) @ coefficients
        )
        local_weights = weights[local] / float(np.sum(weights[local]))
        mean = _weighted_mean(field, local_weights)
        centered = field - mean
        deviation = math.sqrt(_weighted_mean(centered * centered, local_weights))
        if deviation <= 1e-10:
            raise FloatingPointError("random phase field has zero weighted variance")
        global_phase = float(rng.uniform(-math.pi, math.pi))
        phases[local] = (
            centered * (phase_standard_deviation_rad / deviation) + global_phase
        )
        records.append(
            {
                "hemisphere": label,
                "seed": seed,
                "weighted_phase_mean_before_global_offset_rad": 0.0,
                "weighted_phase_standard_deviation_rad": phase_standard_deviation_rad,
                "global_phase_rad": global_phase,
            }
        )
    unit_error = float(np.max(np.abs(np.abs(np.exp(1j * phases)) - 1.0)))
    return phases, {
        "generator": "random Fourier approximation to squared-exponential AP/PD field",
        "correlation_length_intrinsic": correlation_length,
        "features": features,
        "unit_complex_magnitude_maximum_error": unit_error,
        "hemispheres": records,
    }


def project_free_cartesian_leadfield(
    free_leadfield: ArrayLike, directions: ArrayLike
) -> FloatArray:
    free = np.asarray(free_leadfield, dtype=np.float64)
    direction = np.asarray(directions, dtype=np.float64)
    if direction.ndim != 2 or direction.shape[1] != 3:
        raise ValueError("directions must have shape (sources, 3)")
    if free.ndim == 2:
        if free.shape[1] != 3 * len(direction):
            raise ValueError("free leadfield does not have three columns per source")
        free = free.reshape(free.shape[0], len(direction), 3)
    if free.shape[1:] != direction.shape or not np.all(np.isfinite(free)):
        raise ValueError("free leadfield and directions are misaligned")
    norms = np.linalg.norm(direction, axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-6:
        raise ValueError("source directions must be unit vectors")
    return np.einsum("snc,nc->sn", free, direction, optimize=True)


def tilted_directions(directions: ArrayLike, angle_degrees: float) -> tuple[FloatArray, dict[str, float]]:
    """Tilt directions toward a deterministic local tangent axis."""

    direction = np.asarray(directions, dtype=np.float64)
    if direction.ndim != 2 or direction.shape[1] != 3 or not np.all(np.isfinite(direction)):
        raise ValueError("directions must be a finite (sources, 3) array")
    norms = np.linalg.norm(direction, axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-6:
        raise ValueError("directions must be normalized")
    if not 0.0 <= angle_degrees <= 90.0:
        raise ValueError("orientation mismatch angle must lie in [0, 90]")
    axes = (
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    tangent = np.zeros_like(direction)
    unresolved = np.ones(len(direction), dtype=bool)
    for axis in axes:
        projected = axis[None, :] - np.sum(direction * axis[None, :], axis=1)[:, None] * direction
        projected_norm = np.linalg.norm(projected, axis=1)
        use = unresolved & (projected_norm > 1e-8)
        tangent[use] = projected[use] / projected_norm[use, None]
        unresolved[use] = False
    if np.any(unresolved):
        raise FloatingPointError("could not construct a local tangent direction")
    angle = math.radians(angle_degrees)
    tilted = math.cos(angle) * direction + math.sin(angle) * tangent
    tilted /= np.linalg.norm(tilted, axis=1)[:, None]
    cosine = np.sum(tilted * direction, axis=1)
    return tilted, {
        "requested_angle_degrees": angle_degrees,
        "minimum_realized_angle_degrees": float(np.degrees(np.arccos(np.clip(np.max(cosine), -1.0, 1.0)))),
        "maximum_realized_angle_degrees": float(np.degrees(np.arccos(np.clip(np.min(cosine), -1.0, 1.0)))),
        "maximum_unit_norm_error": float(np.max(np.abs(np.linalg.norm(tilted, axis=1) - 1.0))),
    }


def whitened_subspace_projection(
    truth_factor: ArrayLike,
    assumed_factor: ArrayLike,
    nuisance_cholesky: ArrayLike,
) -> dict[str, float]:
    """Fraction of whitened truth energy captured by an assumed subspace."""

    truth = np.asarray(truth_factor, dtype=np.float64)
    assumed = np.asarray(assumed_factor, dtype=np.float64)
    cholesky = np.asarray(nuisance_cholesky, dtype=np.float64)
    if truth.shape[0] != assumed.shape[0] or truth.shape[0] != cholesky.shape[0]:
        raise ValueError("subspace factors have incompatible sensor dimensions")
    truth = truth[:, np.linalg.norm(truth, axis=0) > np.finfo(float).tiny]
    assumed = assumed[:, np.linalg.norm(assumed, axis=0) > np.finfo(float).tiny]
    if truth.shape[1] == 0 or assumed.shape[1] == 0:
        raise ValueError("subspace factors must be nonzero")
    whitened_truth = np.linalg.solve(cholesky, truth)
    whitened_assumed = np.linalg.solve(cholesky, assumed)
    q_truth, _ = np.linalg.qr(whitened_truth, mode="reduced")
    q_assumed, _ = np.linalg.qr(whitened_assumed, mode="reduced")
    captured = float(np.sum((q_assumed.T @ whitened_truth) ** 2))
    total = float(np.sum(whitened_truth**2))
    singular = np.linalg.svd(q_assumed.T @ q_truth, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    maximum_angle = float(np.degrees(np.arccos(np.min(singular))))
    return {
        "whitened_truth_projection_efficiency": captured / total,
        "whitened_subspace_minimum_cosine": float(np.min(singular)),
        "whitened_subspace_maximum_principal_angle_degrees": maximum_angle,
        "truth_whitened_power": total,
        "assumed_whitened_power": float(np.sum(whitened_assumed**2)),
    }


def bilateral_interference(
    left_factor_full_sheet: ArrayLike,
    right_factor_full_sheet: ArrayLike,
) -> dict[str, float]:
    """Decompose bilateral sensor power into unilateral and cross terms."""

    left = np.asarray(left_factor_full_sheet, dtype=np.float64)
    right = np.asarray(right_factor_full_sheet, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left and right factors must have identical shapes")
    left_power = float(np.sum(left * left))
    right_power = float(np.sum(right * right))
    cross = float(2.0 * np.sum(left * right))
    independent = left_power + right_power
    bilateral = independent + cross
    scale = max(independent, np.finfo(float).tiny)
    return {
        "left_power": left_power,
        "right_power": right_power,
        "cross_term": cross,
        "independent_hemisphere_power": independent,
        "bilateral_locked_power": bilateral,
        "bilateral_interference_gain": bilateral / scale,
        "cross_term_fraction_of_independent_power": cross / scale,
    }

