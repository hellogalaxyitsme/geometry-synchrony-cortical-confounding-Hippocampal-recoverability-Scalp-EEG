"""Frozen subject-level and population metrics for the HCP forward cohort.

The quantities in this module are dimensionless or geometry-only numerical
probes.  They must not be relabelled as physiologically calibrated information,
source amplitude, or empirical hippocampal detection performance.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import mne
from mne.io.constants import FIFF
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from theory.recoverability import helmert_reference

from .bundle import sha256_file
from .convergence import sensor_operators, standardized_recoverability_spectrum
from .hcp_forward import scanner_ras_to_bem_transform
from .nested_source import cortical_source_hierarchy, hippocampal_source_hierarchy


FloatArray = NDArray[np.float64]
REGIMES = ("incoherent", "coherent", "traveling_wave")


@dataclass(frozen=True)
class SubjectInputs:
    subject: str
    subject_input: Path
    bem_directory: Path
    forward_directory: Path


def _load_npz(path: Path, names: Iterable[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in names if name not in archive]
        if missing:
            raise ValueError(f"{path} is missing arrays {missing}")
        return {name: np.asarray(archive[name]) for name in names}


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_declared_outputs(directory: Path, report: dict[str, object]) -> bool:
    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return False
    for name, record in outputs.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            return False
        path = directory / name
        if (
            not path.is_file()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            return False
    return True


def _validate_bem_outputs(directory: Path, subject: str, report: dict[str, object]) -> bool:
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        return False
    expected = {
        "bem_surfaces_sha256": directory / "bem" / f"{subject}-ico4-bem.fif",
        "bem_solution_sha256": directory / "bem" / f"{subject}-ico4-bem-sol.fif",
    }
    return all(
        path.is_file() and outputs.get(key) == sha256_file(path)
        for key, path in expected.items()
    )


def _geometry_error(
    rebuilt: dict[str, np.ndarray], frozen: dict[str, np.ndarray]
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name in ("positions_m", "directions", "area_weights_m2"):
        if rebuilt[name].shape != frozen[name].shape:
            raise ValueError(f"reference {name} shape changed")
        errors[name] = float(np.max(np.abs(rebuilt[name] - frozen[name])))
    if not np.array_equal(rebuilt["hemisphere_code"], frozen["hemisphere_code"]):
        raise ValueError("reference hemisphere codes or ordering changed")
    errors["hemisphere_code"] = 0.0
    return errors


def _generalized_spectrum(signal: FloatArray, nuisance: FloatArray) -> FloatArray:
    signal = 0.5 * (np.asarray(signal, dtype=float) + np.asarray(signal, dtype=float).T)
    nuisance = 0.5 * (
        np.asarray(nuisance, dtype=float) + np.asarray(nuisance, dtype=float).T
    )
    values = eigh(signal, nuisance, eigvals_only=True, check_finite=True)[::-1]
    scale = max(float(values[0]), np.finfo(float).tiny)
    if float(values[-1]) < -1e-9 * scale:
        raise FloatingPointError("generalized recoverability spectrum has a negative mode")
    return np.maximum(values, 0.0)


def _spectrum_metrics(values: FloatArray) -> dict[str, float]:
    total = float(values.sum())
    squared = float(np.dot(values, values))
    largest = float(values[0]) if values.size else 0.0
    return {
        "recoverability_index_bits": float(0.5 * np.log2(1.0 + values).sum()),
        "spectrum_sum": total,
        "largest_eigenvalue": largest,
        "participation_ratio": total * total / squared if squared > 0.0 else 0.0,
        "modes_above_relative_1e_3": float(
            np.count_nonzero(values > 1e-3 * max(largest, np.finfo(float).tiny))
        ),
    }


def _surface_record(report: dict[str, object], label: str) -> dict[str, object]:
    surfaces = report.get("surfaces")
    if not isinstance(surfaces, list):
        raise ValueError("BEM report has no surface records")
    matches = [row for row in surfaces if isinstance(row, dict) and row.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"BEM report does not have exactly one {label} surface")
    return matches[0]


def _bem_geometry_report(
    inputs: SubjectInputs,
    solution_report: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    """Return geometry evidence, validating the v3.1 provenance link when used.

    Earlier BEM reports contained geometry and numerical-solution evidence in a
    single JSON document.  The v3.1 exact-import solution gate intentionally
    keeps them separate: the solution report records the SHA-256 of the
    validated surface-gate report that was copied into its immutable bundle.
    """

    name = config.get("bem_geometry_report_name")
    if name is None:
        return solution_report
    if name != "validated_surface_gate_report.json":
        raise ValueError("unexpected BEM geometry report name")
    path = inputs.bem_directory / "bem" / name
    geometry_report = _json(path)
    declared_hash = dict(solution_report.get("input_sha256", {})).get(
        "surface_report"
    )
    expected_protocol = config.get("bem_geometry_protocol")
    if (
        geometry_report.get("ok") is not True
        or geometry_report.get("subject") != inputs.subject
        or declared_hash != sha256_file(path)
        or (
            expected_protocol is not None
            and geometry_report.get("protocol") != expected_protocol
        )
    ):
        raise ValueError("validated BEM geometry provenance failed")
    return geometry_report


def compute_subject_metrics(
    inputs: SubjectInputs,
    output_directory: Path,
    config: dict[str, object],
) -> dict[str, object]:
    """Compute one subject's frozen population metrics and strict QC record."""

    subject = inputs.subject
    output_directory.mkdir(parents=True, exist_ok=False)
    analysis = dict(config["population_analysis"])
    source_levels = list(config["source_levels"])
    hippocampal_levels = [
        {
            "name": row["name"],
            "spatial_bin_mm": row["hippocampal_spatial_bin_mm"],
        }
        for row in source_levels
    ]
    cortical_levels = [
        {
            "name": row["name"],
            "spatial_bin_mm": row["cortical_spatial_bin_mm"],
        }
        for row in source_levels
    ]
    reference_name = "reference"

    bem_report_path = inputs.bem_directory / "bem" / str(
        config.get("bem_report_name", "bem_smoke_report.json")
    )
    bem_marker_path = inputs.bem_directory / str(
        config.get("bem_marker_name", ".hcp_bem_population.json")
    )
    forward_report_path = inputs.forward_directory / "report.json"
    forward_marker_path = inputs.forward_directory / ".hcp_forward_population.json"
    bem_report = _json(bem_report_path)
    bem_marker = _json(bem_marker_path)
    forward_report = _json(forward_report_path)
    forward_marker = _json(forward_marker_path)
    if (
        bem_report.get("ok") is not True
        or bem_report.get("subject") != subject
        or (
            "bem_protocol" in config
            and bem_report.get("protocol") != config["bem_protocol"]
        )
        or bem_marker.get("status") != "complete"
        or bem_marker.get("subject") != subject
        or bem_marker.get("protocol")
        != config.get("bem_protocol", config["protocol"])
        or (
            "bem_config_sha256" in config
            and bem_marker.get("config_sha256") != config["bem_config_sha256"]
        )
        or bem_marker.get("report_sha256") != sha256_file(bem_report_path)
        or not _validate_bem_outputs(inputs.bem_directory, subject, bem_report)
    ):
        raise ValueError("BEM report, marker, or declared products failed integrity checks")
    geometry_report = _bem_geometry_report(inputs, bem_report, config)
    if (
        forward_report.get("ok") is not True
        or forward_report.get("subject") != subject
        or forward_report.get("protocol") != config["protocol"]
        or forward_marker.get("status") != "complete"
        or forward_marker.get("subject") != subject
        or forward_marker.get("protocol") != config["protocol"]
        or forward_marker.get("report_sha256") != sha256_file(forward_report_path)
        or not _validate_declared_outputs(inputs.forward_directory, forward_report)
    ):
        raise ValueError("forward report, marker, or declared products failed integrity checks")

    h_metadata = _load_npz(
        inputs.forward_directory / "hippocampal-source-metadata.npz",
        ("positions_m", "directions", "area_weights_m2", "hemisphere_code"),
    )
    c_metadata = _load_npz(
        inputs.forward_directory / "cortical-source-metadata.npz",
        ("positions_m", "directions", "area_weights_m2", "hemisphere_code"),
    )
    h_leadfield = np.load(
        inputs.forward_directory / "hippocampal-fixed-leadfield.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    c_leadfield = np.load(
        inputs.forward_directory / "cortical-fixed-leadfield.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if h_leadfield.ndim != 2 or c_leadfield.ndim != 2:
        raise ValueError("lead fields must be matrices")
    if h_leadfield.shape[0] != c_leadfield.shape[0]:
        raise ValueError("hippocampal and cortical sensor counts differ")
    if h_leadfield.shape[1] != len(h_metadata["area_weights_m2"]):
        raise ValueError("hippocampal lead field and metadata differ")
    if c_leadfield.shape[1] != len(c_metadata["area_weights_m2"]):
        raise ValueError("cortical lead field and metadata differ")
    if not np.all(np.isfinite(h_leadfield)) or not np.all(np.isfinite(c_leadfield)):
        raise ValueError("lead fields contain non-finite values")

    brain_surface = inputs.bem_directory / "bem" / "brain.surf"
    scanner_to_bem, center_ras_mm = scanner_ras_to_bem_transform(brain_surface)
    h_hierarchy, h_hierarchy_report = hippocampal_source_hierarchy(
        inputs.subject_input / "T1w" / "aparc+aseg.nii.gz",
        scanner_to_bem,
        hippocampal_levels,
        normal_bin_width=float(config["normal_bin_width"]),
        smoothing_sigma_voxels=float(config["hippocampal_smoothing_sigma_voxels"]),
    )
    c_hierarchy, c_hierarchy_report = cortical_source_hierarchy(
        subject,
        inputs.subject_input.parent,
        scanner_to_bem,
        cortical_levels,
        normal_bin_width=float(config["normal_bin_width"]),
        surface_name=str(config["cortical_surface"]),
    )
    h_reference = h_hierarchy[reference_name]
    c_reference = c_hierarchy[reference_name]
    geometry_errors = {
        "hippocampal": _geometry_error(h_reference, h_metadata),
        "cortical": _geometry_error(c_reference, c_metadata),
    }
    maximum_geometry_error = max(
        value for group in geometry_errors.values() for value in group.values()
    )
    exact_nested = bool(h_hierarchy_report["exact_nested"]) and bool(
        c_hierarchy_report["exact_nested"]
    )

    wave_basis = np.vstack(
        (h_reference["longitudinal_cosine"], h_reference["longitudinal_sine"])
    )
    operators = sensor_operators(
        np.asarray(h_leadfield),
        np.asarray(c_leadfield),
        h_metadata["area_weights_m2"],
        c_metadata["area_weights_m2"],
        None,
        hippocampal_wave_basis=wave_basis,
    )
    h_trace = float(np.trace(operators.hippocampal_covariance))
    c_trace = float(np.trace(operators.cortical_covariance))
    coherent_covariance = np.outer(
        operators.hippocampal_coherent_topography,
        operators.hippocampal_coherent_topography,
    )
    coherent_trace = float(np.trace(coherent_covariance))
    wave_trace = float(np.trace(operators.hippocampal_wave_covariance))
    if min(h_trace, c_trace) <= 0.0:
        raise ValueError("sensor covariance traces must be positive")

    sensors = int(h_leadfield.shape[0])
    contrasts = helmert_reference(sensors)
    signals = {
        "incoherent": contrasts @ operators.hippocampal_covariance @ contrasts.T / h_trace,
        "coherent": contrasts @ coherent_covariance @ contrasts.T / h_trace,
        "traveling_wave": contrasts
        @ operators.hippocampal_wave_covariance
        @ contrasts.T
        / h_trace,
    }
    cortical_referenced = contrasts @ operators.cortical_covariance @ contrasts.T
    cortical_referenced = 0.5 * (cortical_referenced + cortical_referenced.T)
    c_values, c_vectors = np.linalg.eigh(cortical_referenced)
    order = np.argsort(c_values)[::-1]
    c_values = np.maximum(c_values[order], 0.0)
    c_vectors = c_vectors[:, order]
    if float(c_values.sum()) <= 0.0:
        raise ValueError("cortical covariance has zero referenced trace")

    ranks = [int(value) for value in analysis["cortical_rank_ladder"]]
    if ranks[-1] != sensors - 1 or any(
        left >= right for left, right in zip(ranks[:-1], ranks[1:])
    ):
        raise ValueError("cortical rank ladder must be increasing and end at sensor rank")
    noise_fraction = float(analysis["standardized_noise_fraction"])
    spectra = np.zeros((len(REGIMES), len(ranks), sensors - 1), dtype=np.float64)
    rank_records: list[dict[str, object]] = []
    for rank_index, rank in enumerate(ranks):
        retained_values = c_values[:rank]
        retained_total = float(retained_values.sum())
        if retained_total <= 0.0:
            raise ValueError("a cortical rank restriction has zero energy")
        basis = c_vectors[:, :rank]
        cortical_fixed_total = (
            basis * (retained_values / retained_total)[None, :]
        ) @ basis.T
        nuisance = cortical_fixed_total + noise_fraction * np.eye(sensors - 1) / (
            sensors - 1
        )
        regime_metrics: dict[str, object] = {}
        for regime_index, regime in enumerate(REGIMES):
            values = _generalized_spectrum(signals[regime], nuisance)
            spectra[regime_index, rank_index] = values
            regime_metrics[regime] = _spectrum_metrics(values)
        rank_records.append(
            {
                "requested_rank": rank,
                "retained_cortical_spectral_energy_fraction": retained_total
                / float(c_values.sum()),
                "fixed_total_cortical_trace": float(np.trace(cortical_fixed_total)),
                "regimes": regime_metrics,
            }
        )

    direct_full = standardized_recoverability_spectrum(
        operators, h_trace, c_trace, noise_fraction
    )
    full_incoherent = spectra[REGIMES.index("incoherent"), -1]
    spectrum_identity_error = float(
        np.linalg.norm(direct_full - full_incoherent)
        / max(np.linalg.norm(direct_full), np.finfo(float).tiny)
    )

    target = np.asarray(operators.hippocampal_coherent_topography)
    projected = contrasts.T @ (contrasts @ target)
    masking_denominator = float(np.dot(target, target))
    masking_residual = float(np.dot(target - projected, target - projected)) / max(
        masking_denominator, np.finfo(float).tiny
    )

    bem_surfaces = mne.read_bem_surfaces(
        inputs.bem_directory / "bem" / f"{subject}-ico4-bem.fif", verbose=False
    )
    by_id = {int(surface["id"]): surface for surface in bem_surfaces}
    scalp_vertices = np.asarray(
        by_id[int(FIFF.FIFFV_BEM_SURF_ID_HEAD)]["rr"], dtype=float
    )
    nearest_depth, _ = cKDTree(scalp_vertices).query(
        np.asarray(h_metadata["positions_m"]), k=1, workers=1
    )
    depth_quantiles = np.quantile(nearest_depth, [0.05, 0.5, 0.95])

    forward_section = dict(forward_report["forward"])
    source_section = dict(forward_report["source_model"])
    containment = dict(forward_report["containment"])
    montage = dict(forward_report["montage"])
    full_rank = int(forward_section["cortical_referenced_rank"])
    expected_rank = int(forward_section["expected_sensor_rank"])
    outside_h = int(dict(containment["hippocampal"])["outside_inner_skull"])
    outside_c = int(dict(containment["cortical"])["outside_inner_skull"])
    tolerance = float(analysis["reference_geometry_maximum_absolute_error"])
    qc = {
        "bem_ok": bem_report.get("ok") is True,
        "closed_genus_zero_topology": geometry_report.get("closed_genus_zero_topology")
        is True,
        "strictly_nested_surface_volumes": geometry_report.get(
            "strictly_nested_surface_volumes"
        )
        is True,
        "bem_solution_finite": bem_report.get("solution_finite") is True,
        "forward_ok": forward_report.get("ok") is True,
        "source_containment": outside_h == 0 and outside_c == 0,
        "expected_sensor_count": sensors == 339 and int(montage["sensors"]) == 339,
        "cortical_full_referenced_rank": full_rank == expected_rank == sensors - 1,
        "exact_nested_hierarchy": exact_nested,
        "reference_geometry_reproduced": maximum_geometry_error <= tolerance,
        "spectrum_implementation_identity": spectrum_identity_error <= 1e-10,
        "full_rank_masking_control": masking_residual
        <= float(analysis["full_rank_masking_residual_fraction_max"]),
        "finite_spectra": bool(np.all(np.isfinite(spectra)) and np.all(spectra >= 0.0)),
    }
    if "bem_inner_surface" in config:
        declared_inner = dict(config["bem_inner_surface"])
        reported_inner = dict(bem_report.get("inner_surface", {}))
        qc["declared_inner_surface_method"] = bool(
            reported_inner.get("method") == declared_inner["method"]
            and reported_inner.get("construction_ico")
            == declared_inner["construction_ico"]
            and reported_inner.get("margin_mm") == declared_inner["margin_mm"]
            and reported_inner.get(
                "representative_brainmask_points_outside_final_inner_skull"
            )
            == 0
            and reported_inner.get("final_inner_vertices_outside_outer_skull") == 0
        )
    if not all(qc.values()):
        raise ValueError(f"subject population QC failed: {qc}")

    full_metrics = {
        regime: rank_records[-1]["regimes"][regime] for regime in REGIMES
    }
    primary = {
        "hippocampal_covariance_trace": h_trace,
        "cortical_covariance_trace": c_trace,
        "hippocampal_to_cortical_trace_ratio": h_trace / c_trace,
        "coherent_cancellation_ratio": coherent_trace / h_trace,
        "traveling_wave_cancellation_ratio": wave_trace / h_trace,
        "recoverability_index_bits_incoherent_rank338": full_metrics["incoherent"][
            "recoverability_index_bits"
        ],
        "recoverability_index_bits_coherent_rank338": full_metrics["coherent"][
            "recoverability_index_bits"
        ],
        "recoverability_index_bits_traveling_wave_rank338": full_metrics[
            "traveling_wave"
        ]["recoverability_index_bits"],
        "largest_recoverability_eigenvalue_incoherent_rank338": full_metrics[
            "incoherent"
        ]["largest_eigenvalue"],
        "effective_recoverability_rank_incoherent_rank338": full_metrics[
            "incoherent"
        ]["participation_ratio"],
        "full_rank_masking_residual_fraction": masking_residual,
        "hippocampal_depth_p05_m": float(depth_quantiles[0]),
        "hippocampal_depth_median_m": float(depth_quantiles[1]),
        "hippocampal_depth_p95_m": float(depth_quantiles[2]),
        "hippocampal_area_m2": float(source_section["hippocampal_area_m2"]),
        "cortical_area_m2": float(source_section["cortical_area_m2"]),
        "inner_skull_volume_m3": float(
            _surface_record(geometry_report, "inner_skull")["absolute_volume_m3"]
        ),
        "outer_skull_volume_m3": float(
            _surface_record(geometry_report, "outer_skull")["absolute_volume_m3"]
        ),
        "outer_skin_volume_m3": float(
            _surface_record(geometry_report, "outer_skin")["absolute_volume_m3"]
        ),
        "outer_skin_surface_area_m2": float(
            _surface_record(geometry_report, "outer_skin")["surface_area_m2"]
        ),
    }
    spectra_path = output_directory / "recoverability_spectra.npy"
    np.save(spectra_path, spectra, allow_pickle=False)
    report = {
        "schema_version": 1,
        "ok": True,
        "protocol": config["protocol"],
        "subject": subject,
        "scope": "geometry-only numerical forward-model probes; not physiological calibration",
        "source_regime_order": list(REGIMES),
        "cortical_rank_order": ranks,
        "primary_metrics": primary,
        "rank_ladder": rank_records,
        "geometry": {
            "hippocampal_sources": int(h_leadfield.shape[1]),
            "cortical_sources": int(c_leadfield.shape[1]),
            "sensors": sensors,
            "scanner_ras_to_bem_surface_ras": scanner_to_bem.tolist(),
            "watershed_center_ras_mm": center_ras_mm.tolist(),
            "reference_geometry_maximum_absolute_error": maximum_geometry_error,
            "reference_geometry_errors": geometry_errors,
            "depth_definition": "Euclidean distance to nearest ico4 outer-skin mesh vertex",
        },
        "diagnostics": {
            "coherent_covariance_trace": coherent_trace,
            "traveling_wave_covariance_trace": wave_trace,
            "cortical_covariance_eigenvalue_sum": float(c_values.sum()),
            "full_spectrum_identity_relative_error": spectrum_identity_error,
            "cortical_referenced_rank": full_rank,
            "expected_sensor_rank": expected_rank,
        },
        "qc": qc,
        "input_sha256": {
            "bem_report": sha256_file(bem_report_path),
            "bem_marker": sha256_file(bem_marker_path),
            "forward_report": sha256_file(forward_report_path),
            "forward_marker": sha256_file(forward_marker_path),
        },
        "outputs": {
            "recoverability_spectra_sha256": sha256_file(spectra_path),
            "recoverability_spectra_bytes": spectra_path.stat().st_size,
        },
        "physiological_inference_authorized": False,
        "histological_laminar_claim_authorized": False,
        "shared_storage_touched": False,
    }
    report_path = output_directory / "metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _bootstrap_median_ci(
    values: FloatArray, seed: int, replicates: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    medians = np.median(values[draws], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def _summarize(values: FloatArray, seed: int, replicates: int) -> dict[str, float]:
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    low, high = _bootstrap_median_ci(values, seed, replicates)
    return {
        "n": float(len(values)),
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "median_bootstrap_95_ci_low": low,
        "median_bootstrap_95_ci_high": high,
    }


def aggregate_population(
    subjects: list[str],
    cohort_csv: Path,
    subjects_root: Path,
    output_directory: Path,
    config: dict[str, object],
) -> dict[str, object]:
    """Aggregate exactly the frozen 50 subjects and no partial cohort."""

    analysis = dict(config["population_analysis"])
    required = int(analysis["minimum_complete_subjects"])
    if len(subjects) != required or len(set(subjects)) != required:
        raise ValueError("aggregation requires exactly the frozen unique 50 subjects")
    with cohort_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        demographics = {row["Subject"]: row for row in csv.DictReader(stream)}
    if set(demographics) != set(subjects):
        raise ValueError("cohort table and frozen subject list disagree")

    reports: list[dict[str, object]] = []
    for subject in subjects:
        directory = subjects_root / subject
        report_path = directory / "metrics.json"
        spectra_path = directory / "recoverability_spectra.npy"
        marker_path = directory / ".hcp_population_metrics.json"
        report = _json(report_path)
        marker = _json(marker_path)
        if (
            report.get("ok") is not True
            or report.get("subject") != subject
            or report.get("protocol") != config["protocol"]
            or not all(dict(report.get("qc", {})).values())
            or marker.get("status") != "complete"
            or marker.get("subject") != subject
            or marker.get("report_sha256") != sha256_file(report_path)
            or marker.get("spectra_sha256") != sha256_file(spectra_path)
            or dict(report["outputs"])["recoverability_spectra_sha256"]
            != sha256_file(spectra_path)
        ):
            raise ValueError(f"subject {subject} failed aggregate integrity checks")
        reports.append(report)

    output_directory.mkdir(parents=True, exist_ok=False)
    primary_names = list(dict(reports[0]["primary_metrics"]))
    if any(list(dict(report["primary_metrics"])) != primary_names for report in reports):
        raise ValueError("subject primary metric schemas differ")
    subject_csv = output_directory / "subject_metrics.csv"
    with subject_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["subject", "gender", "age", "stratum", *primary_names]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for subject, report in zip(subjects, reports):
            row = demographics[subject]
            writer.writerow(
                {
                    "subject": subject,
                    "gender": row["Gender"],
                    "age": row["Age"],
                    "stratum": row["Stratum"],
                    **dict(report["primary_metrics"]),
                }
            )

    rank_csv = output_directory / "rank_ladder_metrics.csv"
    with rank_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "subject",
            "cortical_rank",
            "retained_cortical_spectral_energy_fraction",
            "source_regime",
            "recoverability_index_bits",
            "spectrum_sum",
            "largest_eigenvalue",
            "participation_ratio",
            "modes_above_relative_1e_3",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for subject, report in zip(subjects, reports):
            for rank in list(report["rank_ladder"]):
                for regime in REGIMES:
                    writer.writerow(
                        {
                            "subject": subject,
                            "cortical_rank": rank["requested_rank"],
                            "retained_cortical_spectral_energy_fraction": rank[
                                "retained_cortical_spectral_energy_fraction"
                            ],
                            "source_regime": regime,
                            **dict(rank["regimes"])[regime],
                        }
                    )

    seed = int(analysis["bootstrap_seed"])
    replicates = int(analysis["bootstrap_replicates"])
    summaries = {
        name: _summarize(
            np.asarray([float(dict(report["primary_metrics"])[name]) for report in reports]),
            seed + index,
            replicates,
        )
        for index, name in enumerate(primary_names)
    }
    correlation_pairs = [
        ("hippocampal_covariance_trace", "hippocampal_depth_median_m"),
        ("coherent_cancellation_ratio", "hippocampal_depth_median_m"),
        (
            "recoverability_index_bits_incoherent_rank338",
            "hippocampal_to_cortical_trace_ratio",
        ),
    ]
    correlations: list[dict[str, object]] = []
    for left, right in correlation_pairs:
        x = np.asarray([float(dict(report["primary_metrics"])[left]) for report in reports])
        y = np.asarray([float(dict(report["primary_metrics"])[right]) for report in reports])
        rho = float(spearmanr(x, y).statistic)
        correlations.append(
            {
                "left": left,
                "right": right,
                "spearman_rho": rho,
                "confirmatory_test": False,
                "p_value_reported": False,
            }
        )

    report = {
        "schema_version": 1,
        "ok": True,
        "status": "complete",
        "protocol": config["protocol"],
        "subjects": subjects,
        "subject_count": len(subjects),
        "population_unit": "subject",
        "scope": "geometry-only numerical forward-model population summary",
        "primary_metric_summaries": summaries,
        "predeclared_descriptive_correlations": correlations,
        "bootstrap": {
            "statistic": "median",
            "method": "percentile",
            "confidence_level": 0.95,
            "seed": seed,
            "replicates": replicates,
        },
        "outputs": {
            "subject_metrics_csv_sha256": sha256_file(subject_csv),
            "subject_metrics_csv_bytes": subject_csv.stat().st_size,
            "rank_ladder_metrics_csv_sha256": sha256_file(rank_csv),
            "rank_ladder_metrics_csv_bytes": rank_csv.stat().st_size,
        },
        "all_subject_qc_passed": True,
        "physiological_inference_authorized": False,
        "histological_laminar_claim_authorized": False,
        "shared_storage_touched": False,
    }
    report_path = output_directory / "population_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
