#!/usr/bin/env python3
"""Independent audit arithmetic for the retained-power normalization contract.

The frozen source-ensemble implementation (``anatomical/hippunfold_ensembles.py``)
reports, for a declared source support ``E``,

    z_E(q) = sum_{i in E} u_i q_i g_i
    D_E    = sum_{i in E} u_i ||g_i||^2 = tr(G_E diag(u) G_E^T)
    R_E(q) = ||z_E(q)||^2 / D_E
    P_E(q) = f_E^2 ||z_E(q)||^2 / D_full

with positive midthickness quadrature areas ``a_i``, ``A_E = sum_{i in E} a_i``,
``u_i = a_i / A_E``, unit phases ``|q_i| = 1``, average-referenced
fixed-orientation unit-dipole lead-field columns ``g_i``, and
``f_E = A_E / A_full``.  Weighted Jensen gives ``0 <= R_E <= 1``.

This module recomputes those quantities from the authoritative lead field and
source metadata with arithmetic written out independently of
``phase_locked_metrics`` and ``patch_coherence_metrics``.  The sensor field is an
explicit weighted quadrature sum, the reference is an explicit weighted second
moment of the per-column squared norms, and no function here calls the two
frozen normalization routines.  Reusing the geometry/support construction and
the seeded random-phase generator is deliberate: the audit is an independent
recomputation of the normalization arithmetic, not an independent forward model.

Two neighbouring quantities are deliberately *not* this contract and appear
only as negative controls in the synthetic checks:

* the generic ratio ``||sum_i w_i g_i||^2 / sum_i |w_i|^2 ||g_i||^2``, which is
  not bounded by one and is not what the implementation reports;
* the discrete independent-phase expectation ``E||z_E||^2 = sum_i u_i^2
  ||g_i||^2`` for independent uniform phases, which is generally smaller than
  ``D_E`` because ``u_i^2 <= u_i``; it equals exactly ``D_E / N`` when the ``N``
  conditional weights are equal (``u_i^2 = u_i / N`` termwise) and reaches
  ``D_E`` only in the degenerate single-effective-source case.

Absolute uniform current-density amplitude cancels in ``R_E`` and ``P_E``.
Dentate is excluded from the primary source model upstream and is not
reintroduced here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anatomical.hippunfold_ensembles import (  # noqa: E402
    SourceSupport,
    build_source_supports,
    oriented_intrinsic_coordinates,
    random_correlated_phases,
    support_identifier,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ComplexArray = NDArray[np.complex128]

PROTOCOL = "hcp/source-ensembles-v1.1"
FORMULA_VERSION = "retained-power-normalization/1"
DEFAULT_RELATIVE_TOLERANCE = 1e-9
DEFAULT_ABSOLUTE_TOLERANCE = 1e-12
PHASE_UNIT_TOLERANCE = 1e-12
DENOMINATOR_TOLERANCE = 1e-300

METADATA_ARRAYS = (
    "positions_m",
    "directions",
    "area_weights_m2",
    "longitudinal_coordinate",
    "proximal_distal_coordinate",
    "hemisphere_code",
)

# Saved normalization fields recomputed for every audited row.
NORMALIZATION_FIELDS = (
    "full_sheet_area_fraction",
    "support_incoherent_trace",
    "support_normalized_retained_power_ratio",
    "support_normalized_retained_rms_ratio",
    "whole_sheet_incoherent_power_ratio",
)

# Saved fields that are not part of the retained-power normalization contract.
NOT_RECOMPUTED_COLUMNS = (
    "whole_sheet_recoverability_index_bits",
    "whole_sheet_largest_eigenvalue",
    "whole_sheet_whitened_signal_power",
    "whole_sheet_participation_ratio",
    "whole_sheet_numerical_rank",
    "equal_support_recoverability_index_bits",
    "equal_support_largest_eigenvalue",
    "equal_support_whitened_signal_power",
    "equal_support_participation_ratio",
    "equal_support_numerical_rank",
)


class AuditError(RuntimeError):
    """Raised when an audited artifact violates the normalization contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Report-declared input resolution
# ---------------------------------------------------------------------------


# Container mount prefix used by the frozen conductor image.  Reports produced
# inside the container declare absolute POSIX paths rooted at ``/forward``.
CONTAINER_FORWARD_PREFIX = "/forward/"


@dataclass(frozen=True)
class DeclaredInputResolution:
    """Host resolution of one report-declared input path.

    ``declared`` is the path exactly as written in ``report.json`` (for example
    ``/forward/<subject>/hippunfold-hippocampal-fixed-leadfield.npy``), while
    ``host_path`` is the authoritative file the auditor actually hashes.  The
    two differ whenever the report was written from inside the conductor
    container.
    """

    declared: str
    host_path: Path
    resolved_from_container: bool


def resolve_declared_input(
    subject: str, declared_path: object, forward_root: Path
) -> DeclaredInputResolution:
    """Resolve a report-declared input path to the auditor's host path.

    Per-subject reports are written inside the conductor container and declare
    container paths such as
    ``/forward/<subject>/hippunfold-hippocampal-fixed-leadfield.npy``.  Those
    paths do not exist on the audit host, so a declared path rooted at
    ``/forward/`` is resolved to ``<forward_root>/<subject>/<basename>`` using
    the audited subject identity, not the raw container string.  Any other
    declared path (for example a host-absolute path written by a local fixture)
    is returned verbatim so existing host-generated artifacts keep working.
    """

    declared = str(declared_path)
    posix = declared.replace("\\", "/")
    if posix.startswith(CONTAINER_FORWARD_PREFIX):
        relative = posix[len(CONTAINER_FORWARD_PREFIX) :]
        basename = relative.rsplit("/", 1)[-1]
        if not basename:
            raise AuditError(f"declared container path has no file name: {declared}")
        return DeclaredInputResolution(
            declared=declared,
            host_path=Path(forward_root) / subject / basename,
            resolved_from_container=True,
        )
    return DeclaredInputResolution(
        declared=declared,
        host_path=Path(declared),
        resolved_from_container=False,
    )


# ---------------------------------------------------------------------------
# Independent normalization arithmetic
# ---------------------------------------------------------------------------


def average_reference(leadfield: ArrayLike) -> FloatArray:
    """Return average-referenced lead-field columns.

    The frozen pipeline references through an orthonormal Helmert contrast
    ``H`` with ``H H^T = I`` and ``H 1 = 0``.  For any sensor vector ``v`` the
    identity ``H v = H P v`` with ``P = I - 11^T/m`` holds, so centering each
    column removes the common-average component and preserves every norm used
    by the retained-power ratios.
    """

    matrix = np.asarray(leadfield, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise AuditError("lead field must be a nonempty sensor-by-source matrix")
    if not np.all(np.isfinite(matrix)):
        raise AuditError("lead field contains non-finite entries")
    return matrix - np.mean(matrix, axis=0, keepdims=True)


def _as_columns(columns: ArrayLike, label: str = "columns") -> FloatArray:
    matrix = np.asarray(columns, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise AuditError(f"{label} must be a nonempty sensor-by-source matrix")
    if not np.all(np.isfinite(matrix)):
        raise AuditError(f"{label} contains non-finite entries")
    return matrix


def _as_weights(weights: ArrayLike, count: int, label: str = "weights") -> FloatArray:
    """Validate a positive area vector and return its conditional weights.

    Renormalizing here is what makes the ratios invariant to a uniform rescaling
    of raw quadrature areas, which the synthetic checks assert.
    """

    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value.size != count:
        raise AuditError(f"{label} length {value.size} does not match {count} columns")
    if not np.all(np.isfinite(value)):
        raise AuditError(f"{label} contains non-finite entries")
    if np.any(value <= 0.0):
        raise AuditError(f"{label} contains a non-positive quadrature area")
    total = float(np.sum(value))
    if total <= 0.0:
        raise AuditError(f"{label} has a non-positive total")
    return value / total


def _as_unit_phase(
    unit_phase: ArrayLike, count: int, label: str = "phase"
) -> ComplexArray:
    value = np.asarray(unit_phase, dtype=np.complex128).reshape(-1)
    if value.size != count:
        raise AuditError(f"{label} length {value.size} does not match {count} columns")
    if not np.all(np.isfinite(value)):
        raise AuditError(f"{label} contains non-finite entries")
    magnitude_error = float(np.max(np.abs(np.abs(value) - 1.0))) if value.size else 0.0
    if magnitude_error > PHASE_UNIT_TOLERANCE:
        raise AuditError(
            f"{label} is not unit magnitude (maximum error {magnitude_error:g})"
        )
    return value


def unit_phase_from_angles(angles_rad: ArrayLike) -> ComplexArray:
    """Return ``exp(i angle)``, preserving the input shape."""

    angles = np.asarray(angles_rad, dtype=np.float64)
    if not np.all(np.isfinite(angles)):
        raise AuditError("phase angles contain non-finite entries")
    return np.exp(1j * angles)


def phase_unit_error(unit_phase: ArrayLike) -> float:
    value = np.asarray(unit_phase, dtype=np.complex128).reshape(-1)
    if value.size == 0:
        return 0.0
    return float(np.max(np.abs(np.abs(value) - 1.0)))


def retained_power(
    columns: ArrayLike, weights: ArrayLike, unit_phase: ArrayLike
) -> tuple[float, float, float]:
    """Return ``(||z_E||^2, D_E, R_E)`` by explicit weighted quadrature.

    ``columns`` are the average-referenced lead-field columns ``g_i`` for the
    support, ``weights`` the conditional areas ``u_i`` and ``unit_phase`` the
    unit-modulus phases ``q_i``.
    """

    leadfield = _as_columns(columns)
    weight = _as_weights(weights, leadfield.shape[1])
    phase = _as_unit_phase(unit_phase, leadfield.shape[1])

    # z_E = sum_i u_i q_i g_i, accumulated one column at a time.
    quadrature = weight * phase
    accumulated = np.sum(leadfield * quadrature[None, :], axis=1)
    z_power = float(
        np.dot(accumulated.real, accumulated.real)
        + np.dot(accumulated.imag, accumulated.imag)
    )

    # D_E = sum_i u_i ||g_i||^2, the area-weighted second moment of the
    # individual source-field powers.
    column_power = np.sum(leadfield * leadfield, axis=0)
    incoherent_trace = float(np.dot(weight, column_power))
    if incoherent_trace <= DENOMINATOR_TOLERANCE:
        raise AuditError("incoherent reference trace is non-positive")
    return z_power, incoherent_trace, z_power / incoherent_trace


def independent_phase_power(columns: ArrayLike, weights: ArrayLike) -> float:
    """Return ``sum_i u_i^2 ||g_i||^2``.

    This is the expected sensor power for a fixed-magnitude ensemble with
    independent uniform phases.  It never exceeds ``D_E`` and equals exactly
    ``D_E / N`` when the ``N`` conditional weights are equal, so it must not be
    substituted for the area-weighted incoherent reference.
    """

    leadfield = _as_columns(columns)
    weight = _as_weights(weights, leadfield.shape[1])
    column_power = np.sum(leadfield * leadfield, axis=0)
    return float(np.dot(weight * weight, column_power))


def whole_sheet_power(
    z_power: float, area_fraction: float, full_sheet_trace: float
) -> float:
    """Return ``P_E = f_E^2 ||z_E||^2 / D_full``."""

    if not np.isfinite(area_fraction) or not 0.0 < area_fraction <= 1.0:
        raise AuditError("full-sheet area fraction must lie in (0, 1]")
    if not np.isfinite(full_sheet_trace) or full_sheet_trace <= DENOMINATOR_TOLERANCE:
        raise AuditError("full-sheet incoherent trace is non-positive")
    if not np.isfinite(z_power) or z_power < 0.0:
        raise AuditError("sensor power must be a nonnegative finite scalar")
    return float(area_fraction) ** 2 * float(z_power) / float(full_sheet_trace)


def support_weights(areas: ArrayLike, indices: ArrayLike) -> FloatArray:
    """Return conditional quadrature weights ``u_i = a_i / A_E``."""

    area = np.asarray(areas, dtype=np.float64).reshape(-1)
    index = np.asarray(indices, dtype=np.int64).reshape(-1)
    if area.size == 0 or not np.all(np.isfinite(area)):
        raise AuditError("areas must be a nonempty finite vector")
    if np.any(area <= 0.0):
        raise AuditError("areas must be strictly positive")
    if index.size == 0:
        raise AuditError("support is empty")
    if np.any(index < 0) or np.any(index >= area.size):
        raise AuditError("support index is out of range")
    selected = area[index]
    return selected / float(np.sum(selected))


def incoherent_trace(columns: ArrayLike, weights: ArrayLike) -> float:
    """Return ``D = sum_i u_i ||g_i||^2`` for the declared columns."""

    leadfield = _as_columns(columns)
    weight = _as_weights(weights, leadfield.shape[1])
    column_power = np.sum(leadfield * leadfield, axis=0)
    return float(np.dot(weight, column_power))


def coherent_patch_index(
    anterior_coordinate: ArrayLike,
    proximal_distal_coordinate: ArrayLike,
    hemisphere_code: ArrayLike,
    intrinsic_bin_width: float,
) -> tuple[IntArray, int]:
    """Group sources into independent intrinsic AP/PD patches.

    Sources share a patch when hemisphere, anterior--posterior bin and
    proximal--distal bin agree.  Indices are returned in first-appearance
    order.
    """

    if not 0.0 < float(intrinsic_bin_width) <= 1.0:
        raise AuditError("intrinsic bin width must lie in (0, 1]")
    anterior = np.asarray(anterior_coordinate, dtype=np.float64).reshape(-1)
    proximal_distal = np.asarray(proximal_distal_coordinate, dtype=np.float64).reshape(-1)
    hemispheres = np.asarray(hemisphere_code, dtype=np.int64).reshape(-1)
    if not (anterior.size == proximal_distal.size == hemispheres.size):
        raise AuditError("intrinsic coordinates are misaligned")
    if anterior.size == 0 or not all(
        np.all(np.isfinite(value)) for value in (anterior, proximal_distal)
    ):
        raise AuditError("intrinsic coordinates must be finite and nonempty")
    if np.any((anterior < 0.0) | (anterior > 1.0)):
        raise AuditError("anterior coordinate must lie in [0, 1]")

    width = float(intrinsic_bin_width)
    upper = int(math.ceil(1.0 / width)) - 1
    ap_bin = np.minimum(
        np.floor(anterior / width).astype(np.int64), upper
    )
    pd_bin = np.minimum(
        np.floor(proximal_distal / width).astype(np.int64), upper
    )
    lookup: dict[tuple[int, int, int], int] = {}
    group = np.empty(anterior.size, dtype=np.int64)
    for position in range(anterior.size):
        key = (
            int(hemispheres[position]),
            int(ap_bin[position]),
            int(pd_bin[position]),
        )
        if key not in lookup:
            lookup[key] = len(lookup)
        group[position] = lookup[key]
    return group, len(lookup)


def coherent_patch_factor(
    columns: ArrayLike,
    weights: ArrayLike,
    group: ArrayLike,
    group_count: int,
) -> tuple[FloatArray, FloatArray]:
    """Return the patch quadrature factor and the per-patch area weights."""

    leadfield = _as_columns(columns)
    weight = _as_weights(weights, leadfield.shape[1])
    index = np.asarray(group, dtype=np.int64).reshape(-1)
    if index.size != leadfield.shape[1]:
        raise AuditError("patch assignment does not match lead-field columns")
    if group_count < 1 or np.any(index < 0) or np.any(index >= group_count):
        raise AuditError("patch assignment is out of range")
    factor = np.zeros((leadfield.shape[0], int(group_count)), dtype=np.float64)
    for column in range(leadfield.shape[1]):
        # Each internally coherent patch contributes the area-weighted sum of
        # its member columns.
        factor[:, int(index[column])] += weight[column] * leadfield[:, column]
    patch_area = np.zeros(int(group_count), dtype=np.float64)
    for column in range(leadfield.shape[1]):
        patch_area[int(index[column])] += weight[column]
    return factor, patch_area


def factor_power(factor: ArrayLike) -> float:
    matrix = np.asarray(factor, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise AuditError("patch factor must be a finite matrix")
    return float(np.sum(matrix * matrix))


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_metadata(path: Path) -> dict[str, NDArray]:
    if not path.is_file():
        raise AuditError(f"missing source metadata: {path}")
    with np.load(path, allow_pickle=False) as archive:
        absent = sorted(set(METADATA_ARRAYS).difference(archive.files))
        if absent:
            raise AuditError(f"{path} is missing arrays: {absent}")
        metadata = {name: np.asarray(archive[name]) for name in METADATA_ARRAYS}
    for name, value in metadata.items():
        if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
            raise AuditError(f"{path} array {name} contains non-finite entries")
    areas = np.asarray(metadata["area_weights_m2"], dtype=np.float64).reshape(-1)
    if areas.size == 0 or np.any(areas <= 0.0):
        raise AuditError(f"{path} contains non-positive quadrature areas")
    hemispheres = np.asarray(metadata["hemisphere_code"]).reshape(-1)
    if set(np.unique(hemispheres).tolist()) != {-1, 1}:
        raise AuditError(f"{path} hemisphere codes must be exactly -1 and 1")
    return metadata


def load_fixed_leadfield(path: Path, sensors: int, sources: int) -> FloatArray:
    if not path.is_file():
        raise AuditError(f"missing fixed lead field: {path}")
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (sensors, sources):
        raise AuditError(
            f"fixed lead field shape {matrix.shape} does not match {(sensors, sources)}"
        )
    if not np.all(np.isfinite(matrix)):
        raise AuditError("fixed lead field contains non-finite entries")
    return np.asarray(matrix, dtype=np.float64)


def reconstruct_supports(
    metadata: Mapping[str, NDArray], config: Mapping[str, object]
) -> tuple[FloatArray, FloatArray, dict[str, SourceSupport]]:
    """Rebuild the frozen support ladder from authoritative metadata."""

    support_config = dict(config["support"])  # type: ignore[arg-type]
    anterior, proximal_distal, _ = oriented_intrinsic_coordinates(
        metadata["positions_m"],
        metadata["longitudinal_coordinate"],
        metadata["proximal_distal_coordinate"],
        metadata["hemisphere_code"],
        metadata["area_weights_m2"],
        endpoint_decile=float(support_config["endpoint_decile"]),
        minimum_endpoint_separation_m=float(
            support_config["minimum_endpoint_separation_m"]
        ),
    )
    supports, _ = build_source_supports(
        anterior,
        proximal_distal,
        metadata["hemisphere_code"],
        metadata["area_weights_m2"],
        hemisphere_families=support_config["hemisphere_families"],
        focal_locations=support_config["focal_locations"],
        focal_extents=support_config["focal_extents"],
        include_whole_extent=bool(support_config["include_whole_extent"]),
    )
    return anterior, proximal_distal, supports


def expected_row_counts(config: Mapping[str, object]) -> dict[str, int]:
    support_config = dict(config["support"])  # type: ignore[arg-type]
    random_config = dict(config["random_phase"])  # type: ignore[arg-type]
    coherence_config = dict(config["patch_coherence"])  # type: ignore[arg-type]
    deterministic_config = dict(config["deterministic_phase"])  # type: ignore[arg-type]
    support_count = len(support_config["hemisphere_families"]) * (
        len(support_config["focal_locations"]) * len(support_config["focal_extents"])
        + int(bool(support_config["include_whole_extent"]))
    )
    return {
        "supports": support_count,
        "deterministic": support_count * (1 + len(deterministic_config["wave_cycles"])),
        "patch_coherence": support_count * len(coherence_config["intrinsic_bin_widths"]),
        "random_phase": len(random_config["supports"])
        * len(random_config["intrinsic_correlation_lengths"])
        * int(random_config["replicates"]),
    }


def regime_names(config: Mapping[str, object]) -> list[tuple[str, float]]:
    cycles = list(dict(config["deterministic_phase"])["wave_cycles"])  # type: ignore[arg-type]
    return [("zero_phase", 0.0)] + [
        (f"wave_{float(cycle):g}_cycles", float(cycle)) for cycle in cycles
    ]


def canonical_support_identifier(row: Mapping[str, object]) -> str:
    return support_identifier(
        str(row["hemisphere"]), str(row["location"]), float(row["extent"])
    )


@dataclass(frozen=True)
class SubjectGeometry:
    """Average-referenced geometry and supports for one subject."""

    subject: str
    sensors: int
    sources: int
    areas: FloatArray
    columns: FloatArray
    anterior: FloatArray
    proximal_distal: FloatArray
    hemisphere_code: IntArray
    supports: Mapping[str, SourceSupport]
    full_sheet_trace: float
    support_weights_by_id: Mapping[str, FloatArray]
    full_sheet_area_fraction_by_id: Mapping[str, float]


def build_subject_geometry(
    subject: str,
    metadata: Mapping[str, NDArray],
    leadfield: ArrayLike,
    config: Mapping[str, object],
) -> SubjectGeometry:
    areas = np.asarray(metadata["area_weights_m2"], dtype=np.float64).reshape(-1)
    columns = average_reference(leadfield)
    if columns.shape[1] != areas.size:
        raise AuditError("lead field columns and quadrature areas are misaligned")
    anterior, proximal_distal, supports = reconstruct_supports(metadata, config)
    hemispheres = np.asarray(metadata["hemisphere_code"], dtype=np.int64).reshape(-1)
    full_weights = areas / float(np.sum(areas))
    full_trace = incoherent_trace(columns, full_weights)
    weights_by_id: dict[str, FloatArray] = {}
    fraction_by_id: dict[str, float] = {}
    for identifier, support in supports.items():
        weight = support_weights(areas, support.indices)
        if not np.allclose(weight, support.conditional_weights, rtol=0.0, atol=1e-14):
            raise AuditError(f"{identifier}: reconstructed weights disagree with support")
        weights_by_id[identifier] = weight
        fraction_by_id[identifier] = float(np.sum(areas[support.indices])) / float(
            np.sum(areas)
        )
    return SubjectGeometry(
        subject=subject,
        sensors=int(columns.shape[0]),
        sources=int(columns.shape[1]),
        areas=areas,
        columns=columns,
        anterior=anterior,
        proximal_distal=proximal_distal,
        hemisphere_code=hemispheres,
        supports=supports,
        full_sheet_trace=full_trace,
        support_weights_by_id=weights_by_id,
        full_sheet_area_fraction_by_id=fraction_by_id,
    )


# ---------------------------------------------------------------------------
# Comparison ledger
# ---------------------------------------------------------------------------


@dataclass
class FieldLedger:
    """Accumulate absolute/relative error statistics for one saved field."""

    relative_tolerance: float
    absolute_tolerance: float
    count: int = 0
    maximum_absolute_error: float = 0.0
    maximum_relative_error: float = 0.0
    violations: list[str] = field(default_factory=list)

    def record(self, label: str, actual: float, expected: float) -> bool:
        actual_value = float(actual)
        expected_value = float(expected)
        absolute_error = abs(actual_value - expected_value)
        relative_error = absolute_error / max(
            abs(expected_value), np.finfo(float).tiny
        )
        self.count += 1
        self.maximum_absolute_error = max(self.maximum_absolute_error, absolute_error)
        self.maximum_relative_error = max(self.maximum_relative_error, relative_error)
        limit = self.absolute_tolerance + self.relative_tolerance * abs(expected_value)
        if not absolute_error <= limit:
            self.violations.append(
                f"{label}: recomputed {expected_value!r} != saved {actual_value!r}"
            )
            return False
        return True

    def as_record(self, maximum_violations: int = 20) -> dict[str, object]:
        return {
            "count": self.count,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "violation_count": len(self.violations),
            "violations": self.violations[:maximum_violations],
        }


# ---------------------------------------------------------------------------
# Per-subject recomputation
# ---------------------------------------------------------------------------


@dataclass
class SubjectAudit:
    subject: str
    sensors: int
    sources: int
    support_count: int
    deterministic_rows: int
    patch_coherence_rows: int
    random_phase_rows: int
    maximum_absolute_error: float
    maximum_relative_error: float
    maximum_bound_excess: float
    status: str
    error_count: int
    errors: list[str]

    def as_row(self) -> dict[str, object]:
        row = dict(self.__dict__)
        row["errors"] = "; ".join(self.errors)
        return row


class SubjectRecorder:
    """Collect per-subject recomputation statistics against saved rows."""

    def __init__(
        self,
        geometry: SubjectGeometry,
        config: Mapping[str, object],
        relative_tolerance: float,
        absolute_tolerance: float,
        ledgers: Mapping[str, FieldLedger],
    ) -> None:
        self.geometry = geometry
        self.config = config
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.ledgers = ledgers
        self.errors: list[str] = []
        self.deterministic_rows = 0
        self.patch_rows = 0
        self.random_rows = 0
        self.maximum_bound_excess = 0.0
        self.maximum_absolute_error = 0.0
        self.maximum_relative_error = 0.0
        self.bound_tolerance = float(config.get("maximum_numerical_bound_error", 1e-10))

    # -- helpers ---------------------------------------------------------

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def record(self, field_name: str, label: str, actual: float, expected: float) -> bool:
        """Record one saved-versus-recomputed comparison."""

        ledger = self.ledgers[field_name]
        accepted = ledger.record(label, actual, expected)
        self.maximum_absolute_error = max(
            self.maximum_absolute_error, ledger.maximum_absolute_error
        )
        self.maximum_relative_error = max(
            self.maximum_relative_error, ledger.maximum_relative_error
        )
        if not accepted:
            self.fail(
                f"{label}: saved {actual!r} != recomputed {expected!r} outside the "
                f"{self.relative_tolerance:g}/{self.absolute_tolerance:g} tolerance"
            )
        return accepted

    def compare_fields(
        self,
        prefix: str,
        row: Mapping[str, str],
        z_power: float,
        incoherent: float,
        ratio: float,
        area_fraction: float,
    ) -> None:
        expected_values = {
            "full_sheet_area_fraction": area_fraction,
            "support_incoherent_trace": incoherent,
            "support_normalized_retained_power_ratio": ratio,
            "support_normalized_retained_rms_ratio": math.sqrt(max(ratio, 0.0)),
            "whole_sheet_incoherent_power_ratio": whole_sheet_power(
                z_power, area_fraction, self.geometry.full_sheet_trace
            ),
        }
        for name in NORMALIZATION_FIELDS:
            if name not in row:
                self.fail(f"{prefix}: saved row lacks field {name}")
                continue
            try:
                saved = float(row[name])
            except (TypeError, ValueError):
                self.fail(f"{prefix}: saved field {name} is not numeric")
                continue
            self.record(name, f"{prefix}:{name}", saved, expected_values[name])

        bound_excess = max(0.0, ratio - 1.0)
        self.maximum_bound_excess = max(self.maximum_bound_excess, bound_excess)
        if ratio > 1.0 + self.bound_tolerance:
            self.fail(
                f"{prefix}: retained ratio {ratio!r} exceeds the weighted-Jensen bound"
            )
        if ratio < -self.bound_tolerance:
            self.fail(f"{prefix}: retained ratio {ratio!r} is negative")
        if "analytical_bound_excess" in row:
            self.record(
                "analytical_bound_excess",
                f"{prefix}:analytical_bound_excess",
                float(row["analytical_bound_excess"]),
                bound_excess,
            )

    def check_support_identity(
        self, prefix: str, row: Mapping[str, str], support: SourceSupport
    ) -> None:
        if "source_count" in row and int(float(row["source_count"])) != len(
            support.indices
        ):
            self.fail(f"{prefix}: saved source_count disagrees with the support")

    # -- row families ----------------------------------------------------

    def audit_deterministic(self, rows: Sequence[Mapping[str, str]]) -> None:
        expected = {
            (identifier, regime): cycles
            for identifier in self.geometry.supports
            for regime, cycles in regime_names(self.config)
        }
        seen: set[tuple[str, str]] = set()
        for row in rows:
            identifier = str(row.get("support_id", ""))
            regime = str(row.get("regime", ""))
            key = (identifier, regime)
            prefix = f"deterministic[{identifier}/{regime}]"
            if key not in expected:
                self.fail(f"{prefix}: unexpected saved row")
                continue
            if key in seen:
                self.fail(f"{prefix}: duplicate saved row")
                continue
            seen.add(key)
            support = self.geometry.supports[identifier]
            weights = self.geometry.support_weights_by_id[identifier]
            fraction = self.geometry.full_sheet_area_fraction_by_id[identifier]
            cycles = expected[key]
            phase = unit_phase_from_angles(
                (2.0 * math.pi * cycles) * self.geometry.anterior[support.indices]
            )
            z_power, incoherent, ratio = retained_power(
                self.geometry.columns[:, support.indices], weights, phase
            )
            self.check_support_identity(prefix, row, support)
            self.compare_fields(prefix, row, z_power, incoherent, ratio, fraction)
            self.deterministic_rows += 1
        missing = sorted(set(expected).difference(seen))
        if missing:
            self.fail(f"deterministic: {len(missing)} expected rows are missing")

    def audit_patch_coherence(self, rows: Sequence[Mapping[str, str]]) -> None:
        widths = [float(value) for value in dict(self.config["patch_coherence"])["intrinsic_bin_widths"]]  # type: ignore[arg-type]
        expected = {
            (identifier, width) for identifier in self.geometry.supports for width in widths
        }
        seen: set[tuple[str, float]] = set()
        for row in rows:
            identifier = str(row.get("support_id", ""))
            try:
                width = float(row.get("intrinsic_bin_width", "nan"))
            except (TypeError, ValueError):
                self.fail(f"patch_coherence[{identifier}]: non-numeric bin width")
                continue
            key = (identifier, width)
            prefix = f"patch_coherence[{identifier}/{width:g}]"
            if key not in expected:
                self.fail(f"{prefix}: unexpected saved row")
                continue
            if key in seen:
                self.fail(f"{prefix}: duplicate saved row")
                continue
            seen.add(key)
            support = self.geometry.supports[identifier]
            weights = self.geometry.support_weights_by_id[identifier]
            fraction = self.geometry.full_sheet_area_fraction_by_id[identifier]
            selected = support.indices
            group, group_count = coherent_patch_index(
                self.geometry.anterior[selected],
                self.geometry.proximal_distal[selected],
                self.geometry.hemisphere_code[selected],
                width,
            )
            factor, patch_area = coherent_patch_factor(
                self.geometry.columns[:, selected], weights, group, group_count
            )
            incoherent = incoherent_trace(self.geometry.columns[:, selected], weights)
            z_power = factor_power(factor)
            ratio = z_power / incoherent
            self.check_support_identity(prefix, row, support)
            self.compare_fields(prefix, row, z_power, incoherent, ratio, fraction)
            if "coherent_patch_count" in row and int(
                float(row["coherent_patch_count"])
            ) != group_count:
                self.fail(f"{prefix}: saved coherent patch count disagrees")
            if "coherent_patch_area_minimum" in row:
                self.record(
                    "coherent_patch_area_minimum",
                    f"{prefix}:coherent_patch_area_minimum",
                    float(row["coherent_patch_area_minimum"]),
                    float(np.min(patch_area)),
                )
            if "coherent_patch_area_maximum" in row:
                self.record(
                    "coherent_patch_area_maximum",
                    f"{prefix}:coherent_patch_area_maximum",
                    float(row["coherent_patch_area_maximum"]),
                    float(np.max(patch_area)),
                )
            if "coherent_patch_area_participation_ratio" in row:
                self.record(
                    "coherent_patch_area_participation_ratio",
                    f"{prefix}:coherent_patch_area_participation_ratio",
                    float(row["coherent_patch_area_participation_ratio"]),
                    1.0 / float(np.sum(patch_area * patch_area)),
                )
            self.patch_rows += 1
        missing = sorted(set(expected).difference(seen))
        if missing:
            self.fail(f"patch_coherence: {len(missing)} expected rows are missing")

    def audit_random_phase(self, rows: Sequence[Mapping[str, str]]) -> None:
        random_config = dict(self.config["random_phase"])  # type: ignore[arg-type]
        lengths = [float(value) for value in random_config["intrinsic_correlation_lengths"]]
        replicates = int(random_config["replicates"])
        supports = list(random_config["supports"])
        expected = {
            (canonical_support_identifier(item), length, replicate)
            for item in supports
            for length in lengths
            for replicate in range(replicates)
        }
        seen: set[tuple[str, float, int]] = set()
        features = int(random_config["random_fourier_features"])
        phase_std = float(random_config["marginal_phase_standard_deviation_rad"])
        master_seed = int(random_config["master_seed"])
        for row in rows:
            identifier = str(row.get("support_id", ""))
            try:
                length = float(row.get("intrinsic_correlation_length", "nan"))
                replicate = int(float(row.get("replicate", "nan")))
            except (TypeError, ValueError):
                self.fail(f"random_phase[{identifier}]: non-numeric condition")
                continue
            key = (identifier, length, replicate)
            prefix = f"random_phase[{identifier}/{length:g}/{replicate}]"
            if key not in expected:
                self.fail(f"{prefix}: unexpected saved row")
                continue
            if key in seen:
                self.fail(f"{prefix}: duplicate saved row")
                continue
            seen.add(key)
            support = self.geometry.supports.get(identifier)
            if support is None:
                self.fail(f"{prefix}: unknown support")
                continue
            weights = self.geometry.support_weights_by_id[identifier]
            fraction = self.geometry.full_sheet_area_fraction_by_id[identifier]
            selected = support.indices
            phase, phase_report = random_correlated_phases(
                self.geometry.anterior[selected],
                self.geometry.proximal_distal[selected],
                self.geometry.hemisphere_code[selected],
                weights,
                correlation_length=length,
                features=features,
                phase_standard_deviation_rad=phase_std,
                master_seed=master_seed,
                seed_components=(self.geometry.subject, identifier, length, replicate),
            )
            unit_error = phase_unit_error(np.exp(1j * phase))
            if unit_error > PHASE_UNIT_TOLERANCE:
                self.fail(f"{prefix}: regenerated phase is not unit magnitude")
            manifest = row.get("phase_seed_manifest_json")
            if manifest is not None:
                try:
                    saved_manifest = json.loads(manifest)
                except json.JSONDecodeError:
                    self.fail(f"{prefix}: saved phase seed manifest is not JSON")
                else:
                    if saved_manifest != phase_report["hemispheres"]:
                        self.fail(f"{prefix}: saved phase seed manifest disagrees")
            z_power, incoherent, ratio = retained_power(
                self.geometry.columns[:, selected],
                weights,
                unit_phase_from_angles(phase),
            )
            self.check_support_identity(prefix, row, support)
            self.compare_fields(prefix, row, z_power, incoherent, ratio, fraction)
            self.random_rows += 1
        missing = sorted(set(expected).difference(seen))
        if missing:
            self.fail(f"random_phase: {len(missing)} expected rows are missing")

    def result(self) -> SubjectAudit:
        return SubjectAudit(
            subject=self.geometry.subject,
            sensors=self.geometry.sensors,
            sources=self.geometry.sources,
            support_count=len(self.geometry.supports),
            deterministic_rows=self.deterministic_rows,
            patch_coherence_rows=self.patch_rows,
            random_phase_rows=self.random_rows,
            maximum_absolute_error=self.maximum_absolute_error,
            maximum_relative_error=self.maximum_relative_error,
            maximum_bound_excess=self.maximum_bound_excess,
            status="passed" if not self.errors else "failed",
            error_count=len(self.errors),
            errors=self.errors,
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"missing ensemble table: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise AuditError(f"ensemble table has no header: {path}")
        return [dict(row) for row in reader]


# ---------------------------------------------------------------------------
# Synthetic identity checks
# ---------------------------------------------------------------------------


def _identity_record(
    name: str, passed: bool, detail: Mapping[str, object]
) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": dict(detail)}


def synthetic_identity_checks(seed: int = 20260921) -> list[dict[str, object]]:
    """Run the normalization identities the audit relies on.

    Every check is a small deterministic computation on synthetic columns; no
    subject data are required.
    """

    generator = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    tolerance = 1e-12

    # 1. Aligned identical columns with arbitrary normalized areas reach R = 1.
    identical = np.tile(
        generator.normal(size=(9, 1)), (1, 6)
    )
    areas = generator.uniform(0.5, 2.0, size=6)
    weights = areas / np.sum(areas)
    z_power, incoherent, ratio = retained_power(
        identical, weights, np.ones(6, dtype=np.complex128)
    )
    records.append(
        _identity_record(
            "aligned_identical_columns_unit_ratio",
            abs(ratio - 1.0) <= tolerance,
            {"ratio": ratio, "z_power": z_power, "incoherent_trace": incoherent},
        )
    )

    # 2. Exact cancellation of opposing aligned columns reaches R = 0.
    opposing = np.column_stack(
        (identical[:, 0], identical[:, 0], -identical[:, 0], -identical[:, 0])
    )
    z_power, incoherent, ratio = retained_power(
        opposing, np.full(4, 0.25), np.ones(4, dtype=np.complex128)
    )
    records.append(
        _identity_record(
            "opposing_columns_exact_cancellation",
            abs(ratio) <= tolerance,
            {"ratio": ratio, "z_power": z_power, "incoherent_trace": incoherent},
        )
    )

    # 3. Global lead-field scaling cancels in R and P.
    base_columns = generator.normal(size=(8, 5))
    base_phase = unit_phase_from_angles(generator.uniform(0, 2 * math.pi, size=5))
    base_weights = generator.uniform(0.5, 2.0, size=5)
    base_weights = base_weights / np.sum(base_weights)
    _, base_trace, base_ratio = retained_power(base_columns, base_weights, base_phase)
    base_full_trace = incoherent_trace(base_columns, np.full(5, 0.2))
    base_whole = whole_sheet_power(base_trace * base_ratio, 0.5, base_full_trace)
    scale = 37.5
    scaled_z, scaled_trace, scaled_ratio = retained_power(
        scale * base_columns, base_weights, base_phase
    )
    scaled_whole = whole_sheet_power(scaled_z, 0.5, scale**2 * base_full_trace)
    records.append(
        _identity_record(
            "leadfield_scale_invariance",
            abs(scaled_ratio - base_ratio) <= tolerance
            and abs(scaled_trace - scale**2 * base_trace) <= tolerance * base_trace
            and abs(scaled_whole - base_whole) <= 1e-12 * max(base_whole, 1e-12),
            {
                "base_ratio": base_ratio,
                "scaled_ratio": scaled_ratio,
                "trace_scale_ratio": scaled_trace / base_trace,
                "base_whole_sheet_power": base_whole,
                "scaled_whole_sheet_power": scaled_whole,
            },
        )
    )

    # 4. Uniform raw-area rescaling leaves u, f, R and P unchanged.
    raw_areas = generator.uniform(0.4, 3.0, size=5)
    uniform_z, uniform_trace, uniform_ratio = retained_power(
        base_columns, raw_areas, base_phase
    )
    rescaled_z, rescaled_trace, rescaled_ratio = retained_power(
        base_columns, 4.75 * raw_areas, base_phase
    )
    records.append(
        _identity_record(
            "uniform_area_scale_invariance",
            abs(rescaled_ratio - uniform_ratio) <= tolerance
            and abs(rescaled_trace - uniform_trace) <= tolerance * uniform_trace
            and abs(rescaled_z - uniform_z) <= tolerance * max(uniform_z, 1e-12),
            {"ratio": uniform_ratio, "rescaled_ratio": rescaled_ratio},
        )
    )

    # 5. Splitting a column into identical halves with split areas is inert.
    first_column = base_columns[:, 0][:, None]
    split_columns = np.column_stack((first_column, first_column, base_columns[:, 1:]))
    split_areas = np.concatenate(([raw_areas[0] / 2.0, raw_areas[0] / 2.0], raw_areas[1:]))
    split_phase = np.concatenate(([base_phase[0], base_phase[0]], base_phase[1:]))
    split_z, split_trace, split_ratio = retained_power(
        split_columns, split_areas, split_phase
    )
    records.append(
        _identity_record(
            "mesh_split_of_identical_columns",
            abs(split_ratio - uniform_ratio) <= tolerance
            and abs(split_trace - uniform_trace) <= tolerance * uniform_trace
            and abs(split_z - uniform_z) <= tolerance * max(uniform_z, 1e-12),
            {"baseline_ratio": uniform_ratio, "split_ratio": split_ratio},
        )
    )

    # 6. Weighted Jensen bound on adversarial random columns and phases.
    worst_ratio = 0.0
    negative = False
    for _ in range(64):
        columns = generator.normal(size=(12, 9))
        weight = generator.uniform(0.1, 5.0, size=9)
        weight = weight / np.sum(weight)
        phase = unit_phase_from_angles(generator.uniform(0, 2 * math.pi, size=9))
        _, _, ratio = retained_power(columns, weight, phase)
        worst_ratio = max(worst_ratio, ratio)
        negative = negative or ratio < 0.0
    records.append(
        _identity_record(
            "weighted_jensen_bound",
            worst_ratio <= 1.0 + tolerance and not negative,
            {"maximum_ratio": worst_ratio, "negative_ratio_observed": negative},
        )
    )

    # 7. Independent-phase expectation equals sum u_i^2 ||g_i||^2, not D_E.
    columns = generator.normal(size=(6, 7))
    weight = np.array([0.40, 0.24, 0.14, 0.10, 0.06, 0.04, 0.02])
    analytic = independent_phase_power(columns, weight)
    trials = 400000
    phase = unit_phase_from_angles(
        generator.uniform(0.0, 2.0 * math.pi, size=(trials, 7))
    )
    quadrature = weight[None, :] * phase
    field = quadrature @ columns.T
    monte_carlo = float(np.mean(np.sum(field * field.conjugate(), axis=1).real))
    _, incoherent, _ = retained_power(
        columns, weight, np.ones(7, dtype=np.complex128)
    )
    relative_error = abs(monte_carlo - analytic) / analytic
    records.append(
        _identity_record(
            "independent_phase_expectation",
            relative_error <= 0.01 and abs(analytic - incoherent) / incoherent > 0.1,
            {
                "analytic_sum_u2_g2": analytic,
                "monte_carlo_power": monte_carlo,
                "monte_carlo_relative_error": relative_error,
                "coherent_reference_trace": incoherent,
                "ratio_analytic_to_reference": analytic / incoherent,
            },
        )
    )

    # 8. Equal conditional weights give exactly one N-th of the reference.
    state_count = 7
    equal_weights = np.ones(state_count, dtype=np.float64)
    equal_expectation = independent_phase_power(columns, equal_weights)
    _, equal_reference, _ = retained_power(
        columns, equal_weights, np.ones(state_count, dtype=np.complex128)
    )
    expected_ratio = 1.0 / state_count
    records.append(
        _identity_record(
            "uniform_weights_expectation_is_reference_over_n",
            abs(equal_expectation - equal_reference / state_count)
            <= tolerance * equal_reference,
            {
                "state_count": state_count,
                "expectation": equal_expectation,
                "reference_trace": equal_reference,
                "expectation_over_reference": equal_expectation / equal_reference,
                "expected_ratio": expected_ratio,
            },
        )
    )

    # 9. Whole-sheet-relative power scales with the squared area fraction.
    fraction = 0.25
    z_power, trace, ratio = retained_power(base_columns, base_weights, base_phase)
    full_trace = incoherent_trace(base_columns, np.full(5, 0.2))
    direct = whole_sheet_power(z_power, fraction, full_trace)
    composed = fraction**2 * ratio * trace / full_trace
    doubled = whole_sheet_power(z_power, 2.0 * fraction, full_trace)
    records.append(
        _identity_record(
            "whole_sheet_area_fraction_scaling",
            abs(direct - composed) <= 1e-12 * max(abs(direct), 1e-12)
            and abs(doubled - 4.0 * direct) <= 1e-12 * max(abs(doubled), 1e-12),
            {
                "whole_sheet_power": direct,
                "composed_from_ratio": composed,
                "quadrupled_fraction_power": doubled,
            },
        )
    )
    return records


def synthetic_checks_passed(records: Sequence[Mapping[str, object]]) -> bool:
    return bool(records) and all(bool(record["passed"]) for record in records)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise AuditError(f"refusing to write an empty table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def make_ledgers(
    relative_tolerance: float, absolute_tolerance: float
) -> dict[str, FieldLedger]:
    names = list(NORMALIZATION_FIELDS) + [
        "analytical_bound_excess",
        "coherent_patch_area_minimum",
        "coherent_patch_area_maximum",
        "coherent_patch_area_participation_ratio",
    ]
    return {
        name: FieldLedger(relative_tolerance, absolute_tolerance) for name in names
    }


__all__ = [
    "AuditError",
    "CONTAINER_FORWARD_PREFIX",
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "DeclaredInputResolution",
    "FORMULA_VERSION",
    "FieldLedger",
    "NORMALIZATION_FIELDS",
    "NOT_RECOMPUTED_COLUMNS",
    "PROTOCOL",
    "SubjectGeometry",
    "SubjectRecorder",
    "atomic_write_csv",
    "atomic_write_json",
    "average_reference",
    "build_subject_geometry",
    "canonical_support_identifier",
    "coherent_patch_factor",
    "coherent_patch_index",
    "expected_row_counts",
    "factor_power",
    "incoherent_trace",
    "independent_phase_power",
    "load_fixed_leadfield",
    "load_metadata",
    "make_ledgers",
    "phase_unit_error",
    "read_csv_rows",
    "regime_names",
    "resolve_declared_input",
    "retained_power",
    "sha256_file",
    "support_weights",
    "synthetic_checks_passed",
    "synthetic_identity_checks",
    "unit_phase_from_angles",
    "whole_sheet_power",
]
