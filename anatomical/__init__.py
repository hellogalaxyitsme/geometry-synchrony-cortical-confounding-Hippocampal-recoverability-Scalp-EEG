"""Solver-neutral anatomical lead-field analysis."""

from .analysis import (
    CorticalRestriction,
    analyze_condition,
    build_cortical_restriction,
)
from .bundle import (
    AnatomicalBundle,
    BundleValidationError,
    load_bundle,
    validate_bundle,
    write_bundle,
)
from .cortical_benchmark import (
    CorticalBenchmark,
    CorticalBenchmarkValidationError,
    load_cortical_benchmark,
    validate_cortical_benchmark,
    write_cortical_benchmark,
)
from .operators import (
    density_covariance,
    density_operator,
    farthest_point_order,
)

__all__ = [
    "AnatomicalBundle",
    "BundleValidationError",
    "CorticalBenchmark",
    "CorticalBenchmarkValidationError",
    "CorticalRestriction",
    "analyze_condition",
    "build_cortical_restriction",
    "density_covariance",
    "density_operator",
    "farthest_point_order",
    "load_bundle",
    "load_cortical_benchmark",
    "validate_bundle",
    "validate_cortical_benchmark",
    "write_bundle",
    "write_cortical_benchmark",
]
