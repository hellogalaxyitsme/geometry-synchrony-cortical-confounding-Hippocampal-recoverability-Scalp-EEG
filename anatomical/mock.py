"""Analytic homogeneous-conductor mock for anatomical pipeline validation."""

from __future__ import annotations

from math import pi, sqrt
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .bundle import AnatomicalBundle, write_bundle


FloatArray = NDArray[np.float64]


def fibonacci_sphere(number_of_points: int, radius_m: float) -> FloatArray:
    if number_of_points < 2 or radius_m <= 0:
        raise ValueError("invalid sphere parameters")
    indices = np.arange(number_of_points, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / number_of_points
    azimuth = pi * (3.0 - sqrt(5.0)) * indices
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return radius_m * np.column_stack(
        (radial * np.cos(azimuth), radial * np.sin(azimuth), z)
    )


def infinite_conductor_leadfield(
    electrode_positions_m: FloatArray,
    source_positions_m: FloatArray,
    source_directions: FloatArray,
    conductivity_s_per_m: float,
) -> FloatArray:
    """Unit-dipole potential in an infinite homogeneous conductor."""

    if conductivity_s_per_m <= 0:
        raise ValueError("conductivity must be positive")
    displacement = (
        electrode_positions_m[:, None, :] - source_positions_m[None, :, :]
    )
    distance = np.linalg.norm(displacement, axis=2)
    if np.any(distance == 0):
        raise ValueError("an electrode coincides with a source")
    numerator = np.sum(
        displacement * source_directions[None, :, :], axis=2
    )
    return numerator / (
        4.0 * pi * conductivity_s_per_m * distance**3
    )


def make_mock_bundle(config: Mapping[str, object]) -> AnatomicalBundle:
    sensors = int(config["number_of_electrodes"])
    hippocampal_elements = int(config["number_of_hippocampal_elements"])
    cortical_elements = int(config["number_of_cortical_elements"])
    electrode_positions = fibonacci_sphere(
        sensors, float(config["head_radius_m"])
    )
    cortical_positions = fibonacci_sphere(
        cortical_elements, float(config["brain_radius_m"])
    )
    cortical_directions = cortical_positions / np.linalg.norm(
        cortical_positions, axis=1
    )[:, None]

    xi = (np.arange(hippocampal_elements, dtype=float) + 0.5) / hippocampal_elements
    centered = xi - 0.5
    hippocampal_positions = np.column_stack(
        (
            0.050 * centered,
            -0.012 + 0.004 * np.cos(2.0 * pi * centered),
            -0.010 + 0.006 * np.sin(pi * centered),
        )
    )
    angles = 2.0 * pi * centered
    hippocampal_directions = np.column_stack(
        (
            np.cos(angles),
            np.sin(angles),
            0.2 * np.sin(2.0 * angles),
        )
    )
    hippocampal_directions /= np.linalg.norm(
        hippocampal_directions, axis=1
    )[:, None]

    conductivity = float(config["conductivity_s_per_m"])
    hippocampal_leadfield = infinite_conductor_leadfield(
        electrode_positions,
        hippocampal_positions,
        hippocampal_directions,
        conductivity,
    )
    cortical_leadfield = infinite_conductor_leadfield(
        electrode_positions,
        cortical_positions,
        cortical_directions,
        conductivity,
    )
    hippocampal_weights = np.full(
        hippocampal_elements,
        float(config["hippocampal_total_area_m2"]) / hippocampal_elements,
    )
    cortical_weights = np.full(
        cortical_elements,
        float(config["cortical_total_area_m2"]) / cortical_elements,
    )
    noise = float(config["sensor_noise_variance"]) * np.eye(sensors)
    manifest = {
        "schema_version": 1,
        "subject_id": "analytic-homogeneous-mock",
        "coordinate_frame": "mock-head-cartesian",
        "coordinate_unit": "m",
        "potential_unit": "V",
        "dipole_moment_unit": "A m",
        "leadfield_unit": "V/(A m)",
        "area_unit": "m^2",
        "reference": "common_gauge",
        "sensor_names": [f"E{index + 1:03d}" for index in range(sensors)],
        "solver": {
            "name": "analytic-infinite-conductor",
            "version": "1",
            "method": "V=p_dot_r/(4*pi*sigma*r^3)",
            "conductivity_model": f"homogeneous sigma={conductivity:g} S/m",
        },
        "provenance": (
            "Deterministic analytic software-validation mock; not human anatomy."
        ),
        "arrays_file": "arrays.npz",
        "arrays_sha256": "0" * 64,
    }
    return AnatomicalBundle(
        manifest=manifest,
        hippocampal_leadfield=hippocampal_leadfield,
        cortical_leadfield=cortical_leadfield,
        electrode_positions_m=electrode_positions,
        hippocampal_positions_m=hippocampal_positions,
        cortical_positions_m=cortical_positions,
        hippocampal_directions=hippocampal_directions,
        cortical_directions=cortical_directions,
        hippocampal_area_weights_m2=hippocampal_weights,
        cortical_area_weights_m2=cortical_weights,
        hippocampal_longitudinal_coordinate=xi,
        sensor_noise_covariance=noise,
    )


def write_mock_bundle(
    directory: Path, config: Mapping[str, object]
) -> AnatomicalBundle:
    return write_bundle(directory, make_mock_bundle(config))
