#!/usr/bin/env python3
"""Deterministic numerical tests for stimulation-control primitives and configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from empirical.stimulation_control import (  # noqa: E402
    fiducial_head_coordinates,
    interpolate_rows,
    matrix_attribution_components,
    matrix_false_attribution,
    normalized_dictionary,
    pca_targets,
    robust_channel_scales,
    sparse_basis,
    spherical_interpolation,
    transform_patient_operator,
)
from simulation.cortical_restriction import orthonormal_basis  # noqa: E402
from theory.recoverability import helmert_reference  # noqa: E402


PROTOCOL = "cortical-control/adversarial-stimulation-v1"


def expect_error(function, *args) -> None:
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def run(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["protocol"] == PROTOCOL
    assert config["detector"]["reporting_threshold"] == 0.1
    assert len(config["hippocampal_probes"]["supports"]) * len(config["hippocampal_probes"]["wave_cycles"]) == 12
    assert config["restrictions"]["primary"] == [
        "unrestricted", "empirical_covariance", "smooth", "support", "sparse_k4"
    ]
    assert config["ieeg_waveform_may_be_used_by_detector"] is False

    fiducials = {"LPA": [-100.0, 0.0, 0.0], "NAS": [0.0, 100.0, 0.0], "RPA": [100.0, 0.0, 0.0]}
    points = np.asarray([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    transformed = fiducial_head_coordinates(points, fiducials)
    np.testing.assert_allclose(transformed, points * 1e-3, atol=1e-15)

    phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    theta = np.linspace(0.25, np.pi - 0.25, 8)
    sphere = np.asarray(
        [[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)] for t in theta for p in phi],
        dtype=np.float64,
    )
    interpolation = spherical_interpolation(sphere, sphere, neighbours=8, sigma_degrees=12.0)
    constant = np.ones((len(sphere), 3))
    np.testing.assert_allclose(interpolate_rows(constant, interpolation), constant, atol=1e-14)
    assert np.max(interpolation.nearest_angle_degrees) < 1e-6

    generator = np.random.default_rng(20260830)
    baseline = generator.normal(size=(20, 12, 100))
    scales = robust_channel_scales(baseline)
    assert scales.shape == (12,) and np.all(scales > 0.0)
    expect_error(robust_channel_scales, np.zeros((2, 3, 4)))

    matrix = generator.normal(size=(12, 50))
    targets, retained = pca_targets(matrix, 3)
    assert targets.shape == (12, 3) and 0.0 < retained <= 1.0
    np.testing.assert_allclose(np.sum(targets * targets), np.sum(np.linalg.svd(matrix, compute_uv=False)[:3] ** 2))

    h = np.asarray([[0.0], [1.0], [0.0]])
    nuisance = np.asarray([[1.0], [0.0], [0.0]])
    assert abs(matrix_false_attribution(h, h, nuisance) - 1.0) < 1e-14
    components = matrix_attribution_components(h, h, nuisance)
    assert abs(components["total_attribution_fraction"] - 1.0) < 1e-14
    assert abs(components["cortical_residual_energy_fraction"] - 1.0) < 1e-14
    assert abs(components["conditional_hippocampal_fraction"] - 1.0) < 1e-14
    partly_captured = np.asarray([[1.0], [1.0], [0.0]])
    partial = matrix_attribution_components(partly_captured, partly_captured, nuisance)
    assert abs(partial["total_attribution_fraction"] - 0.5) < 1e-14
    assert abs(partial["cortical_residual_energy_fraction"] - 0.5) < 1e-14
    assert abs(partial["conditional_hippocampal_fraction"] - 1.0) < 1e-14
    assert matrix_false_attribution(nuisance, h, nuisance) < 1e-14
    unrestricted = helmert_reference(3).T
    referenced_h = h - h.mean(axis=0, keepdims=True)
    assert matrix_false_attribution(referenced_h, referenced_h, unrestricted) < 1e-14

    dictionary, keep = normalized_dictionary(np.eye(3))
    sparse, selected = sparse_basis(dictionary, np.asarray([[1.0], [0.0], [0.0]]), 1)
    assert keep.tolist() == [0, 1, 2] and selected.tolist() == [0]
    np.testing.assert_allclose(sparse @ sparse.T, np.diag([1.0, 0.0, 0.0]), atol=1e-14)

    operator = np.arange(20, dtype=np.float64).reshape(5, 4)
    transformed_operator = transform_patient_operator(operator, [0, 2, 4], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(transformed_operator.mean(axis=0), 0.0, atol=1e-14)
    assert orthonormal_basis(transformed_operator).shape[0] == 3

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "ok": True,
        "tests": 25,
        "seed": 20260830,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.config)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
