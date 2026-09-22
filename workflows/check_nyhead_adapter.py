#!/usr/bin/env python3
"""Adversarial tests for the cortical-only benchmark contract."""

from __future__ import annotations

from dataclasses import replace
import json
import platform
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anatomical.cortical_benchmark import (  # noqa: E402
    CorticalBenchmark,
    CorticalBenchmarkValidationError,
    load_cortical_benchmark,
    validate_cortical_benchmark,
    vertex_area_weights,
    write_cortical_benchmark,
)


def fixture() -> CorticalBenchmark:
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.01, 0.01, 0.0], [0.0, 0.01, 0.0]]
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    rng = np.random.default_rng(20260825)
    leadfield = rng.normal(size=(5, 4))
    leadfield -= np.mean(leadfield, axis=0, keepdims=True)
    manifest = {
        "schema_version": 1,
        "benchmark_kind": "cortical_only",
        "subject_id": "fixture",
        "coordinate_frame": "MNI",
        "coordinate_unit": "m",
        "leadfield_reference": "common_average",
        "leadfield_unit": "native_new_york_head_undocumented",
        "leadfield_physical_scale_status": "not_established",
        "amplitude_claims_allowed": False,
        "contains_hippocampal_operator": False,
        "sensor_names": [f"E{index}" for index in range(5)],
        "source_resolution": "fixture",
        "solver": {
            "name": "fixture",
            "version": "1",
            "method": "analytic",
            "conductivity_model": "fixture",
        },
        "provenance": "unit-test fixture",
        "arrays_file": "arrays.npz",
        "arrays_sha256": "0" * 64,
    }
    return CorticalBenchmark(
        manifest=manifest,
        cortical_leadfield_native=leadfield,
        electrode_positions_m=rng.normal(size=(5, 3)),
        electrode_normals=np.tile([0.0, 0.0, 1.0], (5, 1)),
        cortical_positions_m=positions,
        cortical_directions=np.tile([0.0, 0.0, 1.0], (4, 1)),
        cortical_area_weights_m2=vertex_area_weights(positions, triangles),
        cortical_triangles=triangles,
        source_indices_75k=np.array([0, 1, 2, 3], dtype=np.int64),
    )


class CorticalBenchmarkTests(unittest.TestCase):
    def test_roundtrip_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_cortical_benchmark(root / "first", fixture())
            second = write_cortical_benchmark(root / "second", fixture())
            self.assertEqual(
                first.manifest["arrays_sha256"], second.manifest["arrays_sha256"]
            )
            self.assertTrue(validate_cortical_benchmark(first)["valid"])
            self.assertTrue(load_cortical_benchmark(first.directory).manifest)

    def test_contract_rejects_amplitude_claims_and_fake_units(self) -> None:
        benchmark = fixture()
        manifest = dict(benchmark.manifest)
        manifest["amplitude_claims_allowed"] = True
        manifest["leadfield_unit"] = "V/(A m)"
        with self.assertRaises(CorticalBenchmarkValidationError):
            validate_cortical_benchmark(
                replace(benchmark, manifest=manifest), verify_checksum=False
            )

    def test_contract_rejects_gauge_and_direction_errors(self) -> None:
        benchmark = fixture()
        bad_leadfield = benchmark.cortical_leadfield_native.copy()
        bad_leadfield[0] += 1.0
        with self.assertRaises(CorticalBenchmarkValidationError):
            validate_cortical_benchmark(
                replace(benchmark, cortical_leadfield_native=bad_leadfield),
                verify_checksum=False,
            )
        bad_directions = benchmark.cortical_directions.copy()
        bad_directions[0] *= 2.0
        with self.assertRaises(CorticalBenchmarkValidationError):
            validate_cortical_benchmark(
                replace(benchmark, cortical_directions=bad_directions),
                verify_checksum=False,
            )

    def test_contract_rejects_bad_topology_and_quadrature(self) -> None:
        benchmark = fixture()
        bad_triangles = benchmark.cortical_triangles.copy()
        bad_triangles[0, 2] = 99
        with self.assertRaises(CorticalBenchmarkValidationError):
            validate_cortical_benchmark(
                replace(benchmark, cortical_triangles=bad_triangles),
                verify_checksum=False,
            )
        bad_weights = benchmark.cortical_area_weights_m2.copy()
        bad_weights[0] *= 2.0
        with self.assertRaises(CorticalBenchmarkValidationError):
            validate_cortical_benchmark(
                replace(benchmark, cortical_area_weights_m2=bad_weights),
                verify_checksum=False,
            )

    def test_vertex_area_conservation_and_degenerate_rejection(self) -> None:
        benchmark = fixture()
        weights = benchmark.cortical_area_weights_m2
        self.assertAlmostEqual(float(np.sum(weights)), 1e-4, places=15)
        with self.assertRaises(ValueError):
            vertex_area_weights(
                benchmark.cortical_positions_m,
                np.array([[0, 0, 1]], dtype=np.int64),
            )


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CorticalBenchmarkTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema_version": 1,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    output = PROJECT_ROOT / "results" / "nyhead_adapter_tests.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
