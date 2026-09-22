#!/usr/bin/env python3
"""Deterministic unit and adversarial tests for HippUnfold source ingestion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from anatomical.hippunfold import hippunfold_source_hierarchy


def _surface(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    image = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                np.asarray(vertices, dtype=np.float32),
                intent="NIFTI_INTENT_POINTSET",
            ),
            nib.gifti.GiftiDataArray(
                np.asarray(faces, dtype=np.int32),
                intent="NIFTI_INTENT_TRIANGLE",
            ),
        ]
    )
    nib.save(image, str(path))


def _metric(values: np.ndarray, path: Path) -> None:
    image = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                np.asarray(values, dtype=np.float32),
                intent="NIFTI_INTENT_SHAPE",
            )
        ]
    )
    nib.save(image, str(path))


def _grid(nx: int, ny: int, z: float, x_offset: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [(x_offset + x, y, z) for y in range(ny) for x in range(nx)],
        dtype=float,
    )
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(faces, dtype=np.int32)


def _fixture(root: Path, subject: str = "123456") -> None:
    surface_dir = root / f"sub-{subject}" / "surf"
    metric_dir = root / f"sub-{subject}" / "metric"
    surface_dir.mkdir(parents=True)
    metric_dir.mkdir(parents=True)
    for hemisphere, offset in (("L", -20.0), ("R", 20.0)):
        middle, faces = _grid(5, 4, 0.0, offset)
        for label in ("hipp", "dentate"):
            for name, z in (("inner", -1.0), ("midthickness", 0.0), ("outer", 1.0)):
                vertices = middle.copy()
                vertices[:, 2] = z
                filename = (
                    f"sub-{subject}_hemi-{hemisphere}_space-T1w_den-8k_"
                    f"label-{label}_{name}.surf.gii"
                )
                _surface(vertices, faces, surface_dir / filename)
        ap = (middle[:, 0] - middle[:, 0].min()) / np.ptp(middle[:, 0])
        pd = (middle[:, 1] - middle[:, 1].min()) / np.ptp(middle[:, 1])
        for direction, values in (("AP", ap), ("PD", pd)):
            filename = (
                f"sub-{subject}_hemi-{hemisphere}_den-8k_label-hipp_"
                f"desc-laplace_dir-{direction}_coords.shape.gii"
            )
            _metric(values, metric_dir / filename)


class HippUnfoldGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hippunfold-forward-check-")
        self.root = Path(self.temporary.name)
        _fixture(self.root)
        self.levels = [
            {"name": "coarse", "spatial_bin_mm": 4.0},
            {"name": "medium", "spatial_bin_mm": 2.0},
            {"name": "reference", "spatial_bin_mm": 1.0},
        ]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self):
        return hippunfold_source_hierarchy(
            self.root,
            "123456",
            np.eye(4),
            self.levels,
            density="8k",
            normal_bin_width=0.5,
        )

    def test_intrinsic_direction_and_dentate_separation(self) -> None:
        hierarchy, report = self.build()
        reference = hierarchy["reference"]
        np.testing.assert_allclose(reference["directions"][:, 2], 1.0, atol=1e-12)
        np.testing.assert_allclose(reference["directions"][:, :2], 0.0, atol=1e-12)
        self.assertTrue(report["orientation_is_intrinsic_laminar_ribbon_vector"])
        self.assertFalse(report["orientation_is_segmentation_boundary_normal"])
        self.assertFalse(report["dentate"]["included_in_primary_source_model"])
        self.assertEqual(set(np.unique(reference["hemisphere_code"])), {-1, 1})

    def test_hierarchy_is_exactly_nested_and_area_preserving(self) -> None:
        hierarchy, report = self.build()
        self.assertTrue(report["exact_nested"])
        areas = [float(hierarchy[name]["area_weights_m2"].sum()) for name in (
            "coarse", "medium", "reference"
        )]
        np.testing.assert_allclose(areas, areas[0], rtol=0.0, atol=1e-15)
        self.assertLess(len(hierarchy["coarse"]["positions_m"]), len(hierarchy["reference"]["positions_m"]))

    def test_coordinates_survive_aggregation(self) -> None:
        hierarchy, _ = self.build()
        reference = hierarchy["reference"]
        for name in ("longitudinal_coordinate", "proximal_distal_coordinate"):
            self.assertGreaterEqual(float(reference[name].min()), 0.0)
            self.assertLessEqual(float(reference[name].max()), 1.0)

    def test_off_segment_midthickness_is_audited_not_excluded(self) -> None:
        path = next(
            (self.root / "sub-123456" / "surf").glob(
                "*hemi-L*label-hipp_midthickness.surf.gii"
            )
        )
        image = nib.load(str(path))
        image.darrays[0].data[0, 2] = 1.5
        nib.save(image, str(path))
        _, report = self.build()
        left = next(row for row in report["hemispheres"] if row["hemisphere"] == "L")
        audit = left["midthickness_segment_projection_audit"]
        self.assertIn("diagnostic only", audit["policy"])
        self.assertGreater(audit["outside_0_1_vertices"], 0)

    def test_zero_measure_vertex_is_excluded_without_geometry_repair(self) -> None:
        surface_dir = self.root / "sub-123456" / "surf"
        for path in surface_dir.glob("*hemi-L*label-hipp_*.surf.gii"):
            image = nib.load(str(path))
            vertices = np.vstack((image.darrays[0].data, image.darrays[0].data[-1]))
            _surface(vertices, image.darrays[1].data, path)
        metric_dir = self.root / "sub-123456" / "metric"
        for path in metric_dir.glob("*hemi-L*label-hipp*coords.shape.gii"):
            image = nib.load(str(path))
            values = np.append(image.darrays[0].data, image.darrays[0].data[-1])
            _metric(values, path)
        hierarchy, report = self.build()
        left = next(row for row in report["hemispheres"] if row["hemisphere"] == "L")
        qc = left["mesh_measure_qc"]
        self.assertEqual(left["raw_vertices"], 21)
        self.assertEqual(left["source_eligible_vertices"], 20)
        self.assertEqual(qc["excluded_zero_measure_vertices"], 1)
        self.assertEqual(qc["excluded_zero_measure_vertex_indices"], [20])
        self.assertTrue(report["exact_nested"])
        self.assertTrue(np.all(hierarchy["reference"]["area_weights_m2"] > 0.0))

    def test_topology_mismatch_fails_closed(self) -> None:
        path = next((self.root / "sub-123456" / "surf").glob("*hemi-L*label-hipp_outer.surf.gii"))
        image = nib.load(str(path))
        image.darrays[1].data[0] = image.darrays[1].data[0][::-1]
        nib.save(image, str(path))
        with self.assertRaisesRegex(ValueError, "does not share midthickness topology"):
            self.build()

    def test_zero_thickness_fails_closed(self) -> None:
        surface_dir = self.root / "sub-123456" / "surf"
        inner_path = next(surface_dir.glob("*hemi-R*label-hipp_inner.surf.gii"))
        outer_path = next(surface_dir.glob("*hemi-R*label-hipp_outer.surf.gii"))
        inner = nib.load(str(inner_path))
        outer = nib.load(str(outer_path))
        outer.darrays[0].data[:] = inner.darrays[0].data
        nib.save(outer, str(outer_path))
        with self.assertRaisesRegex(ValueError, "ribbon thickness"):
            self.build()

    def test_dentate_topology_mismatch_fails_closed(self) -> None:
        path = next(
            (self.root / "sub-123456" / "surf").glob(
                "*hemi-R*label-dentate_outer.surf.gii"
            )
        )
        image = nib.load(str(path))
        image.darrays[1].data[0] = image.darrays[1].data[0][::-1]
        nib.save(image, str(path))
        with self.assertRaisesRegex(
            ValueError, "dentate outer does not share midthickness topology"
        ):
            self.build()

    def test_out_of_range_coordinate_fails_closed(self) -> None:
        path = next((self.root / "sub-123456" / "metric").glob("*hemi-L*dir-AP*"))
        image = nib.load(str(path))
        image.darrays[0].data[:] = 0.5
        nib.save(image, str(path))
        with self.assertRaisesRegex(ValueError, "AP intrinsic coordinate"):
            self.build()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "hippunfold_geometry_tests.json",
    )
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HippUnfoldGeometryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
