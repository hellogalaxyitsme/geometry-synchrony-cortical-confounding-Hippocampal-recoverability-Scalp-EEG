"""Deterministic closed-surface clearance checks for BEM shell validation.

The routines in this module operate on the final triangle meshes rather than
on vertex-to-vertex distances.  A candidate-pair broad phase is followed by
exact (up to floating-point arithmetic) segment/triangle and distance tests.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def closed_surface_centroid(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return the signed-volume centroid of an oriented closed triangle mesh."""

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    tri = vertices[triangles]
    signed_six_volume = np.einsum(
        "ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])
    )
    denominator = float(signed_six_volume.sum())
    if not np.isfinite(denominator) or abs(denominator) < 1e-15:
        raise ValueError("closed surface has zero or non-finite signed volume")
    centroid = np.sum(
        signed_six_volume[:, None] * tri.sum(axis=1), axis=0
    ) / (4.0 * denominator)
    if not np.all(np.isfinite(centroid)):
        raise ValueError("closed surface centroid is non-finite")
    return centroid


def radial_inset_surface(
    vertices: np.ndarray, triangles: np.ndarray, inset: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inset a star-shaped surface by a fixed radial distance.

    The input and output retain exactly the same topology and vertex ordering.
    The returned center is the volume centroid used for the radial contraction.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    if not np.isfinite(inset) or inset <= 0.0:
        raise ValueError("radial inset must be positive and finite")
    center = closed_surface_centroid(vertices, triangles)
    displacement = vertices - center
    radii = np.linalg.norm(displacement, axis=1)
    if np.any(radii <= inset) or not np.all(np.isfinite(radii)):
        raise ValueError("radial inset exceeds a surface radius")
    result = center + displacement * ((radii - inset) / radii)[:, None]
    return result, center


def _point_triangle_distance_sq(
    point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """Squared distance from a point to a triangle (Ericson region tests)."""

    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.dot(ap, ap))
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.dot(bp, bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        projection = a + v * ab
        delta = point - projection
        return float(np.dot(delta, delta))
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.dot(cp, cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        projection = a + w * ac
        delta = point - projection
        return float(np.dot(delta, delta))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        edge = c - b
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        projection = b + w * edge
        delta = point - projection
        return float(np.dot(delta, delta))
    denominator = va + vb + vc
    if abs(denominator) < 1e-30:
        return min(
            float(np.dot(point - a, point - a)),
            float(np.dot(point - b, point - b)),
            float(np.dot(point - c, point - c)),
        )
    v = vb / denominator
    w = vc / denominator
    projection = a + ab * v + ac * w
    delta = point - projection
    return float(np.dot(delta, delta))


def _segment_segment_distance_sq(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray
) -> float:
    """Squared distance between two closed line segments."""

    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    tiny = 1e-30
    denominator = a * c - b * b
    s_numerator = denominator
    s_denominator = denominator
    t_numerator = denominator
    t_denominator = denominator
    if denominator < tiny:
        s_numerator = 0.0
        s_denominator = 1.0
        t_numerator = e
        t_denominator = c
    else:
        s_numerator = b * e - c * d
        t_numerator = a * e - b * d
        if s_numerator < 0.0:
            s_numerator = 0.0
            t_numerator = e
            t_denominator = c
        elif s_numerator > s_denominator:
            s_numerator = s_denominator
            t_numerator = e + b
            t_denominator = c
    if t_numerator < 0.0:
        t_numerator = 0.0
        if -d < 0.0:
            s_numerator = 0.0
        elif -d > a:
            s_numerator = s_denominator
        else:
            s_numerator = -d
            s_denominator = a
    elif t_numerator > t_denominator:
        t_numerator = t_denominator
        if (-d + b) < 0.0:
            s_numerator = 0.0
        elif (-d + b) > a:
            s_numerator = s_denominator
        else:
            s_numerator = -d + b
            s_denominator = a
    sc = 0.0 if abs(s_numerator) < tiny else s_numerator / s_denominator
    tc = 0.0 if abs(t_numerator) < tiny else t_numerator / t_denominator
    delta = w + sc * u - tc * v
    return float(np.dot(delta, delta))


def _segment_intersects_triangle(
    p0: np.ndarray,
    p1: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tolerance: float = 1e-12,
) -> bool:
    """Möller--Trumbore segment/triangle intersection test."""

    direction = p1 - p0
    edge1 = b - a
    edge2 = c - a
    h = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, h))
    if abs(determinant) <= tolerance:
        return False
    inverse = 1.0 / determinant
    s = p0 - a
    u = inverse * float(np.dot(s, h))
    if u < -tolerance or u > 1.0 + tolerance:
        return False
    q = np.cross(s, edge1)
    v = inverse * float(np.dot(direction, q))
    if v < -tolerance or u + v > 1.0 + tolerance:
        return False
    distance = inverse * float(np.dot(edge2, q))
    return -tolerance <= distance <= 1.0 + tolerance


def triangle_distance_sq(first: np.ndarray, second: np.ndarray) -> float:
    """Squared Euclidean distance between two triangles."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    edges = ((0, 1), (1, 2), (2, 0))
    for start, end in edges:
        if _segment_intersects_triangle(
            first[start], first[end], second[0], second[1], second[2]
        ):
            return 0.0
        if _segment_intersects_triangle(
            second[start], second[end], first[0], first[1], first[2]
        ):
            return 0.0
    distances = [
        _point_triangle_distance_sq(point, second[0], second[1], second[2])
        for point in first
    ]
    distances.extend(
        _point_triangle_distance_sq(point, first[0], first[1], first[2])
        for point in second
    )
    distances.extend(
        _segment_segment_distance_sq(
            first[first_start], first[first_end], second[second_start], second[second_end]
        )
        for first_start, first_end in edges
        for second_start, second_end in edges
    )
    return max(0.0, min(distances))


def triangle_clearance_report(
    inner_vertices: np.ndarray,
    inner_triangles: np.ndarray,
    outer_vertices: np.ndarray,
    outer_triangles: np.ndarray,
    minimum_clearance: float,
) -> dict[str, object]:
    """Verify a minimum triangle-to-triangle clearance between two meshes.

    The broad-phase radius test cannot omit a violating pair: if two triangles
    are closer than ``minimum_clearance``, their bounding spheres necessarily
    overlap after expansion by that amount.
    """

    if not np.isfinite(minimum_clearance) or minimum_clearance <= 0.0:
        raise ValueError("minimum triangle clearance must be positive")
    inner = np.asarray(inner_vertices, dtype=np.float64)[
        np.asarray(inner_triangles, dtype=np.int64)
    ]
    outer = np.asarray(outer_vertices, dtype=np.float64)[
        np.asarray(outer_triangles, dtype=np.int64)
    ]
    inner_centers = inner.mean(axis=1)
    outer_centers = outer.mean(axis=1)
    inner_radii = np.linalg.norm(inner - inner_centers[:, None, :], axis=2).max(axis=1)
    outer_radii = np.linalg.norm(outer - outer_centers[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(outer_centers)
    maximum_outer_radius = float(outer_radii.max())
    threshold_sq = minimum_clearance * minimum_clearance
    candidate_pairs = 0
    violating_pairs = 0
    intersecting_pairs = 0
    smallest_distance_sq = math.inf
    smallest_pair: tuple[int, int] | None = None
    for inner_index, (center, radius) in enumerate(zip(inner_centers, inner_radii)):
        candidates = tree.query_ball_point(
            center, float(radius + maximum_outer_radius + minimum_clearance)
        )
        for outer_index in candidates:
            center_distance = float(np.linalg.norm(center - outer_centers[outer_index]))
            if center_distance > radius + outer_radii[outer_index] + minimum_clearance:
                continue
            candidate_pairs += 1
            distance_sq = triangle_distance_sq(inner[inner_index], outer[outer_index])
            if distance_sq < smallest_distance_sq:
                smallest_distance_sq = distance_sq
                smallest_pair = (inner_index, int(outer_index))
            if distance_sq <= 1e-24:
                intersecting_pairs += 1
            if distance_sq < threshold_sq - 1e-18:
                violating_pairs += 1
    return {
        "required_minimum_clearance_m": float(minimum_clearance),
        "broad_phase_candidate_pairs": int(candidate_pairs),
        "violating_triangle_pairs": int(violating_pairs),
        "intersecting_triangle_pairs": int(intersecting_pairs),
        "smallest_candidate_distance_m": (
            None if not np.isfinite(smallest_distance_sq) else float(math.sqrt(smallest_distance_sq))
        ),
        "smallest_candidate_pair": None if smallest_pair is None else list(smallest_pair),
        "passed": violating_pairs == 0 and intersecting_pairs == 0,
    }
