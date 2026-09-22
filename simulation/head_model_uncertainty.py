"""Geometry perturbation primitives for head-model."""

from __future__ import annotations

import numpy as np


def rotated_directions(directions: np.ndarray, degrees: float, seed: int) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if values.ndim != 2 or values.shape[1] != 3 or np.any(norms <= 0.0):
        raise ValueError("directions must be nonzero 3-vectors")
    unit = values / norms[:, None]
    generator = np.random.default_rng(seed)
    tangent = generator.normal(size=unit.shape)
    tangent -= np.sum(tangent * unit, axis=1)[:, None] * unit
    tangent_norm = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_norm <= 1e-12):
        raise ValueError("orientation perturbation has a degenerate tangent")
    tangent /= tangent_norm[:, None]
    angle = np.deg2rad(float(degrees))
    result = np.cos(angle) * unit + np.sin(angle) * tangent
    return result / np.linalg.norm(result, axis=1)[:, None]


def radial_shift(vertices_mm: np.ndarray, shift_mm: float) -> np.ndarray:
    vertices = np.asarray(vertices_mm, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("surface vertices must be finite 3-vectors")
    center = vertices.mean(axis=0)
    radial = vertices - center
    norm = np.linalg.norm(radial, axis=1)
    if np.any(norm <= 0.0):
        raise ValueError("surface radial direction is undefined")
    return vertices + float(shift_mm) * radial / norm[:, None]


def radial_displacement(vertices_mm: np.ndarray, shifted_mm: np.ndarray) -> dict[str, float]:
    displacement = np.linalg.norm(np.asarray(shifted_mm) - np.asarray(vertices_mm), axis=1)
    return {"minimum_mm": float(np.min(displacement)), "median_mm": float(np.median(displacement)), "maximum_mm": float(np.max(displacement))}
