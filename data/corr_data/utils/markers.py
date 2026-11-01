"""Geometry helpers for reproducible endpoint marker masks.

The DTE data generator stores coordinates as ``(x, y, z)`` voxel indices,
while physical dimensions are specified in millimetres.  This module keeps
the conversion in one dependency-light place so it can be tested without
importing TIGRE or the full generation script.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def make_marker_config(
    enabled: bool = False,
    length_mm: float = 2.0,
    diameter_mm: float = 1.0,
    inset_mm: float = 0.0,
    mu_ratio: float = 2.0,
) -> dict:
    """Return the explicit, pickle-friendly marker generation settings."""
    if not np.isfinite(length_mm) or length_mm <= 0:
        raise ValueError("length_mm must be a finite positive value")
    if not np.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError("diameter_mm must be a finite positive value")
    if not np.isfinite(inset_mm) or inset_mm < 0:
        raise ValueError("inset_mm must be a finite non-negative value")
    if not np.isfinite(mu_ratio) or mu_ratio <= 0:
        raise ValueError("mu_ratio must be a finite positive value")
    return {
        "enabled": bool(enabled),
        "shape": "solid_cylinder",
        "length_mm": float(length_mm),
        "diameter_mm": float(diameter_mm),
        "inset_mm": float(inset_mm),
        "mu_ratio": float(mu_ratio),
    }


def _unit_vector(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must have non-zero length")
    return vector / norm


def _validate_inputs(
    curve_points: np.ndarray,
    spacing: Iterable[float],
    volume_shape: Iterable[int],
    length_mm: float,
    diameter_mm: float,
    inset_mm: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    curve = np.asarray(curve_points, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 2:
        raise ValueError("curve_points must have shape (N, 3), with N >= 2")
    if not np.all(np.isfinite(curve)):
        raise ValueError("curve_points must contain only finite values")

    voxel_spacing = np.asarray(tuple(spacing), dtype=np.float64)
    if voxel_spacing.shape != (3,) or not np.all(np.isfinite(voxel_spacing)):
        raise ValueError("spacing must contain three finite positive values")
    if np.any(voxel_spacing <= 0):
        raise ValueError("spacing must contain three finite positive values")

    shape_values = tuple(int(value) for value in volume_shape)
    if len(shape_values) != 3 or any(value <= 0 for value in shape_values):
        raise ValueError("volume_shape must contain three positive integers")

    if not np.isfinite(length_mm) or length_mm <= 0:
        raise ValueError("length_mm must be a finite positive value")
    if not np.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError("diameter_mm must be a finite positive value")
    if not np.isfinite(inset_mm) or inset_mm < 0:
        raise ValueError("inset_mm must be a finite non-negative value")

    return curve, voxel_spacing, shape_values


def _rasterize_cylinder(
    mask: np.ndarray,
    center_physical: np.ndarray,
    tangent_physical: np.ndarray,
    length_mm: float,
    radius_mm: float,
    spacing: np.ndarray,
) -> None:
    """Rasterize a finite solid cylinder into ``mask`` in-place."""
    half_length = length_mm / 2.0

    # A conservative axis-aligned physical bounding box for a cylinder.  The
    # box is intentionally padded by one voxel so rounding never removes the
    # boundary of a marker at an oblique angle.
    extent = np.abs(tangent_physical) * half_length + radius_mm
    lower = np.floor((center_physical - extent) / spacing).astype(np.int64) - 1
    upper = np.ceil((center_physical + extent) / spacing).astype(np.int64) + 1
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, np.asarray(mask.shape, dtype=np.int64) - 1)
    if np.any(lower > upper):
        return

    axes = [np.arange(lower[axis], upper[axis] + 1, dtype=np.float64) for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    points_physical = grid * spacing
    relative = points_physical - center_physical
    axial = np.sum(relative * tangent_physical, axis=-1)
    radial = relative - axial[..., None] * tangent_physical
    inside = (np.abs(axial) <= half_length) & (
        np.sum(radial * radial, axis=-1) <= radius_mm * radius_mm
    )

    view = mask[
        lower[0] : upper[0] + 1,
        lower[1] : upper[1] + 1,
        lower[2] : upper[2] + 1,
    ]
    view[inside] = 1


def make_endpoint_markers(
    curve_points: np.ndarray,
    spacing: Iterable[float],
    volume_shape: Iterable[int],
    length_mm: float,
    diameter_mm: float,
    inset_mm: float = 0.0,
) -> Tuple[np.ndarray, dict]:
    """Create two endpoint marker cylinders in an ``(x, y, z)`` volume.

    The markers start ``inset_mm`` inside each endpoint and extend toward the
    interior of the centerline.  The returned metadata is deliberately plain
    NumPy/Python data so it can be serialized in the existing pickle config.
    """
    curve, voxel_spacing, shape = _validate_inputs(
        curve_points, spacing, volume_shape, length_mm, diameter_mm, inset_mm
    )
    curve_physical = curve * voxel_spacing
    tangent_start = _unit_vector(curve_physical[1] - curve_physical[0], "start tangent")
    tangent_end = _unit_vector(curve_physical[-1] - curve_physical[-2], "end tangent")

    half_length = float(length_mm) / 2.0
    start_center = curve_physical[0] + tangent_start * (float(inset_mm) + half_length)
    end_center = curve_physical[-1] - tangent_end * (float(inset_mm) + half_length)

    mask = np.zeros(shape, dtype=np.uint8)
    radius_mm = float(diameter_mm) / 2.0
    _rasterize_cylinder(mask, start_center, tangent_start, float(length_mm), radius_mm, voxel_spacing)
    _rasterize_cylinder(mask, end_center, tangent_end, float(length_mm), radius_mm, voxel_spacing)

    endpoint_centers_physical = np.stack([start_center, end_center], axis=0)
    endpoint_centers_voxel = endpoint_centers_physical / voxel_spacing
    info = {
        "shape": "solid_cylinder",
        "length_mm": float(length_mm),
        "diameter_mm": float(diameter_mm),
        "inset_mm": float(inset_mm),
        "spacing_mm": voxel_spacing.copy(),
        "endpoint_centers_physical": endpoint_centers_physical,
        "endpoint_centers_voxel": endpoint_centers_voxel,
        "tangents_physical": np.stack([tangent_start, tangent_end], axis=0),
        "actual_voxel_count": int(mask.sum()),
    }
    return mask, info
