"""Small, dependency-light helpers for optional marker-aware projections."""

from __future__ import annotations

import numpy as np
from typing import Optional


def compose_stent_and_marker(
    base_image: np.ndarray,
    stent_mask: np.ndarray,
    marker_mask: Optional[np.ndarray] = None,
    stent_value: float = 0.08,
    marker_mu_ratio: float = 2.0,
):
    """Compose simulated input attenuation and a binary DTE label.

    ``stent_mask`` and ``marker_mask`` describe occupancy, not attenuation.
    The marker therefore gets a higher value in the input image, while the
    returned label is the binary union of both objects.
    """
    base = np.asarray(base_image, dtype=np.float32)
    stent = np.asarray(stent_mask) > 0
    if base.shape != stent.shape:
        raise ValueError("base_image and stent_mask must have the same shape")

    if marker_mask is None:
        marker = np.zeros_like(stent, dtype=bool)
    else:
        marker = np.asarray(marker_mask) > 0
        if marker.shape != base.shape:
            raise ValueError("marker_mask must have the same shape as base_image")

    if not np.isfinite(stent_value) or stent_value < 0:
        raise ValueError("stent_value must be a finite non-negative value")
    if not np.isfinite(marker_mu_ratio) or marker_mu_ratio <= 0:
        raise ValueError("marker_mu_ratio must be a finite positive value")

    image = np.array(base, copy=True, dtype=np.float32)
    image[stent] = np.float32(stent_value)
    image[marker] = np.float32(stent_value * marker_mu_ratio)
    label = np.logical_or(stent, marker).astype(np.float32)
    return image, label
