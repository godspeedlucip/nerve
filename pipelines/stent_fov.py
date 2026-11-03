"""Place a stent within every cone-beam view of a fixed detector."""

from __future__ import annotations

import hashlib

import numpy as np


def configure_safe_stent_geometry(geo, stent_mask, sample_name, *, margin_pixels=16,
                                  randomize=True):
    """Validate the stent FOV and optionally choose a detector offset.

    The mask uses TIGRE's Python volume order (Z, Y, X). A sphere around its
    complete voxel bounding box bounds every view after any in-plane rotation.
    The sphere's closest possible source distance gives a conservative bound
    on projected U and V, including the outer faces of occupied voxels.
    """
    mask = np.asarray(stent_mask) > 0
    if mask.shape != tuple(geo.nVoxel):
        raise ValueError("stent_mask shape must match geo.nVoxel")
    if not mask.any():
        raise ValueError("stent_mask is empty")
    if margin_pixels < 1 or int(margin_pixels) != margin_pixels:
        raise ValueError("margin_pixels must be a positive integer")

    indices = np.nonzero(mask)
    low = np.array([axis.min() for axis in indices], dtype=np.float64)
    high = np.array([axis.max() for axis in indices], dtype=np.float64)
    spacing = np.asarray(geo.dVoxel, dtype=np.float64)
    center_mm = ((low + high) / 2 - (np.asarray(mask.shape) - 1) / 2) * spacing
    half_extent_mm = (high - low + 1) * spacing / 2
    radius_mm = float(np.linalg.norm(half_extent_mm))

    dso = float(geo.DSO)
    dsd = float(geo.DSD)
    if dso <= radius_mm or dsd <= 0:
        raise ValueError("stent does not fit between source and detector")
    projected_radius_mm = dsd * radius_mm / (dso - radius_mm)
    detector_half_mm = np.asarray(geo.sDetector, dtype=np.float64) / 2
    margin_mm = margin_pixels * np.asarray(geo.dDetector, dtype=np.float64)
    limit_mm = detector_half_mm - margin_mm - projected_radius_mm
    if np.any(limit_mm <= 0):
        raise ValueError("stent does not fit in the detector FOV with the requested margin")

    if randomize:
        seed = int.from_bytes(
            hashlib.blake2b(str(sample_name).encode("utf-8"), digest_size=8).digest(),
            "little",
        )
        rng = np.random.default_rng(seed)
        principal_axis = int(rng.integers(0, 2))
        other_axis = 1 - principal_axis
        detector_offset = np.empty(2, dtype=np.float64)
        detector_offset[principal_axis] = (
            rng.choice([-1, 1]) * rng.uniform(0.55, 0.9) * limit_mm[principal_axis]
        )
        detector_offset[other_axis] = rng.uniform(-0.9, 0.9) * limit_mm[other_axis]
        geo.offOrigin = -center_mm
        geo.offDetector = detector_offset
    else:
        # A rotating volume can move across U by its radial XY displacement.
        displacement = np.array([abs(center_mm[0]), np.linalg.norm(center_mm[1:])])
        projected_displacement = dsd * displacement / (dso - radius_mm)
        if np.any(projected_displacement > limit_mm):
            raise ValueError("translated stent does not fit in the detector FOV")
        geo.offOrigin = np.zeros(3, dtype=np.float64)
        geo.offDetector = np.zeros(2, dtype=np.float64)
    return {
        "sample": str(sample_name),
        "DSO_mm": dso,
        "DSD_mm": dsd,
        "nDetector": np.asarray(geo.nDetector).tolist(),
        "dDetector_mm": np.asarray(geo.dDetector).tolist(),
        "nVoxel": np.asarray(geo.nVoxel).tolist(),
        "dVoxel_mm": spacing.tolist(),
        "offOrigin_mm": geo.offOrigin.tolist(),
        "offDetector_mm": geo.offDetector.tolist(),
        "margin_pixels": int(margin_pixels),
        "stent_bound_radius_mm": radius_mm,
        "projected_bound_radius_mm": projected_radius_mm,
        "safe_detector_offset_limit_mm": limit_mm.tolist(),
    }


def apply_saved_stent_geometry(geo, info):
    """Restore a projection geometry on the matching reconstruction volume."""
    if not np.array_equal(geo.nDetector, info["nDetector"]) or not np.allclose(
        geo.dDetector, info["dDetector_mm"]
    ):
        raise ValueError("saved detector geometry does not match reconstruction geometry")
    if not np.array_equal(geo.nVoxel, info["nVoxel"]) or not np.allclose(
        geo.dVoxel, info["dVoxel_mm"]
    ):
        raise ValueError("saved voxel geometry does not match reconstruction geometry")
    if not np.isclose(geo.DSO, info["DSO_mm"]) or not np.isclose(geo.DSD, info["DSD_mm"]):
        raise ValueError("saved source geometry does not match reconstruction geometry")
    geo.offOrigin = np.asarray(info["offOrigin_mm"], dtype=np.float64)
    geo.offDetector = np.asarray(info["offDetector_mm"], dtype=np.float64)


def validate_stent_projection(projection, *, margin_pixels=16):
    """Reject empty angles or projected label pixels touching the safety border."""
    projected = np.asarray(projection)
    if projected.ndim != 3:
        raise ValueError("projection must have shape (angles, rows, columns)")
    if margin_pixels < 1 or 2 * margin_pixels >= min(projected.shape[1:]):
        raise ValueError("invalid margin_pixels for projection shape")
    occupied = projected > 0
    if not np.all(np.any(occupied, axis=(1, 2))):
        raise ValueError("stent projection is empty in at least one angle")
    edge = occupied[:, :margin_pixels, :].any()
    edge |= occupied[:, -margin_pixels:, :].any()
    edge |= occupied[:, :, :margin_pixels].any()
    edge |= occupied[:, :, -margin_pixels:].any()
    if edge:
        raise ValueError("stent projection touches the detector edge margin")


def save_stent_projection_slice(
    image, label, roi_mask, save_dir, sample_name, angle_index, *,
    include_label_in_roi=True, save_full_every=20,
):
    """Save legacy tight crops and fixed-size canvases for one projection."""
    import os

    from utils.file_rw import write_sp

    image = np.asarray(image)
    label = np.asarray(label)
    roi = np.asarray(roi_mask) > 0
    if include_label_in_roi:
        roi = roi | (label > 0)
    if image.ndim != 2 or image.shape != label.shape or image.shape != roi.shape:
        raise ValueError("image, label, and ROI projection must have the same 2D shape")
    coords = np.where(roi)
    if len(coords[0]) == 0:
        raise ValueError("ROI projection is empty")
    r0, r1 = int(coords[0].min()), int(coords[0].max()) + 1
    c0, c1 = int(coords[1].min()), int(coords[1].max()) + 1
    rectangle = np.zeros(image.shape, dtype=bool)
    rectangle[r0:r1, c0:c1] = True
    base = f"{sample_name}_{angle_index:03d}"

    views = {
        "rec": (image[r0:r1, c0:c1], label[r0:r1, c0:c1]),
        "rec_all": (image * rectangle, label * rectangle),
        "cir_all": (image * roi, label * roi),
        "cir": ((image * roi)[r0:r1, c0:c1], (label * roi)[r0:r1, c0:c1]),
    }
    for view, (image_view, label_view) in views.items():
        write_sp(image_view, os.path.join(save_dir, "image_proj", view, sample_name, base + "_0000.npz"))
        write_sp(label_view, os.path.join(save_dir, "label_proj", view, sample_name, base + ".npz"))
    if save_full_every and angle_index % save_full_every == 0:
        write_sp(image, os.path.join(save_dir, "image_proj", "full", sample_name, base + "_0000.npz"))
        write_sp(label, os.path.join(save_dir, "label_proj", "full", sample_name, base + ".npz"))
    write_sp(roi.astype(np.uint8), os.path.join(save_dir, "roi_proj", sample_name, base + ".npz"))


def place_roi_prediction(prediction, roi_mask):
    """Place either a fixed canvas or a legacy tight crop on the ROI mask."""
    prediction = np.asarray(prediction)
    roi = np.asarray(roi_mask) > 0
    if roi.ndim != 2 or prediction.ndim != 2 or not roi.any():
        raise ValueError("prediction and nonempty ROI must be 2D")
    if prediction.shape == roi.shape:
        return prediction * roi
    rows, columns = np.where(roi)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(columns.min()), int(columns.max()) + 1
    if prediction.shape != (r1 - r0, c1 - c0):
        raise ValueError("prediction shape does not match ROI crop or full detector")
    complete = np.zeros(roi.shape, dtype=prediction.dtype)
    complete[r0:r1, c0:c1] = prediction
    return complete * roi
