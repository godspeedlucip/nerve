"""Calibrate newly projected TopBrain head frames to real phantom RAW values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from pipelines.geo.reconstruct_real import find_projections
from pipelines.utils import projection_stack_path, read_frame, save_frame


DEFAULT_REAL_HEAD_PROJ_DIR = Path(
    r"D:\desktop\stent\interior\真实数据\real_data_2\data\new\head\proj\100446_proj"
)
QUANTILES = np.linspace(0.005, 0.995, 199)


def apply_head_polyfit(values: np.ndarray, coefficients: np.ndarray,
                       input_range: tuple[float, float]) -> np.ndarray:
    """Apply a fitted mapping without cubic extrapolation in the tails."""
    source = np.asarray(values, dtype=np.float32)
    if len(coefficients) > 2:
        source = np.clip(source, *input_range)
    return np.polynomial.polynomial.polyval(source, coefficients).astype(np.float32)


def calibrate_head_projections(
    work_dir: Path,
    angles_deg: list[float],
    view_indices: list[int],
    output_mode: str,
    reference_dir: Path = DEFAULT_REAL_HEAD_PROJ_DIR,
    *,
    max_fit_views: int = 21,
) -> dict:
    """Fit on sampled view quantiles, then transform every saved head frame.

    Views are paired by acquisition progress because the phantom and TopBrain
    subjects are different and their exact angular registration is unverified.
    Odd sampled views validate the fit; the fit uses the even sampled views.
    """
    work_dir, reference_dir = Path(work_dir), Path(reference_dir)
    if output_mode not in ("2D", "3D") or not angles_deg or len(angles_deg) != len(view_indices):
        raise ValueError("head calibration needs matching nonempty views in 2D or 3D mode")
    if max_fit_views < 1:
        raise ValueError("max_fit_views must be positive")
    reference_files, reference_shape = find_projections(reference_dir)
    count = len(view_indices)
    selected = np.unique(np.rint(
        np.linspace(0, count - 1, min(count, max_fit_views))
    ).astype(int))
    stack_path = projection_stack_path(work_dir, "head")

    def read_head(page: int) -> np.ndarray:
        if output_mode == "2D":
            return read_frame(work_dir, "head", angles_deg[page])
        with tifffile.TiffFile(stack_path) as source:
            return source.pages[page].asarray()

    pairs = []
    for page in selected:
        head = read_head(int(page))
        if head.shape != reference_shape or not np.isfinite(head).all():
            raise ValueError("head projections must match finite real RAW frame shape")
        reference_index = round(int(page) / max(count - 1, 1) * (len(reference_files) - 1))
        real = np.fromfile(reference_files[reference_index], dtype="<f4").reshape(reference_shape)
        if not np.isfinite(real).all():
            raise ValueError(f"non-finite reference projection: {reference_files[reference_index]}")
        pairs.append((int(page), int(reference_index),
                      np.quantile(head, QUANTILES), np.quantile(real, QUANTILES)))

    train = pairs[::2]
    validation = pairs[1::2] or train
    x_train = np.concatenate([pair[2] for pair in train])
    y_train = np.concatenate([pair[3] for pair in train])
    x_valid = np.concatenate([pair[2] for pair in validation])
    y_valid = np.concatenate([pair[3] for pair in validation])
    input_min, input_max = float(x_train.min()), float(x_train.max())
    if input_max == input_min:
        coefficients = np.array([float(np.median(y_train) - input_min), 1.0])
    else:
        linear = np.polynomial.Polynomial.fit(x_train, y_train, 1).convert()
        coefficients = linear.coef
        linear_mae = float(np.mean(np.abs(linear(x_valid) - y_valid)))
        if len(np.unique(x_train)) >= 4:
            cubic = np.polynomial.Polynomial.fit(x_train, y_train, 3).convert()
            grid = np.linspace(input_min, input_max, 256)
            cubic_mae = float(np.mean(np.abs(
                cubic(np.clip(x_valid, input_min, input_max)) - y_valid)))
            if (np.isfinite(cubic.coef).all() and np.min(cubic.deriv()(grid)) > 0
                    and linear_mae - cubic_mae > max(0.01 * linear_mae, 1e-4)):
                coefficients = cubic.coef
    if not np.isfinite(coefficients).all():
        raise ValueError("head polyfit produced non-finite coefficients")

    if output_mode == "2D":
        for page in range(count):
            calibrated = apply_head_polyfit(read_head(page), coefficients,
                                            (input_min, input_max))
            save_frame(work_dir, "head", angles_deg[page], calibrated)
    else:
        temporary = stack_path.with_name("head.polyfit.tmp.tif")
        try:
            with tifffile.TiffWriter(temporary) as writer:
                for page in range(count):
                    calibrated = apply_head_polyfit(read_head(page), coefficients,
                                                    (input_min, input_max))
                    writer.write(calibrated, compression="zlib",
                                 compressionargs={"level": 1},
                                 photometric="minisblack", metadata=None)
            temporary.replace(stack_path)
        finally:
            temporary.unlink(missing_ok=True)

    return {
        "applied": True,
        "reference_dir": str(reference_dir),
        "reference_view_count": len(reference_files),
        "pairing": "normalized acquisition progress; not registered pixel by pixel",
        "training_view_indices": [pair[0] for pair in train],
        "validation_view_indices": [pair[0] for pair in validation],
        "reference_indices": [pair[1] for pair in pairs],
        "quantile_range": [float(QUANTILES[0]), float(QUANTILES[-1])],
        "input_range": [input_min, input_max],
        "degree": len(coefficients) - 1,
        "coefficients_ascending": coefficients.tolist(),
        "tail_policy": ("clip input to training quantile range"
                        if len(coefficients) > 2 else "linear extrapolation"),
        "validation_quantile_mae_before": float(np.mean(np.abs(x_valid - y_valid))),
        "validation_quantile_mae_after": float(np.mean(np.abs(
            apply_head_polyfit(x_valid, coefficients, (input_min, input_max)) - y_valid))),
    }
