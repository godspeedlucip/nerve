"""Small helpers shared by the four pipeline stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import convolve

from pipelines.medical_image import MedicalImage


def weighted_fill(image: np.ndarray, mask: np.ndarray, max_iters: int = 50) -> tuple[np.ndarray, int]:
    """Fill masked voxels in layers using 26-neighbour face/edge/corner weights."""
    filled = np.asarray(image, dtype=np.float32).copy()
    remaining = np.asarray(mask, dtype=bool).copy()
    kernel = np.zeros((3, 3, 3), dtype=np.float64)
    weights_by_distance = (0.0, 1.0, 0.7, 0.5)
    for index in np.ndindex(kernel.shape):
        distance = sum(abs(axis - 1) for axis in index)
        kernel[index] = weights_by_distance[distance]

    for layer in range(max_iters):
        if not remaining.any():
            return filled, layer
        valid = ~remaining
        weights = convolve(valid.astype(np.float64), kernel, mode="constant", cval=0.0)
        boundary = remaining & (weights > 0)
        if not boundary.any():
            raise RuntimeError("masked region has no valid boundary")
        sums = convolve(np.where(valid, filled, 0.0).astype(np.float64), kernel,
                        mode="constant", cval=0.0)
        filled[boundary] = (sums[boundary] / weights[boundary]).astype(np.float32)
        remaining[boundary] = False
    if remaining.any():
        raise RuntimeError(f"{int(remaining.sum())} masked voxels remain unfilled")
    return filled, max_iters


def frame_path(work_dir: Path, kind: str, angle: float) -> Path:
    value = float(angle)
    if not np.isfinite(value):
        raise ValueError("projection angle must be finite")
    whole, dot, fraction = f"{value:.12f}".rstrip("0").rstrip(".").partition(".")
    name = f"{int(whole):03d}" + (f".{fraction}" if dot else "")
    return work_dir / "projections" / kind / f"{name}.tif"


def projection_stack_path(work_dir: Path, kind: str) -> Path:
    return work_dir / "projections" / f"{kind}.tif"


def prune_projection_outputs(work_dir: Path, kind: str,
                             angles_deg: list[float], output_mode: str) -> None:
    """Remove generated files from a previous run in the other format or angle set."""
    stack_path = projection_stack_path(work_dir, kind)
    keep = {frame_path(work_dir, kind, angle) for angle in angles_deg} if output_mode == "2D" else set()
    if output_mode == "2D":
        stack_path.unlink(missing_ok=True)
    frame_dir = stack_path.parent / kind
    if frame_dir.is_dir():
        for path in frame_dir.glob("*.tif"):
            if path not in keep:
                path.unlink()
        if output_mode == "3D" and not any(frame_dir.iterdir()):
            frame_dir.rmdir()


def save_frame(work_dir: Path, kind: str, angle: float, values: np.ndarray) -> None:
    MedicalImage(np.asarray(values, dtype=np.float32)).save(
        frame_path(work_dir, kind, angle), compression="zlib", compressionargs={"level": 1}
    )


def read_frame(work_dir: Path, kind: str, angle: float) -> np.ndarray:
    return np.asarray(MedicalImage.read(frame_path(work_dir, kind, angle)).data,
                      dtype=np.float32)


def read_projection_frame(work_dir: Path, kind: str, page_index: int,
                          frame_index: int, *, stack_path: Path | None = None) -> np.ndarray:
    """Read one view from a 3D TIFF stack, with support for legacy frame files."""
    path = Path(stack_path) if stack_path is not None else projection_stack_path(work_dir, kind)
    if path.is_file():
        return np.asarray(
            MedicalImage.read_frames(path, page_index, page_index + 1).data[0],
            dtype=np.float32,
        )
    return read_frame(work_dir, kind, frame_index)


def hu_to_attenuation(hu: np.ndarray, mu_water: float = 0.0227) -> np.ndarray:
    if mu_water <= 0:
        raise ValueError("mu_water must be positive")
    return (np.asarray(hu, dtype=np.float32) / 1000 + 1) * np.float32(mu_water)


def parse_angles(spec: str) -> list[int]:
    """Accept `all`, an inclusive `start:stop`, or comma-separated degrees."""
    if spec == "all":
        return list(range(360))
    if ":" in spec:
        start, stop = (int(part) for part in spec.split(":"))
        angles = list(range(start, stop + 1))
    else:
        angles = [int(part) for part in spec.split(",")]
    if not angles or len(angles) != len(set(angles)) or any(not 0 <= angle < 360 for angle in angles):
        raise ValueError("angles must be unique integers from 0 through 359")
    return angles
