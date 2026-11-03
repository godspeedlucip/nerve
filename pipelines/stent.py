"""Stage 2: turn a PLY mesh into a complete 3D stent and marker metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from pipelines.medical_image import MedicalImage

default_mesh_dir = repo_root / "pipelines" / "ply"
default_mesh = default_mesh_dir / "train" / "stent1.ply"
default_save_path = repo_root / "output" / "stent_models"
default_spacing_mm = (0.188946, 0.188946, 0.188946)
default_shape = (512, 512, 533)
default_stent_length_mm = 21.0
default_stent_diameter_mm = 2.5
default_variants = 10


def read_stent_points(mesh_path: Path) -> np.ndarray:
    from plyfile import PlyData

    vertices = PlyData.read(str(mesh_path))["vertex"]
    return np.stack([np.asarray(vertices[axis]) for axis in ("x", "y", "z")], axis=1).astype(np.float64)


def centerline_offsets_mm(along: np.ndarray, seed: int | None = None,
                          length_mm: float = default_stent_length_mm) -> np.ndarray:
    """Smooth lateral offsets for a reproducible, multi-bend 3D centerline."""
    t = np.asarray(along, dtype=np.float64)
    if seed is None:
        x = (0.75 * np.sin(2 * np.pi * t)
             + 0.40 * np.sin(4 * np.pi * t)
             + 0.25 * np.sin(6 * np.pi * t))
        y = (5.0 * (t - 0.5)
             + 0.65 * np.sin(np.pi * t)
             - 0.55 * np.sin(3 * np.pi * t)
             + 0.30 * np.sin(4 * np.pi * t))
    else:
        rng = np.random.default_rng(seed)
        phases_x = rng.uniform(-0.35, 0.35, 3)
        phases_y = rng.uniform(-0.35, 0.35, 3)
        amp_x = np.array([0.75, 0.40, 0.25]) * rng.uniform(0.8, 1.2, 3)
        amp_y = np.array([0.65, -0.55, 0.30]) * rng.uniform(0.8, 1.2, 3)
        x = np.sin(np.pi * t) * sum(
            amp * np.sin(2 * np.pi * harmonic * t + phase)
            for harmonic, (amp, phase) in enumerate(zip(amp_x, phases_x), start=1)
        )
        y = (5.0 * (t - 0.5) + np.sin(np.pi * t) * sum(
            amp * np.sin(np.pi * harmonic * t + phase)
            for harmonic, (amp, phase) in enumerate(zip(amp_y, phases_y), start=1)
        ))
    return np.stack((x, y), axis=-1) * (length_mm / default_stent_length_mm)


def add_box(mask: np.ndarray, center: np.ndarray, half_size: np.ndarray) -> None:
    """Fill a small axis-aligned marker box in an (x, y, z) mask."""
    low = np.maximum(0, np.floor(center - half_size).astype(int))
    high = np.minimum(mask.shape, np.ceil(center + half_size + 1).astype(int))
    region = tuple(slice(int(start), int(stop)) for start, stop in zip(low, high))
    axes = np.ogrid[region]
    inside = [np.abs(axis - point) <= radius
              for axis, point, radius in zip(axes, center, half_size)]
    mask[region] |= np.logical_and.reduce(np.broadcast_arrays(*inside))


def sample_translation_voxels(coords: np.ndarray, shape: tuple[int, int, int],
                              spacing: np.ndarray, seed: int) -> np.ndarray:
    """Choose a repeatable translation that leaves room for both end markers."""
    marker_margin = np.ceil(np.array([0.35, 0.35, 1.7]) / spacing).astype(int) + 1
    max_move = np.floor(np.array([20.0, 20.0, 8.0]) / spacing).astype(int)
    low = np.maximum(marker_margin - coords.min(axis=0), -max_move)
    high = np.minimum(np.asarray(shape) - 1 - marker_margin - coords.max(axis=0), max_move)
    if np.any(low > high):
        raise ValueError("stent and markers do not fit in the volume after translation")
    rng = np.random.default_rng(seed ^ 0x9E3779B9)
    shift = rng.integers(low, high + 1)
    if not np.any(shift) and np.any((low < 0) | (high > 0)):
        axis = int(np.argmax(high - low))
        shift[axis] = 1 if high[axis] > 0 else -1
    return shift.astype(int)


def create_stent_masks(points: np.ndarray, shape: tuple[int, int, int],
                       spacing_mm: np.ndarray, *,
                       length_mm: float = default_stent_length_mm,
                       diameter_mm: float = default_stent_diameter_mm,
                       variation_seed: int | None = None
                       ) -> tuple[np.ndarray, np.ndarray, list[list[float]], np.ndarray]:
    """Place a 21 mm multi-bend stent and 0.7 mm end markers."""
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    vertices = np.asarray(points, dtype=np.float64).copy()
    if (spacing.shape != (3,) or np.any(spacing <= 0)
            or vertices.ndim != 2 or vertices.shape[1] != 3):
        raise ValueError("expected positive x/y/z spacing and Nx3 mesh points")
    if not np.isfinite(length_mm) or length_mm <= 0 or not np.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError("stent length and diameter must be positive values in mm")
    vertices[:, :2] -= (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2
    vertices[:, :2] *= diameter_mm / np.ptp(vertices[:, :2], axis=0).max()
    vertices[:, 2] -= vertices[:, 2].min()
    vertices[:, 2] *= length_mm / vertices[:, 2].max()
    along = vertices[:, 2] / length_mm
    lateral = centerline_offsets_mm(along, variation_seed, length_mm)
    center_mm = (np.asarray(shape) - 1) * spacing / 2
    bent_mm = np.stack([
        center_mm[0] + vertices[:, 0] + lateral[:, 0],
        center_mm[1] + vertices[:, 1] + lateral[:, 1],
        center_mm[2] - length_mm / 2 + vertices[:, 2],
    ], axis=1)
    coords = np.rint(bent_mm / spacing).astype(np.int32)
    if not np.all((coords >= 0) & (coords < np.asarray(shape))):
        raise ValueError("stent mesh extends beyond the CT volume")
    translation = (np.zeros(3, dtype=int) if variation_seed is None else
                   sample_translation_voxels(coords, shape, spacing, variation_seed))
    coords += translation
    stent = np.zeros(shape, dtype=np.uint8)
    stent[tuple(coords.T)] = 1

    marker = np.zeros(shape, dtype=np.uint8)
    centers_mm = []
    stent_z = np.flatnonzero(stent.any(axis=(0, 1)))
    for tip, direction in ((int(stent_z.min()), -1), (int(stent_z.max()), 1)):
        xy = np.argwhere(stent[:, :, tip]).mean(axis=0)
        center = np.array([xy[0], xy[1], tip + direction * 1.2 / spacing[2]])
        add_box(marker, center, np.array([0.7, 0.7, 1.0]) / (2 * spacing))
        centers_mm.append((center * spacing).tolist())

    marker_z = np.flatnonzero(marker.any(axis=(0, 1)))
    if np.any(stent & marker) or np.any(marker[:, :, stent_z.min():stent_z.max() + 1]):
        raise ValueError("marker overlaps stent")
    if not (marker_z.min() < stent_z.min() and marker_z.max() > stent_z.max()):
        raise ValueError("markers must lie beyond both stent ends")
    return stent, marker, centers_mm, translation


def save_stent_outputs(output_dir: Path, stent: np.ndarray,
                       marker: np.ndarray, affine: np.ndarray) -> None:
    """Save the binary union and each marker's occupied voxel coordinates."""
    combined = np.bitwise_or(stent, marker)
    MedicalImage(combined, affine).save(output_dir / "stent_real_grid.nii.gz")

    stent_z = np.flatnonzero(stent.any(axis=(0, 1)))
    marker_voxels = np.argwhere(marker > 0)
    entries = []
    for end, selected in (
        ("low_z", marker_voxels[:, 2] < stent_z.min()),
        ("high_z", marker_voxels[:, 2] > stent_z.max()),
    ):
        voxels = marker_voxels[selected]
        center = voxels.mean(axis=0)
        world = np.asarray(affine) @ np.append(center, 1.0)
        entries.append({
            "end": end,
            "center_voxel_ijk": center.tolist(),
            "center_world_mm": world[:3].tolist(),
            "voxel_count": len(voxels),
            "voxels_ijk": voxels.tolist(),
        })
    (output_dir / "marker.json").write_text(
        json.dumps({"coordinate_order": "i,j,k", "markers": entries}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "marker_real_grid.nii.gz").unlink(missing_ok=True)


def prune_stale_variants(series_dir: Path, nums: int) -> None:
    """Remove only older generated variants outside the requested 1..nums set."""
    known_files = {"stent.json", "marker.json", "stent_real_grid.nii.gz",
                   "marker_real_grid.nii.gz"}
    root = series_dir.resolve()
    for old in series_dir.iterdir():
        if not old.is_dir() or not old.name.isdigit() or int(old.name) <= nums:
            continue
        if not old.resolve().is_relative_to(root):
            raise ValueError(f"variant path escapes output series: {old}")
        report_path = old / "stent.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("series") != series_dir.name or report.get("variant") != int(old.name):
            continue
        if any(path.name not in known_files for path in old.iterdir()):
            raise ValueError(f"cannot prune variant with unexpected files: {old}")
        shutil.rmtree(old)


def make_stent(work_dir: Path, mesh_path: Path = default_mesh, *,
               length_mm: float = default_stent_length_mm,
               diameter_mm: float = default_stent_diameter_mm) -> dict:
    ct = nib.load(str(work_dir / "topbrain_filled_real_grid.nii.gz"))
    shape = ct.shape
    spacing = np.asarray(ct.header.get_zooms()[:3])
    seed_key = f"{mesh_path.resolve()}:{work_dir.resolve()}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:4], "little")
    stent, marker, centers_mm, translation = create_stent_masks(
        read_stent_points(mesh_path), shape, spacing,
        length_mm=length_mm, diameter_mm=diameter_mm, variation_seed=seed,
    )
    save_stent_outputs(work_dir, stent, marker, ct.affine)
    report = {
        "mesh": str(mesh_path), "volume_shape": list(shape), "seed": seed,
        "stent_length_mm": length_mm, "stent_diameter_mm": diameter_mm,
        "marker_dimensions_mm": [0.7, 0.7, 1.0], "marker_outward_mm": 1.2,
        "stent_voxels": int(stent.sum()), "marker_voxels": int(marker.sum()),
        "combined_voxels": int(stent.sum() + marker.sum()),
        "marker_centers_mm": centers_mm,
        "translation_voxels_ijk": translation.tolist(),
        "translation_mm_xyz": (translation * spacing).tolist(),
    }
    (work_dir / "stent.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def generate_stent_series(input_dir: Path, nums: int, series: str,
                          save_path: Path, spacing_mm: tuple[float, float, float],
                          shape: tuple[int, int, int] = default_shape,
                          length_mm: float = default_stent_length_mm,
                          diameter_mm: float = default_stent_diameter_mm,
                          *, split: str = "all") -> dict:
    """Generate `nums` deterministic curve variants for each selected PLY."""
    if nums <= 0:
        raise ValueError("nums must be a positive integer")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"PLY input directory not found: {input_dir}")
    if split not in ("all", "train", "test"):
        raise ValueError("split must be all, train or test")
    meshes = sorted(path for path in input_dir.rglob("*.ply") if path.is_file())
    if split != "all":
        meshes = [path for path in meshes
                  if ("test" if "test" in path.parts else "train") == split]
    if series.lower() != "all":
        meshes = [path for path in meshes if path.stem.lower() == series.lower()]
        if not meshes:
            raise FileNotFoundError(f"No PLY named '{series}' in {input_dir}")
    if not meshes:
        raise FileNotFoundError(f"No PLY files in {input_dir}")
    if len({path.stem for path in meshes}) != len(meshes):
        raise ValueError("PLY filenames must be unique across train/test splits")

    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("voxel spacing must contain three positive values in mm")
    shape = tuple(int(value) for value in shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("shape must contain three positive dimensions")

    save_path.mkdir(parents=True, exist_ok=True)
    reports = []
    affine = np.diag([*spacing, 1.0])
    for mesh_path in meshes:
        points = read_stent_points(mesh_path)
        mesh_report = []
        for index in range(1, nums + 1):
            seed_key = f"{mesh_path.resolve()}:{index}".encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:4], "little")
            stent, marker, centers_mm, translation = create_stent_masks(
                points, shape, spacing, length_mm=length_mm,
                diameter_mm=diameter_mm, variation_seed=seed
            )
            variant_dir = save_path / mesh_path.stem / f"{index:03d}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            save_stent_outputs(variant_dir, stent, marker, affine)
            report = {
                "series": mesh_path.stem, "variant": index,
                "split": "test" if "test" in mesh_path.parts else "train",
                "mesh": str(mesh_path), "volume_shape": list(shape),
                "voxel_spacing_mm": spacing.tolist(), "seed": seed,
                "stent_length_mm": length_mm, "stent_diameter_mm": diameter_mm,
                "marker_dimensions_mm": [0.7, 0.7, 1.0],
                "marker_outward_mm": 1.2,
                "stent_voxels": int(stent.sum()),
                "marker_voxels": int(marker.sum()),
                "combined_voxels": int(stent.sum() + marker.sum()),
                "marker_centers_mm": centers_mm,
                "translation_voxels_ijk": translation.tolist(),
                "translation_mm_xyz": (translation * spacing).tolist(),
            }
            (variant_dir / "stent.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            mesh_report.append(report)
        prune_stale_variants(save_path / mesh_path.stem, nums)
        reports.append({"series": mesh_path.stem, "variants": mesh_report})
    return {"count": len(reports) * nums, "output_dir": str(save_path), "series": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=default_mesh_dir,
                        help="folder containing PLY files (searched recursively)")
    parser.add_argument("--nums", type=int, default=default_variants,
                        help="number of stent variants to generate for each PLY")
    parser.add_argument("--series", default="all",
                        help="PLY filename stem to generate, or 'all'")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all",
                        help="select PLY split from train/test folders (default: all)")
    parser.add_argument("--save_path", type=Path, default=default_save_path)
    parser.add_argument("--voxel", type=float, nargs=3, default=default_spacing_mm,
                        metavar=("X", "Y", "Z"),
                        help="voxel spacing in mm; defaults to the real data spacing")
    parser.add_argument("--shape", type=int, nargs=3, default=default_shape,
                        metavar=("X", "Y", "Z"), help="output volume shape")
    parser.add_argument("--length-mm", type=float, default=default_stent_length_mm,
                        help="stent length in mm (default: 21.0)")
    parser.add_argument("--diameter-mm", type=float, default=default_stent_diameter_mm,
                        help="stent diameter in mm (default: 2.5)")
    args = parser.parse_args()
    print(json.dumps(generate_stent_series(
        args.input_dir, args.nums, args.series, args.save_path,
        tuple(args.voxel), tuple(args.shape), args.length_mm, args.diameter_mm,
        split=args.split,
    ), ensure_ascii=False, indent=2))
