"""Stage 1: fill a TopBrain CT's vessels and sample the target 3D grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform, binary_dilation, generate_binary_structure

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from pipelines.utils import weighted_fill
from pipelines.medical_image import MedicalImage

default_real_recon = Path(
    "D:/desktop/stent/interior/真实数据/real_data_2/data/new/head/recon/100446_recon"
)


def sample_files(sample_dir: Path, sample: str,
                 label_path: Path) -> tuple[Path, Path]:
    """Find one named CT and the explicitly supplied vessel label."""
    image_path = sample_dir / f"{sample}_0000.nii.gz"
    for path in (image_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(f"Sample file not found: {path}")
    return image_path, label_path


def sample_split(sample_dir: Path, sample: str, test_count: int = 2) -> str:
    """Assign the last `test_count` sorted CT cases to the test set."""
    if test_count < 0:
        raise ValueError("test_count must be nonnegative")
    suffix = "_0000.nii.gz"
    samples = sorted(path.name.removesuffix(suffix)
                     for path in sample_dir.glob(f"*{suffix}") if path.is_file())
    if sample not in samples:
        raise ValueError(f"Sample {sample} not found in {sample_dir}")
    return "test" if sample in samples[-test_count:] and test_count else "train"


def target_grid(source_affine: np.ndarray, center_world: np.ndarray,
                spacing_mm: tuple[float, float, float],
                shape: tuple[int, int, int]) -> np.ndarray:
    """Keep the CT orientation and center a physical target grid on the vessel."""
    if (len(shape) != 3 or len(spacing_mm) != 3
            or min((*shape, *spacing_mm)) <= 0
            or not np.all(np.isfinite(spacing_mm))):
        raise ValueError("spacing and shape must each have three positive values")
    directions = np.array(source_affine[:3, :3], dtype=np.float64, copy=True)
    directions /= np.linalg.norm(directions, axis=0)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = directions * np.asarray(spacing_mm)
    affine[:3, 3] = center_world - affine[:3, :3] @ ((np.asarray(shape) - 1) / 2)
    return affine


def resample(volume: np.ndarray, source_affine: np.ndarray,
             target_affine: np.ndarray, shape: tuple[int, int, int],
             *, order: int, outside_value: float) -> np.ndarray:
    transform = np.linalg.inv(source_affine) @ target_affine
    return affine_transform(
        np.asarray(volume, dtype=np.float32), transform[:3, :3], transform[:3, 3],
        output_shape=shape, output=np.float32, order=order,
        mode="constant", cval=outside_value, prefilter=order > 1,
    )


def dicom_grid(real_recon: Path) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    import pydicom

    slices = sorted(real_recon.glob("*.dcm"))
    if not slices:
        raise FileNotFoundError(f"No DICOM slices in {real_recon}")
    header = pydicom.dcmread(slices[0], stop_before_pixels=True)
    spacing = (
        float(header.PixelSpacing[0]), float(header.PixelSpacing[1]),
        float(getattr(header, "SpacingBetweenSlices", header.SliceThickness)),
    )
    shape = (int(header.Rows), int(header.Columns), len(slices))
    return spacing, shape


def prepare(sample_dir: Path, sample: str, output: Path, *,
            label_path: Path,
            test_count: int = 2,
            spacing_mm: tuple[float, float, float] | None = None,
            shape: tuple[int, int, int] | None = None,
            real_recon: Path = default_real_recon) -> dict:
    """Write filled CT and vessel mask in one sample's output directory."""
    image_path, label_path = sample_files(sample_dir, sample, label_path)
    if (spacing_mm is None) != (shape is None):
        raise ValueError("spacing_mm and shape must be supplied together")
    if spacing_mm is None:
        spacing_mm, shape = dicom_grid(real_recon)
    spacing_mm = tuple(float(value) for value in spacing_mm)
    shape = tuple(int(value) for value in shape)

    image = MedicalImage.read(image_path)
    label = MedicalImage.read(label_path)
    if image.data.shape != label.data.shape or not np.allclose(image.affine, label.affine):
        raise ValueError("CT and vessel label grids do not match")
    vessel = label.data > 0
    if not vessel.any():
        raise ValueError("empty vessel mask")

    expanded = binary_dilation(vessel, structure=generate_binary_structure(3, 2), iterations=3)
    points = np.argwhere(expanded)
    low = np.maximum(points.min(axis=0) - 16, 0)
    high = np.minimum(points.max(axis=0) + 17, image.data.shape)
    region = tuple(slice(int(start), int(stop)) for start, stop in zip(low, high))
    filled = np.array(image.data, dtype=np.float32, copy=True)
    filled[region], layers = weighted_fill(filled[region], expanded[region])
    np.clip(filled, -1000, 3000, out=filled)

    center_vox = (points.min(axis=0) + points.max(axis=0)) / 2
    center_world = image.affine[:3, :3] @ center_vox + image.affine[:3, 3]
    affine = target_grid(image.affine, center_world, spacing_mm, shape)
    matched = resample(filled, image.affine, affine, shape, order=3, outside_value=-1000)
    np.clip(matched, -1000, 3000, out=matched)
    matched_vessel = resample(vessel, image.affine, affine, shape,
                              order=0, outside_value=0) > 0

    output.mkdir(parents=True, exist_ok=True)
    MedicalImage(filled, image.affine).save(output / "filled_native.nii.gz")
    MedicalImage(matched, affine).save(output / "topbrain_filled_real_grid.nii.gz")
    MedicalImage(matched_vessel.astype(np.uint8), affine).save(output / "vessel_mask_real_grid.nii.gz")
    report = {
        "sample": sample,
        "split": sample_split(sample_dir, sample, test_count),
        "test_count": test_count,
        "source_image": str(image_path),
        "source_vessel_mask": str(label_path),
        "fill_before_resample": True,
        "filled_mask_voxels": int(expanded.sum()),
        "fill_layers": layers,
        "target_shape": list(shape),
        "target_spacing_mm": list(spacing_mm),
        "target_center_world_mm": center_world.tolist(),
    }
    (output / "preparation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def prepare_all(sample_dir: Path, label_dir: Path, output_root: Path, *,
                test_count: int = 2,
                spacing_mm: tuple[float, float, float] | None = None,
                shape: tuple[int, int, int] | None = None,
                real_recon: Path = default_real_recon) -> dict:
    """Prepare every CT in `sample_dir` under its own output folder."""
    suffix = "_0000.nii.gz"
    samples = sorted(path.name.removesuffix(suffix)
                     for path in sample_dir.glob(f"*{suffix}") if path.is_file())
    if not samples:
        raise FileNotFoundError(f"No *{suffix} files in {sample_dir}")
    if test_count < 0:
        raise ValueError("test_count must be nonnegative")
    for sample in samples:
        sample_files(sample_dir, sample, label_dir / f"{sample}.nii.gz")
    if (spacing_mm is None) != (shape is None):
        raise ValueError("spacing_mm and shape must be supplied together")
    if spacing_mm is None:
        spacing_mm, shape = dicom_grid(real_recon)

    for index, sample in enumerate(samples, start=1):
        prepare(sample_dir, sample, output_root / sample,
                label_path=label_dir / f"{sample}.nii.gz",
                test_count=test_count,
                spacing_mm=spacing_mm, shape=shape, real_recon=real_recon)
        print(f"Prepared {sample} ({index}/{len(samples)})", flush=True)
    return {"count": len(samples), "samples": samples,
            "train_samples": samples[:-test_count] if test_count else samples,
            "test_samples": samples[-test_count:] if test_count else [],
            "output_root": str(output_root)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True,
                        help="sample ID, or 'all' to prepare every CT in --sample-dir")
    parser.add_argument("--sample-dir", type=Path, required=True,
                        help="directory containing <sample>_0000.nii.gz CT files")
    parser.add_argument("--label", type=Path,
                        help="vessel label NIfTI for one sample")
    parser.add_argument("--label-dir", type=Path,
                        help="directory containing <sample>.nii.gz vessel labels for --sample all")
    parser.add_argument("--output", type=Path,
                        help="one sample folder; with --sample all, parent folder for all samples")
    parser.add_argument("--real-recon", type=Path, default=default_real_recon)
    parser.add_argument("--spacing-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--shape", type=int, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--test-count", type=int, default=2,
                        help="number of last sorted CT cases reserved for test (default: 2)")
    args = parser.parse_args()
    if args.sample == "all":
        if args.label_dir is None or args.label is not None:
            parser.error("--sample all requires --label-dir and does not use --label")
        output = args.output or repo_root / "output" / "topbrain_pipeline"
        report = prepare_all(args.sample_dir, args.label_dir, output,
                             test_count=args.test_count,
                             spacing_mm=args.spacing_mm, shape=args.shape,
                             real_recon=args.real_recon)
    else:
        if args.label is None or args.label_dir is not None:
            parser.error("one sample requires --label and does not use --label-dir")
        output = args.output or repo_root / "output" / "topbrain_pipeline" / args.sample
        report = prepare(args.sample_dir, args.sample, output,
                         label_path=args.label,
                         test_count=args.test_count,
                         spacing_mm=args.spacing_mm, shape=args.shape,
                         real_recon=args.real_recon)
    print(json.dumps(report, ensure_ascii=False, indent=2))
