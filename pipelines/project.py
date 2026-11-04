"""Stage 3: project the prepared head, stent and markers at selected angles."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import nibabel as nib
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, label as connected_components

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from pipelines.utils import (frame_path, hu_to_attenuation, projection_stack_path,
                             prune_projection_outputs, save_frame)
from pipelines.medical_image import MedicalImage
from pipelines.stent_fov import (
    configure_safe_stent_geometry,
    validate_stent_projection,
)
from pipelines.tool import TigreTool
from pipelines.calibration.auto_polyfit import (
    DEFAULT_REAL_HEAD_PROJ_DIR, calibrate_head_projections,
)


N_DETECTOR = (1056, 1056)
D_DETECTOR_MM = (0.150, 0.150)
DSO_MM = 765.0
DSD_MM = 1200.0
VOXEL_SPACING_MULTIPLIER = 1.0
DEFAULT_N_PROJECTIONS = 31
DEFAULT_ANGLE_SPAN_DEG = 300.0
DEFAULT_OUTPUT_MODE = "2D"
DEFAULT_STENT_MU = 0.17
DEFAULT_MARKER_MU = 1.02


def _variant_matches(directory_name: str, requested: str | None) -> bool:
    if requested is None:
        return True
    if directory_name == str(requested):
        return True
    return directory_name.isdigit() and str(requested).isdigit() and int(directory_name) == int(requested)


def to_tigre_volume(xyz: np.ndarray) -> np.ndarray:
    """NIfTI uses (x, y, z); TIGRE expects (z, y, x)."""
    return np.ascontiguousarray(np.transpose(xyz, (2, 1, 0)), dtype=np.float32)


def implant_attenuation(stent: np.ndarray, marker: np.ndarray,
                        stent_mu: float, marker_mu: float) -> np.ndarray:
    """Assign marker attenuation once, even when stent includes the marker."""
    volume = np.zeros(stent.shape, dtype=np.float32)
    volume[stent > 0] = stent_mu
    volume[marker > 0] = marker_mu
    return volume


def read_marker_mask(path: Path, combined: np.ndarray) -> np.ndarray:
    """Rebuild marker occupancy from the saved voxel lists in marker.json."""
    metadata = json.loads(path.read_text(encoding="utf-8"))
    mask = np.zeros(combined.shape, dtype=np.uint8)
    for entry in metadata["markers"]:
        voxels = np.asarray(entry["voxels_ijk"], dtype=np.int64)
        if voxels.ndim != 2 or voxels.shape[1] != 3 or not len(voxels):
            raise ValueError(f"invalid marker voxels in {path}")
        if np.any(voxels < 0) or np.any(voxels >= np.asarray(combined.shape)):
            raise ValueError(f"marker voxels outside stent volume in {path}")
        if not np.all(combined[tuple(voxels.T)] > 0):
            raise ValueError(f"marker voxels missing from stent volume in {path}")
        mask[tuple(voxels.T)] = 1
    return mask


def project_volume(volume: np.ndarray, geometry, angles_rad: np.ndarray,
                   view_indices: list[int],
                   work_dir: Path, kind: str, *, blur_px: float = 0,
                   check_margin: bool = False, output_mode: str = "3D",
                   angles_deg: list[float] | None = None) -> None:
    if output_mode not in ("2D", "3D"):
        raise ValueError("output_mode must be 2D or 3D")
    if output_mode == "2D":
        if angles_deg is None or len(angles_deg) != len(view_indices):
            raise ValueError("2D output requires one angle per view")
        paths = [frame_path(work_dir, kind, angle) for angle in angles_deg]
        if len(paths) != len(set(paths)):
            raise ValueError("2D projection angles produce duplicate filenames")
    stack_path = projection_stack_path(work_dir, kind)
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    context = tifffile.TiffWriter(stack_path) if output_mode == "3D" else nullcontext()
    with context as writer:
        for start in range(0, len(view_indices), 12):
            batch_indices = view_indices[start:start + 12]
            batch = TigreTool.project_tigre(
                volume, geometry, angles_rad[start:start + 12]
            )
            if batch.ndim != 3 or batch.shape[0] != len(batch_indices):
                raise RuntimeError(
                    f"expected {len(batch_indices)} projections from TIGRE, got {batch.shape}"
                )
            if check_margin:
                validate_stent_projection(batch, margin_pixels=16)
            for offset, values in enumerate(batch):
                if blur_px > 0:
                    values = gaussian_filter(values, sigma=blur_px)
                if output_mode == "2D":
                    save_frame(work_dir, kind, angles_deg[start + offset], values)
                else:
                    writer.write(
                        np.asarray(values, dtype=np.float32),
                        compression="zlib", compressionargs={"level": 1},
                        photometric="minisblack", metadata=None,
                    )
    prune_projection_outputs(work_dir, kind, angles_deg or [], output_mode)


def project_marker_metadata(marker_volume: np.ndarray, geometry,
                            angles_rad: np.ndarray, angles_deg: list[float],
                            view_indices: list[int], output_dir: Path) -> None:
    """Write one JSON per view with marker center and occupied detector pixels."""
    metadata_dir = output_dir / "marker_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(view_indices), 12):
        batch_indices = view_indices[start:start + 12]
        projections = TigreTool.project_tigre(
            marker_volume, geometry, angles_rad[start:start + 12]
        )
        if projections.ndim != 3 or projections.shape[0] != len(batch_indices):
            raise RuntimeError(
                f"expected {len(batch_indices)} marker projections from TIGRE, got {projections.shape}"
            )
        for offset, (view_index, values) in enumerate(zip(batch_indices, projections)):
            occupied = values > 0
            pixels = np.argwhere(occupied)
            components, count = connected_components(occupied)
            centers = [
                np.argwhere(components == component).mean(axis=0).tolist()
                for component in range(1, count + 1)
            ]
            record = {
                "view_index": int(view_index),
                "angle_deg": float(angles_deg[start + offset]),
                "coordinate_order": "row_column",
                "index_base": 0,
                "marker_center_rc": pixels.mean(axis=0).tolist() if len(pixels) else None,
                "marker_centers_rc": centers,
                "marker_pixels_rc": pixels.tolist(),
            }
            (metadata_dir / f"{view_index:03d}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
    expected = {f"{view_index:03d}.json" for view_index in view_indices}
    for path in metadata_dir.glob("*.json"):
        if path.stem.isdecimal() and path.name not in expected:
            path.unlink()


def project(work_dir: Path, angles: list[int] | None = None, *,
            n_projections: int = 600,
            angle_span_deg: float = 200.0) -> dict:
    """Use one geometry for all three volumes, then record the projected angles."""
    tigre_tool = TigreTool()
    if angles is None:
        if n_projections < 1 or not 0 < angle_span_deg <= 360:
            raise ValueError("n_projections must be positive and angle_span_deg in (0, 360]")
        angles_rad = tigre_tool.set_angles(0, angle_span_deg, n_projections)
        angles_deg = np.linspace(
            0, angle_span_deg, n_projections, endpoint=False
        ).tolist()
        view_indices = list(range(n_projections))
    else:
        if (not angles or len(set(angles)) != len(angles)
                or any(not isinstance(angle, int) or not 0 <= angle < 360
                       for angle in angles)):
            raise ValueError("angles must be unique integers from 0 through 359")
        angles_deg = angles
        angles_rad = np.deg2rad(np.asarray(angles, dtype=np.float32))
        view_indices = angles
    preparation = json.loads((work_dir / "preparation.json").read_text(encoding="utf-8"))
    ct_path = work_dir / "topbrain_filled_real_grid.nii.gz"
    ct_header = nib.load(str(ct_path))
    shape = ct_header.shape
    spacing = np.asarray(ct_header.header.get_zooms()[:3])
    geometry = tigre_tool.set_geometry(
        volume_shape=np.asarray(shape)[[2, 1, 0]],
        voxel_spacing=spacing[[2, 1, 0]] * VOXEL_SPACING_MULTIPLIER,
        n_detector=N_DETECTOR,
        d_detector=D_DETECTOR_MM,
        dso=DSO_MM,
        dsd=DSD_MM,
    )
    geometry.offOrigin = np.zeros(3, dtype=np.float32)
    geometry.offDetector = np.zeros(2, dtype=np.float32)

    stent = MedicalImage.read(work_dir / "stent_real_grid.nii.gz")
    if stent.data.shape != shape or not np.allclose(stent.affine, ct_header.affine):
        raise ValueError("implant mask and prepared CT grids do not match")
    marker = read_marker_mask(work_dir / "marker.json", stent.data)
    geometry_info = configure_safe_stent_geometry(
        geometry, np.transpose(stent.data > 0, (2, 1, 0)),
        preparation["sample"], randomize=False,
    )
    marker_volume = to_tigre_volume(marker)
    project_volume(
        to_tigre_volume(implant_attenuation(
            stent.data, marker, DEFAULT_STENT_MU, DEFAULT_MARKER_MU
        )),
        geometry, angles_rad, view_indices, work_dir, "implant", check_margin=True,
    )
    project_marker_metadata(
        marker_volume, geometry, angles_rad, angles_deg, view_indices, work_dir
    )
    del stent
    del marker
    gc.collect()

    ct = MedicalImage.read(ct_path)
    head = to_tigre_volume(hu_to_attenuation(ct.data))
    del ct
    gc.collect()
    project_volume(head, geometry, angles_rad, view_indices, work_dir,
                   "head", blur_px=1.0)

    report = {
        "sample": preparation["sample"],
        "angles": angles_deg,
        "angles_deg": angles_deg,
        "view_indices": view_indices,
        "n_projections": len(view_indices),
        "angle_start_deg": 0.0 if angles is None else None,
        "angle_span_deg": angle_span_deg if angles is None else None,
        "angle_endpoint_included": False if angles is None else None,
        "head_source": str(ct_path),
        "head_sigma_px": 1.0,
        "mu_water": 0.0227,
        "voxel_spacing_multiplier": VOXEL_SPACING_MULTIPLIER,
        "stent_mu": DEFAULT_STENT_MU,
        "marker_mu": DEFAULT_MARKER_MU,
        "geometry": geometry_info,
    }
    (work_dir / "projection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _new_projection_setup(shape, spacing, *, n_projections: int,
                          angle_span_deg: float):
    if n_projections < 1 or not 0 < angle_span_deg <= 360:
        raise ValueError("n_projections must be positive and angle_span_deg in (0, 360]")
    shape = tuple(shape)
    spacing = np.asarray(spacing, dtype=np.float64)
    tigre_tool = TigreTool()
    geometry = tigre_tool.set_geometry(
        volume_shape=np.asarray(shape)[[2, 1, 0]],
        voxel_spacing=spacing[[2, 1, 0]] * VOXEL_SPACING_MULTIPLIER,
        n_detector=N_DETECTOR,
        d_detector=D_DETECTOR_MM,
        dso=DSO_MM,
        dsd=DSD_MM,
    )
    angles_deg = np.linspace(0, angle_span_deg, n_projections, endpoint=True).tolist()
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float32))
    return geometry, angles_rad, angles_deg, list(range(n_projections))


def _project_stent_variant(output_dir: Path, stent_dir: Path,
                           series: str, variant: str, *, n_projections: int,
                           angle_span_deg: float, stent_mu: float,
                           marker_mu: float, output_mode: str) -> dict:
    variant_dir = stent_dir / series / str(variant)
    stent_info = json.loads((variant_dir / "stent.json").read_text(encoding="utf-8"))
    if stent_info.get("split") not in ("train", "test"):
        raise ValueError(f"split missing or invalid for stent {series}/{variant}; regenerate model metadata")
    shape = tuple(stent_info["volume_shape"])
    spacing = np.asarray(stent_info["voxel_spacing_mm"], dtype=np.float64)
    geometry, angles_rad, angles_deg, view_indices = _new_projection_setup(
        shape, spacing, n_projections=n_projections, angle_span_deg=angle_span_deg
    )
    stent = MedicalImage.read(variant_dir / "stent_real_grid.nii.gz")
    if stent.data.shape != shape:
        raise ValueError("stent mask must match stent.json volume_shape")
    marker = read_marker_mask(variant_dir / "marker.json", stent.data)
    geometry_info = configure_safe_stent_geometry(
        geometry,
        np.transpose(stent.data > 0, (2, 1, 0)),
        f"{series}:{variant}", randomize=False,
    )
    marker_volume = to_tigre_volume(marker)
    implant_volume = to_tigre_volume(implant_attenuation(
        stent.data, marker, stent_mu, marker_mu
    ))
    project_volume(implant_volume, geometry, angles_rad, view_indices,
                   output_dir, "implant", check_margin=True,
                   output_mode=output_mode, angles_deg=angles_deg)
    project_marker_metadata(
        marker_volume, geometry, angles_rad, angles_deg, view_indices, output_dir
    )
    report = {
        "stent_series": series,
        "stent_variant": str(variant),
        "split": stent_info.get("split"),
        "stent_source": str(variant_dir),
        "stent_mu": stent_mu,
        "marker_mu": marker_mu,
        "marker_metadata_dir": str(output_dir / "marker_metadata"),
        "angles": angles_deg,
        "angles_deg": angles_deg,
        "view_indices": view_indices,
        "n_projections": len(view_indices),
        "output_mode": output_mode,
        "angle_start_deg": 0.0,
        "angle_span_deg": angle_span_deg,
        "angle_endpoint_included": True,
        "voxel_spacing_multiplier": VOXEL_SPACING_MULTIPLIER,
        "geometry": geometry_info,
    }
    (output_dir / "projection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def project_stent(stent_dir: Path, output_root: Path,
                  series: str | None = None,
                  variant: str | None = None, *,
                  n_projections: int = DEFAULT_N_PROJECTIONS,
                  angle_span_deg: float = DEFAULT_ANGLE_SPAN_DEG,
                  stent_mu: float = DEFAULT_STENT_MU,
                  marker_mu: float = DEFAULT_MARKER_MU,
                  output_mode: str = DEFAULT_OUTPUT_MODE) -> list[dict]:
    """Project the variants matching optional series and variant filters."""
    stent_dir, output_root = Path(stent_dir), Path(output_root)
    series_dirs = sorted(path for path in stent_dir.iterdir() if path.is_dir())
    if series is not None:
        series_dirs = [path for path in series_dirs if path.name == series]
    if not series_dirs:
        raise FileNotFoundError(f"No matching stent series in {stent_dir}")

    selected = []
    for series_path in series_dirs:
        variant_dirs = sorted(path for path in series_path.iterdir() if path.is_dir())
        if variant is not None:
            variant_dirs = [path for path in variant_dirs if _variant_matches(path.name, variant)]
        selected.extend((series_path.name, path.name, path) for path in variant_dirs)
    if not selected:
        raise FileNotFoundError(
            f"No matching variants for series={series!r}, variant={variant!r} in {stent_dir}"
        )

    reports = []
    for series_name, variant_name, variant_dir in selected:
        output_dir = output_root / series_name / variant_name
        output_dir.mkdir(parents=True, exist_ok=True)
        reports.append(_project_stent_variant(
            output_dir, stent_dir, series_name, variant_name,
            n_projections=n_projections, angle_span_deg=angle_span_deg,
            stent_mu=stent_mu, marker_mu=marker_mu, output_mode=output_mode,
        ))
    return reports


def project_topbrain(topbrain_dir: Path, series: str | None = None, *,
                     n_projections: int = DEFAULT_N_PROJECTIONS,
                     angle_span_deg: float = DEFAULT_ANGLE_SPAN_DEG,
                     output_mode: str = DEFAULT_OUTPUT_MODE,
                     real_head_proj_dir: Path = DEFAULT_REAL_HEAD_PROJ_DIR,
                     auto_polyfit: bool = True) -> dict | list[dict]:
    """Project CT directly with the geometry defined by this pipeline."""
    topbrain_dir = Path(topbrain_dir)
    if series is None or str(series).strip().lower() == "all":
        sample_dirs = sorted(
            path for path in topbrain_dir.iterdir()
            if path.is_dir() and (path / "topbrain_filled_real_grid.nii.gz").is_file()
        ) if topbrain_dir.is_dir() else []
        if not sample_dirs:
            raise FileNotFoundError(
                f"No prepared TopBrain samples containing "
                f"topbrain_filled_real_grid.nii.gz in {topbrain_dir}"
            )
        return [
            project_topbrain(
                topbrain_dir, sample_dir.name,
                n_projections=n_projections,
                angle_span_deg=angle_span_deg,
                output_mode=output_mode,
                real_head_proj_dir=real_head_proj_dir,
                auto_polyfit=auto_polyfit,
            )
            for sample_dir in sample_dirs
        ]

    series = str(series)
    work_dir = Path(topbrain_dir) / series
    preparation = work_dir / "preparation.json"
    split = (json.loads(preparation.read_text(encoding="utf-8")).get("split")
             if preparation.is_file() else None)
    if split not in ("train", "test"):
        raise ValueError(f"split missing or invalid for CT {series}; regenerate preparation metadata")
    if auto_polyfit and not Path(real_head_proj_dir).is_dir():
        raise FileNotFoundError(f"real head RAW projection directory not found: {real_head_proj_dir}")
    ct_path = work_dir / "topbrain_filled_real_grid.nii.gz"
    ct_header = nib.load(str(ct_path))
    spacing = np.asarray(ct_header.header.get_zooms()[:3])
    geometry, angles_rad, angles_deg, view_indices = _new_projection_setup(
        ct_header.shape, spacing,
        n_projections=n_projections, angle_span_deg=angle_span_deg,
    )
    ct = MedicalImage.read(ct_path)
    head = to_tigre_volume(hu_to_attenuation(ct.data))
    geometry_info = {
        "DSO_mm": DSO_MM,
        "DSD_mm": DSD_MM,
        "nDetector": list(N_DETECTOR),
        "dDetector_mm": list(D_DETECTOR_MM),
        "nVoxel": geometry.nVoxel.tolist(),
        "dVoxel_mm": geometry.dVoxel.tolist(),
        "offOrigin_mm": geometry.offOrigin.tolist(),
        "offDetector_mm": geometry.offDetector.tolist(),
    }
    project_volume(head, geometry, angles_rad, view_indices, work_dir,
                   "head", blur_px=1.0, output_mode=output_mode,
                   angles_deg=angles_deg)
    polyfit_report = (calibrate_head_projections(
        work_dir, angles_deg, view_indices, output_mode, real_head_proj_dir
    ) if auto_polyfit else {"applied": False})
    report = {
        "sample": series,
        "split": split,
        "angles": angles_deg,
        "angles_deg": angles_deg,
        "view_indices": view_indices,
        "n_projections": len(view_indices),
        "output_mode": output_mode,
        "angle_start_deg": 0.0,
        "angle_span_deg": angle_span_deg,
        "angle_endpoint_included": True,
        "head_source": str(ct_path),
        "head_sigma_px": 1.0,
        "head_polyfit": polyfit_report,
        "mu_water": 0.0227,
        "voxel_spacing_multiplier": VOXEL_SPACING_MULTIPLIER,
        "geometry": geometry_info,
    }
    (work_dir / "projection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stent_parser = subparsers.add_parser("project_stent")
    stent_parser.add_argument("--dir", type=Path, default=repo_root / "output" / "stent_models")
    stent_parser.add_argument("--output-dir", type=Path,
                              default=repo_root / "output" / "stent_projections")
    stent_parser.add_argument("--series", help="optional stent series filter; defaults to all")
    stent_parser.add_argument("--variant", help="optional variant filter; defaults to all")
    stent_parser.add_argument("--n-projections", type=int, default=DEFAULT_N_PROJECTIONS)
    stent_parser.add_argument("--angle-span-deg", type=float, default=DEFAULT_ANGLE_SPAN_DEG)
    stent_parser.add_argument("--output-mode", choices=("2D", "3D"),
                              default=DEFAULT_OUTPUT_MODE)
    stent_parser.add_argument("--stent-mu", type=float, default=DEFAULT_STENT_MU)
    stent_parser.add_argument("--marker-mu", type=float, default=DEFAULT_MARKER_MU)

    head_parser = subparsers.add_parser("project_topbrain")
    head_parser.add_argument("--dir", type=Path, default=repo_root / "output" / "topbrain_pipeline")
    head_parser.add_argument(
        "--series", help="TopBrain sample directory; omitted or 'all' projects every prepared sample"
    )
    head_parser.add_argument("--n-projections", type=int, default=DEFAULT_N_PROJECTIONS)
    head_parser.add_argument("--angle-span-deg", type=float, default=DEFAULT_ANGLE_SPAN_DEG)
    head_parser.add_argument("--output-mode", choices=("2D", "3D"),
                             default=DEFAULT_OUTPUT_MODE)
    head_parser.add_argument("--real-head-proj-dir", type=Path,
                             default=DEFAULT_REAL_HEAD_PROJ_DIR,
                             help="real head phantom RAW projection directory for automatic polyfit")
    head_parser.add_argument("--no-auto-polyfit", action="store_true",
                             help="save original TopBrain head projections without calibration")
    args = parser.parse_args()
    if args.command == "project_stent":
        result = project_stent(
            args.dir, args.output_dir, args.series, args.variant,
            n_projections=args.n_projections, angle_span_deg=args.angle_span_deg,
            stent_mu=args.stent_mu, marker_mu=args.marker_mu,
            output_mode=args.output_mode,
        )
    else:
        result = project_topbrain(
            args.dir, args.series,
            n_projections=args.n_projections,
            angle_span_deg=args.angle_span_deg,
            output_mode=args.output_mode,
            real_head_proj_dir=args.real_head_proj_dir,
            auto_polyfit=not args.no_auto_polyfit,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
