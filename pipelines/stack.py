"""Stage 4: stack every stent variant projection with every image projection."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from pipelines.utils import (frame_path, projection_stack_path, prune_projection_outputs,
                             read_frame, read_projection_frame, save_frame)


def compose_projection(head: np.ndarray, implant: np.ndarray,
                       implant_sigma_px: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """Return (head + smoothed implant, smoothed implant)."""
    if head.shape != implant.shape:
        raise ValueError("head and implant projections must have the same shape")
    implant = np.asarray(implant, dtype=np.float32)
    if implant_sigma_px > 0:
        implant = gaussian_filter(implant, sigma=implant_sigma_px)
    implant = implant.astype(np.float32, copy=False)
    return (np.asarray(head, dtype=np.float32) + implant).astype(np.float32), implant


def stack(work_dir: Path, stent_projection_dir: Path, *,
          output_dir: Path | None = None, implant_sigma_px: float = 1.2,
          _head_frames: list[np.ndarray] | None = None) -> dict:
    """Combine TopBrain and stent projections in their shared output mode.

    ``work_dir`` is the TopBrain sample directory containing ``projection.json``
    and head projections. ``stent_projection_dir`` contains the selected
    variant's ``projection.json``, implant projections and marker JSONs.
    """
    work_dir = Path(work_dir)
    stent_projection_dir = Path(stent_projection_dir)
    if implant_sigma_px < 0:
        raise ValueError("implant_sigma_px must be nonnegative")

    head_info = json.loads((work_dir / "projection.json").read_text(encoding="utf-8"))
    stent_info = json.loads(
        (stent_projection_dir / "projection.json").read_text(encoding="utf-8")
    )
    split = head_info.get("split")
    if split != stent_info.get("split"):
        raise ValueError("head and stent train/test splits must match")
    angles = head_info.get("angles_deg", head_info.get("angles"))
    stent_angles = stent_info.get("angles_deg", stent_info.get("angles"))
    view_indices = head_info.get("view_indices", head_info.get("angles"))
    stent_indices = stent_info.get("view_indices", stent_info.get("angles"))
    if angles != stent_angles or view_indices != stent_indices:
        raise ValueError("head and stent projection angles/view indices must match")
    if not view_indices:
        raise ValueError("projection metadata contains no views")
    output_mode = head_info.get("output_mode", "3D")
    if output_mode not in ("2D", "3D") or output_mode != stent_info.get("output_mode", "3D"):
        raise ValueError("head and stent output modes must match (2D or 3D)")
    head_geometry = head_info.get("geometry", {})
    stent_geometry = stent_info.get("geometry", {})
    for key in ("DSO_mm", "DSD_mm", "nDetector", "dDetector_mm"):
        if key not in head_geometry or key not in stent_geometry:
            raise ValueError(f"projection geometry is missing shared parameter {key}")
        if not np.allclose(head_geometry[key], stent_geometry[key]):
            raise ValueError(f"head and stent geometry differ for {key}")

    output_dir = Path(output_dir) if output_dir else (
        work_dir / "projections" / stent_info["stent_series"]
        / str(stent_info["stent_variant"])
    )
    output_paths = {
        kind: projection_stack_path(output_dir, kind)
        for kind in ("image", "label", "implant_smoothed")
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    head_stack = projection_stack_path(work_dir, "head")
    implant_stack = projection_stack_path(stent_projection_dir, "implant")
    with ExitStack() as resources:
        writers = ({kind: resources.enter_context(tifffile.TiffWriter(path))
                    for kind, path in output_paths.items()}
                   if output_mode == "3D" else {})
        for page_index, view_index in enumerate(view_indices):
            angle = angles[page_index]
            if _head_frames is not None:
                head = _head_frames[page_index]
            elif output_mode == "2D":
                head = read_frame(work_dir, "head", angle)
            else:
                head = read_projection_frame(
                    work_dir, "head", page_index, view_index, stack_path=head_stack
                )
            if output_mode == "2D":
                implant_source = read_frame(stent_projection_dir, "implant", angle)
            else:
                implant_source = read_projection_frame(
                    stent_projection_dir, "implant", page_index, view_index,
                    stack_path=implant_stack,
                )
            if not np.isfinite(head).all() or not np.isfinite(implant_source).all():
                raise ValueError(f"non-finite projection at view {view_index}")
            image, implant = compose_projection(
                head, implant_source, implant_sigma_px=implant_sigma_px
            )
            for kind, values in (
                ("image", image), ("implant_smoothed", implant),
                ("label", implant_source > 0),
            ):
                if output_mode == "2D":
                    save_frame(output_dir, kind, angle, values)
                else:
                    writers[kind].write(
                        np.asarray(values, dtype=np.float32),
                        compression="zlib", compressionargs={"level": 1},
                        photometric="minisblack", metadata=None,
                    )
    for kind in output_paths:
        prune_projection_outputs(output_dir, kind, angles, output_mode)

    source_metadata = stent_projection_dir / "marker_metadata"
    target_metadata = output_dir / "marker_metadata"
    expected_metadata = set()
    if source_metadata.is_dir():
        expected_metadata = {path.name for path in source_metadata.glob("*.json")}
        shutil.copytree(source_metadata, target_metadata, dirs_exist_ok=True)
    if target_metadata.is_dir():
        for path in target_metadata.glob("*.json"):
            if path.stem.isdecimal() and path.name not in expected_metadata:
                path.unlink()
    combined_projection = {
        "sample": head_info.get("sample"),
        "split": split,
        "stent_series": stent_info["stent_series"],
        "stent_variant": str(stent_info["stent_variant"]),
        "angles": angles,
        "angles_deg": angles,
        "view_indices": view_indices,
        "output_mode": output_mode,
        "head_geometry": head_geometry,
        "stent_geometry": stent_geometry,
        "head_projection_source": str(work_dir),
        "stent_projection_source": str(stent_projection_dir),
    }
    (output_dir / "projection.json").write_text(
        json.dumps(combined_projection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        **combined_projection,
        "stent_mu": stent_info.get("stent_mu", 0.17),
        "marker_mu": stent_info.get("marker_mu", 1.02),
        "implant_sigma_px": implant_sigma_px,
        "formula": "image = head + Gaussian(combined implant projection)",
        "output_dir": str(output_dir),
    }
    (output_dir / "stack.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _projection_files_ready(directory: Path, kind: str, info: dict) -> bool:
    """Check every declared view, not just the presence of projection.json."""
    angles = info.get("angles_deg", info.get("angles", []))
    if not angles:
        return False
    if info.get("output_mode", "3D") == "2D":
        return all(frame_path(directory, kind, angle).is_file() for angle in angles)
    return projection_stack_path(directory, kind).is_file()


def _completed_stack(output: Path, head: Path, implant: Path,
                     info: dict, implant_sigma_px: float) -> bool:
    report_path = output / "stack.json"
    if not report_path.is_file():
        return False
    if any((source / "projection.json").stat().st_mtime_ns > report_path.stat().st_mtime_ns
           for source in (head, implant)):
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (report.get("head_projection_source") != str(head)
            or report.get("stent_projection_source") != str(implant)
            or report.get("implant_sigma_px") != implant_sigma_px
            or report.get("angles_deg") != info.get("angles_deg", info.get("angles"))
            or report.get("output_mode") != info.get("output_mode", "3D")):
        return False
    return all(_projection_files_ready(output, kind, info)
               for kind in ("image", "label", "implant_smoothed"))


def stack_all(stent_dir: Path, image_dir: Path, *,
              implant_sigma_px: float = 1.2) -> dict:
    """Stack every currently available projected CT and stent variant.

    ``stent_dir`` may point to stent projections directly or to the sibling
    stent_models folder. Existing complete stacks are skipped on reruns.
    """
    stent_dir = Path(stent_dir)
    topbrain_dir = Path(image_dir)
    stent_projection_root = (stent_dir.parent / "stent_projections"
                             if stent_dir.name == "stent_models" else stent_dir)
    output_root = topbrain_dir.parent / "stacked"
    if implant_sigma_px < 0:
        raise ValueError("implant_sigma_px must be nonnegative")
    if not topbrain_dir.is_dir() or not stent_projection_root.is_dir():
        raise FileNotFoundError("image and stent projection directories must exist")

    heads = []
    for sample in sorted(path for path in topbrain_dir.iterdir() if path.is_dir()):
        metadata = sample / "projection.json"
        if not metadata.is_file():
            continue
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info.get("split") not in ("train", "test"):
            raise ValueError(f"split missing or invalid for CT projection {sample}; regenerate metadata")
        if _projection_files_ready(sample, "head", info):
            heads.append((sample, info))

    implants = []
    for series in sorted(path for path in stent_projection_root.iterdir() if path.is_dir()):
        for projection in sorted(path for path in series.iterdir() if path.is_dir()):
            metadata = projection / "projection.json"
            if not metadata.is_file():
                continue
            info = json.loads(metadata.read_text(encoding="utf-8"))
            if info.get("split") not in ("train", "test"):
                raise ValueError(f"split missing or invalid for stent projection {projection}; regenerate metadata")
            if _projection_files_ready(projection, "implant", info):
                implants.append((projection, info))

    total = sum(head_info["split"] == implant_info["split"]
                for _, head_info in heads for _, implant_info in implants)
    completed = skipped = 0
    for sample, head_info in heads:
        pending = []
        for projection, implant_info in implants:
            if head_info["split"] != implant_info["split"]:
                continue
            output = (output_root / head_info["split"] / sample.name
                      / projection.parent.name / projection.name)
            if _completed_stack(output, sample, projection,
                                head_info, implant_sigma_px):
                skipped += 1
                continue
            pending.append((projection, output))
        if not pending:
            continue
        angles = head_info.get("angles_deg", head_info.get("angles"))
        indices = head_info["view_indices"]
        if head_info.get("output_mode", "3D") == "2D":
            head_frames = [read_frame(sample, "head", angle) for angle in angles]
        else:
            source_stack = projection_stack_path(sample, "head")
            head_frames = [read_projection_frame(
                sample, "head", page, index, stack_path=source_stack
            ) for page, index in enumerate(indices)]
        for projection, output in pending:
            stack(sample, projection, output_dir=output,
                  implant_sigma_px=implant_sigma_px, _head_frames=head_frames)
            completed += 1
            print(f"Stacked {completed + skipped}/{total}: {sample.name} + "
                  f"{projection.parent.name}/{projection.name}", flush=True)
    return {"images": len(heads), "stent_variants": len(implants),
            "total": total, "completed": completed, "skipped": skipped}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stent-dir", type=Path, required=True,
                        help="stent_models or stent_projections folder")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="folder containing projected image samples")
    args = parser.parse_args()
    result = stack_all(args.stent_dir, args.image_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
