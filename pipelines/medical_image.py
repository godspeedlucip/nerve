"""Medical image data and format-aware image I/O.

The small data container follows the MedicalImage idea in 课题2/utils/tool.py.
All image file access in the projection pipeline goes through this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MedicalImage:
    data: np.ndarray
    affine: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.affine is not None:
            self.affine = np.asarray(self.affine)

    @property
    def spacing(self) -> tuple[float, ...]:
        header = self.metadata.get("header")
        if header is not None:
            return tuple(float(v) for v in header.get_zooms()[: self.data.ndim])
        if self.affine is not None:
            return tuple(float(v) for v in np.linalg.norm(self.affine[:3, :3], axis=0))
        raise ValueError("Image has no spatial metadata")

    def with_data(self, data: np.ndarray) -> MedicalImage:
        """Keep the source spatial metadata for a derived volume."""
        return MedicalImage(data, self.affine, self.metadata.copy())

    def crop(self, start: np.ndarray, end: np.ndarray) -> MedicalImage:
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
        affine = None if self.affine is None else self.affine.copy()
        if affine is not None:
            affine[:3, 3] += affine[:3, :3] @ np.asarray(start, dtype=float)
        return MedicalImage(self.data[slices], affine, self.metadata.copy())

    @staticmethod
    def _suffix(path: Path) -> str:
        return ".nii.gz" if path.name.lower().endswith(".nii.gz") else path.suffix.lower()

    @classmethod
    def read(cls, path: str | Path) -> MedicalImage:
        path = Path(path)
        suffix = cls._suffix(path)
        if suffix in {".nii", ".nii.gz"}:
            import nibabel as nib

            source = nib.load(str(path))
            return cls(np.asanyarray(source.dataobj), source.affine, {"header": source.header.copy()})
        if suffix in {".tif", ".tiff"}:
            import tifffile

            return cls(tifffile.imread(path))
        if suffix == ".npy":
            return cls(np.load(path, allow_pickle=False))
        if suffix == ".npz":
            import scipy.sparse as sparse

            with np.load(path, allow_pickle=False) as archive:
                if "format" in archive.files and "shape" in archive.files:
                    return cls(sparse.load_npz(path).toarray(), metadata={"sparse": True})
                if not archive.files:
                    raise ValueError(f"Empty NPZ image: {path}")
                key = "data" if "data" in archive.files else archive.files[0]
                return cls(archive[key], metadata={"npz_key": key})
        raise ValueError(f"Unsupported image format: {path}")

    @classmethod
    def read_frames(cls, path: str | Path, start: int, stop: int) -> MedicalImage:
        """Read a slice of a 3D TIFF stack without materializing the full stack."""
        path = Path(path)
        if cls._suffix(path) not in {".tif", ".tiff"}:
            raise ValueError("read_frames requires a TIFF stack")
        import tifffile

        with tifffile.TiffFile(path) as source:
            shape = source.series[0].shape
        if len(shape) == 2 and start == 0 and stop == 1:
            return cls(np.asarray(tifffile.imread(path))[np.newaxis, ...])
        if len(shape) != 3 or not (0 <= start < stop <= shape[0]):
            raise ValueError(f"Invalid TIFF frame range {start}:{stop} for shape {shape}")
        try:
            frames = tifffile.memmap(path, mode="r")[start:stop]
        except ValueError:
            frames = tifffile.imread(path, key=range(start, stop))
            if frames.ndim == 2:
                frames = frames[np.newaxis, ...]
        return cls(np.asarray(frames))

    def save(
        self, path: str | Path, *, sparse: bool | None = None,
        compression: str | None = None, compressionargs: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self._suffix(path)
        if suffix in {".nii", ".nii.gz"}:
            import nibabel as nib

            affine = self.affine if self.affine is not None else np.eye(4)
            image = nib.Nifti1Image(self.data, affine, header=self.metadata.get("header"))
            nib.save(image, str(path))
        elif suffix in {".tif", ".tiff"}:
            import tifffile

            tifffile.imwrite(path, self.data, compression=compression, compressionargs=compressionargs)
        elif suffix == ".npy":
            np.save(path, self.data)
        elif suffix == ".npz":
            use_sparse = self.metadata.get("sparse", False) if sparse is None else sparse
            if use_sparse:
                import scipy.sparse as sparse_module

                if self.data.ndim != 2:
                    raise ValueError("Sparse NPZ requires a 2D image")
                sparse_module.save_npz(path, sparse_module.csr_matrix(self.data), compressed=True)
            else:
                np.savez_compressed(path, **{self.metadata.get("npz_key", "data"): self.data})
        else:
            raise ValueError(f"Unsupported image format: {path}")
        return path
