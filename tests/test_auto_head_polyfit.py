"""Head projection calibration fits each newly generated sample to RAW phantom data."""

import json
import types

import numpy as np
import tifffile

from pipelines.calibration.auto_polyfit import (
    apply_head_polyfit, calibrate_head_projections,
)
from pipelines.project import project_topbrain
from pipelines.utils import read_frame, save_frame
from pipelines.medical_image import MedicalImage


def _reference_frames(directory, values):
    directory.mkdir()
    for index, frame in enumerate(values, 1):
        np.asarray(frame, dtype="<f4").tofile(
            directory / f"IMG{index:03d}_{frame.shape[0]}-{frame.shape[1]}.raw")


def test_fit_and_apply_new_head_projections_in_2d(tmp_path):
    head_dir = tmp_path / "ct"
    real_dir = tmp_path / "real"
    angles = [0, 10, 20, 30, 40]
    grid = np.linspace(0.2, 3.0, 64, dtype=np.float32).reshape(8, 8)
    synthetic = [grid + index * 0.05 for index in range(5)]
    for angle, frame in zip(angles, synthetic):
        save_frame(head_dir, "head", angle, frame)
    _reference_frames(real_dir, [2 + 0.8 * frame for frame in synthetic])

    report = calibrate_head_projections(
        head_dir, angles, list(range(5)), "2D", real_dir)

    assert report["reference_view_count"] == 5
    assert report["degree"] == 1
    assert report["validation_quantile_mae_after"] < 1e-5
    for angle, frame in zip(angles, synthetic):
        np.testing.assert_allclose(read_frame(head_dir, "head", angle),
                                   2 + 0.8 * frame, atol=1e-5)


def test_fit_and_apply_new_head_projections_in_3d(tmp_path):
    head_dir = tmp_path / "ct"
    real_dir = tmp_path / "real"
    (head_dir / "projections").mkdir(parents=True)
    grid = np.linspace(0.2, 3.0, 64, dtype=np.float32).reshape(8, 8)
    synthetic = np.stack([grid, grid + 0.1, grid + 0.2])
    path = head_dir / "projections/head.tif"
    tifffile.imwrite(path, synthetic, photometric="minisblack")
    _reference_frames(real_dir, [1.5 + 1.2 * frame for frame in synthetic])

    report = calibrate_head_projections(
        head_dir, [0, 10, 20], [0, 1, 2], "3D", real_dir)

    assert report["degree"] == 1
    np.testing.assert_allclose(tifffile.imread(path),
                               1.5 + 1.2 * synthetic, atol=1e-5)


def test_cubic_polyfit_does_not_extrapolate_outside_training_range():
    values = np.array([-2, 0, 1, 2, 5], dtype=np.float32)
    coefficients = np.array([1, 1, 0, 0.25])
    result = apply_head_polyfit(values, coefficients, (0, 2))
    np.testing.assert_array_equal(result, [1, 1, 2.25, 5, 5])


def test_project_topbrain_automatically_calibrates_after_projection(tmp_path, monkeypatch):
    sample = tmp_path / "topbrain" / "case"
    sample.mkdir(parents=True)
    MedicalImage(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)).save(
        sample / "topbrain_filled_real_grid.nii.gz")
    (sample / "preparation.json").write_text(json.dumps({"split": "train"}))
    real_dir = tmp_path / "real"
    grid = np.linspace(0.2, 3, 64, dtype=np.float32).reshape(8, 8)
    synthetic = [grid + 0.1 * index for index in range(3)]
    _reference_frames(real_dir, [2 + 0.8 * frame for frame in synthetic])

    geometry = types.SimpleNamespace(
        nVoxel=np.array([4, 4, 4]), dVoxel=np.ones(3),
        offOrigin=np.zeros(3), offDetector=np.zeros(2))
    monkeypatch.setattr("pipelines.project._new_projection_setup",
                        lambda *args, **kwargs: (
                            geometry, np.deg2rad([0, 10, 20]), [0, 10, 20], [0, 1, 2]))

    def fake_project_volume(volume, geo, radians, indices, directory, kind, **kwargs):
        for angle, frame in zip([0, 10, 20], synthetic):
            save_frame(directory, kind, angle, frame)

    monkeypatch.setattr("pipelines.project.project_volume", fake_project_volume)
    report = project_topbrain(tmp_path / "topbrain", "case", n_projections=3,
                              angle_span_deg=20, real_head_proj_dir=real_dir)

    assert report["head_polyfit"]["applied"] is True
    assert report["head_polyfit"]["reference_dir"] == str(real_dir)
    np.testing.assert_allclose(read_frame(sample, "head", 10),
                               2 + 0.8 * synthetic[1], atol=1e-5)
    assert json.loads((sample / "projection.json").read_text())["head_polyfit"]["degree"] == 1
