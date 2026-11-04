"""Gaussian projection noise remains reproducible across stacked outputs."""

import json

import numpy as np
import pytest
import tifffile

from pipelines.stack import compose_projection, stack, stack_all
from pipelines.utils import save_frame


def test_compose_adds_seeded_noise_to_head_only():
    head = np.full((32, 32), 2.0, dtype=np.float32)
    implant = np.zeros_like(head)
    implant[16, 16] = 1.0
    image, smoothed = compose_projection(
        head, implant, implant_sigma_px=0, noise_std=0.2,
        rng=np.random.default_rng(17),
    )
    expected = np.random.default_rng(17).normal(0, 0.2, head.shape)
    np.testing.assert_allclose(image, head + implant + expected, atol=1e-6)
    np.testing.assert_array_equal(smoothed, implant)
    assert image.dtype == np.float32


def test_batch_noise_seed_reproduces_image_and_changed_std_forces_rerun(tmp_path):
    heads = tmp_path / "topbrain_pipeline"
    sample = heads / "case"
    implant_dir = tmp_path / "stent_projections" / "series" / "001"
    geometry = {"DSO_mm": 765, "DSD_mm": 1200,
                "nDetector": [16, 16], "dDetector_mm": [0.15, 0.15]}
    common = {"angles_deg": [0], "view_indices": [0], "split": "train",
              "output_mode": "2D", "geometry": geometry}
    sample.mkdir(parents=True)
    implant_dir.mkdir(parents=True)
    (sample / "projection.json").write_text(json.dumps({**common, "sample": "case"}))
    (implant_dir / "projection.json").write_text(json.dumps({
        **common, "stent_series": "series", "stent_variant": "001"}))
    save_frame(sample, "head", 0, np.full((16, 16), 2, dtype=np.float32))
    implant = np.zeros((16, 16), dtype=np.float32)
    implant[8, 8] = 1
    save_frame(implant_dir, "implant", 0, implant)

    def output_image():
        path = tmp_path / "stacked" / "train" / "case" / "series" / "001"
        return tifffile.imread(path / "projections" / "image" / "000.tif"), path

    result = stack_all(implant_dir.parent.parent, heads,
                       implant_sigma_px=0, noise_std=0.2, noise_seed=17)
    image, output = output_image()
    assert result["completed"] == 1
    np.testing.assert_array_equal(
        tifffile.imread(output / "projections" / "label" / "000.tif"), implant > 0)
    assert stack_all(implant_dir.parent.parent, heads, implant_sigma_px=0,
                     noise_std=0.2, noise_seed=17)["skipped"] == 1
    assert stack_all(implant_dir.parent.parent, heads, implant_sigma_px=0,
                     noise_std=0.4, noise_seed=17)["completed"] == 1
    changed, _ = output_image()
    assert not np.array_equal(changed, image)
    assert stack_all(implant_dir.parent.parent, heads, implant_sigma_px=0,
                     noise_std=0.2, noise_seed=17)["completed"] == 1
    np.testing.assert_array_equal(output_image()[0], image)
    report = json.loads((output / "stack.json").read_text())
    assert report["noise_std"] == 0.2
    assert report["noise_seed"] == 17
    separate = tmp_path / "stacked_noise_020"
    assert stack_all(implant_dir.parent.parent, heads, implant_sigma_px=0,
                     noise_std=0.2, noise_seed=17,
                     output_root=separate)["completed"] == 1
    np.testing.assert_array_equal(
        tifffile.imread(separate / "train/case/series/001/projections/image/000.tif"),
        image,
    )
    default_root = tmp_path / "stacked_default_noise"
    stack_all(implant_dir.parent.parent, heads, implant_sigma_px=0,
              output_root=default_root)
    default_output = default_root / "train/case/series/001"
    default_report = json.loads((default_output / "stack.json").read_text())
    assert default_report["noise_std"] == 0.03
    default_image = tifffile.imread(default_output / "projections/image/000.tif")
    assert 0.02 < np.std(default_image - 2 - implant) < 0.04

    calibration = tmp_path / "head_polyfit.json"
    calibration.write_text(json.dumps({"degree": 1,
                                       "coefficients_ascending": [2.0, 0.8]}))
    calibrated_root = tmp_path / "stacked_calibrated"
    result = stack_all(implant_dir.parent.parent, heads, implant_sigma_px=0,
                       noise_std=0, output_root=calibrated_root,
                       head_polyfit=calibration)
    calibrated_output = calibrated_root / "train/case/series/001"
    assert result["completed"] == 1
    np.testing.assert_allclose(
        tifffile.imread(calibrated_output / "projections/image/000.tif"),
        3.6 + implant, atol=1e-6)
    np.testing.assert_array_equal(
        tifffile.imread(calibrated_output / "projections/label/000.tif"), implant > 0)
    assert stack_all(implant_dir.parent.parent, heads, implant_sigma_px=0,
                     noise_std=0, output_root=calibrated_root,
                     head_polyfit=calibration)["skipped"] == 1
    calibration.write_text(json.dumps({"degree": 1,
                                       "coefficients_ascending": [2.1, 0.8]}))
    assert stack_all(implant_dir.parent.parent, heads, implant_sigma_px=0,
                     noise_std=0, output_root=calibrated_root,
                     head_polyfit=calibration)["completed"] == 1
    head_info = json.loads((sample / "projection.json").read_text())
    head_info["head_polyfit"] = {"applied": True}
    (sample / "projection.json").write_text(json.dumps(head_info))
    with pytest.raises(ValueError, match="already calibrated"):
        stack(sample, implant_dir, head_polyfit=calibration)
