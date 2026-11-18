"""验证新旧 Br 目录、FITS 波长兜底以及预览和导出的一致性。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from astropy.io import fits

import app


class DisplayChannelTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = np.linspace(-800, 800, 64, dtype=np.float32).reshape(8, 8)

    def tearDown(self):
        self.temp.cleanup()

    def write_fits(self, filename="image.fits", **keywords):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(path, self.data, fits.Header(keywords), overwrite=True)
        return path

    def test_old_and_new_br_names_use_gauss_limits(self):
        settings = app.DisplaySettings(br_min=-250, br_max=600)
        for name in ["hmi.B_720s/Br.fits", "hmi.B_720s/Br_sub.fits", "hmi_Br", "hmi_Br_submap", "hmi.B_720s.20240618_200000_TAI.Br_submap.fits"]:
            with self.subTest(name=name):
                self.assertTrue(app.is_br_channel(name))
                self.assertEqual(app.display_limits(self.data, name, settings), (-250, 600))
        for name in ["hmi_Bphi", "hmi_Btheta", "hmi_b720", "inclination_submap", "131sub_map"]:
            self.assertFalse(app.is_br_channel(name))

    def test_br_filename_in_generic_folder_controls_rendering(self):
        path = self.write_fits("magnetic_series/20240618_200000_Br.fits")
        settings = app.DisplaySettings(br_min=-400, br_max=400)
        expected = app.colorize_preview_data(self.data, "Br", settings)
        actual = app.colorize_preview_data(self.data, "magnetic_series", settings, path=path)
        np.testing.assert_array_equal(actual, expected)

    def test_new_continuum_names_use_full_linear_range(self):
        # 两端的极值必须保留，不能被百分位裁剪或默认 Levels 扩展掉。
        for name in ["hmi_con", "hmi_con_submap", "hmi_continuum", "hmi.Ic_45ssub_map"]:
            with self.subTest(name=name):
                self.assertTrue(app.is_hmi_continuum(name))
                settings = app.default_display_settings(name)
                self.assertEqual(app.base_display_range(self.data, name), (-800, 800))
                self.assertEqual(app.display_limits(self.data, name, settings), (-800, 800))
                self.assertEqual(app.display_stretch(name, settings), "linear")

    def test_aligned_continuum_orientation_is_preserved(self):
        np.testing.assert_array_equal(app.orient_image_for_display(self.data, "hmi_con_submap"), self.data)
        np.testing.assert_array_equal(app.orient_image_for_display(self.data, "hmi.Ic_45ssub_map"), np.rot90(self.data, 2))

    def test_continuum_filename_fallback(self):
        path = self.write_fits("custom/hmi.Ic_45s.20240618_200000_TAI.continuum_submap.fits", INSTRUME="HMI", WAVELNTH=6173)
        resolved = app.resolve_display_channel("custom", path)
        self.assertTrue(app.is_hmi_continuum(resolved))
        settings = app.default_display_settings(resolved)
        actual = app.colorize_preview_data(self.data, "custom", settings, path=path)
        expected = app.colorize_preview_data(self.data, "hmi_con_submap", settings)
        np.testing.assert_array_equal(actual, expected)

    def test_aia_names_and_leading_zero(self):
        for wave in app.AIA_WAVELENGTHS:
            self.assertEqual(app.aia_wavelength(f"{wave}_submap"), wave)
            self.assertNotEqual(app.channel_cmap(f"{wave}_submap"), "gray")
        self.assertEqual(app.aia_wavelength("094_submap"), 94)
        self.assertIsNone(app.aia_wavelength("20240618_200000"))

    def test_header_fallback_is_per_frame_and_used_by_preview(self):
        for wave in [94, 131, 171]:
            path = self.write_fits(f"custom/image_{wave // 10}.fits", INSTRUME="AIA", WAVELNTH=wave, WAVEUNIT="angstrom")
            resolved = app.resolve_display_channel("custom", path)
            self.assertEqual(app.aia_wavelength(resolved), wave)
            self.assertEqual(app.default_display_settings(resolved).stretch, "asinh")
            rgb = app.make_preview_image(path, "custom", 100)
            expected = app.make_preview_image(path, f"{wave}_submap", 100)
            np.testing.assert_array_equal(rgb, expected)
            self.assertTrue(np.any(rgb[:, :, 0] != rgb[:, :, 1]))

    def test_image_extension_header_and_wavelength_units(self):
        path = self.root / "extension.fits"
        primary = fits.PrimaryHDU(header=fits.Header({"WAVELNTH": 94}))
        image = fits.ImageHDU(self.data, fits.Header({"INSTRUME": "AIA", "WAVELNTH": 13.1, "WAVEUNIT": "nm"}))
        with fits.HDUList([primary, image]) as hdul:
            hdul.writeto(path)
        self.assertEqual(app.aia_wavelength(app.resolve_display_channel("custom", path)), 131)

    def test_legacy_name_priority_and_unknown_header(self):
        path = self.write_fits(INSTRUME="AIA", WAVELNTH=94)
        self.assertEqual(app.resolve_display_channel("131sub_map", path), "131sub_map")
        self.assertEqual(app.resolve_display_channel("hmi.B_720s/Br_sub.fits", path), "hmi.B_720s/Br_sub.fits")
        for keywords in [{}, {"INSTRUME": "HMI", "WAVELNTH": 6173}, {"WAVELNTH": 94, "WAVEUNIT": "invalid"}]:
            path = self.write_fits(**keywords)
            self.assertEqual(app.resolve_display_channel("custom", path), "custom")

    def test_all_wavelengths_have_fallback_without_sunpy(self):
        original = app.sunpy_color_tables
        try:
            app.sunpy_color_tables = None
            for wave in app.AIA_WAVELENGTHS:
                self.assertNotEqual(app.channel_cmap(str(wave)), "gray")
        finally:
            app.sunpy_color_tables = original

    def test_worker_cache_and_export_use_same_header_fallback(self):
        path = self.write_fits("custom/frame.fits", INSTRUME="AIA", WAVELNTH=94)
        settings = app.DisplaySettings(level_high=180, stretch="log")
        expected = app.make_preview_image(path, "custom", 100, settings)
        original_cache_dir = app.PREVIEW_CACHE_DIR
        try:
            app.PREVIEW_CACHE_DIR = self.root / "cache"
            for use_disk_cache in [True, False]:
                _, rgb_path, _ = app.build_preview_worker(str(path), "custom", 100, use_disk_cache, settings)
                with app.Image.open(rgb_path) as image:
                    np.testing.assert_array_equal(np.asarray(image), expected)
        finally:
            app.PREVIEW_CACHE_DIR = original_cache_dir
        viewer = app.FitsViewerApp.__new__(app.FitsViewerApp)
        viewer.preview_cache, viewer.preview_cache_keys, viewer.scalar_cache = {}, {}, {}
        actual = viewer.export_image_for_frame(app.scan_frame(path), "custom", settings, 100)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
