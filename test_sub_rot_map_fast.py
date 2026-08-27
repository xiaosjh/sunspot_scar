import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from sub_rot_map_fast import cleanup_original_fits, scan_missing_records


def write_test_fits(path):
    fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32)).writeto(path)


class SubRotMapFastCleanupTests(unittest.TestCase):
    def test_cleanup_deletes_originals_when_cutouts_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp) / "20120701_191100_UTC"
            input_dir = event_dir / "131"
            output_dir = event_dir / "131sub_map"
            input_dir.mkdir(parents=True)
            output_dir.mkdir()

            src = input_dir / "aia.lev1.131.fits"
            out = output_dir / src.name
            src.write_text("full disk")
            write_test_fits(out)

            deleted, records = cleanup_original_fits(event_dir, ["131"])

            self.assertEqual(deleted, 1)
            self.assertFalse(src.exists())
            self.assertFalse(input_dir.exists())
            self.assertEqual(records, [])

    def test_cleanup_keeps_nonempty_original_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp) / "20120701_191100_UTC"
            input_dir = event_dir / "131"
            output_dir = event_dir / "131sub_map"
            input_dir.mkdir(parents=True)
            output_dir.mkdir()

            src = input_dir / "aia.lev1.131.fits"
            sidecar = input_dir / "note.txt"
            out = output_dir / src.name
            src.write_text("full disk")
            sidecar.write_text("keep me")
            write_test_fits(out)

            deleted, records = cleanup_original_fits(event_dir, ["131"])

            self.assertEqual(deleted, 1)
            self.assertFalse(src.exists())
            self.assertTrue(input_dir.exists())
            self.assertTrue(sidecar.exists())
            self.assertEqual(records, [])

    def test_cleanup_keeps_original_and_records_missing_cutout(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp) / "20120701_191100_UTC"
            input_dir = event_dir / "131"
            input_dir.mkdir(parents=True)
            src = input_dir / "aia.lev1.131.fits"
            src.write_text("full disk")

            deleted, records = cleanup_original_fits(event_dir, ["131"])

            self.assertEqual(deleted, 0)
            self.assertTrue(src.exists())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["issue"], "missing_cutout")

    def test_cleanup_keeps_original_when_cutout_is_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp) / "20120701_191100_UTC"
            input_dir = event_dir / "131"
            output_dir = event_dir / "131sub_map"
            input_dir.mkdir(parents=True)
            output_dir.mkdir()

            src = input_dir / "aia.lev1.131.fits"
            out = output_dir / src.name
            src.write_text("full disk")
            out.write_bytes(b"SIMPLE  =                    T")

            deleted, records = cleanup_original_fits(event_dir, ["131"])

            self.assertEqual(deleted, 0)
            self.assertTrue(src.exists())
            self.assertTrue(input_dir.exists())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["issue"], "missing_cutout")

    def test_cleanup_never_deletes_br_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp) / "20120701_191100_UTC"
            input_dir = event_dir / "hmi.B_720s"
            input_dir.mkdir(parents=True)
            br = input_dir / "Br.fits"
            br_sub = input_dir / "Br_sub.fits"
            br.write_text("computed br")
            br_sub.write_text("cutout br")

            deleted, records = cleanup_original_fits(event_dir, ["hmi.B_720s"])

            self.assertEqual(deleted, 0)
            self.assertTrue(br.exists())
            self.assertEqual(records, [])

    def test_scan_missing_records_missing_passband_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp) / "20120701_191100_UTC"
            event_dir.mkdir()

            records = scan_missing_records(event_dir, ["131"], crop_br=False)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["issue"], "missing_passband_dir")

    def test_scan_records_corrupt_cutout_even_without_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp) / "20120701_191100_UTC"
            output_dir = event_dir / "131sub_map"
            output_dir.mkdir(parents=True)
            (output_dir / "bad.fits").write_bytes(b"SIMPLE  =                    T")

            records = scan_missing_records(event_dir, ["131"], crop_br=False)

            self.assertTrue(any(record["issue"] == "corrupt_cutout" for record in records))


if __name__ == "__main__":
    unittest.main()
