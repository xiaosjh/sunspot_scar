from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import astropy.units as u
import pandas as pd
import sunpy.map
from astropy.io import fits
from astropy.coordinates import SkyCoord
from sunpy.coordinates import propagate_with_solar_surface


COPY_META_KEYS = (
    "telescop",
    "instrume",
    "detector",
    "wavelnth",
    "waveunit",
    "bunit",
    "exptime",
)


def _record(event_dir: Path, passband: str, issue: str, path: Path | str = "", detail: str = ""):
    return {
        "event": event_dir.name,
        "passband": passband,
        "issue": issue,
        "path": str(path),
        "detail": detail,
    }


def _valid_existing_file(path: Path):
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _valid_fits_file(path: Path):
    if not _valid_existing_file(path):
        return False
    try:
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                if data is not None:
                    _ = data.shape
                    return True
    except Exception:
        return False
    return False


def _dedupe_records(records):
    seen = set()
    unique = []
    for record in records:
        key = (
            record.get("event", ""),
            record.get("passband", ""),
            record.get("issue", ""),
            record.get("path", ""),
            record.get("detail", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def scan_missing_records(event_dir: Path, passbands, *, crop_br: bool = True):
    records = []

    for passband in passbands:
        input_dir = event_dir / passband
        output_dir = event_dir / f"{passband}sub_map"

        if passband == "hmi.B_720s":
            continue

        input_files = sorted(input_dir.glob("*.fits")) if input_dir.exists() else []
        output_files = sorted(output_dir.glob("*.fits")) if output_dir.exists() else []

        for out_path in output_files:
            if not _valid_fits_file(out_path):
                records.append(_record(event_dir, passband, "corrupt_cutout", out_path))

        if not input_dir.exists() and not output_files:
            records.append(_record(event_dir, passband, "missing_passband_dir", input_dir))
            continue

        if input_dir.exists() and not input_files and not output_files:
            records.append(_record(event_dir, passband, "missing_fits", input_dir))
            continue

        for src_path in input_files:
            out_path = output_dir / src_path.name
            if not _valid_fits_file(out_path):
                records.append(_record(event_dir, passband, "missing_cutout", out_path))

    if crop_br:
        br_path = event_dir / "hmi.B_720s" / "Br.fits"
        br_sub_path = event_dir / "hmi.B_720s" / "Br_sub.fits"
        if not _valid_fits_file(br_path):
            records.append(_record(event_dir, "hmi.B_720s", "missing_br", br_path))
        elif not _valid_fits_file(br_sub_path):
            records.append(_record(event_dir, "hmi.B_720s", "missing_br_sub", br_sub_path))

    return records


def cleanup_original_fits(event_dir: Path, passbands):
    """Delete original full-disk FITS only after the matching cutout exists."""
    deleted = 0
    records = []

    for passband in passbands:
        if passband == "hmi.B_720s":
            continue

        input_dir = event_dir / passband
        output_dir = event_dir / f"{passband}sub_map"
        input_files = sorted(input_dir.glob("*.fits")) if input_dir.exists() else []

        for src_path in input_files:
            out_path = output_dir / src_path.name
            if _valid_fits_file(out_path):
                try:
                    src_path.unlink()
                    deleted += 1
                except Exception as exc:
                    records.append(
                        _record(event_dir, passband, "delete_error", src_path, str(exc))
                    )
            else:
                records.append(_record(event_dir, passband, "missing_cutout", out_path))

        if input_dir.exists() and not any(input_dir.iterdir()):
            try:
                input_dir.rmdir()
            except Exception as exc:
                records.append(_record(event_dir, passband, "delete_dir_error", input_dir, str(exc)))

    return deleted, records


def _crop_box(smap, center, width_arcsec):
    half = width_arcsec / 2
    bl = SkyCoord(
        center.Tx - half * u.arcsec,
        center.Ty - half * u.arcsec,
        frame=smap.coordinate_frame,
    )
    tr = SkyCoord(
        center.Tx + half * u.arcsec,
        center.Ty + half * u.arcsec,
        frame=smap.coordinate_frame,
    )
    return smap.submap(bl, top_right=tr)


def _event_center(smap, x_arcsec, y_arcsec):
    return SkyCoord(x_arcsec * u.arcsec, y_arcsec * u.arcsec, frame=smap.coordinate_frame)


def _copy_observing_meta(dst_map, src_map):
    for key in COPY_META_KEYS:
        if key in src_map.meta:
            dst_map.meta[key] = src_map.meta[key]
    dst_map.meta.pop("BLANK", None)
    return dst_map


def process_passband(
    event_dir: Path,
    passband: str,
    x_arcsec: float,
    y_arcsec: float,
    *,
    width_arcsec: int = 800,
    source_margin_arcsec: int = 180,
    overwrite: bool = False,
):
    """Crop first frame, then rotate-align later frames to the first cutout WCS.

    The speedup comes from cutting each source image to a local patch before
    reprojection. Increase source_margin_arcsec if frames span a long time.
    """
    input_dir = event_dir / passband
    output_dir = event_dir / f"{passband}sub_map"
    output_dir.mkdir(exist_ok=True)

    files = sorted(input_dir.glob("*.fits")) if input_dir.exists() else []
    output_files = sorted(output_dir.glob("*.fits"))
    valid_output_files = [path for path in output_files if _valid_fits_file(path)]
    if not files:
        if valid_output_files:
            return 0, f"{event_dir.name}/{passband}: originals already cleaned"
        if output_files:
            return 0, f"{event_dir.name}/{passband}: only corrupt cutouts and no originals"
        return 0, f"{event_dir.name}/{passband}: no fits"

    if not overwrite and all(_valid_fits_file(output_dir / src.name) for src in files):
        return 0, f"{event_dir.name}/{passband}: cutouts already exist"

    saved = 0
    all_names = sorted({path.name for path in files} | {path.name for path in output_files})
    ref_name = all_names[0]
    ref_out = output_dir / ref_name
    ref_src = input_dir / ref_name

    if not overwrite and _valid_fits_file(ref_out):
        ref_submap = sunpy.map.Map(ref_out)
        ref_center = _event_center(ref_submap, x_arcsec, y_arcsec)
        files_to_align = files
    elif ref_src.exists():
        ref_map = sunpy.map.Map(ref_src)
        ref_center = _event_center(ref_map, x_arcsec, y_arcsec)
        ref_submap = _crop_box(ref_map, ref_center, width_arcsec)
        ref_submap.meta.pop("BLANK", None)
        ref_submap.save(ref_out, overwrite=True)
        saved += 1
        files_to_align = [path for path in files if path.name != ref_name]
    elif valid_output_files and not overwrite:
        ref_submap = sunpy.map.Map(valid_output_files[0])
        ref_center = _event_center(ref_submap, x_arcsec, y_arcsec)
        files_to_align = files
    else:
        raise FileNotFoundError(f"No usable reference FITS for {event_dir.name}/{passband}")

    target_wcs = ref_submap.wcs
    source_crop_width = width_arcsec + 2 * source_margin_arcsec

    for src_path in files_to_align:
        out_path = output_dir / src_path.name
        if _valid_fits_file(out_path) and not overwrite:
            continue

        src_map = sunpy.map.Map(src_path)

        with propagate_with_solar_surface():
            src_center = ref_center.transform_to(src_map.coordinate_frame)

        try:
            src_cutout = _crop_box(src_map, src_center, source_crop_width)
        except Exception:
            # Fallback for limb/edge cases where the padded cutout is partly outside.
            src_cutout = src_map

        with propagate_with_solar_surface():
            aligned = src_cutout.reproject_to(target_wcs, preserve_date_obs=True)

        aligned = _copy_observing_meta(aligned, src_map)
        aligned = sunpy.map.Map(aligned.data.astype("float32", copy=False), aligned.meta)
        aligned.save(out_path, overwrite=True)
        saved += 1

    return saved, f"{event_dir.name}/{passband}: saved {saved}/{len(files)}"


def process_br(
    event_dir: Path,
    x_arcsec: float,
    y_arcsec: float,
    *,
    width_arcsec: int = 800,
    overwrite: bool = False,
):
    br_path = event_dir / "hmi.B_720s" / "Br.fits"
    out_path = event_dir / "hmi.B_720s" / "Br_sub.fits"

    if out_path.exists() and not overwrite:
        return 0, f"{event_dir.name}/hmi.B_720s: Br_sub exists"

    if not br_path.exists():
        return 0, f"{event_dir.name}/hmi.B_720s: Br.fits not found"

    smap = sunpy.map.Map(br_path)
    center = _event_center(smap, x_arcsec, y_arcsec)
    submap = _crop_box(smap, center, width_arcsec)
    submap.meta.pop("BLANK", None)
    submap.save(out_path, overwrite=True)
    return 1, f"{event_dir.name}/hmi.B_720s: saved Br_sub.fits"


def process_event(
    index: int,
    event_dir: Path,
    x_arcsec: float,
    y_arcsec: float,
    *,
    passbands,
    width_arcsec: int,
    source_margin_arcsec: int,
    overwrite: bool,
    crop_br: bool,
    cleanup_originals: bool,
):
    messages = []
    records = []
    saved_total = 0
    deleted_total = 0

    for passband in passbands:
        try:
            saved, message = process_passband(
                event_dir,
                passband,
                x_arcsec,
                y_arcsec,
                width_arcsec=width_arcsec,
                source_margin_arcsec=source_margin_arcsec,
                overwrite=overwrite,
            )
        except Exception as exc:
            saved, message = 0, f"{event_dir.name}/{passband}: ERROR {exc}"
            records.append(_record(event_dir, passband, "process_error", detail=str(exc)))
        saved_total += saved
        messages.append(message)

        if cleanup_originals:
            deleted, cleanup_records = cleanup_original_fits(event_dir, [passband])
            deleted_total += deleted
            records.extend(cleanup_records)
            if deleted:
                messages.append(f"{event_dir.name}/{passband}: deleted originals {deleted}")

    if crop_br:
        try:
            saved, message = process_br(
                event_dir,
                x_arcsec,
                y_arcsec,
                width_arcsec=width_arcsec,
                overwrite=overwrite,
            )
        except Exception as exc:
            saved, message = 0, f"{event_dir.name}/hmi.B_720s: ERROR {exc}"
            records.append(_record(event_dir, "hmi.B_720s", "process_error", detail=str(exc)))
        saved_total += saved
        messages.append(message)

    records.extend(scan_missing_records(event_dir, passbands, crop_br=crop_br))

    return index, saved_total, deleted_total, messages, records


def run_sub_rot_map(
    root_dir=r"C:\Learning\PHD2nd\sunspotscar\data\M",
    excel_path="../dataset/M级耀斑.xlsx",
    *,
    n_events: int = 100,
    passbands=("131", "211", "304", "hmi.Ic_45s"),
    width_arcsec: int = 800,
    source_margin_arcsec: int = 180,
    max_workers: int = 2,
    overwrite: bool = False,
    crop_br: bool = True,
    cleanup_originals: bool = True,
    missing_report_path=None,
):
    root_dir = Path(root_dir)
    event_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())[:n_events]

    df = pd.read_excel(excel_path)
    x_values = df["X, arcsec"].to_numpy()
    y_values = df["Y, arcsec"].to_numpy()

    if len(event_dirs) > len(x_values):
        raise ValueError(f"Need {len(event_dirs)} flare rows, but Excel has {len(x_values)} rows")

    futures = []
    all_records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, event_dir in enumerate(event_dirs):
            futures.append(
                executor.submit(
                    process_event,
                    index,
                    event_dir,
                    float(x_values[index]),
                    float(y_values[index]),
                    passbands=passbands,
                    width_arcsec=width_arcsec,
                    source_margin_arcsec=source_margin_arcsec,
                    overwrite=overwrite,
                    crop_br=crop_br,
                    cleanup_originals=cleanup_originals,
                )
            )

        for future in as_completed(futures):
            index, saved_total, deleted_total, messages, records = future.result()
            all_records.extend(records)
            print(f"[{index:03d}] saved_total={saved_total} deleted_total={deleted_total}")
            for message in messages:
                print("  " + message)

    if missing_report_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        missing_report_path = root_dir / f"sub_rot_map_missing_report_{timestamp}.xlsx"
    missing_report_path = Path(missing_report_path)

    report_df = pd.DataFrame(
        _dedupe_records(all_records),
        columns=["event", "passband", "issue", "path", "detail"],
    )
    report_df.to_excel(missing_report_path, index=False)
    print(f"Missing/error report saved to: {missing_report_path}")
    return report_df


if __name__ == "__main__":
    run_sub_rot_map()
