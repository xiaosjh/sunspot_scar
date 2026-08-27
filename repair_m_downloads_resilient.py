r'''
使用说明
cd C:\Learning\PHD2nd\sunspotscar\program

# 先检查剩余缺失
python repair_m_downloads_resilient.py --check

# 先验证某一个事件 JSOC 是否有记录，不下载
python repair_m_downloads_resilient.py --preflight-only --only-event 20160212_103600_UTC --sleep 5 --max-retries 3

# 正式补下载，建议 sleep 设长一点，避免 JSOC pending 队列卡死
python repair_m_downloads_resilient.py --download --sleep 60 --max-retries 8

'''

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path
from typing import Any

import astropy.units as u
import drms
import pandas as pd
from astropy.time import Time


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "dataset" / "M级耀斑.xlsx"
ROOT_DIR = BASE_DIR.parent / "data" / "M"

JSOC_EMAIL = "652024260007@smail.nju.edu.cn"
VECTOR_JSOC_EMAIL = "19374325@buaa.edu.cn"

AIA_SERIES = "aia.lev1_euv_12s"
AIA_WAVELENGTHS = [131, 211, 304]
AIA_CADENCE = "36s"

HMI_SERIES_FULL = "hmi.Ic_45s"
HMI_SEGMENT = "continuum"
HMI_FOLDER_NAME = "hmi.Ic_45s"

HMI_VECTOR_SERIES = "hmi.B_720s"
HMI_VECTOR_FOLDER_NAME = "hmi.B_720s"
HMI_VECTOR_SEGMENTS = ["field", "inclination", "azimuth", "disambig"]

EXPECTED_SUBDIRS = ["131", "211", "304", HMI_FOLDER_NAME, HMI_VECTOR_FOLDER_NAME]


def load_m_events() -> pd.DataFrame:
    df = pd.read_excel(DATASET_PATH)
    df["Start Time"] = pd.to_datetime(df["Start Time"])
    df["event_name"] = df["Start Time"].apply(start_time_to_folder_name)
    return df


def start_time_to_folder_name(t_utc: Any) -> str:
    t = Time(t_utc, scale="utc")
    return t.utc.strftime("%Y%m%d_%H%M%S_UTC")


def get_40min_tai_window(t_utc: Any) -> tuple[Time, str, str]:
    t_utc = Time(t_utc, scale="utc")
    t_tai = t_utc.tai
    t_start = t_tai - 30 * u.minute
    t_end = t_tai + 10 * u.minute
    return (
        t_utc,
        t_start.strftime("%Y.%m.%d_%H:%M:%S_TAI"),
        t_end.strftime("%Y.%m.%d_%H:%M:%S_TAI"),
    )


def fits_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return len(list(path.rglob("*.fits")))


def scan_missing(root_dir: Path = ROOT_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    df = load_m_events()
    for index, row in df.iterrows():
        event_name = row["event_name"]
        event_dir = root_dir / event_name
        for subdir in EXPECTED_SUBDIRS:
            path = event_dir / subdir
            count = fits_count(path)
            if count == 0:
                rows.append(
                    {
                        "index": int(index),
                        "start_time": row["Start Time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "event": event_name,
                        "empty_subdir": subdir,
                        "exists": path.is_dir(),
                        "file_count": count,
                        "path": str(path),
                    }
                )
    return rows


def read_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rescan_input_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only re-check the rows from the input CSV, not the whole data directory."""
    remaining: list[dict[str, Any]] = []
    for row in rows:
        event = row["event"]
        subdir = row["empty_subdir"]
        path = ROOT_DIR / event / subdir
        count = fits_count(path)
        if count == 0:
            updated = dict(row)
            updated["exists"] = path.is_dir()
            updated["file_count"] = count
            updated["path"] = str(path)
            remaining.append(updated)
    return remaining


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> Path:
    fieldnames = [
        "index",
        "start_time",
        "event",
        "empty_subdir",
        "exists",
        "file_count",
        "path",
        "query",
        "records",
        "attempts",
        "status",
        "error",
    ]
    try:
        target = csv_path
        f = target.open("w", newline="", encoding="utf-8-sig")
    except PermissionError:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        target = csv_path.with_name(f"{csv_path.stem}_{stamp}{csv_path.suffix}")
        print(f"  [WARNING] {csv_path.name} is locked; writing {target.name} instead")
        f = target.open("w", newline="", encoding="utf-8-sig")

    with f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target


def is_transient_jsoc_error(exc: Exception) -> bool:
    text = repr(exc)
    transient_markers = [
        "502",
        "503",
        "504",
        "Bad Gateway",
        "timed out",
        "TimeoutError",
        "pending export requests",
        "[status=7]",
        "Connection",
        "Remote end closed connection",
    ]
    return any(marker in text for marker in transient_markers)


def pending_request_id(exc: Exception) -> str | None:
    match = re.search(r"(JSOC_\d+_\d+)", repr(exc))
    return match.group(1) if match else None


def sleep_with_jitter(seconds: float) -> None:
    delay = seconds + random.uniform(0, min(5.0, seconds * 0.25))
    print(f"  Sleep {delay:.1f}s before retry/next request")
    time.sleep(delay)


def query_record_count(client: drms.Client, qstr: str, max_retries: int, base_sleep: float) -> int:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            res = client.query(qstr, key="T_REC")
            return 0 if res is None else len(res)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries or not is_transient_jsoc_error(exc):
                raise
            print(f"  [QUERY RETRY {attempt}/{max_retries}] {repr(exc)}")
            sleep_with_jitter(base_sleep * attempt)
    raise RuntimeError(f"query failed after retries: {last_error!r}")


def safe_export_and_download(
    client: drms.Client,
    qstr: str,
    out_dir: Path,
    email: str,
    max_retries: int,
    base_sleep: float,
) -> tuple[list[Path], int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(out_dir.glob("*.fits"))
    if existing_files:
        print(f"  [SKIP] {out_dir} already has {len(existing_files)} FITS files")
        return existing_files, 0

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Exporting attempt {attempt}/{max_retries}: {qstr}")
            print(f"Save to: {out_dir}")
            export_req = client.export(
                qstr,
                method="url_quick",
                protocol="fits",
                email=email,
            )

            if len(export_req.urls) == 0:
                print("  [WARNING] export returned no URLs")
                return [], attempt

            dl = export_req.download(str(out_dir))

            downloaded: list[Path] = []
            for item in getattr(dl, "download", []):
                if item is None:
                    continue
                downloaded.append(Path(item))

            disk_files = list(out_dir.glob("*.fits"))
            if disk_files:
                print(f"  Saved {len(disk_files)} FITS files on disk")
                return disk_files, attempt

            print(f"  [WARNING] download completed but no FITS on disk; raw download list={downloaded}")
            return downloaded, attempt
        except Exception as exc:
            last_error = exc
            rid = pending_request_id(exc)
            if rid:
                print(f"  [JSOC pending/request issue] request id: {rid}")

            if attempt == max_retries or not is_transient_jsoc_error(exc):
                raise

            print(f"  [DOWNLOAD RETRY {attempt}/{max_retries}] {repr(exc)}")
            sleep_with_jitter(base_sleep * attempt)

    raise RuntimeError(f"download failed after retries: {last_error!r}")


def jsoc_trec_to_astropy_time(trec_str: Any) -> Time:
    s = trec_str.decode("utf-8") if isinstance(trec_str, (bytes, bytearray)) else str(trec_str)
    if s.endswith("_TAI"):
        s = s[:-4]
    s = s.replace(".", "-", 2).replace("_", "T", 1)
    return Time(s, format="isot", scale="tai")


def find_previous_hmi_vector_720s_record(
    client: drms.Client,
    t_utc: Any,
    max_retries: int,
    base_sleep: float,
    lookback=2 * u.hour,
) -> str | None:
    t_utc = Time(t_utc, scale="utc")
    t_tai = t_utc.tai

    t_start = (t_tai - lookback).strftime("%Y.%m.%d_%H:%M:%S_TAI")
    t_end = t_tai.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    q = f"{HMI_VECTOR_SERIES}[{t_start}-{t_end}]"
    print(f"Query previous HMI vector: {q}")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            res = client.query(q, key="T_REC")
            if res is None or len(res) == 0:
                print(f"  [WARNING] No {HMI_VECTOR_SERIES} record before {t_utc.isot}")
                return None

            trec_strs = [
                s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                for s in res["T_REC"]
            ]
            trec_times = Time([jsoc_trec_to_astropy_time(s) for s in trec_strs])
            valid = trec_times <= t_tai
            if not valid.any():
                print(f"  [WARNING] No previous record <= {t_utc.isot}")
                return None

            valid_indices = [i for i, ok in enumerate(valid) if ok]
            dt = t_tai - trec_times[valid]
            idx = valid_indices[dt.argmin()]
            nearest_previous_trec = trec_strs[idx]
            print(f"  Previous nearest T_REC: {nearest_previous_trec}")
            print(f"  Delta before Start Time: {(t_tai - trec_times[idx]).to(u.minute):.2f}")
            return nearest_previous_trec
        except Exception as exc:
            last_error = exc
            if attempt == max_retries or not is_transient_jsoc_error(exc):
                raise
            print(f"  [VECTOR QUERY RETRY {attempt}/{max_retries}] {repr(exc)}")
            sleep_with_jitter(base_sleep * attempt)

    raise RuntimeError(f"vector query failed after retries: {last_error!r}")


def build_query(row: dict[str, Any], df: pd.DataFrame, clients: dict[str, drms.Client], max_retries: int, base_sleep: float):
    event_index = int(row["index"])
    channel = str(row["empty_subdir"])
    t_utc = df.loc[event_index, "Start Time"]
    _, t_start_str, t_end_str = get_40min_tai_window(t_utc)

    if channel in {"131", "211", "304"}:
        qstr = f"{AIA_SERIES}[{t_start_str}-{t_end_str}@{AIA_CADENCE}][{channel}]{{image}}"
        return qstr, ROOT_DIR / row["event"] / channel, clients["main"], JSOC_EMAIL

    if channel == HMI_FOLDER_NAME:
        qstr = f"{HMI_SERIES_FULL}[{t_start_str}-{t_end_str}]{{{HMI_SEGMENT}}}"
        return qstr, ROOT_DIR / row["event"] / HMI_FOLDER_NAME, clients["main"], JSOC_EMAIL

    if channel == HMI_VECTOR_FOLDER_NAME:
        previous_trec = find_previous_hmi_vector_720s_record(
            clients["vector"],
            t_utc,
            max_retries=max_retries,
            base_sleep=base_sleep,
        )
        if previous_trec is None:
            return None, ROOT_DIR / row["event"] / HMI_VECTOR_FOLDER_NAME, clients["vector"], VECTOR_JSOC_EMAIL
        segment_str = ",".join(HMI_VECTOR_SEGMENTS)
        qstr = f"{HMI_VECTOR_SERIES}[{previous_trec}]{{{segment_str}}}"
        return qstr, ROOT_DIR / row["event"] / HMI_VECTOR_FOLDER_NAME, clients["vector"], VECTOR_JSOC_EMAIL

    raise ValueError(f"Unknown channel: {channel}")


def repair_one(
    row: dict[str, Any],
    df: pd.DataFrame,
    clients: dict[str, drms.Client],
    max_retries: int,
    base_sleep: float,
    preflight_only: bool,
) -> dict[str, Any]:
    result = dict(row)
    try:
        qstr, out_dir, client, email = build_query(row, df, clients, max_retries, base_sleep)
        result["query"] = qstr or ""

        if qstr is None:
            result["records"] = 0
            result["attempts"] = 0
            result["status"] = "no_vector_record"
            result["error"] = ""
            return result

        records = query_record_count(client, qstr, max_retries=max_retries, base_sleep=base_sleep)
        result["records"] = records
        if records == 0:
            result["attempts"] = 0
            result["status"] = "no_records"
            result["error"] = ""
            return result

        if preflight_only:
            result["attempts"] = 0
            result["status"] = "preflight_ok"
            result["error"] = ""
            return result

        files, attempts = safe_export_and_download(
            client,
            qstr,
            out_dir,
            email=email,
            max_retries=max_retries,
            base_sleep=base_sleep,
        )
        result["attempts"] = attempts
        result["status"] = "ok" if files else "empty_after_download"
        result["error"] = ""
        return result
    except Exception as exc:
        result["attempts"] = result.get("attempts", "")
        result["status"] = "error"
        result["error"] = repr(exc)
        return result


def check() -> list[dict[str, Any]]:
    rows = scan_missing()
    csv_path = write_csv(rows, BASE_DIR / "missing_M_downloads.csv")
    print(f"Missing/empty channel rows: {len(rows)}")
    print(f"Events with any missing/empty channel: {len({row['event'] for row in rows})}")
    print(f"Wrote: {csv_path}")
    return rows


def download(args: argparse.Namespace) -> None:
    if args.failed_csv:
        rows = read_rows_from_csv(Path(args.failed_csv))
        print(f"Using input CSV only: {args.failed_csv}")
    else:
        rows = scan_missing()
        print("No input CSV was given; using global scan_missing().")

    if args.only_event:
        rows = [row for row in rows if row["event"] == args.only_event]

    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"Limit enabled: only processing first {args.limit} missing rows")

    current_run_csv = write_csv(rows, BASE_DIR / "missing_M_downloads_current_run.csv")
    print(f"Rows to process: {len(rows)}")
    print(f"Wrote: {current_run_csv}")

    if not rows:
        print("No missing downloads found.")
        return

    df = load_m_events()
    clients = {
        "main": drms.Client(email=JSOC_EMAIL),
        "vector": drms.Client(email=VECTOR_JSOC_EMAIL),
    }

    results: list[dict[str, Any]] = []
    for n, row in enumerate(rows, start=1):
        print("\n" + "=" * 80)
        print(f"[{n}/{len(rows)}] index={row['index']} event={row['event']} channel={row['empty_subdir']}")
        result = repair_one(
            row,
            df,
            clients,
            max_retries=args.max_retries,
            base_sleep=args.sleep,
            preflight_only=args.preflight_only,
        )
        results.append(result)
        attempts_csv = write_csv(results, BASE_DIR / "redownload_attempts_M_resilient.csv")

        if result["status"] == "error" and is_transient_jsoc_error(Exception(result.get("error", ""))):
            sleep_with_jitter(args.sleep * 2)
        elif not args.preflight_only:
            sleep_with_jitter(args.sleep)

    print(f"Wrote: {attempts_csv}")

    if args.failed_csv:
        remaining = rescan_input_rows(rows)
    else:
        remaining = scan_missing()
    failed_csv = write_csv(remaining, BASE_DIR / "redownload_failed_M_resilient.csv")
    if args.failed_csv:
        print(f"Remaining missing/empty rows from input CSV: {len(remaining)}")
        print(f"Remaining input-CSV events with any missing/empty channel: {len({row['event'] for row in remaining})}")
    else:
        print(f"Remaining missing/empty channel rows: {len(remaining)}")
        print(f"Remaining events with any missing/empty channel: {len({row['event'] for row in remaining})}")
    print(f"Wrote: {failed_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only scan empty/missing channel folders.")
    parser.add_argument("--download", action="store_true", help="Download missing/empty channel folders.")
    parser.add_argument("--preflight-only", action="store_true", help="Query JSOC record counts without downloading.")
    parser.add_argument(
        "--failed-csv",
        default=None,
        help="Use this CSV as the only input rows; no global scan is done for download/final failed rows.",
    )
    parser.add_argument("--only-event", default=None, help="Process only one event folder, e.g. 20160212_103600_UTC.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N missing rows.")
    parser.add_argument("--max-retries", type=int, default=6, help="Retries for transient JSOC errors.")
    parser.add_argument("--sleep", type=float, default=30.0, help="Base sleep seconds between JSOC requests/retries.")
    args = parser.parse_args()

    if args.download or args.preflight_only:
        download(args)
    else:
        check()


if __name__ == "__main__":
    main()
