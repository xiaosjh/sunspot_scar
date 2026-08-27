from __future__ import annotations

import argparse
import csv
from pathlib import Path

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


def start_time_to_folder_name(t_utc) -> str:
    t = Time(t_utc, scale="utc")
    return t.utc.strftime("%Y%m%d_%H%M%S_UTC")


def get_40min_tai_window(t_utc):
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


def scan_missing(root_dir: Path = ROOT_DIR) -> list[dict]:
    rows = []
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


def write_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "index",
        "start_time",
        "event",
        "empty_subdir",
        "exists",
        "file_count",
        "path",
        "status",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_and_download(client: drms.Client, qstr: str, out_dir: Path, email: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(out_dir.glob("*.fits"))
    if existing_files:
        print(f"  [SKIP] {out_dir} already has {len(existing_files)} FITS files")
        return existing_files

    print(f"Exporting: {qstr}")
    print(f"Save to: {out_dir}")
    export_req = client.export(
        qstr,
        method="url_quick",
        protocol="fits",
        email=email,
    )

    if len(export_req.urls) == 0:
        print("  [WARNING] export returned no URLs")
        return []

    dl = export_req.download(str(out_dir))
    files = [Path(f) for f in dl.download]
    print(f"  Saved {len(files)} FITS files")
    return files


def jsoc_trec_to_astropy_time(trec_str) -> Time:
    s = trec_str.decode("utf-8") if isinstance(trec_str, (bytes, bytearray)) else str(trec_str)
    if s.endswith("_TAI"):
        s = s[:-4]
    s = s.replace(".", "-", 2).replace("_", "T", 1)
    return Time(s, format="isot", scale="tai")


def find_previous_hmi_vector_720s_record(client: drms.Client, t_utc, lookback=2 * u.hour):
    t_utc = Time(t_utc, scale="utc")
    t_tai = t_utc.tai

    t_start = (t_tai - lookback).strftime("%Y.%m.%d_%H:%M:%S_TAI")
    t_end = t_tai.strftime("%Y.%m.%d_%H:%M:%S_TAI")

    q = f"{HMI_VECTOR_SERIES}[{t_start}-{t_end}]"
    print(f"Query previous HMI vector: {q}")

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


def download_one_missing(row: dict, df: pd.DataFrame, clients: dict[str, drms.Client]) -> dict:
    event_index = int(row["index"])
    channel = row["empty_subdir"]
    t_utc = df.loc[event_index, "Start Time"]
    event_dir = ROOT_DIR / row["event"]

    try:
        _, t_start_str, t_end_str = get_40min_tai_window(t_utc)

        if channel in {"131", "211", "304"}:
            qstr = (
                f"{AIA_SERIES}"
                f"[{t_start_str}-{t_end_str}@{AIA_CADENCE}]"
                f"[{channel}]"
                f"{{image}}"
            )
            files = export_and_download(
                clients["main"],
                qstr,
                event_dir / channel,
                JSOC_EMAIL,
            )
        elif channel == HMI_FOLDER_NAME:
            qstr = (
                f"{HMI_SERIES_FULL}"
                f"[{t_start_str}-{t_end_str}]"
                f"{{{HMI_SEGMENT}}}"
            )
            files = export_and_download(
                clients["main"],
                qstr,
                event_dir / HMI_FOLDER_NAME,
                JSOC_EMAIL,
            )
        elif channel == HMI_VECTOR_FOLDER_NAME:
            previous_trec = find_previous_hmi_vector_720s_record(clients["vector"], t_utc)
            if previous_trec is None:
                files = []
            else:
                segment_str = ",".join(HMI_VECTOR_SEGMENTS)
                qstr = f"{HMI_VECTOR_SERIES}[{previous_trec}]{{{segment_str}}}"
                files = export_and_download(
                    clients["vector"],
                    qstr,
                    event_dir / HMI_VECTOR_FOLDER_NAME,
                    VECTOR_JSOC_EMAIL,
                )
        else:
            raise ValueError(f"Unknown channel: {channel}")

        row["status"] = "ok" if len(files) > 0 else "empty_after_download"
        row["error"] = ""
    except Exception as exc:
        row["status"] = "error"
        row["error"] = repr(exc)

    return row


def check() -> list[dict]:
    rows = scan_missing()
    write_csv(rows, BASE_DIR / "missing_M_downloads.csv")
    print(f"Missing/empty channel rows: {len(rows)}")
    print(f"Events with any missing/empty channel: {len({row['event'] for row in rows})}")
    print(f"Wrote: {BASE_DIR / 'missing_M_downloads.csv'}")
    return rows


def download(limit: int | None = None) -> None:
    rows = check()
    if limit is not None:
        rows = rows[:limit]
        print(f"Limit enabled: only downloading first {limit} missing rows")

    if not rows:
        print("No missing downloads found.")
        return

    df = load_m_events()
    clients = {
        "main": drms.Client(email=JSOC_EMAIL),
        "vector": drms.Client(email=VECTOR_JSOC_EMAIL),
    }

    results = []
    for n, row in enumerate(rows, start=1):
        print("\n" + "=" * 80)
        print(f"[{n}/{len(rows)}] index={row['index']} event={row['event']} channel={row['empty_subdir']}")
        results.append(download_one_missing(row, df, clients))

    write_csv(results, BASE_DIR / "redownload_attempts_M.csv")
    print(f"Wrote: {BASE_DIR / 'redownload_attempts_M.csv'}")

    remaining = scan_missing()
    write_csv(remaining, BASE_DIR / "redownload_failed_M.csv")
    print(f"Remaining missing/empty channel rows: {len(remaining)}")
    print(f"Remaining events with any missing/empty channel: {len({row['event'] for row in remaining})}")
    print(f"Wrote: {BASE_DIR / 'redownload_failed_M.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only scan empty/missing channel folders.")
    parser.add_argument("--download", action="store_true", help="Download missing/empty channel folders.")
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N missing rows.")
    args = parser.parse_args()

    if args.download:
        download(args.limit)
    else:
        check()


if __name__ == "__main__":
    main()
