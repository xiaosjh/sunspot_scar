from __future__ import annotations

import math
import os
import re
import shutil
import threading
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from tkinter import (
    BooleanVar,
    Button,
    Checkbutton,
    Canvas,
    DoubleVar,
    Entry,
    Frame,
    IntVar,
    Label,
    LabelFrame,
    Listbox,
    StringVar,
    Scrollbar,
    Tk,
    colorchooser,
    filedialog,
    messagebox,
)
from tkinter import ttk

import astropy.units as u
import imageio.v2 as imageio
import numpy as np
from astropy.io import fits
from matplotlib import colormaps
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.patheffects as path_effects
from PIL import Image

try:
    from sunpy.visualization.colormaps import color_tables as sunpy_color_tables
except Exception:
    sunpy_color_tables = None


DEFAULT_DATA_ROOT = Path(r"C:\Learning\PHD2nd\sunspotscar\data\M")
FITS_EXTENSIONS = {".fits", ".fit", ".fts"}
# Edit these names if you want different channels checked by default.
DEFAULT_CHANNEL_HINTS: tuple[str, ...] = (
    "131sub_map",
    "304sub_map",
    "211sub_map",
    "hmi.B_720s/Br_sub.fits",
    "hmi.B_720s/inclination_calc_sub.fits",
    "hmi.Ic_45ssub_map",
    "hmi.Ic_45s_sub_map",
)
# Edit these values if you want a different default Br display range.
DEFAULT_BR_MIN_G = -800.0
DEFAULT_BR_MAX_G = 800.0
# Change to "Curve" if you want freehand annotations by default.
DEFAULT_ANNOTATION_MODE = "Line"
ANNOTATION_MODES = ("Line", "Curve")
PREVIEW_CACHE_VERSION = "preview-v10"
PREVIEW_CACHE_DIR = Path(__file__).resolve().parent / ".preview_cache"
AIA_STRETCHES = ("asinh", "log", "linear")

FALLBACK_AIA_CMAPS = {
    131: LinearSegmentedColormap.from_list(
        "AIA 131 fallback",
        ["#000000", "#001a18", "#006d70", "#16d9d2", "#d8ffff"],
        N=256,
    ),
    211: LinearSegmentedColormap.from_list(
        "AIA 211 fallback",
        ["#000000", "#16002f", "#4c177d", "#b95fd3", "#fff0ff"],
        N=256,
    ),
    304: LinearSegmentedColormap.from_list(
        "AIA 304 fallback",
        ["#000000", "#2a0000", "#990000", "#ff6a00", "#fff4b0"],
        N=256,
    ),
}


@dataclass(frozen=True)
class FrameInfo:
    path: Path
    time: datetime | None
    label: str


@dataclass
class ChannelSequence:
    name: str
    frames: list[FrameInfo]


@dataclass(frozen=True)
class SourceOption:
    key: str
    path: Path
    label: str
    selected: bool = False


@dataclass(frozen=True)
class DisplaySettings:
    level_low: float = -100.0
    level_high: float = 500.0
    br_min: float = DEFAULT_BR_MIN_G
    br_max: float = DEFAULT_BR_MAX_G
    stretch: str = "asinh"
    vmax_percentile: float = 99.99

    def cache_key(self) -> str:
        return (
            f"levels={self.level_low:.4g},{self.level_high:.4g}"
            f"|br={self.br_min:.4g},{self.br_max:.4g}|stretch={self.stretch}"
            f"|vmaxpct={self.vmax_percentile:.6g}"
        )


@dataclass(frozen=True)
class DrawnPanel:
    ax: object
    image_artist: object
    text_artist: object
    frame: FrameInfo
    sequence_name: str
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class AnnotationStroke:
    points: tuple[tuple[float, float], ...]


def is_fits(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in FITS_EXTENSIONS


def parse_time_text(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    candidates = [
        text.replace("Z", ""),
        text.replace("_TAI", "").replace(".", "-", 2).replace("_", " "),
    ]
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass

    match = re.search(r"(\d{4})[-.]?(\d{2})[-.]?(\d{2})[T_ ]?(\d{2})(\d{2})(\d{2})", text)
    if match:
        y, m, d, hh, mm, ss = map(int, match.groups())
        return datetime(y, m, d, hh, mm, ss)
    return None


def time_from_filename(path: Path) -> datetime | None:
    return parse_time_text(path.name)


def find_image_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        data = hdu.data
        if isinstance(data, np.ndarray) and data.ndim >= 2:
            return hdu
    raise ValueError("No 2-D image data found in this FITS file.")


def scan_frame(path: Path) -> FrameInfo:
    time_value = None
    try:
        with fits.open(path, memmap=True) as hdul:
            hdu = find_image_hdu(hdul)
            header = hdu.header
            time_value = (
                parse_time_text(header.get("T_OBS"))
                or parse_time_text(header.get("DATE-OBS"))
                or parse_time_text(header.get("DATE_OBS"))
            )
    except Exception:
        time_value = None

    if time_value is None:
        time_value = time_from_filename(path)
    label = time_value.isoformat(sep="T", timespec="milliseconds") if time_value else path.stem
    return FrameInfo(path=path, time=time_value, label=label)


def scan_channel(channel_dir: Path) -> ChannelSequence:
    frames = [scan_frame(path) for path in sorted(channel_dir.iterdir()) if is_fits(path)]
    frames.sort(key=lambda item: (item.time is None, item.time or datetime.min, item.path.name))
    return ChannelSequence(name=channel_dir.name, frames=frames)


def scan_source(source: SourceOption) -> ChannelSequence:
    if source.path.is_file():
        return ChannelSequence(name=source.key, frames=[scan_frame(source.path)])
    return ChannelSequence(name=source.key, frames=scan_channel(source.path).frames)


@lru_cache(maxsize=16)
def load_fits_image(path_text: str) -> np.ndarray:
    with fits.open(path_text, memmap=True) as hdul:
        hdu = find_image_hdu(hdul)
        data = np.asarray(hdu.data, dtype=np.float32)
    if data.ndim > 2:
        data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D FITS image, got shape {data.shape}")
    return data


def is_hmi_continuum(sequence_name: str) -> bool:
    lower = sequence_name.lower()
    return "hmi.ic" in lower or "continuum" in lower


def is_br_channel(sequence_name: str) -> bool:
    lower = sequence_name.lower()
    return "br.fits" in lower or "br_sub.fits" in lower


def is_inclination_channel(sequence_name: str) -> bool:
    lower = sequence_name.lower()
    return "inclination_calc" in lower or "inclination" in lower


def is_default_selected_channel(name: str) -> bool:
    lower = name.lower()
    if name in DEFAULT_CHANNEL_HINTS:
        return True
    if lower.endswith("br_sub.fits"):
        return True
    if "sub_map" not in lower:
        return False
    return (
        lower.startswith("131")
        or lower.startswith("304")
        or "hmi.ic_45s" in lower
        or "hmi.ic45s" in lower
    )


def orient_image_for_display(data: np.ndarray, sequence_name: str) -> np.ndarray:
    if is_hmi_continuum(sequence_name):
        return np.rot90(data, 2)
    return data


def base_display_range(
    data: np.ndarray,
    sequence_name: str = "",
    vmax_percentile: float = 99.99,
) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    if is_hmi_continuum(sequence_name):
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
        if low == high:
            high = low + 1.0
        return low, high
    if finite.size > 250_000:
        step = math.ceil(finite.size / 250_000)
        finite = finite[::step]
    lower = sequence_name.lower()
    if "hmi.b" in lower or "mag" in lower:
        limit = float(np.nanpercentile(np.abs(finite), 99.2))
        if np.isfinite(limit) and limit > 0:
            return -limit, limit
    if aia_wavelength(sequence_name) is not None:
        upper = min(100.0, max(90.0, float(vmax_percentile)))
        low, high = np.nanpercentile(finite, [1, upper])
    else:
        low, high = np.nanpercentile(finite, [1, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def display_limits(
    data: np.ndarray,
    sequence_name: str = "",
    settings: DisplaySettings | None = None,
    base_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    settings = settings or DisplaySettings()
    if is_br_channel(sequence_name):
        low, high = float(settings.br_min), float(settings.br_max)
        if low > high:
            low, high = high, low
        if low == high:
            high = low + 1.0
        return low, high
    if is_inclination_channel(sequence_name):
        return 0.0, 90.0

    base_low, base_high = base_range or base_display_range(data, sequence_name, settings.vmax_percentile)
    if base_low > base_high:
        base_low, base_high = base_high, base_low
    if base_low == base_high:
        base_high = base_low + 1.0
    level_low = min(max(-100.0, settings.level_low), 1000.0)
    level_high = min(max(-100.0, settings.level_high), 1000.0)
    if level_low >= level_high:
        midpoint = (level_low + level_high) / 2.0
        level_low = max(-100.0, midpoint - 0.05)
        level_high = min(1000.0, midpoint + 0.05)
    span = base_high - base_low
    low = base_low + span * ((level_low + 100.0) / 200.0)
    high = base_low + span * ((level_high + 100.0) / 200.0)
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def nearest_frame(sequence: ChannelSequence, target: FrameInfo, index: int) -> FrameInfo:
    if not sequence.frames:
        raise ValueError(f"No frames loaded for {sequence.name}")
    if target.time is None:
        return sequence.frames[min(index, len(sequence.frames) - 1)]
    timed = [frame for frame in sequence.frames if frame.time is not None]
    if not timed:
        return sequence.frames[min(index, len(sequence.frames) - 1)]
    return min(timed, key=lambda frame: abs((frame.time - target.time).total_seconds()))


def aia_wavelength(sequence_name: str) -> int | None:
    match = re.search(r"(?<!\d)(131|211|304)(?!\d)", sequence_name)
    if not match:
        return None
    return int(match.group(1))


def channel_cmap(sequence_name: str):
    if is_inclination_channel(sequence_name):
        return colormaps["Blues_r"]
    wavelength = aia_wavelength(sequence_name)
    if wavelength is None:
        return "gray"
    if sunpy_color_tables is not None:
        try:
            return sunpy_color_tables.aia_color_table(wavelength * u.angstrom)
        except Exception:
            pass
    return FALLBACK_AIA_CMAPS[wavelength]


def display_stretch(sequence_name: str, settings: DisplaySettings) -> str:
    if aia_wavelength(sequence_name) is None:
        return "linear"
    stretch = settings.stretch.lower()
    return stretch if stretch in AIA_STRETCHES else "asinh"


def default_display_settings(sequence_name: str = "") -> DisplaySettings:
    if is_hmi_continuum(sequence_name):
        return DisplaySettings(level_low=-100.0, level_high=100.0, stretch="linear")
    if aia_wavelength(sequence_name) is not None:
        return DisplaySettings(level_low=-100.0, level_high=500.0, stretch="asinh")
    return DisplaySettings(level_low=-100.0, level_high=500.0, stretch="linear")


def apply_stretch(scaled: np.ndarray, stretch: str) -> np.ndarray:
    scaled = np.clip(scaled, 0.0, 1.0)
    if stretch == "log":
        return np.log1p(1000.0 * scaled) / np.log1p(1000.0)
    if stretch == "asinh":
        # SunPy's AIA plotting defaults use an asinh-style stretch; this keeps faint EUV structure visible.
        a = 0.01
        return np.arcsinh(scaled / a) / np.arcsinh(1.0 / a)
    return scaled


def channel_display_name(sequence_name: str) -> str:
    if is_inclination_channel(sequence_name):
        return "HMI inclination"
    wavelength = aia_wavelength(sequence_name)
    if wavelength is not None:
        return f"AIA {wavelength}"
    lower = sequence_name.lower()
    if "hmi.ic" in lower:
        return "HMI continuum"
    if "hmi.b" in lower:
        return "HMI magnetogram"
    return sequence_name


def overlay_text(frame: FrameInfo, sequence_name: str) -> str:
    return f"{frame.label} {channel_display_name(sequence_name)}"


def grid_shape(count: int) -> tuple[int, int]:
    if count <= 0:
        return 1, 1
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    if count == 2:
        return 1, 2
    return rows, cols


def read_preview_data(path: Path, max_side: int) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        hdu = find_image_hdu(hdul)
        data = np.squeeze(hdu.data)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2-D FITS image, got shape {data.shape}")
        stride = max(1, math.ceil(max(data.shape) / max_side))
        if stride > 1:
            data = data[::stride, ::stride]
        return np.asarray(data, dtype=np.float32)


def make_preview_image(path: Path, sequence_name: str, max_side: int, settings: DisplaySettings | None = None) -> np.ndarray:
    settings = settings or DisplaySettings()
    data = orient_image_for_display(read_preview_data(path, max_side), sequence_name)
    return colorize_preview_data(data, sequence_name, settings)


def colorize_preview_data(
    data: np.ndarray,
    sequence_name: str,
    settings: DisplaySettings | None = None,
    base_range: tuple[float, float] | None = None,
) -> np.ndarray:
    settings = settings or DisplaySettings()
    vmin, vmax = display_limits(data, sequence_name, settings, base_range)
    scaled = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    scaled = apply_stretch(scaled, display_stretch(sequence_name, settings))

    cmap = channel_cmap(sequence_name)
    if cmap == "gray":
        gray = (scaled * 255).astype(np.uint8)
        rgb = np.dstack([gray, gray, gray])
    else:
        rgb = (cmap(scaled)[:, :, :3] * 255).astype(np.uint8)

    return rgb


def scalar_disk_cache_path(path: Path, max_side: int) -> Path:
    stat = path.stat()
    key = f"{PREVIEW_CACHE_VERSION}|scalar|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{max_side}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return PREVIEW_CACHE_DIR / f"{digest}.npy"


def preview_disk_cache_path(path: Path, max_side: int, settings: DisplaySettings | None = None) -> Path:
    settings = settings or DisplaySettings()
    stat = path.stat()
    key = f"{PREVIEW_CACHE_VERSION}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{max_side}|{settings.cache_key()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return PREVIEW_CACHE_DIR / f"{digest}.png"


def clear_preview_disk_cache() -> None:
    """Remove only this application's preview-cache directory."""
    cache_dir = PREVIEW_CACHE_DIR.resolve()
    expected_dir = (Path(__file__).resolve().parent / ".preview_cache").resolve()
    if cache_dir != expected_dir:
        raise RuntimeError(f"Refusing to clear unexpected cache path: {cache_dir}")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def load_or_make_preview_image(
    path: Path,
    sequence_name: str,
    max_side: int,
    use_disk_cache: bool,
    settings: DisplaySettings | None = None,
) -> np.ndarray:
    settings = settings or DisplaySettings()
    cache_path = preview_disk_cache_path(path, max_side, settings)
    if use_disk_cache and cache_path.exists():
        try:
            return np.asarray(Image.open(cache_path).convert("RGB"))
        except Exception:
            cache_path.unlink(missing_ok=True)

    rgb = make_preview_image(path, sequence_name, max_side, settings)
    if use_disk_cache:
        PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".{threading.get_ident()}.tmp.png")
        try:
            Image.fromarray(rgb).save(tmp_path)
            tmp_path.replace(cache_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return rgb


def build_preview_worker(
    path_text: str,
    sequence_name: str,
    max_side: int,
    use_disk_cache: bool,
    settings: DisplaySettings | None = None,
) -> tuple[str, str, str]:
    settings = settings or DisplaySettings()
    path = Path(path_text)
    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scalar_path = scalar_disk_cache_path(path, max_side)
    if scalar_path.exists():
        data = np.load(scalar_path)
    else:
        data = orient_image_for_display(read_preview_data(path, max_side), sequence_name)
        tmp_scalar = scalar_path.with_suffix(f".{os.getpid()}.tmp.npy")
        try:
            np.save(tmp_scalar, data)
            tmp_scalar.replace(scalar_path)
        finally:
            tmp_scalar.unlink(missing_ok=True)

    if use_disk_cache:
        cache_path = preview_disk_cache_path(path, max_side, settings)
        if not cache_path.exists():
            rgb = colorize_preview_data(data, sequence_name, settings)
            tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.png")
            try:
                Image.fromarray(rgb).save(tmp_path)
                tmp_path.replace(cache_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        return path_text, str(cache_path), str(scalar_path)

    rgb = colorize_preview_data(data, sequence_name, settings)
    digest = hashlib.sha1(f"memory|{os.getpid()}|{threading.get_ident()}|{path_text}".encode("utf-8")).hexdigest()
    tmp_path = PREVIEW_CACHE_DIR / f"{digest}.tmp.png"
    Image.fromarray(rgb).save(tmp_path)
    return path_text, str(tmp_path), str(scalar_path)


def draw_sequence_grid(
    fig: Figure,
    sequences: list[ChannelSequence],
    timeline: list[FrameInfo],
    index: int,
    preview_cache: dict[str, np.ndarray] | None = None,
    detail_cache: dict[str, np.ndarray] | None = None,
    view_state: tuple[float, float, float, float] | None = None,
    settings: DisplaySettings | None = None,
    settings_by_sequence: dict[str, DisplaySettings] | None = None,
    annotations: list[AnnotationStroke] | None = None,
    annotation_color: str = "#ffe45c",
    annotation_width: float = 0.5,
) -> tuple[list[str], list[DrawnPanel]]:
    settings = settings or DisplaySettings()
    target = timeline[index]
    rows, cols = grid_shape(len(sequences))
    fig.clear()
    fig.patch.set_facecolor("black")
    labels = []
    panels = []
    panel_items = []

    for sequence in sequences:
        frame = nearest_frame(sequence, target, index)
        path_key = str(frame.path)
        preview = None
        if detail_cache is not None and path_key in detail_cache:
            preview = detail_cache[path_key]
        elif preview_cache is not None:
            preview = preview_cache.get(path_key)
        panel_items.append((sequence, frame, preview))

    image_aspects = []
    for sequence, frame, preview in panel_items:
        if preview is not None:
            height, width = preview.shape[:2]
        else:
            data = orient_image_for_display(load_fits_image(str(frame.path)), sequence.name)
            height, width = data.shape[:2]
        image_aspects.append(width / height if height else 1.0)

    cell_aspect = float(np.median(image_aspects)) if image_aspects else 1.0
    figure_aspect = fig.get_figwidth() / fig.get_figheight()
    content_aspect = cols * cell_aspect / rows
    if figure_aspect > content_aspect:
        content_width = content_aspect / figure_aspect
        left = (1.0 - content_width) / 2.0
        fig.subplots_adjust(left=left, right=1.0 - left, bottom=0, top=1, wspace=0, hspace=0)
    else:
        content_height = figure_aspect / content_aspect
        bottom = (1.0 - content_height) / 2.0
        fig.subplots_adjust(left=0, right=1, bottom=bottom, top=1.0 - bottom, wspace=0, hspace=0)

    for panel, (sequence, frame, preview) in enumerate(panel_items, start=1):

        ax = fig.add_subplot(rows, cols, panel)
        ax.set_facecolor("black")
        if preview is None:
            data = orient_image_for_display(load_fits_image(str(frame.path)), sequence.name)
            sequence_settings = settings_by_sequence.get(sequence.name, settings) if settings_by_sequence else settings
            preview = colorize_preview_data(data, sequence.name, sequence_settings)
            height, width = data.shape[:2]
            image_artist = ax.imshow(preview, origin="lower", aspect="equal", extent=(0, width, 0, height))
        else:
            height, width = preview.shape[:2]
            image_artist = ax.imshow(preview, origin="lower", aspect="equal", extent=(0, width, 0, height))
        ax.set_aspect("equal", adjustable="box")
        if view_state is None:
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
        else:
            x0, x1, y0, y1 = view_state
            ax.set_xlim(x0 * width, x1 * width)
            ax.set_ylim(y0 * height, y1 * height)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        text = overlay_text(frame, sequence.name)
        labels.append(f"{channel_display_name(sequence.name)}: {frame.label}")
        artist = ax.text(
            0.012,
            0.018,
            text,
            transform=ax.transAxes,
            color=(0.84, 0.84, 0.84, 0.86),
            fontsize=14,
            ha="left",
            va="bottom",
        )
        artist.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground=(0, 0, 0, 0.55))])
        if annotations:
            for stroke in annotations:
                line_artist = ax.plot(
                    [point[0] * width for point in stroke.points],
                    [point[1] * height for point in stroke.points],
                    color=annotation_color,
                    linewidth=annotation_width,
                    alpha=0.95,
                    zorder=30,
                )[0]
                line_artist.set_path_effects(
                    [path_effects.withStroke(linewidth=annotation_width + 1.6, foreground=(0, 0, 0, 0.45))]
                )
        panels.append(
            DrawnPanel(ax=ax, image_artist=image_artist, text_artist=artist, frame=frame, sequence_name=sequence.name, image_shape=(height, width))
        )

    return labels, panels


class FitsViewerApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Local FITS Solar Viewer")
        self.root.geometry("1280x820")

        self.data_root = StringVar(value=str(DEFAULT_DATA_ROOT))
        self.event_name = StringVar()
        self.status = StringVar(value="Choose an event and load channels.")
        self.frame_index = IntVar(value=0)
        self.start_index = IntVar(value=0)
        self.end_index = IntVar(value=0)
        self.fps = IntVar(value=8)
        self.preview_max_side = IntVar(value=4096)
        self.detail_max_side = IntVar(value=4096)
        self.cache_workers = IntVar(value=min(12, max(1, os.cpu_count() or 1)))
        self.selected_display_layer = StringVar(value="")
        self.selected_br_layer = StringVar(value="")
        self.aia_low_level = StringVar(value="-100")
        self.aia_high_level = StringVar(value="500")
        self.aia_low_label = StringVar(value="-100%")
        self.aia_high_label = StringVar(value="500%")
        self.aia_low_slider = DoubleVar(value=-100.0)
        self.aia_high_slider = DoubleVar(value=500.0)
        self.levels_active_handle: str | None = None
        self.aia_stretch = StringVar(value="asinh")
        self.aia_vmax_percentile = DoubleVar(value=99.99)
        self.aia_vmax_text = StringVar(value="99.99")
        self.br_min_level = StringVar(value=f"{DEFAULT_BR_MIN_G:g}")
        self.br_max_level = StringVar(value=f"{DEFAULT_BR_MAX_G:g}")
        self.br_symmetric = BooleanVar(value=False)
        self.display_settings = DisplaySettings()
        self.layer_settings: dict[str, DisplaySettings] = {}
        self.pending_display_redraw = None
        self.pending_slider_redraw = None
        self.pending_slider_index: int | None = None
        self.pending_canvas_redraw = None
        self.use_process_pool = BooleanVar(value=True)
        self.use_disk_cache = BooleanVar(value=True)
        self.playing = False
        self.ignore_slider_callback = False
        self.source_options: dict[str, SourceOption] = {}
        self.channel_vars: dict[str, BooleanVar] = {}
        self.channel_order: list[str] = []
        self.channel_drag_name: str | None = None
        self.channel_drag_target: int | None = None
        self.channel_row_frames: list[Frame] = []
        self.sequences: list[ChannelSequence] = []
        self.timeline: list[FrameInfo] = []
        self.preview_cache: dict[str, np.ndarray] = {}
        self.preview_cache_keys: dict[str, str] = {}
        self.scalar_cache: dict[str, np.ndarray] = {}
        self.preview_cache_side = 0
        self.cached_event_path: Path | None = None
        self.detail_cache: dict[str, np.ndarray] = {}
        self.detail_cache_keys: dict[str, str] = {}
        self.detail_cache_side = 0
        self.view_state: tuple[float, float, float, float] | None = None
        self.drawn_panels: list[DrawnPanel] = []
        self.drag_state: tuple[float, float, tuple[float, float, float, float], object] | None = None
        self.annotation_lines: list[AnnotationStroke] = []
        self.annotation_artists: list[object] = []
        self.annotation_draft_artists: list[object] = []
        self.annotation_points: list[tuple[float, float]] | None = None
        self.annotation_source_ax: object | None = None
        self.annotation_start_canvas_point: tuple[float, float] | None = None
        self.annotation_last_canvas_point: tuple[float, float] | None = None
        self.annotation_path_length_px = 0.0
        self.annotation_active_mode = DEFAULT_ANNOTATION_MODE
        self.shift_pressed = False
        self.annotations_visible = BooleanVar(value=True)
        self.annotation_mode = StringVar(value=DEFAULT_ANNOTATION_MODE)
        self.annotation_color = StringVar(value="#ffe45c")
        self.annotation_width = DoubleVar(value=0.5)
        self.axes = []
        self.images = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Left>", lambda _event: self.step_frame(-1))
        self.root.bind("<Right>", lambda _event: self.step_frame(1))
        self.root.bind_all("<KeyPress-Shift_L>", self.on_shift_press)
        self.root.bind_all("<KeyPress-Shift_R>", self.on_shift_press)
        self.root.bind_all("<KeyRelease-Shift_L>", self.on_shift_release)
        self.root.bind_all("<KeyRelease-Shift_R>", self.on_shift_release)
        self.root.bind_all("<Shift-BackSpace>", self.undo_annotation)
        self.root.bind_all("<Shift-Delete>", self.undo_annotation)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.refresh_events()

    def _build_ui(self) -> None:
        self.root.configure(bg="#171a1f")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TCombobox", padding=3)
        style.configure("TScale", background="#171a1f")

        top = Frame(self.root, padx=8, pady=8, bg="#f2f4f7")
        top.pack(side="top", fill="x")

        Label(top, text="Data root").grid(row=0, column=0, sticky="w")
        Entry(top, textvariable=self.data_root, width=74).grid(row=0, column=1, sticky="w", padx=6)
        Button(top, text="Browse", command=self.browse_root).grid(row=0, column=2, padx=2)
        Button(top, text="Refresh", command=self.refresh_events).grid(row=0, column=3, padx=2)
        Checkbutton(top, text="Annotation", variable=self.annotations_visible, command=self.redraw_annotations).grid(
            row=0, column=5, padx=(10, 2)
        )
        Button(top, text="Clear lines", command=self.clear_annotations).grid(row=0, column=6, padx=2)

        Label(top, text="Draw").grid(row=1, column=5, sticky="e", padx=(10, 2), pady=(6, 0))
        self.annotation_mode_combo = ttk.Combobox(
            top,
            textvariable=self.annotation_mode,
            values=ANNOTATION_MODES,
            state="readonly",
            width=8,
        )
        self.annotation_mode_combo.grid(row=1, column=6, sticky="w", padx=2, pady=(6, 0))
        Label(top, text="FPS").grid(row=1, column=7, sticky="e", padx=(10, 2), pady=(6, 0))
        ttk.Spinbox(top, from_=1, to=60, textvariable=self.fps, width=6).grid(row=1, column=8, sticky="w", pady=(6, 0))

        Label(top, text="Event").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.event_combo = ttk.Combobox(top, textvariable=self.event_name, state="readonly", width=48)
        self.event_combo.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        self.event_combo.bind("<<ComboboxSelected>>", self.on_event_changed)
        top.columnconfigure(4, weight=1)

        bottom = Frame(self.root, padx=8, pady=8, bg="#f2f4f7")
        bottom.pack(side="bottom", fill="x")
        self.load_progress = ttk.Progressbar(bottom, orient="horizontal", mode="determinate", length=160)
        self.load_progress.pack(side="left", padx=(0, 8), fill="x")
        Label(bottom, textvariable=self.status, anchor="w", bg="#f2f4f7").pack(side="left", fill="x")

        main = ttk.PanedWindow(self.root, orient="horizontal")
        main.pack(side="top", fill="both", expand=True)

        side_outer = Frame(main, bg="#f2f4f7", width=400)
        main.add(side_outer, weight=0)
        side_canvas = Canvas(side_outer, bg="#f2f4f7", highlightthickness=0, width=380)
        side_scrollbar = Scrollbar(side_outer, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scrollbar.set)
        side_scrollbar.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)
        side = Frame(side_canvas, padx=8, pady=4, bg="#f2f4f7")
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda _event: side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>", lambda event: side_canvas.itemconfigure(side_window, width=event.width))
        def _scroll_side(event):
            side_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        side_canvas.bind("<Enter>", lambda _event: side_canvas.bind_all("<MouseWheel>", _scroll_side))
        side_canvas.bind("<Leave>", lambda _event: side_canvas.unbind_all("<MouseWheel>"))

        channels_box = LabelFrame(side, text="Channels", padx=6, pady=6)
        channels_box.pack(side="top", fill="both", expand=False)
        channel_buttons = Frame(channels_box)
        channel_buttons.pack(fill="x", pady=(0, 6))
        Button(channel_buttons, text="Sub maps", command=self.select_submaps).pack(side="left", fill="x", expand=True)
        Button(channel_buttons, text="All", command=lambda: self.set_channel_selection(True)).pack(side="left", fill="x", expand=True, padx=3)
        Button(channel_buttons, text="Clear", command=lambda: self.set_channel_selection(False)).pack(side="left", fill="x", expand=True)
        Button(channels_box, text="Load selected", command=self.load_selected).pack(fill="x", pady=(0, 6))
        self.channels_frame = Frame(channels_box)
        self.channels_frame.pack(fill="both", expand=True)

        display_box = LabelFrame(side, text="Display", padx=6, pady=6)
        display_box.pack(side="top", fill="x", pady=(10, 0))
        Label(display_box, text="Layer").grid(row=0, column=0, sticky="w")
        self.display_layer_combo = ttk.Combobox(display_box, textvariable=self.selected_display_layer, state="readonly", width=14)
        self.display_layer_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=4)
        self.display_layer_combo.bind("<<ComboboxSelected>>", self.on_display_layer_selected)
        Label(display_box, text="Levels").grid(row=1, column=0, sticky="w")
        self.levels_canvas = Canvas(display_box, height=34, bg="#f2f4f7", highlightthickness=0, width=190)
        self.levels_canvas.grid(row=1, column=1, sticky="ew", padx=4)
        self.levels_canvas.bind("<Configure>", lambda _event: self.redraw_levels_control())
        self.levels_canvas.bind("<ButtonPress-1>", self.on_levels_press)
        self.levels_canvas.bind("<B1-Motion>", self.on_levels_drag)
        self.levels_canvas.bind("<ButtonRelease-1>", self.on_levels_release)
        level_values = Frame(display_box, bg="#f2f4f7")
        level_values.grid(row=1, column=2, sticky="e")
        Label(level_values, textvariable=self.aia_low_label, bg="#f2f4f7", anchor="e", width=7).pack(anchor="e")
        Label(level_values, textvariable=self.aia_high_label, bg="#f2f4f7", anchor="e", width=7).pack(anchor="e")
        self.stretch_label = Label(display_box, text="AIA stretch")
        self.stretch_combo = ttk.Combobox(display_box, textvariable=self.aia_stretch, state="readonly", values=AIA_STRETCHES, width=8)
        self.stretch_combo.bind("<<ComboboxSelected>>", self.on_stretch_selected)
        self.vmax_label = Label(display_box, text="Vmax percentile")
        self.vmax_scale = ttk.Scale(
            display_box,
            from_=90.0,
            to=100.0,
            orient="horizontal",
            variable=self.aia_vmax_percentile,
            command=self.on_vmax_slider,
        )
        self.vmax_entry = Entry(display_box, textvariable=self.aia_vmax_text, justify="right", width=9)
        self.vmax_entry.bind("<Return>", self.apply_vmax_entry)
        self.apply_layer_button = Button(display_box, text="Apply layer", command=self.apply_display_settings)
        self.apply_all_button = Button(display_box, text="Apply all layers", command=self.apply_all_loaded_display_settings)
        self.layout_display_controls("")
        display_box.columnconfigure(1, weight=1)
        display_box.columnconfigure(2, weight=2)

        br_box = LabelFrame(side, text="Br display", padx=6, pady=6)
        br_box.pack(side="top", fill="x", pady=(10, 0))
        Label(br_box, text="Layer").grid(row=0, column=0, sticky="w")
        self.br_layer_combo = ttk.Combobox(br_box, textvariable=self.selected_br_layer, state="readonly", width=14)
        self.br_layer_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=4)
        self.br_layer_combo.bind("<<ComboboxSelected>>", self.on_br_layer_selected)
        Label(br_box, text="Br min G").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.br_min_entry = Entry(br_box, textvariable=self.br_min_level, width=8)
        self.br_min_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=(4, 0))
        self.br_min_entry.bind("<Return>", lambda _event: self.commit_br_range("min"))
        Label(br_box, text="Br max G").grid(row=2, column=0, sticky="w")
        self.br_max_entry = Entry(br_box, textvariable=self.br_max_level, width=8)
        self.br_max_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=4)
        self.br_max_entry.bind("<Return>", lambda _event: self.commit_br_range("max"))
        self.br_symmetric_check = Checkbutton(
            br_box,
            text="Symmetric ±",
            variable=self.br_symmetric,
            command=self.on_br_symmetric_toggle,
            anchor="w",
        )
        self.br_symmetric_check.grid(row=3, column=0, columnspan=3, sticky="w")
        Button(br_box, text="Apply Br", command=self.apply_br_display_settings).grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )
        br_box.columnconfigure(1, weight=1)
        br_box.columnconfigure(2, weight=1)

        annotation_box = LabelFrame(side, text="Annotation", padx=6, pady=6)
        annotation_box.pack(side="top", fill="x", pady=(10, 0))
        Label(annotation_box, text="Width").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(annotation_box, from_=0.5, to=8.0, increment=0.5, textvariable=self.annotation_width, width=7).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        Button(annotation_box, text="Color", command=self.choose_annotation_color).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.annotation_color_preview = Label(annotation_box, textvariable=self.annotation_color, bg=self.annotation_color.get(), width=9)
        self.annotation_color_preview.grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        Button(annotation_box, text="Apply lines", command=self.redraw_annotations).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        annotation_box.columnconfigure(1, weight=1)

        export_box = LabelFrame(side, text="Export", padx=6, pady=6)
        export_box.pack(side="top", fill="x", pady=(10, 0))
        Button(export_box, text="Export MP4/GIF", command=self.export_video).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )
        Label(export_box, text="Start").grid(row=1, column=0, sticky="w")
        self.start_spin = ttk.Spinbox(export_box, from_=0, to=0, textvariable=self.start_index, width=7)
        self.start_spin.grid(row=1, column=1, sticky="ew", padx=4)
        Label(export_box, text="End").grid(row=2, column=0, sticky="w")
        self.end_spin = ttk.Spinbox(export_box, from_=0, to=0, textvariable=self.end_index, width=7)
        self.end_spin.grid(row=2, column=1, sticky="ew", padx=4)
        Label(export_box, text="FPS").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(export_box, from_=1, to=60, textvariable=self.fps, width=7).grid(row=3, column=1, sticky="ew", padx=4)
        Label(export_box, text="Cache px").grid(row=4, column=0, sticky="w")
        cache_spin = ttk.Spinbox(export_box, from_=300, to=4096, increment=100, textvariable=self.preview_max_side, width=7)
        cache_spin.grid(
            row=4, column=1, sticky="ew", padx=4
        )
        cache_spin.bind("<Return>", self.apply_cache_px_change)
        cache_spin.bind("<FocusOut>", self.apply_cache_px_change)
        Label(export_box, text="Zoom px").grid(row=5, column=0, sticky="w")
        zoom_spin = ttk.Spinbox(export_box, from_=900, to=4096, increment=100, textvariable=self.detail_max_side, width=7)
        zoom_spin.grid(
            row=5, column=1, sticky="ew", padx=4
        )
        zoom_spin.bind("<Return>", self.apply_zoom_px_change)
        zoom_spin.bind("<FocusOut>", self.apply_zoom_px_change)
        Label(export_box, text="Workers").grid(row=6, column=0, sticky="w")
        ttk.Spinbox(export_box, from_=1, to=32, textvariable=self.cache_workers, width=7).grid(row=6, column=1, sticky="ew", padx=4)
        Checkbutton(export_box, text="Processes", variable=self.use_process_pool, anchor="w").grid(row=7, column=0, columnspan=2, sticky="w")
        Checkbutton(export_box, text="Disk cache", variable=self.use_disk_cache, anchor="w").grid(row=8, column=0, columnspan=2, sticky="w")
        Button(export_box, text="Reset view", command=self.reset_view).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        export_box.columnconfigure(1, weight=1)

        self.info_list = Listbox(side, height=9)
        self.info_list.pack(side="top", fill="x", pady=(10, 0))

        plot_area = Frame(main, bg="black")
        main.add(plot_area, weight=1)
        timeline_bar = Frame(plot_area, padx=8, pady=6, bg="#f2f4f7")
        timeline_bar.pack(side="top", fill="x")
        Button(timeline_bar, text="Play/Pause", command=self.toggle_play).pack(side="left")
        self.slider = ttk.Scale(timeline_bar, from_=0, to=0, orient="horizontal", command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)
        Label(timeline_bar, textvariable=self.status, anchor="e", bg="#f2f4f7", width=38).pack(side="left", padx=(6, 0))
        self.figure = Figure(figsize=(9, 6), dpi=100, facecolor="black")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)


    def browse_root(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.data_root.get() or str(DEFAULT_DATA_ROOT))
        if chosen:
            self.data_root.set(chosen)
            self.refresh_events()

    def event_path(self) -> Path:
        return Path(self.data_root.get()) / self.event_name.get()

    def refresh_events(self) -> None:
        root = Path(self.data_root.get())
        events = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.exists() else []
        self.event_combo["values"] = events
        if events and self.event_name.get() not in events:
            self.event_name.set(events[0])
        self.refresh_channels()
        self.status.set(f"Found {len(events)} event folders.")

    def clear_event_cache(self, clear_disk: bool = False) -> None:
        self.preview_cache.clear()
        self.preview_cache_keys.clear()
        self.scalar_cache.clear()
        self.preview_cache_side = 0
        self.detail_cache.clear()
        self.detail_cache_keys.clear()
        self.detail_cache_side = 0
        self.cached_event_path = None
        load_fits_image.cache_clear()
        if clear_disk:
            clear_preview_disk_cache()

    def clear_event_cache_with_warning(self) -> None:
        try:
            self.clear_event_cache(clear_disk=True)
        except OSError as exc:
            messagebox.showwarning(
                "Cache cleanup incomplete",
                f"Some preview-cache files could not be removed:\n{exc}",
            )

    def on_close(self) -> None:
        self.playing = False
        self.status.set("Clearing preview cache...")
        self.root.update_idletasks()
        self.clear_event_cache_with_warning()
        self.root.destroy()

    def on_event_changed(self, _event=None) -> None:
        self.status.set("Clearing previous event cache...")
        self.root.update_idletasks()
        self.clear_event_cache_with_warning()
        self.sequences = []
        self.timeline = []
        self.view_state = None
        self.drawn_panels = []
        self.annotation_artists = []
        self.annotation_draft_artists = []
        self.cancel_annotation_draft()
        self.figure.clear()
        self.canvas.draw_idle()
        self.refresh_channels()

    def refresh_channels(self) -> None:
        for child in self.channels_frame.winfo_children():
            child.destroy()
        self.source_options.clear()
        self.channel_vars.clear()
        self.channel_order.clear()
        self.channel_row_frames.clear()

        event = self.event_path()
        if not event.exists():
            return

        options = self.build_source_options(event)
        for option in options:
            selected = option.selected or is_default_selected_channel(option.key)
            var = BooleanVar(value=selected)
            self.source_options[option.key] = option
            self.channel_vars[option.key] = var
            self.channel_order.append(option.key)
        self.render_channel_rows()

    def render_channel_rows(self) -> None:
        for child in self.channels_frame.winfo_children():
            child.destroy()
        self.channel_row_frames.clear()
        for row, name in enumerate(self.channel_order):
            option = self.source_options[name]
            row_frame = Frame(self.channels_frame)
            row_frame.grid(row=row, column=0, sticky="ew")
            drag_handle = Label(row_frame, text="☰", cursor="fleur", padx=3)
            drag_handle.pack(side="left")
            Checkbutton(row_frame, text=option.label, variable=self.channel_vars[name], anchor="w").pack(
                side="left", fill="x", expand=True
            )
            drag_handle.bind("<ButtonPress-1>", lambda event, key=name: self.on_channel_drag_start(event, key))
            drag_handle.bind("<B1-Motion>", self.on_channel_drag_motion)
            drag_handle.bind("<ButtonRelease-1>", self.on_channel_drag_release)
            self.channel_row_frames.append(row_frame)
        self.channels_frame.columnconfigure(0, weight=1)

    def channel_row_at_pointer(self, event) -> int | None:
        if not self.channel_order:
            return None
        local_y = event.y_root - self.channels_frame.winfo_rooty()
        centers = [frame.winfo_y() + frame.winfo_height() / 2.0 for frame in self.channel_row_frames]
        return min(range(len(centers)), key=lambda index: abs(centers[index] - local_y))

    def on_channel_drag_start(self, _event, name: str) -> None:
        self.channel_drag_name = name
        self.channel_drag_target = self.channel_order.index(name)

    def on_channel_drag_motion(self, event) -> None:
        if self.channel_drag_name is not None:
            self.channel_drag_target = self.channel_row_at_pointer(event)

    def on_channel_drag_release(self, event) -> None:
        name = self.channel_drag_name
        target = self.channel_row_at_pointer(event)
        self.channel_drag_name = None
        self.channel_drag_target = None
        if name is None or target is None:
            return
        source = self.channel_order.index(name)
        if source == target:
            return
        self.channel_order.pop(source)
        self.channel_order.insert(target, name)
        self.render_channel_rows()
        rank = {key: index for index, key in enumerate(self.channel_order)}
        if self.sequences:
            self.sequences.sort(key=lambda sequence: rank.get(sequence.name, len(rank)))
            self.update_display_layer_options()
            self.draw_frame(self.frame_index.get(), load_detail=False)
        self.status.set(f"Moved {self.source_options[name].label} to position {target + 1}.")

    def build_source_options(self, event: Path) -> list[SourceOption]:
        options: list[SourceOption] = []
        for channel_dir in sorted(path for path in event.iterdir() if path.is_dir()):
            if channel_dir.name == "hmi.B_720s":
                for filename in ("Br_sub.fits", "Br.fits", "inclination_calc_sub.fits"):
                    source = channel_dir / filename
                    if source.exists():
                        options.append(SourceOption(key=f"{channel_dir.name}/{filename}", path=source, label=f"{filename} (1)"))
                fits_count = sum(1 for path in channel_dir.iterdir() if is_fits(path))
                options.append(SourceOption(key=channel_dir.name, path=channel_dir, label=f"{channel_dir.name} all ({fits_count})"))
                continue

            count = sum(1 for path in channel_dir.iterdir() if is_fits(path))
            options.append(SourceOption(key=channel_dir.name, path=channel_dir, label=f"{channel_dir.name} ({count})"))
        return options

    def set_channel_selection(self, selected: bool) -> None:
        for var in self.channel_vars.values():
            var.set(selected)

    def select_submaps(self) -> None:
        for name, var in self.channel_vars.items():
            var.set(name.endswith("sub_map") or name.endswith("Br_sub.fits"))

    def current_display_settings(self) -> DisplaySettings:
        try:
            level_low = float(self.aia_low_level.get())
            level_high = float(self.aia_high_level.get())
        except ValueError as exc:
            raise ValueError("Display values must be numbers.") from exc
        name = self.selected_sequence_name()
        previous = self.settings_for_sequence(name) if name is not None else DisplaySettings()
        stretch = self.aia_stretch.get().lower()
        if stretch not in AIA_STRETCHES:
            stretch = "asinh"
        vmax_percentile = min(100.0, max(90.0, float(self.aia_vmax_percentile.get())))
        level_low = min(max(-100.0, level_low), 1000.0)
        level_high = min(max(-100.0, level_high), 1000.0)
        if level_low >= level_high:
            level_high = min(1000.0, level_low + 0.1)
            if level_low >= level_high:
                level_low = max(-100.0, level_high - 0.1)
        return DisplaySettings(
            level_low=level_low,
            level_high=level_high,
            br_min=previous.br_min,
            br_max=previous.br_max,
            stretch=stretch,
            vmax_percentile=vmax_percentile,
        )

    def current_br_display_settings(self) -> DisplaySettings:
        name = self.selected_br_sequence_name()
        if name is None:
            raise ValueError("No Br layer is loaded.")
        previous = self.settings_for_sequence(name)
        try:
            br_min = float(self.br_min_level.get())
            br_max = float(self.br_max_level.get())
        except ValueError as exc:
            raise ValueError("Br min and max must be numbers.") from exc
        if self.br_symmetric.get():
            magnitude = max(abs(br_min), abs(br_max))
            if magnitude == 0:
                raise ValueError("Symmetric Br range must be greater than zero.")
            br_min, br_max = -magnitude, magnitude
            self.br_min_level.set(f"{br_min:g}")
            self.br_max_level.set(f"{br_max:g}")
        if br_min == br_max:
            raise ValueError("Br min and max must be different.")
        return DisplaySettings(
            level_low=previous.level_low,
            level_high=previous.level_high,
            br_min=br_min,
            br_max=br_max,
            stretch=previous.stretch,
            vmax_percentile=previous.vmax_percentile,
        )

    def commit_br_range(self, source: str) -> None:
        if self.br_symmetric.get():
            variable = self.br_min_level if source == "min" else self.br_max_level
            try:
                magnitude = abs(float(variable.get()))
            except ValueError:
                messagebox.showerror("Invalid Br range", "Br range must be a number.")
                return
            if magnitude == 0:
                messagebox.showerror("Invalid Br range", "Symmetric Br range must be greater than zero.")
                return
            self.br_min_level.set(f"{-magnitude:g}")
            self.br_max_level.set(f"{magnitude:g}")
        self.apply_br_display_settings()

    def on_br_symmetric_toggle(self) -> None:
        if not self.br_symmetric.get():
            return
        try:
            magnitude = max(abs(float(self.br_min_level.get())), abs(float(self.br_max_level.get())))
        except ValueError:
            messagebox.showerror("Invalid Br range", "Br range must be a number.")
            self.br_symmetric.set(False)
            return
        if magnitude == 0:
            messagebox.showerror("Invalid Br range", "Symmetric Br range must be greater than zero.")
            self.br_symmetric.set(False)
            return
        self.br_min_level.set(f"{-magnitude:g}")
        self.br_max_level.set(f"{magnitude:g}")
        self.apply_br_display_settings()

    def settings_for_sequence(self, sequence_name: str) -> DisplaySettings:
        return self.layer_settings.get(sequence_name, default_display_settings(sequence_name))

    def selected_sequence_name(self) -> str | None:
        selected = self.selected_display_layer.get()
        if selected and not is_br_channel(selected):
            return selected
        return next((sequence.name for sequence in self.sequences if not is_br_channel(sequence.name)), None)

    def selected_br_sequence_name(self) -> str | None:
        selected = self.selected_br_layer.get()
        if selected and is_br_channel(selected):
            return selected
        return next((sequence.name for sequence in self.sequences if is_br_channel(sequence.name)), None)

    def update_display_layer_options(self) -> None:
        names = [sequence.name for sequence in self.sequences if not is_br_channel(sequence.name)]
        self.display_layer_combo["values"] = names
        if names and self.selected_display_layer.get() not in names:
            self.selected_display_layer.set(names[0])
            self.load_display_settings_to_controls(names[0])
        elif not names:
            self.selected_display_layer.set("")
            self.layout_display_controls("")
        self.update_br_layer_options()

    def update_br_layer_options(self) -> None:
        names = [sequence.name for sequence in self.sequences if is_br_channel(sequence.name)]
        self.br_layer_combo["values"] = names
        if names and self.selected_br_layer.get() not in names:
            self.selected_br_layer.set(names[0])
            self.load_br_settings_to_controls(names[0])
        elif not names:
            self.selected_br_layer.set("")

    def load_br_settings_to_controls(self, sequence_name: str) -> None:
        settings = self.settings_for_sequence(sequence_name)
        self.br_min_level.set(f"{settings.br_min:.2f}".rstrip("0").rstrip("."))
        self.br_max_level.set(f"{settings.br_max:.2f}".rstrip("0").rstrip("."))

    def on_br_layer_selected(self, _event=None) -> None:
        name = self.selected_br_layer.get()
        if name:
            self.load_br_settings_to_controls(name)

    def load_display_settings_to_controls(self, sequence_name: str) -> None:
        settings = self.settings_for_sequence(sequence_name)
        self.aia_low_level.set(f"{settings.level_low:.2f}".rstrip("0").rstrip("."))
        self.aia_high_level.set(f"{settings.level_high:.2f}".rstrip("0").rstrip("."))
        low, high = self.clamp_level_pair(settings.level_low, settings.level_high)
        self.aia_low_slider.set(low)
        self.aia_high_slider.set(high)
        self.update_level_labels()
        self.redraw_levels_control()
        self.aia_stretch.set(display_stretch(sequence_name, settings))
        self.aia_vmax_percentile.set(settings.vmax_percentile)
        self.update_vmax_label()
        self.layout_display_controls(sequence_name)

    def layout_display_controls(self, sequence_name: str) -> None:
        row = 2
        if aia_wavelength(sequence_name) is not None:
            self.vmax_label.grid(row=row, column=0, sticky="w")
            self.vmax_scale.grid(row=row, column=1, sticky="ew", padx=4)
            self.vmax_entry.grid(row=row, column=2, sticky="e")
            row += 1
            self.stretch_label.grid(row=row, column=0, sticky="w")
            self.stretch_combo.grid(row=row, column=1, columnspan=2, sticky="ew", padx=4)
            row += 1
        else:
            self.vmax_label.grid_remove()
            self.vmax_scale.grid_remove()
            self.vmax_entry.grid_remove()
            self.stretch_label.grid_remove()
            self.stretch_combo.grid_remove()

        self.apply_layer_button.grid(row=row, column=0, sticky="ew", pady=(6, 0))
        self.apply_all_button.grid(row=row, column=1, columnspan=2, sticky="ew", pady=(6, 0), padx=(4, 0))

    def update_level_labels(self) -> None:
        low, high = self.clamp_level_pair(float(self.aia_low_slider.get()), float(self.aia_high_slider.get()))
        self.aia_low_level.set(f"{low:.2f}".rstrip("0").rstrip("."))
        self.aia_high_level.set(f"{high:.2f}".rstrip("0").rstrip("."))
        self.aia_low_label.set(f"{low:.0f}%")
        self.aia_high_label.set(f"{high:.0f}%")

    def clamp_level_pair(self, low: float, high: float) -> tuple[float, float]:
        low = min(max(-100.0, low), 1000.0)
        high = min(max(-100.0, high), 1000.0)
        if low >= high:
            if self.levels_active_handle == "low":
                low = max(-100.0, high - 0.1)
            else:
                high = min(1000.0, low + 0.1)
            if low >= high:
                low, high = -100.0, 1000.0
        return low, high

    def level_value_to_x(self, value: float) -> float:
        width = max(80, self.levels_canvas.winfo_width())
        pad = 12
        return pad + ((value + 100.0) / 1100.0) * (width - 2 * pad)

    def level_x_to_value(self, x: float) -> float:
        width = max(80, self.levels_canvas.winfo_width())
        pad = 12
        fraction = min(1.0, max(0.0, (x - pad) / max(1, width - 2 * pad)))
        return -100.0 + fraction * 1100.0

    def redraw_levels_control(self) -> None:
        if not hasattr(self, "levels_canvas"):
            return
        canvas = self.levels_canvas
        canvas.delete("all")
        width = max(80, canvas.winfo_width())
        y = 17
        pad = 12
        low, high = self.clamp_level_pair(float(self.aia_low_slider.get()), float(self.aia_high_slider.get()))
        low_x = self.level_value_to_x(low)
        high_x = self.level_value_to_x(high)
        canvas.create_line(pad, y, width - pad, y, fill="#8b8f96", width=3)
        canvas.create_line(low_x, y, high_x, y, fill="#4d79ff", width=3)
        canvas.create_oval(low_x - 7, y - 7, low_x + 7, y + 7, fill="#a7abb1", outline="#70757d", width=1)
        canvas.create_oval(high_x - 7, y - 7, high_x + 7, y + 7, fill="#a7abb1", outline="#70757d", width=1)

    def on_levels_press(self, event) -> None:
        low_x = self.level_value_to_x(float(self.aia_low_slider.get()))
        high_x = self.level_value_to_x(float(self.aia_high_slider.get()))
        self.levels_active_handle = "low" if abs(event.x - low_x) <= abs(event.x - high_x) else "high"
        self.on_levels_drag(event)

    def on_levels_drag(self, event) -> None:
        if self.levels_active_handle == "low":
            value = min(max(-100.0, self.level_x_to_value(event.x)), float(self.aia_high_slider.get()) - 0.1)
            self.aia_low_slider.set(value)
        elif self.levels_active_handle == "high":
            value = max(min(1000.0, self.level_x_to_value(event.x)), float(self.aia_low_slider.get()) + 0.1)
            self.aia_high_slider.set(value)
        self.on_level_slider()

    def on_levels_release(self, _event=None) -> None:
        self.levels_active_handle = None

    def on_display_layer_selected(self, _event=None) -> None:
        name = self.selected_display_layer.get()
        if name:
            self.load_display_settings_to_controls(name)

    def on_stretch_selected(self, _event=None) -> None:
        self.apply_aia_control_change()

    def update_vmax_label(self) -> None:
        value = min(100.0, max(90.0, float(self.aia_vmax_percentile.get())))
        self.aia_vmax_text.set(f"{value:.5f}".rstrip("0").rstrip("."))

    def apply_vmax_entry(self, _event=None) -> None:
        try:
            value = float(self.aia_vmax_text.get().strip().rstrip("%"))
        except ValueError:
            self.update_vmax_label()
            self.status.set("Vmax percentile must be a number between 90 and 100.")
            return
        value = min(100.0, max(90.0, value))
        self.aia_vmax_percentile.set(value)
        self.update_vmax_label()
        self.apply_aia_control_change()

    def on_vmax_slider(self, _value: str | None = None) -> None:
        value = min(100.0, max(90.0, float(self.aia_vmax_percentile.get())))
        self.aia_vmax_percentile.set(value)
        self.update_vmax_label()
        self.apply_aia_control_change(deferred=True)

    def apply_aia_control_change(self, deferred: bool = False) -> None:
        try:
            settings = self.current_display_settings()
        except ValueError:
            return
        name = self.selected_sequence_name()
        if name is None:
            return
        self.layer_settings[name] = settings
        self.detail_cache.clear()
        self.detail_cache_keys.clear()
        self.detail_cache_side = 0
        if deferred:
            if self.pending_display_redraw is not None:
                self.root.after_cancel(self.pending_display_redraw)
            self.pending_display_redraw = self.root.after(10, self.recolor_current_frame)
        else:
            self.recolor_current_frame()

    def on_level_slider(self, _value: str | None = None) -> None:
        low, high = self.clamp_level_pair(float(self.aia_low_slider.get()), float(self.aia_high_slider.get()))
        self.aia_low_slider.set(low)
        self.aia_high_slider.set(high)
        self.update_level_labels()
        self.redraw_levels_control()
        try:
            settings = self.current_display_settings()
        except ValueError:
            return
        name = self.selected_sequence_name()
        if name is None:
            return
        self.layer_settings[name] = settings
        self.detail_cache.clear()
        self.detail_cache_keys.clear()
        self.detail_cache_side = 0
        if self.pending_display_redraw is not None:
            self.root.after_cancel(self.pending_display_redraw)
        self.pending_display_redraw = self.root.after(10, self.recolor_current_frame)

    def recolor_current_frame(self) -> None:
        self.pending_display_redraw = None
        if not self.sequences or not self.timeline:
            return
        selected = self.selected_sequence_name()
        changed = False
        for panel in self.drawn_panels:
            if selected is not None and panel.sequence_name != selected:
                continue
            key = str(panel.frame.path)
            data = self.scalar_cache.get(key)
            if data is not None:
                rgb = colorize_preview_data(
                    data,
                    panel.sequence_name,
                    self.settings_for_sequence(panel.sequence_name),
                )
                self.preview_cache[key] = rgb
                self.preview_cache_keys[key] = self.settings_for_sequence(panel.sequence_name).cache_key()
                panel.image_artist.set_data(rgb)
                changed = True
        if changed:
            self.canvas.draw_idle()

    def apply_display_settings(self) -> None:
        try:
            settings = self.current_display_settings()
        except ValueError as exc:
            messagebox.showerror("Invalid display settings", str(exc))
            return
        name = self.selected_sequence_name()
        if name is None:
            return
        self.layer_settings[name] = settings
        low, high = self.clamp_level_pair(settings.level_low, settings.level_high)
        self.aia_low_slider.set(low)
        self.aia_high_slider.set(high)
        self.update_level_labels()
        self.redraw_levels_control()
        recolored = 0
        for sequence in self.sequences:
            if sequence.name != name:
                continue
            for frame in sequence.frames:
                key = str(frame.path)
                data = self.scalar_cache.get(key)
                if data is not None:
                    self.preview_cache[key] = colorize_preview_data(
                        data,
                        sequence.name,
                        settings,
                    )
                    self.preview_cache_keys[key] = settings.cache_key()
                    recolored += 1
        self.detail_cache.clear()
        self.detail_cache_keys.clear()
        self.detail_cache_side = 0
        if self.sequences:
            if recolored == 0:
                self.build_preview_cache()
            self.draw_frame(self.frame_index.get())

    def apply_br_display_settings(self) -> None:
        try:
            settings = self.current_br_display_settings()
        except ValueError as exc:
            messagebox.showerror("Invalid Br display settings", str(exc))
            return
        name = self.selected_br_sequence_name()
        if name is None:
            return
        self.layer_settings[name] = settings
        recolored = 0
        for sequence in self.sequences:
            if sequence.name != name:
                continue
            for frame in sequence.frames:
                key = str(frame.path)
                data = self.scalar_cache.get(key)
                if data is not None:
                    self.preview_cache[key] = colorize_preview_data(data, sequence.name, settings)
                    self.preview_cache_keys[key] = settings.cache_key()
                    recolored += 1
        self.detail_cache.clear()
        self.detail_cache_keys.clear()
        self.detail_cache_side = 0
        if self.sequences:
            if recolored == 0:
                self.build_preview_cache()
            self.draw_frame(self.frame_index.get())

    def apply_all_loaded_display_settings(self) -> None:
        try:
            settings = self.current_display_settings()
        except ValueError as exc:
            messagebox.showerror("Invalid display settings", str(exc))
            return
        target_sequences = [sequence for sequence in self.sequences if not is_br_channel(sequence.name)]
        for sequence in target_sequences:
            self.layer_settings.setdefault(sequence.name, default_display_settings(sequence.name))
            self.layer_settings[sequence.name] = settings
        for sequence in target_sequences:
            for frame in sequence.frames:
                key = str(frame.path)
                data = self.scalar_cache.get(key)
                if data is not None:
                    self.preview_cache[key] = colorize_preview_data(
                        data,
                        sequence.name,
                        settings,
                    )
                    self.preview_cache_keys[key] = settings.cache_key()
        self.detail_cache.clear()
        self.detail_cache_keys.clear()
        self.detail_cache_side = 0
        if self.sequences:
            self.draw_frame(self.frame_index.get())

    def load_selected(self) -> None:
        event = self.event_path()
        selected_sources = [
            self.source_options[name]
            for name in self.channel_order
            if self.channel_vars[name].get()
        ]
        if not event.exists() or not selected_sources:
            messagebox.showinfo("Nothing to load", "Please choose an event and at least one channel.")
            return
        self.status.set("Scanning FITS headers...")
        self.root.update_idletasks()
        resolved_event = event.resolve()
        if self.cached_event_path != resolved_event:
            self.clear_event_cache_with_warning()
            self.cached_event_path = resolved_event
        self.detail_cache.clear()
        self.detail_cache_side = 0
        self.view_state = None
        self.drawn_panels = []
        self.annotation_artists = []
        self.annotation_draft_artists = []
        self.cancel_annotation_draft()
        load_fits_image.cache_clear()
        sequences = [scan_source(source) for source in selected_sources]
        sequences = [sequence for sequence in sequences if sequence.frames]
        if not sequences:
            messagebox.showerror("Load failed", "No FITS images were found in the selected channels.")
            return

        self.sequences = sequences
        for sequence in self.sequences:
            self.layer_settings.setdefault(sequence.name, default_display_settings(sequence.name))
        self.update_display_layer_options()
        self.update_timeline_controls(index=0, reset_export=True)
        self.set_slider_without_callback(0)
        skipped = self.build_preview_cache()
        if not self.sequences or not self.timeline:
            messagebox.showerror("Load failed", "All selected FITS images failed to load.")
            return
        self.update_timeline_controls(index=0, reset_export=True)
        self.set_slider_without_callback(0)
        self.draw_frame(0)
        suffix = f" Skipped {skipped} bad FITS file(s)." if skipped else ""
        self.status.set(f"Loaded {len(self.sequences)} channels, {len(self.timeline)} timeline frames.{suffix}")

    def update_timeline_controls(self, index: int | None = None, reset_export: bool = False) -> None:
        if not self.sequences:
            self.timeline = []
            self.slider.configure(from_=0, to=0)
            self.start_spin.configure(from_=0, to=0)
            self.end_spin.configure(from_=0, to=0)
            self.frame_index.set(0)
            self.start_index.set(0)
            self.end_index.set(0)
            return
        self.timeline = max(self.sequences, key=lambda sequence: len(sequence.frames)).frames
        last = max(0, len(self.timeline) - 1)
        self.slider.configure(from_=0, to=last)
        self.start_spin.configure(from_=0, to=last)
        self.end_spin.configure(from_=0, to=last)
        next_index = self.frame_index.get() if index is None else index
        next_index = max(0, min(next_index, last))
        self.frame_index.set(next_index)
        if reset_export:
            self.start_index.set(0)
            self.end_index.set(last)

    def drop_failed_frames(self, failed_paths: set[str]) -> int:
        if not failed_paths:
            return 0
        kept_sequences: list[ChannelSequence] = []
        removed = 0
        for sequence in self.sequences:
            frames = [frame for frame in sequence.frames if str(frame.path) not in failed_paths]
            removed += len(sequence.frames) - len(frames)
            if frames:
                kept_sequences.append(ChannelSequence(sequence.name, frames))
        self.sequences = kept_sequences
        self.update_display_layer_options()
        self.update_timeline_controls()
        return removed

    def build_preview_cache(self) -> int:
        paths: list[tuple[Path, str]] = []
        seen: set[str] = set()
        for sequence in self.sequences:
            for frame in sequence.frames:
                key = str(frame.path)
                if key not in seen:
                    seen.add(key)
                    paths.append((frame.path, sequence.name))

        total = len(paths)
        max_side = max(100, int(self.preview_max_side.get()))
        if self.preview_cache_side not in (0, max_side):
            self.preview_cache.clear()
            self.preview_cache_keys.clear()
            self.scalar_cache.clear()
        self.preview_cache_side = max_side
        paths = [
            (path, sequence_name)
            for path, sequence_name in paths
            if str(path) not in self.preview_cache or str(path) not in self.scalar_cache
        ]
        if not paths:
            self.status.set("Reusing cached previews for selected channels.")
            self.load_progress["value"] = 0
            self.root.update_idletasks()
            return 0
        total = len(paths)
        self.load_progress["maximum"] = total
        self.load_progress["value"] = 0
        use_disk_cache = self.use_disk_cache.get()
        workers = max(1, min(total, int(self.cache_workers.get()), 32))
        executor_cls = ProcessPoolExecutor if self.use_process_pool.get() and workers > 1 else ThreadPoolExecutor
        pool_name = "processes" if executor_cls is ProcessPoolExecutor else "threads"
        failed_paths: set[str] = set()
        failed_messages: list[str] = []
        with executor_cls(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    build_preview_worker,
                    str(path),
                    sequence_name,
                    max_side,
                    use_disk_cache,
                    self.settings_for_sequence(sequence_name),
                ): (
                    path,
                    sequence_name,
                )
                for path, sequence_name in paths
            }
            for number, future in enumerate(as_completed(futures), start=1):
                path, _sequence_name = futures[future]
                try:
                    path_text, cache_path, scalar_path = future.result()
                except Exception as exc:
                    failed_paths.add(str(path))
                    failed_messages.append(f"{path.name}: {exc}")
                    self.load_progress["value"] = number
                    self.status.set(f"Skipped bad FITS {len(failed_paths)}: {path.name}")
                    self.root.update_idletasks()
                    continue
                self.preview_cache[path_text] = np.asarray(Image.open(cache_path).convert("RGB"))
                self.preview_cache_keys[path_text] = self.settings_for_sequence(_sequence_name).cache_key()
                self.scalar_cache[path_text] = np.load(scalar_path)
                if not use_disk_cache:
                    Path(cache_path).unlink(missing_ok=True)
                    Path(scalar_path).unlink(missing_ok=True)
                self.load_progress["value"] = number
                self.status.set(f"Loading {number}/{total} with {workers} {pool_name}: {path.name}")
                self.root.update_idletasks()
        self.load_progress["value"] = 0
        removed = self.drop_failed_frames(failed_paths)
        if removed:
            details = "; ".join(failed_messages[:3])
            more = "" if len(failed_messages) <= 3 else f"; +{len(failed_messages) - 3} more"
            self.status.set(f"Skipped {removed} bad FITS file(s): {details}{more}")
        load_fits_image.cache_clear()
        return removed

    def on_slider(self, value: str) -> None:
        if self.ignore_slider_callback:
            return
        if not self.timeline:
            return
        index = int(round(float(value)))
        if index != self.frame_index.get():
            self.frame_index.set(index)
        self.pending_slider_index = index
        if self.pending_slider_redraw is None:
            self.pending_slider_redraw = self.root.after(16, self.flush_slider_redraw)

    def flush_slider_redraw(self) -> None:
        self.pending_slider_redraw = None
        if self.pending_slider_index is None:
            return
        index = self.pending_slider_index
        self.pending_slider_index = None
        self.draw_frame(index, load_detail=True)

    def on_slider_release(self, _event=None) -> None:
        if self.pending_slider_redraw is not None:
            self.root.after_cancel(self.pending_slider_redraw)
            self.pending_slider_redraw = None
        index = self.pending_slider_index if self.pending_slider_index is not None else self.frame_index.get()
        self.pending_slider_index = None
        self.frame_index.set(index)
        self.draw_frame(index, load_detail=True)

    def set_slider_without_callback(self, index: int) -> None:
        self.ignore_slider_callback = True
        self.slider.set(index)
        self.ignore_slider_callback = False

    def step_frame(self, delta: int) -> None:
        if not self.timeline:
            return
        if self.pending_slider_redraw is not None:
            self.root.after_cancel(self.pending_slider_redraw)
            self.pending_slider_redraw = None
            self.pending_slider_index = None
        index = max(0, min(self.frame_index.get() + delta, len(self.timeline) - 1))
        self.frame_index.set(index)
        self.set_slider_without_callback(index)
        self.draw_frame(index)

    def current_zoom(self) -> float:
        if self.view_state is None:
            return 1.0
        x0, x1, y0, y1 = self.view_state
        return 1.0 / max(x1 - x0, y1 - y0)

    def reset_view(self) -> None:
        self.view_state = None
        self.drag_state = None
        self.draw_frame(self.frame_index.get())

    def apply_cache_px_change(self, _event=None) -> None:
        if not self.sequences:
            return
        self.preview_cache.clear()
        self.preview_cache_keys.clear()
        self.scalar_cache.clear()
        self.preview_cache_side = 0
        self.detail_cache.clear()
        self.detail_cache_keys.clear()
        self.detail_cache_side = 0
        self.status.set("Rebuilding preview cache with new Cache px...")
        self.root.update_idletasks()
        self.build_preview_cache()
        self.draw_frame(self.frame_index.get(), load_detail=False)

    def apply_zoom_px_change(self, _event=None) -> None:
        self.detail_cache.clear()
        self.detail_cache_keys.clear()
        self.detail_cache_side = 0
        if self.sequences:
            self.draw_frame(self.frame_index.get(), load_detail=True)

    def request_canvas_redraw(self, delay_ms: int = 16) -> None:
        if self.pending_canvas_redraw is None:
            self.pending_canvas_redraw = self.root.after(delay_ms, self.flush_canvas_redraw)

    def flush_canvas_redraw(self) -> None:
        self.pending_canvas_redraw = None
        self.canvas.draw_idle()

    def apply_view_to_existing_axes(self, throttled: bool = True) -> None:
        for panel in self.drawn_panels:
            height, width = panel.image_shape
            if self.view_state is None:
                panel.ax.set_xlim(0, width)
                panel.ax.set_ylim(0, height)
            else:
                x0, x1, y0, y1 = self.view_state
                panel.ax.set_xlim(x0 * width, x1 * width)
                panel.ax.set_ylim(y0 * height, y1 * height)
        if throttled:
            self.request_canvas_redraw()
        else:
            self.canvas.draw_idle()

    def on_shift_press(self, _event=None) -> None:
        self.shift_pressed = True

    def on_shift_release(self, _event=None) -> None:
        self.shift_pressed = False
        self.finish_annotation_stroke()

    def is_shift_event(self, event) -> bool:
        if self.shift_pressed or event.key == "shift":
            return True
        gui_event = getattr(event, "guiEvent", None)
        state = getattr(gui_event, "state", 0)
        return bool(state & 0x0001)

    def is_left_button_drag_event(self, event) -> bool:
        buttons = getattr(event, "buttons", None)
        if buttons is not None:
            return MouseButton.LEFT in buttons
        gui_event = getattr(event, "guiEvent", None)
        state = getattr(gui_event, "state", 0)
        if state:
            return bool(state & 0x0100)
        return event.button in (1, MouseButton.LEFT)

    def panel_for_axes(self, ax) -> DrawnPanel | None:
        for panel in self.drawn_panels:
            if panel.ax is ax:
                return panel
        return None

    def current_cache_key(self, sequence_name: str) -> str:
        return self.settings_for_sequence(sequence_name).cache_key()

    def cached_image_for_frame(self, frame: FrameInfo, sequence_name: str, prefer_detail: bool = True) -> np.ndarray | None:
        key = str(frame.path)
        settings_key = self.current_cache_key(sequence_name)
        if (
            prefer_detail
            and self.view_state is not None
            and key in self.detail_cache
            and self.detail_cache_keys.get(key) == settings_key
        ):
            return self.detail_cache[key]
        data = self.scalar_cache.get(key)
        if data is not None:
            if key not in self.preview_cache or self.preview_cache_keys.get(key) != settings_key:
                self.preview_cache[key] = colorize_preview_data(
                    data,
                    sequence_name,
                    self.settings_for_sequence(sequence_name),
                )
                self.preview_cache_keys[key] = settings_key
            return self.preview_cache[key]
        return self.preview_cache.get(key)

    def remove_annotation_artists(self, artists: list[object]) -> None:
        while artists:
            artist = artists.pop()
            try:
                artist.remove()
            except Exception:
                pass

    def annotation_line_width(self) -> float:
        try:
            return max(0.5, min(8.0, float(self.annotation_width.get())))
        except Exception:
            return 1.6

    def choose_annotation_color(self) -> None:
        _rgb, color = colorchooser.askcolor(color=self.annotation_color.get(), title="Annotation color")
        if color:
            self.annotation_color.set(color)
            self.annotation_color_preview.configure(bg=color)
            self.redraw_annotations()

    def redraw_annotations(self, draw: bool = True) -> None:
        self.remove_annotation_artists(self.annotation_artists)
        if not self.annotations_visible.get():
            self.remove_annotation_artists(self.annotation_draft_artists)
        if self.annotations_visible.get():
            color = self.annotation_color.get()
            width_px = self.annotation_line_width()
            for panel in self.drawn_panels:
                height, width = panel.image_shape
                for stroke in self.annotation_lines:
                    artist = panel.ax.plot(
                        [point[0] * width for point in stroke.points],
                        [point[1] * height for point in stroke.points],
                        color=color,
                        linewidth=width_px,
                        alpha=0.95,
                        zorder=30,
                    )[0]
                    artist.set_path_effects([path_effects.withStroke(linewidth=width_px + 1.6, foreground=(0, 0, 0, 0.45))])
                    self.annotation_artists.append(artist)
        if draw:
            self.canvas.draw_idle()

    def clear_annotations(self) -> None:
        self.annotation_lines.clear()
        self.cancel_annotation_draft()
        self.remove_annotation_artists(self.annotation_artists)
        self.canvas.draw_idle()

    def undo_annotation(self, _event=None) -> str:
        if self.annotation_lines:
            self.annotation_lines.pop()
            self.redraw_annotations()
        return "break"

    def cancel_annotation_draft(self) -> None:
        self.annotation_points = None
        self.annotation_source_ax = None
        self.annotation_start_canvas_point = None
        self.annotation_last_canvas_point = None
        self.annotation_path_length_px = 0.0
        self.remove_annotation_artists(self.annotation_draft_artists)

    def finish_annotation_stroke(self) -> None:
        if self.annotation_points is None:
            return
        points = tuple(self.annotation_points)
        path_length_px = self.annotation_path_length_px
        self.cancel_annotation_draft()
        if len(points) >= 2 and path_length_px >= 3.0:
            self.annotation_lines.append(AnnotationStroke(points))
            self.redraw_annotations()
        else:
            self.canvas.draw_idle()

    def draw_annotation_draft(self, stroke: AnnotationStroke) -> None:
        if not self.annotations_visible.get():
            self.remove_annotation_artists(self.annotation_draft_artists)
            return
        width_px = self.annotation_line_width()
        if len(self.annotation_draft_artists) != len(self.drawn_panels):
            self.remove_annotation_artists(self.annotation_draft_artists)
            for panel in self.drawn_panels:
                height, width = panel.image_shape
                artist = panel.ax.plot(
                    [point[0] * width for point in stroke.points],
                    [point[1] * height for point in stroke.points],
                    color=self.annotation_color.get(),
                    linewidth=width_px,
                    linestyle="--",
                    alpha=0.85,
                    zorder=31,
                )[0]
                artist.set_path_effects(
                    [path_effects.withStroke(linewidth=width_px + 1.5, foreground=(0, 0, 0, 0.55))]
                )
                self.annotation_draft_artists.append(artist)
        else:
            for panel, artist in zip(self.drawn_panels, self.annotation_draft_artists):
                height, width = panel.image_shape
                artist.set_data(
                    [point[0] * width for point in stroke.points],
                    [point[1] * height for point in stroke.points],
                )
        self.request_canvas_redraw()

    def normalized_point_from_event(self, event, source_ax=None) -> tuple[float, float] | None:
        ax = source_ax or event.inaxes
        if ax is None:
            return None
        panel = self.panel_for_axes(ax)
        if panel is None:
            return None
        height, width = panel.image_shape
        if width <= 0 or height <= 0:
            return None

        if event.inaxes is not ax:
            return None
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None or not np.isfinite(xdata) or not np.isfinite(ydata):
            return None
        if not (0.0 <= xdata <= width and 0.0 <= ydata <= height):
            return None
        x = min(1.0, max(0.0, xdata / width))
        y = min(1.0, max(0.0, ydata / height))
        return x, y

    def append_annotation_point(self, event, force: bool = False) -> bool:
        if self.annotation_points is None or self.annotation_source_ax is None:
            return False
        point = self.normalized_point_from_event(event, self.annotation_source_ax)
        if point is None:
            return False

        canvas_point = (float(event.x), float(event.y))
        if self.annotation_last_canvas_point is not None:
            distance = math.hypot(
                canvas_point[0] - self.annotation_last_canvas_point[0],
                canvas_point[1] - self.annotation_last_canvas_point[1],
            )
            if not force and distance < 1.5:
                return False
            self.annotation_path_length_px += distance
        if not self.annotation_points or point != self.annotation_points[-1]:
            self.annotation_points.append(point)
        self.annotation_last_canvas_point = canvas_point
        return True

    def set_annotation_line_endpoint(self, event) -> bool:
        if self.annotation_points is None or self.annotation_source_ax is None:
            return False
        point = self.normalized_point_from_event(event, self.annotation_source_ax)
        if point is None:
            return False
        canvas_point = (float(event.x), float(event.y))
        start_canvas_point = self.annotation_start_canvas_point or canvas_point
        self.annotation_path_length_px = math.hypot(
            canvas_point[0] - start_canvas_point[0],
            canvas_point[1] - start_canvas_point[1],
        )
        self.annotation_points[1:] = [point]
        self.annotation_last_canvas_point = canvas_point
        return True

    def capture_annotation_point(self, event, force: bool = False) -> bool:
        if self.annotation_active_mode == "Line":
            return self.set_annotation_line_endpoint(event)
        return self.append_annotation_point(event, force=force)

    def clamp_view(self, view: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
        x0, x1, y0, y1 = view
        width = min(1.0, max(0.02, x1 - x0))
        height = min(1.0, max(0.02, y1 - y0))
        x0 = min(max(0.0, x0), 1.0 - width)
        y0 = min(max(0.0, y0), 1.0 - height)
        x1 = x0 + width
        y1 = y0 + height
        if width >= 0.995 and height >= 0.995:
            return None
        return x0, x1, y0, y1

    def on_scroll(self, event) -> None:
        if not self.timeline or event.inaxes is None:
            return
        panel = self.panel_for_axes(event.inaxes)
        if panel is None:
            return
        height, width = panel.image_shape
        current = self.view_state or (0.0, 1.0, 0.0, 1.0)
        x0, x1, y0, y1 = current
        if event.xdata is None or event.ydata is None:
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
        else:
            cx = min(1.0, max(0.0, event.xdata / width))
            cy = min(1.0, max(0.0, event.ydata / height))

        factor = 0.72 if event.button == "up" else 1.38
        new_view = (
            cx - (cx - x0) * factor,
            cx + (x1 - cx) * factor,
            cy - (cy - y0) * factor,
            cy + (y1 - cy) * factor,
        )
        self.view_state = self.clamp_view(new_view)
        self.draw_frame(self.frame_index.get())

    def on_mouse_press(self, event) -> None:
        if event.dblclick:
            self.reset_view()
            return
        if event.button != 1 or event.inaxes is None:
            return
        if self.is_shift_event(event):
            point = self.normalized_point_from_event(event)
            if point is not None:
                self.annotation_points = [point]
                self.annotation_source_ax = event.inaxes
                self.annotation_start_canvas_point = (float(event.x), float(event.y))
                self.annotation_last_canvas_point = self.annotation_start_canvas_point
                self.annotation_path_length_px = 0.0
                selected_mode = self.annotation_mode.get()
                self.annotation_active_mode = selected_mode if selected_mode in ANNOTATION_MODES else DEFAULT_ANNOTATION_MODE
                self.remove_annotation_artists(self.annotation_draft_artists)
            return
        panel = self.panel_for_axes(event.inaxes)
        if panel is None:
            return
        self.drag_state = (event.x, event.y, self.view_state or (0.0, 1.0, 0.0, 1.0), panel.ax)

    def on_mouse_release(self, event) -> None:
        if self.annotation_points is not None:
            if self.is_shift_event(event):
                self.capture_annotation_point(event, force=True)
            self.finish_annotation_stroke()
            return
        self.drag_state = None
        if self.timeline:
            self.apply_view_to_existing_axes(throttled=False)

    def on_mouse_motion(self, event) -> None:
        if self.annotation_points is not None:
            if not self.is_shift_event(event) or not self.is_left_button_drag_event(event):
                self.finish_annotation_stroke()
                return
            if self.capture_annotation_point(event):
                self.draw_annotation_draft(AnnotationStroke(tuple(self.annotation_points)))
            return
        if self.drag_state is None:
            return
        start_x, start_y, start_view, start_ax = self.drag_state
        if start_ax is None:
            return
        bbox = start_ax.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return
        x0, x1, y0, y1 = start_view
        view_width = x1 - x0
        view_height = y1 - y0
        dx = (event.x - start_x) / bbox.width * view_width
        dy = (event.y - start_y) / bbox.height * view_height
        self.view_state = self.clamp_view((x0 - dx, x1 - dx, y0 - dy, y1 - dy))
        self.apply_view_to_existing_axes()

    def ensure_detail_for_current_frame(self, index: int) -> None:
        if self.view_state is None or self.current_zoom() < 1.15:
            return
        detail_side = max(int(self.preview_max_side.get()), min(4096, int(self.detail_max_side.get())))
        if self.preview_cache_side >= detail_side:
            return
        if self.detail_cache_side != detail_side:
            self.detail_cache.clear()
            self.detail_cache_keys.clear()
            self.detail_cache_side = detail_side

        target = self.timeline[index]
        frames: list[tuple[FrameInfo, str]] = []
        for sequence in self.sequences:
            frame = nearest_frame(sequence, target, index)
            if str(frame.path) not in self.detail_cache:
                frames.append((frame, sequence.name))
        if not frames:
            return

        self.status.set(f"Loading zoom detail for current frame at {detail_side}px...")
        self.root.update_idletasks()
        for frame, sequence_name in frames:
            cache_key = self.current_cache_key(sequence_name)
            self.detail_cache[str(frame.path)] = load_or_make_preview_image(
                frame.path,
                sequence_name,
                detail_side,
                self.use_disk_cache.get(),
                self.settings_for_sequence(sequence_name),
            )
            self.detail_cache_keys[str(frame.path)] = cache_key
        load_fits_image.cache_clear()

    def update_frame_fast(self, index: int) -> bool:
        if not self.drawn_panels or len(self.drawn_panels) != len(self.sequences):
            return False
        target = self.timeline[index]
        labels = []
        for panel, sequence in zip(self.drawn_panels, self.sequences):
            if panel.sequence_name != sequence.name:
                return False
            frame = nearest_frame(sequence, target, index)
            image = self.cached_image_for_frame(frame, sequence.name)
            if image is None:
                return False
            height, width = image.shape[:2]
            panel.image_artist.set_data(image)
            panel.image_artist.set_extent((0, width, 0, height))
            panel.text_artist.set_text(overlay_text(frame, sequence.name))
            object.__setattr__(panel, "frame", frame)
            object.__setattr__(panel, "image_shape", (height, width))
            labels.append(f"{channel_display_name(sequence.name)}: {frame.label}")

            if self.view_state is None:
                panel.ax.set_xlim(0, width)
                panel.ax.set_ylim(0, height)
            else:
                x0, x1, y0, y1 = self.view_state
                panel.ax.set_xlim(x0 * width, x1 * width)
                panel.ax.set_ylim(y0 * height, y1 * height)

        self.info_list.delete(0, "end")
        for label in labels:
            self.info_list.insert("end", label)
        self.redraw_annotations(draw=False)
        self.canvas.draw_idle()
        self.status.set(f"Frame {index + 1}/{len(self.timeline)}  {target.label}")
        return True

    def draw_frame(self, index: int, load_detail: bool = True) -> None:
        if not self.sequences or not self.timeline:
            return
        index = max(0, min(index, len(self.timeline) - 1))
        target = self.timeline[index]
        if load_detail:
            self.ensure_detail_for_current_frame(index)
        if self.update_frame_fast(index):
            return

        self.info_list.delete(0, "end")
        settings_by_sequence = {sequence.name: self.settings_for_sequence(sequence.name) for sequence in self.sequences}
        active_detail_cache = self.detail_cache if self.view_state is not None else None
        labels, self.drawn_panels = draw_sequence_grid(
            self.figure,
            self.sequences,
            self.timeline,
            index,
            self.preview_cache,
            active_detail_cache,
            self.view_state,
            self.display_settings,
            settings_by_sequence,
        )
        for label in labels:
            self.info_list.insert("end", label)
        self.redraw_annotations(draw=False)
        self.canvas.draw_idle()
        self.status.set(f"Frame {index + 1}/{len(self.timeline)}  {target.label}")

    def toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.play_next()

    def play_next(self) -> None:
        if not self.playing or not self.timeline:
            return
        index = self.frame_index.get() + 1
        if index >= len(self.timeline):
            index = 0
        self.frame_index.set(index)
        self.set_slider_without_callback(index)
        self.draw_frame(index, load_detail=False)
        delay_ms = max(20, int(1000 / max(1, self.fps.get())))
        self.root.after(delay_ms, self.play_next)

    def export_video(self) -> None:
        if not self.sequences or not self.timeline:
            messagebox.showinfo("Nothing to export", "Load an event first.")
            return
        start = max(0, min(self.start_index.get(), len(self.timeline) - 1))
        end = max(0, min(self.end_index.get(), len(self.timeline) - 1))
        if end < start:
            start, end = end, start

        output = filedialog.asksaveasfilename(
            title="Export video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("GIF animation", "*.gif")],
            initialfile=f"{self.event_name.get()}_{start:03d}_{end:03d}.mp4",
        )
        if not output:
            return

        settings_snapshot = {sequence.name: self.settings_for_sequence(sequence.name) for sequence in self.sequences}
        annotations_snapshot = list(self.annotation_lines) if self.annotations_visible.get() else []
        annotation_color = self.annotation_color.get()
        annotation_width = self.annotation_line_width()
        preview_side = max(100, int(self.preview_max_side.get()))
        view_state_snapshot = self.view_state

        self.status.set("Exporting video...")
        thread = threading.Thread(
            target=self._export_worker,
            args=(
                Path(output),
                start,
                end,
                self.fps.get(),
                settings_snapshot,
                annotations_snapshot,
                annotation_color,
                annotation_width,
                preview_side,
                view_state_snapshot,
            ),
            daemon=True,
        )
        thread.start()

    def export_figure_size(self, index: int, preview_side: int) -> tuple[float, float]:
        """Size the export canvas to the image grid so no outer side bars are added."""
        rows, cols = grid_shape(len(self.sequences))
        target = self.timeline[index]
        aspects: list[float] = []
        for sequence in self.sequences:
            frame = nearest_frame(sequence, target, index)
            data = self.scalar_cache.get(str(frame.path))
            if data is None:
                data = orient_image_for_display(read_preview_data(frame.path, preview_side), sequence.name)
            height, width = data.shape[:2]
            aspects.append(width / height if height else 1.0)
        cell_aspect = float(np.median(aspects)) if aspects else 1.0
        content_aspect = cols * cell_aspect / rows
        figure_width = 5.2 * cols
        return figure_width, figure_width / content_aspect

    def _export_worker(
        self,
        output: Path,
        start: int,
        end: int,
        fps: int,
        settings_by_sequence: dict[str, DisplaySettings],
        annotations: list[AnnotationStroke],
        annotation_color: str,
        annotation_width: float,
        preview_side: int,
        view_state: tuple[float, float, float, float] | None,
    ) -> None:
        try:
            frames = range(start, end + 1)
            figure_size = self.export_figure_size(start, preview_side)
            fig = Figure(figsize=figure_size, dpi=120, facecolor="black")
            canvas = FigureCanvasAgg(fig)
            writer_kwargs = {"fps": max(1, fps)}
            if output.suffix.lower() == ".mp4":
                writer_kwargs.update(
                    {
                        "codec": "libx264",
                        "macro_block_size": 1,
                        "output_params": ["-preset", "veryfast", "-crf", "18"],
                    }
                )
            with imageio.get_writer(output, **writer_kwargs) as writer:
                for index in frames:
                    writer.append_data(
                        self.render_export_frame(
                            index,
                            settings_by_sequence,
                            annotations,
                            annotation_color,
                            annotation_width,
                            preview_side,
                            view_state,
                            fig,
                            canvas,
                        )
                    )
                    self.root.after(0, self.status.set, f"Exporting frame {index + 1}/{end + 1}...")
            self.root.after(0, self.status.set, f"Exported {output}")
            self.root.after(0, messagebox.showinfo, "Export complete", f"Saved:\n{output}")
        except Exception as exc:
            self.root.after(0, self.status.set, "Export failed.")
            self.root.after(0, messagebox.showerror, "Export failed", str(exc))

    def export_image_for_frame(
        self,
        frame: FrameInfo,
        sequence_name: str,
        settings: DisplaySettings,
        preview_side: int,
    ) -> np.ndarray:
        key = str(frame.path)
        settings_key = settings.cache_key()
        if key in self.preview_cache and self.preview_cache_keys.get(key) == settings_key:
            return self.preview_cache[key]
        data = self.scalar_cache.get(key)
        if data is None:
            data = orient_image_for_display(read_preview_data(frame.path, preview_side), sequence_name)
        rgb = colorize_preview_data(
            data,
            sequence_name,
            settings,
        )
        self.preview_cache[key] = rgb
        self.preview_cache_keys[key] = settings_key
        return rgb

    def render_export_frame(
        self,
        index: int,
        settings_by_sequence: dict[str, DisplaySettings],
        annotations: list[AnnotationStroke],
        annotation_color: str,
        annotation_width: float,
        preview_side: int,
        view_state: tuple[float, float, float, float] | None = None,
        fig: Figure | None = None,
        canvas: FigureCanvasAgg | None = None,
    ) -> np.ndarray:
        if fig is None:
            fig = Figure(figsize=self.export_figure_size(index, preview_side), dpi=120, facecolor="black")
        if canvas is None:
            canvas = FigureCanvasAgg(fig)
        target = self.timeline[index]
        export_cache: dict[str, np.ndarray] = {}
        for sequence in self.sequences:
            frame = nearest_frame(sequence, target, index)
            settings = settings_by_sequence.get(sequence.name, DisplaySettings())
            export_cache[str(frame.path)] = self.export_image_for_frame(frame, sequence.name, settings, preview_side)
        draw_sequence_grid(
            fig,
            self.sequences,
            self.timeline,
            index,
            export_cache,
            view_state=view_state,
            settings=DisplaySettings(),
            settings_by_sequence=settings_by_sequence,
            annotations=annotations,
            annotation_color=annotation_color,
            annotation_width=annotation_width,
        )
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return rgba[:, :, :3].copy()


def main() -> None:
    root = Tk()
    app = FitsViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
