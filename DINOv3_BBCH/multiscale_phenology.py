import ast
import functools
import itertools
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

try:
    import torchvision.models as tvm
    import torchvision.transforms as T
except Exception:  # pragma: no cover - imported in notebook/runtime environments
    tvm = None
    T = None

try:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel
except Exception:  # pragma: no cover - optional DINOv3 dependency
    AutoConfig = None
    AutoImageProcessor = None
    AutoModel = None

try:
    from huggingface_hub import get_token as get_huggingface_token
except Exception:  # pragma: no cover - installed with transformers in DINO runs
    get_huggingface_token = None


DATE_RE = re.compile(r"(\d{4})[_-](\d{2})[_-](\d{2})|(\d{4})(\d{2})(\d{2})")
CURRENT_DATA_RE = re.compile(
    r"(?P<station>\d{2}[_.,]\d{2})-(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
)
CURRENT_DATA_TIMESTAMP_RE = re.compile(
    r"(?P<station>\d{2}[_.,]\d{2})-(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
    r"[-_](?P<hour>[01]\d|2[0-3])[_:-](?P<minute>[0-5]\d)"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DAYS_SINCE_PLANTING_FEATURE = "days_since_planting"
DEFAULT_TEMPORAL_FEATURE_COLUMNS = (DAYS_SINCE_PLANTING_FEATURE,)
WEATHER_VALUE_FEATURE_COLUMNS = (
    "weather_tavg_norm",
    "weather_tmin_norm",
    "weather_tmax_norm",
    "weather_prcp_norm",
    "weather_gdd_norm",
    "weather_gdd_cum_norm",
)
WEATHER_MISSING_FEATURE_COLUMNS = (
    "weather_tavg_missing",
    "weather_tmin_missing",
    "weather_tmax_missing",
    "weather_prcp_missing",
)
WEATHER_TEMPORAL_FEATURE_COLUMNS = (
    *WEATHER_VALUE_FEATURE_COLUMNS,
    *WEATHER_MISSING_FEATURE_COLUMNS,
)
WEATHER_FEATURE_SETS = {
    "cumulative": ("weather_gdd_cum_norm",),
    "daily_cumulative": ("weather_gdd_norm", "weather_gdd_cum_norm"),
    "full": WEATHER_TEMPORAL_FEATURE_COLUMNS,
}
LOCATION_FEATURE_COLUMNS = (
    "location_latitude_norm",
    "location_longitude_norm",
    "location_elevation_norm",
)
LOCATION_LATITUDE_CENTER = 39.0
LOCATION_LATITUDE_SCALE = 3.0
LOCATION_LONGITUDE_CENTER = 35.0
LOCATION_LONGITUDE_SCALE = 5.0
LOCATION_ELEVATION_CENTER = 500.0
LOCATION_ELEVATION_SCALE = 500.0
TEMPORAL_FEATURE_DIM = len(DEFAULT_TEMPORAL_FEATURE_COLUMNS)
DINO_DEFAULT_BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_BACKBONE_ALIASES = {
    "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3-vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
DAYS_SINCE_PLANTING_SCALE = 365.0
WEATHER_TEMP_SCALE = 40.0
WEATHER_PRCP_SCALE = 50.0
WEATHER_GDD_SCALE = 30.0
WEATHER_CUM_GDD_SCALE = 2500.0

BASE_CLASSES = [
    "OffSeason",
    "BBCH0",
    "BBCH1",
    "BBCH2",
    "BBCH3",
    "BBCH5",
    "BBCH6_7",
    "BBCH8",
]

# The BBCH table contains eight milestones that define seven ordered intervals.
# BBCH0 starts at sowing, so no leading boundary is skipped.
PHENOLOGY_BOUNDARY_OFFSET = 0

STAGE_COLUMNS = [
    "1-Sowing",
    "2 - Emergence",
    "3 - Tillering",
    "4 - Stem Elongation",
    "5 - Heading",
    "6 - Flowering",
    "7 - Maturity",
    "8 - Harvest",
]

STAGE_COLUMN_ALIASES = {
    "1-Sowing": ["1-Sowing", "1 - Sowing"],
    "2 - Emergence": ["2 - Emergence", "2-Emergence"],
    "3 - Tillering": ["3 - Tillering", "3-Tillering"],
    "4 - Stem Elongation": ["4 - Stem Elongation", "4-Stem Elongation"],
    "5 - Heading": ["5 - Heading", "5-Heading"],
    "6 - Flowering": ["6 - Flowering", "6-Flowering"],
    "7 - Maturity": ["7 - Maturity", "7-Maturity"],
    "8 - Harvest": ["8 - Harvest", "8-Harvest"],
}

STATION_COORDINATES = {
    "01": (36.983, 35.317), # Adana
    "02": (37.76, 38.2761), # Adıyaman
    "06": (39.9288, 32.8547), # Ankara
    "11": (40.150, 29.983), # Bilecik
    "26": (39.7713, 30.5183), # Eskişehir
    "27": (37.0658, 37.3780) # Gaziantep
}

STATION_ELEVATIONS = { # average, by meters
    "01": 23, # Adana
    "02": 669, # Adıyaman
    "06": 938, # Ankara
    "11": 513, # Bilecik
    "26": 792, # Eskişehir
    "27": 838 # Gaziantep
}


def _station_family(value: object) -> str:
    try:
        text = f"{float(value):05.2f}"
    except (TypeError, ValueError):
        text = str(value).strip()
    text = text.replace("_", ".").replace(",", ".")
    return text.split(".", 1)[0]


def _station_coordinates(value: object) -> Optional[Tuple[float, float]]:
    return STATION_COORDINATES.get(_station_family(value))


def station_location_features(value: object, strict: bool = False) -> torch.Tensor:
    family = _station_family(value)
    coordinates = STATION_COORDINATES.get(family)
    elevation = STATION_ELEVATIONS.get(family)
    if coordinates is None or elevation is None:
        if strict:
            raise ValueError(
                f"No complete latitude/longitude/elevation metadata configured for station family {family!r}"
            )
        return torch.zeros(len(LOCATION_FEATURE_COLUMNS), dtype=torch.float32)
    latitude, longitude = coordinates
    return torch.tensor(
        [
            (float(latitude) - LOCATION_LATITUDE_CENTER) / LOCATION_LATITUDE_SCALE,
            (float(longitude) - LOCATION_LONGITUDE_CENTER) / LOCATION_LONGITUDE_SCALE,
            (float(elevation) - LOCATION_ELEVATION_CENTER) / LOCATION_ELEVATION_SCALE,
        ],
        dtype=torch.float32,
    )


def add_location_metadata(daily_df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    out = daily_df.copy()
    families = out["station_code"].map(_station_family)
    missing = sorted(
        {
            family
            for family in families.unique()
            if family not in STATION_COORDINATES or family not in STATION_ELEVATIONS
        }
    )
    if missing and strict:
        raise ValueError(
            "Missing station coordinates/elevations for station families: " + ", ".join(missing)
        )

    out["location_latitude"] = families.map(
        lambda family: STATION_COORDINATES.get(family, (np.nan, np.nan))[0]
    )
    out["location_longitude"] = families.map(
        lambda family: STATION_COORDINATES.get(family, (np.nan, np.nan))[1]
    )
    out["location_elevation_m"] = families.map(STATION_ELEVATIONS)
    out["location_latitude_norm"] = (
        out["location_latitude"] - LOCATION_LATITUDE_CENTER
    ) / LOCATION_LATITUDE_SCALE
    out["location_longitude_norm"] = (
        out["location_longitude"] - LOCATION_LONGITUDE_CENTER
    ) / LOCATION_LONGITUDE_SCALE
    out["location_elevation_norm"] = (
        out["location_elevation_m"] - LOCATION_ELEVATION_CENTER
    ) / LOCATION_ELEVATION_SCALE
    for column in LOCATION_FEATURE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype(float)
    return out


def _normalize_col(name: object) -> str:
    import unicodedata

    text = str(name).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text if ch.isalnum())


def _extract_date(path_or_name: str) -> Optional[pd.Timestamp]:
    name = os.path.basename(path_or_name)

    current_match = CURRENT_DATA_RE.search(name)
    if current_match:
        y = current_match.group("year")
        m = current_match.group("month")
        d = current_match.group("day")
        try:
            return pd.Timestamp(f"{y}-{m}-{d}")
        except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
            return None

    match = DATE_RE.search(name)
    if not match:
        return None
    if match.group(1):
        y, m, d = match.group(1), match.group(2), match.group(3)
    else:
        y, m, d = match.group(4), match.group(5), match.group(6)
    try:
        return pd.Timestamp(f"{y}-{m}-{d}")
    except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
        return None


def _extract_capture_timestamp(path_or_name: str) -> Optional[pd.Timestamp]:
    match = CURRENT_DATA_TIMESTAMP_RE.search(os.path.basename(path_or_name))
    if not match:
        return None
    try:
        return pd.Timestamp(
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
            hour=int(match.group("hour")),
            minute=int(match.group("minute")),
        )
    except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
        return None


def _daily_path_sort_key(path: str) -> Tuple[bool, pd.Timestamp, str]:
    captured_at = _extract_capture_timestamp(path)
    return captured_at is None, captured_at or pd.Timestamp.max, path.lower()


def _select_daily_image_paths(candidates: Dict[str, List[str]]) -> Dict[str, str]:
    """Choose deterministic, time-aligned 1X/10X captures for a day."""
    available = {
        scale: sorted(set(paths), key=_daily_path_sort_key)
        for scale, paths in candidates.items()
        if paths
    }
    macro_paths = available.get("macro", [])
    micro_paths = available.get("micro", [])
    if not macro_paths or not micro_paths:
        return {scale: paths[0] for scale, paths in available.items()}

    timestamped_pairs = []
    for macro_path in macro_paths:
        macro_time = _extract_capture_timestamp(macro_path)
        if macro_time is None:
            continue
        for micro_path in micro_paths:
            micro_time = _extract_capture_timestamp(micro_path)
            if micro_time is None:
                continue
            timestamped_pairs.append(
                (
                    abs(macro_time - micro_time),
                    max(macro_time, micro_time),
                    macro_time,
                    micro_time,
                    macro_path.lower(),
                    micro_path.lower(),
                    macro_path,
                    micro_path,
                )
            )
    if timestamped_pairs:
        best_pair = min(timestamped_pairs)
        return {"macro": best_pair[-2], "micro": best_pair[-1]}
    return {"macro": macro_paths[0], "micro": micro_paths[0]}


def _camera_scale(path: str) -> Optional[str]:
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    filename = os.path.basename(path).lower()
    if any(part == "10x" for part in parts) or "-10x" in filename or "_10x" in filename:
        return "micro"
    if any(part == "1x" for part in parts) or "-1x" in filename or "_1x" in filename:
        return "macro"
    return None


def _camera_name(path: str) -> Optional[str]:
    for part in os.path.normpath(path).split(os.sep):
        if re.fullmatch(r"k\d+", part, flags=re.IGNORECASE):
            return part.upper()
    match = re.search(r"-(K\d+)-", os.path.basename(path), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _normalize_camera_value(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"K{int(value)}"
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL", "-"}:
        return None
    text = text.replace(" ", "")
    if text == "ALL":
        return "ALL"
    if text == "AUTO":
        return None
    match = re.fullmatch(r"K?(\d+)(?:\.0+)?", text)
    if match:
        return f"K{int(match.group(1))}"
    return text


def _record_camera_preference(record: Dict) -> Optional[str]:
    camera_keys = {"kamera", "camera", "camerano", "kamerano"}
    for key, value in record.items():
        if _normalize_col(key) in camera_keys:
            camera = _normalize_camera_value(value)
            if camera:
                return camera
    return None


def _resolve_camera_preference(preferred_camera: Optional[str], record_camera: Optional[str]) -> Optional[str]:
    if preferred_camera is None:
        return None
    requested = _normalize_camera_value(preferred_camera)
    if requested is None:
        return record_camera or "AUTO"
    if requested == "AUTO":
        return record_camera or "AUTO"
    return requested


def _station_folder_variants(station_raw: object) -> List[str]:
    try:
        numeric = f"{float(station_raw):05.2f}"
    except (TypeError, ValueError):
        numeric = str(station_raw).strip()
    return [
        numeric,
        numeric.replace(".", "_"),
        numeric.replace(".", ","),
        numeric.lstrip("0"),
    ]


def _map_excel_stage_columns(df: pd.DataFrame) -> pd.DataFrame:
    norm_to_actual = {_normalize_col(c): c for c in df.columns}
    rename = {}
    for canonical, aliases in STAGE_COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            key = _normalize_col(alias)
            if key in norm_to_actual:
                found = norm_to_actual[key]
                break
        if found is not None and found != canonical:
            rename[found] = canonical
    return df.rename(columns=rename)


def _read_phenology_table(label_path: str) -> pd.DataFrame:
    label_path = os.path.abspath(label_path)
    ext = os.path.splitext(label_path)[1].lower()
    if ext == ".csv":
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin1"):
            try:
                return pd.read_csv(label_path, sep=None, engine="python", encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    return pd.read_excel(label_path)


def _parse_stage_date_column(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    iso_mask = text.str.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", na=False)

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(text.loc[iso_mask], format="%Y-%m-%d", errors="coerce")
    if (~iso_mask).any():
        parsed.loc[~iso_mask] = pd.to_datetime(values.loc[~iso_mask], dayfirst=True, errors="coerce")
    return parsed


def _load_phenology_excel(excel_path: str) -> pd.DataFrame:
    df = _read_phenology_table(excel_path)
    df = _map_excel_stage_columns(df)
    for col in STAGE_COLUMNS:
        if col in df.columns:
            df[col] = _parse_stage_date_column(df[col])
        else:
            df[col] = pd.NaT
    return df


def _stage_boundaries(record: Dict) -> List[object]:
    return [record.get(col) for col in STAGE_COLUMNS]


def _ordered_stage_labels(class_to_idx: Dict[str, int]) -> List[str]:
    """Return active phenology classes in their configured ordinal order."""
    return [
        label
        for label, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
        if label != "OffSeason"
    ]


def _find_station_path(root_dir: str, station_raw: object) -> Tuple[Optional[str], Optional[str]]:
    variants = _station_folder_variants(station_raw)
    for variant in variants:
        candidate = os.path.join(root_dir, variant)
        if os.path.isdir(candidate):
            return candidate, variant

    compact_variants = {v.replace(".", "").replace("_", "").replace(",", "") for v in variants}
    if not os.path.isdir(root_dir):
        return None, None
    for folder in os.listdir(root_dir):
        compact = folder.replace(".", "").replace("_", "").replace(",", "")
        if compact in compact_variants:
            return os.path.join(root_dir, folder), folder
    return None, None


def _iter_images(path: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTS:
                yield os.path.join(dirpath, filename)


def _year_folder(root_dir: str, year: object) -> Optional[str]:
    try:
        year_text = str(int(float(year)))
    except (TypeError, ValueError):
        year_text = str(year).strip()
    candidate = os.path.join(root_dir, year_text)
    return candidate if os.path.isdir(candidate) else None


def _year_values(year_or_years: object) -> List[object]:
    if isinstance(year_or_years, range):
        return list(year_or_years)
    if isinstance(year_or_years, (list, tuple, set)):
        return list(year_or_years)
    return [year_or_years]


@functools.lru_cache(maxsize=64)
def _year_file_index(year_path: str) -> Dict[str, Tuple[str, ...]]:
    by_name: Dict[str, List[str]] = {}
    for path in _iter_images(year_path):
        by_name.setdefault(os.path.basename(path).lower(), []).append(path)
    return {name: tuple(sorted(paths)) for name, paths in by_name.items()}


def _local_status_path(path_from_csv: object, year_path: str) -> Optional[str]:
    if not isinstance(path_from_csv, str) or not path_from_csv:
        return None
    parts = re.split(r"[\\/]+", path_from_csv)
    for i, part in enumerate(parts):
        if re.fullmatch(r"K\d+", part, flags=re.IGNORECASE) and i + 2 < len(parts):
            candidate = os.path.join(year_path, *parts[i:])
            if os.path.isfile(candidate):
                return candidate
    candidate = os.path.join(year_path, os.path.basename(path_from_csv))
    if os.path.isfile(candidate):
        return candidate
    matches = _year_file_index(os.path.abspath(year_path)).get(os.path.basename(path_from_csv).lower(), ())
    return matches[0] if matches else None


def _iter_current_data_images(
    station_path: str,
    year: object,
    preferred_camera: Optional[str] = "AUTO",
    use_status_csv: bool = True,
) -> Iterable[str]:
    """
    Iterate the repository's current layout:

    data/<station>/<year>/K1|K2/1X|10X/<station>-YYYY_MM_DD-HH_MM-K*-*X.jpeg

    If a day_image_status CSV exists, valid rows from it are preferred because it
    already excludes corrupt files. Otherwise, the function falls back to scanning
    the year folder.
    """
    camera_preference = preferred_camera.upper() if preferred_camera else None
    for year_value in _year_values(year):
        year_path = _year_folder(station_path, year_value)
        if year_path is None:
            continue

        status_files = [
            os.path.join(year_path, f)
            for f in os.listdir(year_path)
            if f.lower().startswith("day_image_status") and f.lower().endswith(".csv")
        ]
        if use_status_csv and status_files:
            status_frames = [pd.read_csv(status_file) for status_file in status_files]
            status_df = pd.concat(status_frames, ignore_index=True)
            if camera_preference == "AUTO":
                valid_cameras = sorted(
                    str(camera).upper()
                    for camera in status_df.loc[status_df["status"].astype(str).str.lower() == "valid", "camera"].dropna().unique()
                )
                camera_filter = "K1" if "K1" in valid_cameras else valid_cameras[0] if valid_cameras else None
            elif camera_preference == "ALL":
                camera_filter = None
            else:
                camera_filter = camera_preference

            for status_file in status_files:
                one_status_df = pd.read_csv(status_file)
                for row in one_status_df.itertuples(index=False):
                    if str(getattr(row, "status", "")).lower() != "valid":
                        continue
                    camera = str(getattr(row, "camera", "")).upper()
                    if camera_filter and camera != camera_filter:
                        continue
                    local_path = _local_status_path(getattr(row, "valid_file", None), year_path)
                    if local_path is not None:
                        yield local_path
            continue

        if camera_preference == "AUTO":
            valid_cameras = sorted(
                folder.upper()
                for folder in os.listdir(year_path)
                if os.path.isdir(os.path.join(year_path, folder)) and re.fullmatch(r"k\d+", folder, flags=re.IGNORECASE)
            )
            camera_filter = "K1" if "K1" in valid_cameras else valid_cameras[0] if valid_cameras else None
        elif camera_preference == "ALL":
            camera_filter = None
        else:
            camera_filter = camera_preference

        for image_path in _iter_images(year_path):
            camera = _camera_name(image_path)
            if camera_filter and camera != camera_filter:
                continue
            yield image_path


def _soft_interval_label(
    date: pd.Timestamp,
    boundaries: Sequence[pd.Timestamp],
    class_to_idx: Dict[str, int],
    transition_days: int,
) -> Optional[np.ndarray]:
    target = np.zeros(len(class_to_idx), dtype=np.float32)
    active_start, harvest = boundaries[PHENOLOGY_BOUNDARY_OFFSET], boundaries[-1]
    stage_labels = _ordered_stage_labels(class_to_idx)
    if PHENOLOGY_BOUNDARY_OFFSET + len(stage_labels) >= len(boundaries):
        raise ValueError(
            "Not enough stage boundaries for configured classes: "
            f"offset={PHENOLOGY_BOUNDARY_OFFSET}, stages={len(stage_labels)}, boundaries={len(boundaries)}"
        )

    if date < active_start:
        if "OffSeason" in class_to_idx:
            target[class_to_idx["OffSeason"]] = 1.0
        else:
            target[class_to_idx["Dormancy"]] = 1.0
        return target
    if date >= harvest:
        if "OffSeason" in class_to_idx:
            target[class_to_idx["OffSeason"]] = 1.0
        else:
            target[class_to_idx["PostHarvest"]] = 1.0
        return target
    hard_idx = None
    for i, label in enumerate(stage_labels):
        left_boundary = PHENOLOGY_BOUNDARY_OFFSET + i
        if boundaries[left_boundary] <= date < boundaries[left_boundary + 1]:
            hard_idx = class_to_idx[label]
            break
    if hard_idx is None:
        return None

    target[hard_idx] = 1.0
    if transition_days <= 0:
        return target

    # Blend neighboring classes near biological transition boundaries.
    for boundary_i in range(1, len(stage_labels)):
        boundary = boundaries[PHENOLOGY_BOUNDARY_OFFSET + boundary_i]
        delta = abs((date - boundary).days)
        if delta > transition_days:
            continue
        left = class_to_idx[stage_labels[boundary_i - 1]]
        right = class_to_idx[stage_labels[boundary_i]]
        # Keep transition targets soft without creating a 50/50 argmax tie on
        # the exact boundary. The interval containing the date remains primary.
        blend = 0.4 * (1.0 - delta / max(transition_days, 1))
        target[:] = 0.0
        if date < boundary:
            target[left] = 1.0 - blend
            target[right] = blend
        else:
            target[left] = blend
            target[right] = 1.0 - blend
        return target

    return target


def _distance_to_interval(date: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> int:
    if start <= date <= end:
        return 0
    if date < start:
        return int((start - date).days)
    return int((date - end).days)


def _stream_path_columns(stream: str = "both") -> List[Tuple[str, str]]:
    if stream == "micro":
        return [("micro", "micro_path")]
    if stream == "macro":
        return [("macro", "macro_path")]
    return [("macro", "macro_path"), ("micro", "micro_path")]


def _display_path(path: object, base_dir: Optional[str] = None) -> str:
    if path is None or pd.isna(path):
        return ""
    text = str(path)
    if not text:
        return ""
    try:
        base = base_dir or os.getcwd()
        return os.path.relpath(text, base)
    except (OSError, ValueError):
        return text


def station_image_edge_summary(
    daily_df: pd.DataFrame,
    stream: str = "both",
    base_dir: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    if daily_df.empty:
        return pd.DataFrame(rows)
    for station_year, group in daily_df.sort_values(["station_year", "date"]).groupby("station_year", sort=True):
        group_id = group["group_id"].iloc[0] if "group_id" in group.columns and not group.empty else None
        for stream_name, path_col in _stream_path_columns(stream):
            if path_col not in group.columns:
                continue
            valid = group.loc[group[path_col].notna()].sort_values("date")
            if valid.empty:
                rows.append(
                    {
                        "station_year": station_year,
                        "group_id": group_id,
                        "stream": stream_name,
                        "image_count": 0,
                        "first_date": "",
                        "first_label": "",
                        "first_path": "",
                        "last_date": "",
                        "last_label": "",
                        "last_path": "",
                    }
                )
                continue
            first = valid.iloc[0]
            last = valid.iloc[-1]
            rows.append(
                {
                    "station_year": station_year,
                    "group_id": group_id,
                    "stream": stream_name,
                    "image_count": int(len(valid)),
                    "first_date": str(pd.Timestamp(first["date"]).date()),
                    "first_label": first.get("label", ""),
                    "first_path": _display_path(first[path_col], base_dir),
                    "last_date": str(pd.Timestamp(last["date"]).date()),
                    "last_label": last.get("label", ""),
                    "last_path": _display_path(last[path_col], base_dir),
                }
            )
    return pd.DataFrame(rows)


def print_station_image_edges(
    daily_df: pd.DataFrame,
    stream: str = "both",
    base_dir: Optional[str] = None,
    printer=print,
    title: str = "First/last resolved images by station-year",
) -> None:
    summary = station_image_edge_summary(daily_df, stream=stream, base_dir=base_dir)
    printer(title)
    if summary.empty:
        printer("  no station-year image rows found")
        return
    for row in summary.itertuples(index=False):
        printer(
            f"  {row.station_year} [{row.stream}] count={row.image_count} "
            f"first={row.first_date} {row.first_label} | {row.first_path} "
            f"last={row.last_date} {row.last_label} | {row.last_path}"
        )


def _date_window_scores(
    date: pd.Timestamp,
    boundaries: Sequence[pd.Timestamp],
    class_to_idx: Dict[str, int],
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
    tolerance_days: int,
) -> np.ndarray:
    scores = np.zeros(len(class_to_idx), dtype=np.float32)
    tolerance_days = max(0, int(tolerance_days))
    stage_labels = _ordered_stage_labels(class_to_idx)
    if PHENOLOGY_BOUNDARY_OFFSET + len(stage_labels) >= len(boundaries):
        raise ValueError(
            "Not enough stage boundaries for configured classes: "
            f"offset={PHENOLOGY_BOUNDARY_OFFSET}, stages={len(stage_labels)}, boundaries={len(boundaries)}"
        )

    def score_from_distance(distance: int) -> float:
        if distance <= 0:
            return 1.0
        if tolerance_days <= 0:
            return 0.0
        return max(0.0, 1.0 - (distance / tolerance_days))

    if "OffSeason" in class_to_idx:
        pre_dist = _distance_to_interval(date, first_date, boundaries[PHENOLOGY_BOUNDARY_OFFSET] - pd.Timedelta(days=1))
        post_dist = _distance_to_interval(date, boundaries[-1] + pd.Timedelta(days=1), last_date)
        scores[class_to_idx["OffSeason"]] = score_from_distance(min(pre_dist, post_dist))

    for i, label in enumerate(stage_labels):
        if label not in class_to_idx:
            continue
        boundary_i = PHENOLOGY_BOUNDARY_OFFSET + i
        start = boundaries[boundary_i]
        end = boundaries[boundary_i + 1] - pd.Timedelta(days=1)
        scores[class_to_idx[label]] = score_from_distance(_distance_to_interval(date, start, end))

    return scores


def generate_group_folds(
    meta_df: pd.DataFrame,
    group_col: str = "group_id",
    n_train: int = 8,
    n_test: int = 2,
    num_folds: Optional[int] = None,
    random_state: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = sorted(list(pd.unique(meta_df[group_col])))
    if n_train + n_test > len(groups):
        raise ValueError(f"Not enough groups ({len(groups)}) for n_train={n_train} + n_test={n_test}")

    all_test_combs = list(itertools.combinations(groups, n_test))
    if num_folds is not None and num_folds < len(all_test_combs):
        rng = random.Random(random_state)
        chosen = rng.sample(all_test_combs, num_folds)
    else:
        chosen = all_test_combs

    folds = []
    for test_groups in chosen:
        test_set = set(test_groups)
        train_groups = [group for group in groups if group not in test_set]
        train_idx = meta_df.index[meta_df[group_col].isin(train_groups)].to_numpy()
        test_idx = meta_df.index[meta_df[group_col].isin(test_set)].to_numpy()
        folds.append((train_idx, test_idx))
    return folds


def generate_group_train_val_test_folds(
    meta_df: pd.DataFrame,
    group_col: str = "group_id",
    n_train: int = 7,
    n_val: int = 1,
    n_test: int = 2,
    num_folds: Optional[int] = None,
    random_state: Optional[int] = None,
    station_col: str = "station_code",
    require_diverse_val_stations: bool = False,
    require_diverse_test_stations: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    groups = sorted(list(pd.unique(meta_df[group_col])))
    if n_train + n_val + n_test > len(groups):
        raise ValueError(
            f"Not enough groups ({len(groups)}) for "
            f"n_train={n_train} + n_val={n_val} + n_test={n_test}"
        )

    if station_col in meta_df.columns:
        group_to_family = {}
        for group, group_df in meta_df.groupby(group_col):
            values = group_df[station_col].dropna().unique()
            group_to_family[group] = _station_family(values[0]) if len(values) else str(group)
    else:
        group_to_family = {group: str(group) for group in groups}

    def has_station_diversity(group_values: Sequence[object], required: bool) -> bool:
        if not required or len(group_values) <= 1:
            return True
        families = {group_to_family.get(group, str(group)) for group in group_values}
        return len(families) >= 2

    all_splits = []
    for test_groups in itertools.combinations(groups, n_test):
        if not has_station_diversity(test_groups, require_diverse_test_stations):
            continue
        remaining = [group for group in groups if group not in set(test_groups)]
        for val_groups in itertools.combinations(remaining, n_val):
            if not has_station_diversity(val_groups, require_diverse_val_stations):
                continue
            val_set = set(val_groups)
            train_groups = [group for group in remaining if group not in val_set]
            all_splits.append((train_groups, list(val_groups), list(test_groups)))
    if not all_splits:
        raise ValueError(
            "No valid train/validation/test folds were generated. "
            "Relax station diversity requirements or reduce validation/test station counts."
        )

    if num_folds is not None and num_folds < len(all_splits):
        rng = random.Random(random_state)
        chosen = rng.sample(all_splits, num_folds)
    else:
        chosen = all_splits

    folds = []
    for train_groups, val_groups, test_groups in chosen:
        train_idx = np.flatnonzero(meta_df[group_col].isin(train_groups).to_numpy())
        val_idx = np.flatnonzero(meta_df[group_col].isin(val_groups).to_numpy())
        test_idx = np.flatnonzero(meta_df[group_col].isin(test_groups).to_numpy())
        folds.append((train_idx, val_idx, test_idx))
    return folds


def generate_loso_train_val_test_folds(
    meta_df: pd.DataFrame,
    group_col: str = "station_code",
    n_val: int = 2,
    random_state: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create one fold per group, with each group used as test exactly once.

    Validation groups are selected from a seeded circular ordering. This makes
    their use balanced: every group appears in validation exactly ``n_val``
    times across the complete LOSO run.
    """
    groups = sorted(list(pd.unique(meta_df[group_col])))
    if n_val < 1:
        raise ValueError(f"LOSO requires at least one validation group, got n_val={n_val}")
    if len(groups) < n_val + 2:
        raise ValueError(
            f"Not enough groups ({len(groups)}) for LOSO with "
            f"n_val={n_val}, one test group, and at least one training group"
        )

    ordered_groups = list(groups)
    random.Random(random_state).shuffle(ordered_groups)
    folds = []
    for test_position, test_group in enumerate(ordered_groups):
        val_groups = {
            ordered_groups[(test_position + offset) % len(ordered_groups)]
            for offset in range(1, n_val + 1)
        }
        test_groups = {test_group}
        train_groups = [
            group
            for group in ordered_groups
            if group not in test_groups and group not in val_groups
        ]
        train_idx = np.flatnonzero(meta_df[group_col].isin(train_groups).to_numpy())
        val_idx = np.flatnonzero(meta_df[group_col].isin(val_groups).to_numpy())
        test_idx = np.flatnonzero(meta_df[group_col].isin(test_groups).to_numpy())
        folds.append((train_idx, val_idx, test_idx))
    return folds


def build_multiscale_daily_dataframe(
    excel_path: str,
    root_dir: str,
    include_preplant_days: int = 30,
    include_postharvest_days: int = 30,
    transition_days: int = 2,
    date_tolerance_days: int = 7,
    classes: Sequence[str] = BASE_CLASSES,
    preferred_camera: Optional[str] = "AUTO",
    use_status_csv: bool = True,
) -> pd.DataFrame:
    """
    Build one row per station-year-date with paired 1x/10x image paths and soft labels.

    The function accepts the existing folder style used by the project:

        data/<station>/<year>/K1|K2/1X|10X/*.jpeg

    It infers macro/micro from the 1X and 10X folders, uses the
    day_image_status CSV when present, and keeps dates from pre-planting dormancy
    through post-harvest closure.
    """
    class_to_idx = {name: i for i, name in enumerate(classes)}

    df = _load_phenology_excel(excel_path)
    root_dir = os.path.abspath(root_dir)

    rows = []
    for record in df.to_dict("records"):
        station_path, station_folder = _find_station_path(root_dir, record.get("Station Code"))
        if station_path is None:
            continue

        label_camera = _record_camera_preference(record)
        effective_camera = _resolve_camera_preference(preferred_camera, label_camera)
        boundaries = _stage_boundaries(record)
        if any(pd.isna(x) for x in boundaries):
            continue

        boundaries = [pd.Timestamp(x).normalize() for x in boundaries]
        first_date = boundaries[0] - pd.Timedelta(days=include_preplant_days)
        last_date = boundaries[-1] + pd.Timedelta(days=include_postharvest_days)
        scan_years = range(first_date.year, last_date.year + 1)

        by_date: Dict[pd.Timestamp, Dict[str, List[str]]] = {}
        for image_path in _iter_current_data_images(
            station_path,
            scan_years,
            preferred_camera=effective_camera,
            use_status_csv=use_status_csv,
        ):
            image_date = _extract_date(image_path)
            if image_date is None:
                continue
            image_date = image_date.normalize()
            if image_date < first_date or image_date > last_date:
                continue
            scale = _camera_scale(image_path)
            if scale is None:
                continue
            by_date.setdefault(image_date, {}).setdefault(scale, []).append(image_path)

        for date, daily_candidates in sorted(by_date.items()):
            paths = _select_daily_image_paths(daily_candidates)
            if "macro" not in paths and "micro" not in paths:
                continue
            soft = _soft_interval_label(date, boundaries, class_to_idx, transition_days)
            if soft is None:
                continue
            date_score = _date_window_scores(
                date,
                boundaries,
                class_to_idx,
                first_date,
                last_date,
                date_tolerance_days,
            )
            hard = classes[int(np.argmax(soft))]
            rows.append(
                {
                    "station_year": f"{station_folder}_{record.get('Year')}",
                    "group_id": record.get("ID"),
                    "station_code": record.get("Station Code"),
                    "year": record.get("Year"),
                    "label_camera": label_camera,
                    "camera_preference": effective_camera,
                    "date": date,
                    "planting_date": boundaries[0],
                    "macro_path": paths.get("macro"),
                    "micro_path": paths.get("micro"),
                    "label": hard,
                    "target": soft.tolist(),
                    "date_score": date_score.tolist(),
                }
            )

    return pd.DataFrame(rows)


def _fetch_meteostat_weather_for_station(
    station_family: str,
    latitude: float,
    longitude: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    try:
        import meteostat as ms
    except ImportError as exc:
        raise ImportError(
            "meteostat is required for --use-weather-metadata. "
            "Install it with: python -m pip install meteostat"
        ) from exc

    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    point = ms.Point(float(latitude), float(longitude))

    if hasattr(ms, "stations") and hasattr(ms, "daily") and hasattr(ms, "interpolate"):
        stations = ms.stations.nearby(point, limit=4)
        daily = ms.daily(stations, start, end)
        weather_df = ms.interpolate(daily, point).fetch()
    else:
        daily_cls = getattr(ms, "Daily")
        weather_df = daily_cls(point, start, end).fetch()

    if weather_df is None or len(weather_df) == 0:
        return pd.DataFrame()

    weather_df = weather_df.reset_index()
    date_col = "time" if "time" in weather_df.columns else weather_df.columns[0]
    keep_cols = [col for col in ["tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres", "tsun"] if col in weather_df.columns]
    weather_df = weather_df[[date_col] + keep_cols].copy()
    weather_df = weather_df.rename(columns={date_col: "date"})
    weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce").dt.normalize()
    weather_df["station_family"] = station_family
    return weather_df.dropna(subset=["date"])


def build_or_load_meteostat_weather_cache(
    daily_df: pd.DataFrame,
    cache_path: Optional[str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    if cache_path and os.path.isfile(cache_path) and not force_refresh:
        weather_df = pd.read_csv(cache_path)
        weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce").dt.normalize()
        if "station_family" in weather_df.columns:
            # CSV inference turns 01/02/06 into integers. Canonicalize the key
            # before merging so leading-zero station families still match.
            weather_df["station_family"] = weather_df["station_family"].map(_station_family)
        return weather_df.dropna(subset=["date"])

    if daily_df.empty:
        return pd.DataFrame()

    records = []
    work_df = daily_df.copy()
    work_df["station_family"] = work_df["station_code"].map(_station_family)
    for family, family_df in work_df.groupby("station_family"):
        coords = STATION_COORDINATES.get(str(family))
        if coords is None:
            print(f"Warning: no coordinates configured for station family {family}; weather will be zero-filled.", flush=True)
            continue
        start_date = pd.to_datetime(family_df["date"], errors="coerce").min()
        end_date = pd.to_datetime(family_df["date"], errors="coerce").max()
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        try:
            fetched = _fetch_meteostat_weather_for_station(str(family), coords[0], coords[1], start_date, end_date)
        except Exception as exc:
            print(f"Warning: Meteostat fetch failed for station family {family}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not fetched.empty:
            records.append(fetched)

    weather_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        weather_df.to_csv(cache_path, index=False)
    return weather_df


def add_weather_metadata(
    daily_df: pd.DataFrame,
    cache_path: Optional[str],
    force_refresh: bool = False,
    gdd_base_temp: float = 0.0,
) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df

    weather_df = build_or_load_meteostat_weather_cache(daily_df, cache_path, force_refresh=force_refresh)
    out = daily_df.copy()
    out["station_family"] = out["station_code"].map(_station_family)

    if weather_df.empty:
        out["weather_gdd_raw"] = 0.0
        out["weather_gdd_cum_raw"] = 0.0
        for col in WEATHER_VALUE_FEATURE_COLUMNS:
            out[col] = 0.0
        for col in WEATHER_MISSING_FEATURE_COLUMNS:
            out[col] = 1.0
        return out

    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce").dt.normalize()
    weather_df["station_family"] = weather_df["station_family"].map(_station_family)
    numeric_cols = [col for col in ["tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres", "tsun"] if col in weather_df.columns]
    for col in numeric_cols:
        weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
    # Preserve whether each model input was observed before interpolation. A
    # filled value and a measured value should not be indistinguishable.
    for source_col in ("tavg", "tmin", "tmax", "prcp"):
        missing_col = f"weather_{source_col}_missing"
        if source_col in weather_df.columns:
            weather_df[missing_col] = weather_df[source_col].isna().astype(float)
        else:
            weather_df[missing_col] = 1.0
    weather_df = weather_df.sort_values(["station_family", "date"])
    if numeric_cols:
        weather_df[numeric_cols] = weather_df.groupby("station_family", group_keys=False)[numeric_cols].apply(
            lambda frame: frame.interpolate(limit_direction="both").ffill().bfill()
        )

    merge_cols = ["station_family", "date"] + numeric_cols + list(WEATHER_MISSING_FEATURE_COLUMNS)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.merge(weather_df[merge_cols], on=["station_family", "date"], how="left")

    # A date absent from the weather table is missing for every weather input.
    for col in WEATHER_MISSING_FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1.0).astype(float)

    if "tavg" not in out.columns:
        if "tmin" in out.columns and "tmax" in out.columns:
            out["tavg"] = (out["tmin"] + out["tmax"]) / 2.0
        elif "tmin" in out.columns:
            out["tavg"] = out["tmin"]
        elif "tmax" in out.columns:
            out["tavg"] = out["tmax"]
        else:
            out["tavg"] = 0.0
    if "tmin" not in out.columns:
        out["tmin"] = out["tavg"]
    if "tmax" not in out.columns:
        out["tmax"] = out["tavg"]
    if "prcp" not in out.columns:
        out["prcp"] = 0.0

    out["tavg"] = out["tavg"].fillna((out["tmin"] + out["tmax"]) / 2.0).fillna(0.0)
    out["tmin"] = out["tmin"].fillna(out["tavg"]).fillna(0.0)
    out["tmax"] = out["tmax"].fillna(out["tavg"]).fillna(0.0)
    out["prcp"] = out["prcp"].fillna(0.0).clip(lower=0.0)

    out["weather_tavg_norm"] = out["tavg"] / WEATHER_TEMP_SCALE
    out["weather_tmin_norm"] = out["tmin"] / WEATHER_TEMP_SCALE
    out["weather_tmax_norm"] = out["tmax"] / WEATHER_TEMP_SCALE
    out["weather_prcp_norm"] = np.log1p(out["prcp"].clip(lower=0.0, upper=WEATHER_PRCP_SCALE)) / np.log1p(WEATHER_PRCP_SCALE)
    out["weather_gdd_raw"] = (out["tavg"] - float(gdd_base_temp)).clip(lower=0.0)
    out["weather_gdd_norm"] = out["weather_gdd_raw"] / WEATHER_GDD_SCALE
    out["weather_gdd_cum_raw"] = 0.0
    out["weather_gdd_cum_norm"] = 0.0

    weather_by_family = {
        str(family): frame.sort_values("date").set_index("date")
        for family, frame in weather_df.groupby("station_family")
    }
    for _, group_idx in out.groupby("group_id").groups.items():
        group = out.loc[group_idx].sort_values("date")
        planting_values = pd.to_datetime(group["planting_date"], errors="coerce").dropna()
        planting_date = planting_values.iloc[0].normalize() if len(planting_values) else None
        family = str(group["station_family"].iloc[0])
        family_weather = weather_by_family.get(family)
        if family_weather is None or family_weather.empty:
            cumulative_gdd = np.zeros(len(group), dtype=np.float64)
        else:
            full_tavg = family_weather.get("tavg")
            if full_tavg is None:
                full_tmin = family_weather.get("tmin", pd.Series(index=family_weather.index, dtype=float))
                full_tmax = family_weather.get("tmax", pd.Series(index=family_weather.index, dtype=float))
                full_tavg = (full_tmin + full_tmax) / 2.0
            full_tavg = pd.to_numeric(full_tavg, errors="coerce").interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
            full_gdd = (full_tavg - float(gdd_base_temp)).clip(lower=0.0)
            if planting_date is not None:
                full_gdd = full_gdd.where(full_gdd.index >= planting_date, 0.0)
            cumulative_by_date = full_gdd.cumsum()
            cumulative_gdd = cumulative_by_date.reindex(group["date"], method="ffill").fillna(0.0).to_numpy()
        out.loc[group.index, "weather_gdd_cum_raw"] = cumulative_gdd
        out.loc[group.index, "weather_gdd_cum_norm"] = cumulative_gdd / WEATHER_CUM_GDD_SCALE

    for col in WEATHER_TEMPORAL_FEATURE_COLUMNS:
        default = 1.0 if col in WEATHER_MISSING_FEATURE_COLUMNS else 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default).astype(float)
    out["weather_gdd_cum_raw"] = pd.to_numeric(out["weather_gdd_cum_raw"], errors="coerce").fillna(0.0).astype(float)
    return out


@dataclass
class WindowConfig:
    window_days: int = 31
    center_offset: Optional[int] = None
    require_center_image: bool = True
    classes: Tuple[str, ...] = tuple(BASE_CLASSES)
    stream: str = "micro"
    temporal_feature_columns: Tuple[str, ...] = DEFAULT_TEMPORAL_FEATURE_COLUMNS
    weather_feature_columns: Tuple[str, ...] = tuple()
    location_feature_columns: Tuple[str, ...] = tuple()

    @property
    def center(self) -> int:
        return self.window_days // 2 if self.center_offset is None else self.center_offset

    @property
    def temporal_feature_dim(self) -> int:
        return len(self.temporal_feature_columns)

    @property
    def weather_feature_dim(self) -> int:
        return len(self.weather_feature_columns)

    @property
    def location_feature_dim(self) -> int:
        return len(self.location_feature_columns)


def _exact_row(group: pd.DataFrame, date: pd.Timestamp) -> Optional[pd.Series]:
    if date not in group.index:
        return None
    row = group.loc[date]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _build_coverage_filtered_samples(
    groups: Dict[str, pd.DataFrame],
    config: WindowConfig,
    has_required_stream,
    min_stage_support_days: int = 0,
    min_window_coverage_days: int = 0,
) -> Tuple[List[Tuple[str, pd.Timestamp]], Dict[Tuple[str, pd.Timestamp], Dict[str, int]], Dict[str, object]]:
    min_stage_support_days = max(0, int(min_stage_support_days))
    min_window_coverage_days = max(0, int(min_window_coverage_days))
    if min_window_coverage_days > config.window_days:
        raise ValueError(
            f"min_window_coverage_days={min_window_coverage_days} exceeds "
            f"window_days={config.window_days}"
        )

    samples: List[Tuple[str, pd.Timestamp]] = []
    sample_coverage: Dict[Tuple[str, pd.Timestamp], Dict[str, int]] = {}
    stage_support_summary: Dict[str, int] = {}
    excluded_stage_support: Dict[str, int] = {}
    excluded_window_coverage: Dict[str, int] = {}
    skipped_missing_target = 0

    for station_year, group in groups.items():
        support_dates: Dict[str, set] = {}
        for date, row in group.iterrows():
            if has_required_stream(row):
                support_dates.setdefault(str(row.get("label", "")), set()).add(pd.Timestamp(date))
        stage_support = {label: len(dates) for label, dates in support_dates.items()}
        for label, count in stage_support.items():
            stage_support_summary[f"{station_year}|{label}"] = int(count)

        for date, row in group.iterrows():
            center_date = pd.Timestamp(date)
            label = str(row.get("label", ""))
            if label not in config.classes:
                continue
            if config.require_center_image and not has_required_stream(row):
                skipped_missing_target += 1
                continue

            support_days = int(stage_support.get(label, 0))
            station_stage = f"{station_year}|{label}"
            if (
                min_stage_support_days > 0
                and label != "OffSeason"
                and support_days < min_stage_support_days
            ):
                excluded_stage_support[station_stage] = excluded_stage_support.get(station_stage, 0) + 1
                continue

            start = center_date - pd.Timedelta(days=config.center)
            coverage_days = 0
            for step in range(config.window_days):
                exact = _exact_row(group, start + pd.Timedelta(days=step))
                if exact is not None and has_required_stream(exact):
                    coverage_days += 1
            if min_window_coverage_days > 0 and coverage_days < min_window_coverage_days:
                excluded_window_coverage[station_stage] = excluded_window_coverage.get(station_stage, 0) + 1
                continue

            key = (station_year, center_date)
            samples.append(key)
            sample_coverage[key] = {
                "window_coverage_days": int(coverage_days),
                "stage_support_days": support_days,
            }

    summary = {
        "min_stage_support_days": min_stage_support_days,
        "min_window_coverage_days": min_window_coverage_days,
        "kept_samples": len(samples),
        "skipped_missing_target": skipped_missing_target,
        "excluded_stage_support_samples": int(sum(excluded_stage_support.values())),
        "excluded_window_coverage_samples": int(sum(excluded_window_coverage.values())),
        "excluded_stage_support_by_station_stage": excluded_stage_support,
        "excluded_window_coverage_by_station_stage": excluded_window_coverage,
        "stage_support_days_by_station_stage": stage_support_summary,
    }
    return samples, sample_coverage, summary


class MultiScaleWindowDataset(Dataset):
    def __init__(
        self,
        daily_df: pd.DataFrame,
        config: WindowConfig,
        transform=None,
        fallback_to_nearest: bool = True,
        min_stage_support_days: int = 0,
        min_window_coverage_days: int = 0,
    ):
        if T is None and transform is None:
            raise ImportError("torchvision is required when transform is not provided")
        self.df = daily_df.copy()
        self.config = config
        self.fallback_to_nearest = fallback_to_nearest
        self.transform = transform or T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.normalize()
        self.df = self.df.sort_values(["station_year", "date"]).reset_index(drop=True)
        self.groups = {k: g.set_index("date") for k, g in self.df.groupby("station_year")}
        self.samples, self.sample_coverage, self.filter_summary = _build_coverage_filtered_samples(
            self.groups,
            self.config,
            self._has_required_stream,
            min_stage_support_days=min_stage_support_days,
            min_window_coverage_days=min_window_coverage_days,
        )

    def _has_required_stream(self, row: pd.Series) -> bool:
        if self.config.stream == "micro":
            return pd.notna(row.get("micro_path"))
        if self.config.stream == "macro":
            return pd.notna(row.get("macro_path"))
        return pd.notna(row.get("macro_path")) and pd.notna(row.get("micro_path"))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Optional[str]) -> Tuple[torch.Tensor, bool]:
        if path is None or pd.isna(path):
            return torch.zeros(3, 224, 224), False
        try:
            with Image.open(path) as img:
                return self.transform(img.convert("RGB")), True
        except (OSError, UnidentifiedImageError) as exc:
            print(
                f"Warning: could not load image {path}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return torch.zeros(3, 224, 224), False

    def _row_for_date(self, group: pd.DataFrame, date: pd.Timestamp) -> Tuple[Optional[pd.Series], float]:
        if date in group.index:
            row = group.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row, 1.0
        if not self.fallback_to_nearest or len(group.index) == 0:
            return None, 0.0
        nearest_pos = np.argmin(np.abs((group.index - date).days))
        nearest_date = group.index[int(nearest_pos)]
        row = group.loc[nearest_date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row, 0.0

    def __getitem__(self, idx: int):
        station_year, center_date = self.samples[idx]
        coverage = self.sample_coverage[(station_year, center_date)]
        group = self.groups[station_year]
        start = center_date - pd.Timedelta(days=self.config.center)
        planting_date = _planting_date_from_group(group)

        macro_frames, micro_frames, mask, temporal_features = [], [], [], []
        for step in range(self.config.window_days):
            current_date = start + pd.Timedelta(days=step)
            row, present = self._row_for_date(group, current_date)
            macro_path = None if row is None else row.get("macro_path")
            micro_path = None if row is None else row.get("micro_path")
            temporal_features.append(
                _temporal_features_for_date(
                    current_date,
                    planting_date,
                    row=row,
                    feature_columns=self.config.temporal_feature_columns,
                )
            )
            macro_frame, macro_ok = self._load_image(macro_path if self.config.stream in {"macro", "both"} else None)
            micro_frame, micro_ok = self._load_image(micro_path if self.config.stream in {"micro", "both"} else None)
            macro_frames.append(macro_frame)
            micro_frames.append(micro_frame)
            if row is None:
                mask.append(0.0)
            elif self.config.stream == "micro":
                mask.append(float(present and micro_ok))
            elif self.config.stream == "macro":
                mask.append(float(present and macro_ok))
            else:
                mask.append(float(present and macro_ok and micro_ok))

        center_row = group.loc[center_date]
        if isinstance(center_row, pd.DataFrame):
            center_row = center_row.iloc[0]
        target = torch.tensor(center_row["target"], dtype=torch.float32)
        date_score = _date_score_to_tensor(center_row)
        label = int(torch.argmax(target).item())

        return {
            "macro": torch.stack(macro_frames),
            "micro": torch.stack(micro_frames),
            "temporal_features": torch.stack(temporal_features),
            "location_features": _location_features_from_row(
                center_row,
                self.config.location_feature_columns,
            ),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target": target,
            "date_score": date_score,
            "label": torch.tensor(label, dtype=torch.long),
            "station_year": station_year,
            "date": str(center_date.date()),
            "window_coverage_days": torch.tensor(coverage["window_coverage_days"], dtype=torch.long),
            "stage_support_days": torch.tensor(coverage["stage_support_days"], dtype=torch.long),
        }


def _path_key(path: Optional[str]) -> Optional[str]:
    if path is None or pd.isna(path):
        return None
    return os.path.abspath(os.path.normpath(str(path)))


def _target_to_tensor(value: object) -> torch.Tensor:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return torch.tensor(value, dtype=torch.float32)


def _date_score_to_tensor(row: pd.Series) -> torch.Tensor:
    value = row.get("date_score")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        value = row.get("target")
    return _target_to_tensor(value)


def _planting_date_from_group(group: pd.DataFrame) -> Optional[pd.Timestamp]:
    if "planting_date" not in group.columns:
        return None
    values = pd.to_datetime(group["planting_date"], errors="coerce").dropna()
    if values.empty:
        return None
    return pd.Timestamp(values.iloc[0]).normalize()


def _location_features_from_row(
    row: pd.Series,
    feature_columns: Sequence[str] = LOCATION_FEATURE_COLUMNS,
) -> torch.Tensor:
    if not feature_columns:
        return torch.zeros(0, dtype=torch.float32)
    if all(column in row.index and pd.notna(row.get(column)) for column in feature_columns):
        return torch.tensor([float(row.get(column)) for column in feature_columns], dtype=torch.float32)
    return station_location_features(row.get("station_code"), strict=False)


def _temporal_features_for_date(
    date: pd.Timestamp,
    planting_date: Optional[pd.Timestamp],
    row: Optional[pd.Series] = None,
    feature_columns: Sequence[str] = DEFAULT_TEMPORAL_FEATURE_COLUMNS,
) -> torch.Tensor:
    values = []
    for column in feature_columns:
        if column == DAYS_SINCE_PLANTING_FEATURE:
            if planting_date is None or pd.isna(planting_date):
                values.append(0.0)
            else:
                days_since_planting = (pd.Timestamp(date).normalize() - pd.Timestamp(planting_date).normalize()).days
                values.append(days_since_planting / DAYS_SINCE_PLANTING_SCALE)
            continue
        missing_default = 1.0 if column in WEATHER_MISSING_FEATURE_COLUMNS else 0.0
        if row is None or column not in row.index:
            values.append(missing_default)
            continue
        value = row.get(column)
        if value is None or pd.isna(value):
            values.append(missing_default)
        else:
            values.append(float(value))
    return torch.tensor(values, dtype=torch.float32)


class MultiScaleEmbeddingWindowDataset(Dataset):
    def __init__(
        self,
        daily_df: pd.DataFrame,
        config: WindowConfig,
        embedding_cache: Dict,
        fallback_to_nearest: bool = True,
        use_augmentation: bool = False,
        augmentation_multiplier: Optional[int] = None,
        min_stage_support_days: int = 0,
        min_window_coverage_days: int = 0,
    ):
        self.df = daily_df.copy()
        self.config = config
        self.fallback_to_nearest = fallback_to_nearest
        self.feature_dim = int(embedding_cache["feature_dim"])
        tiling = embedding_cache.get("tiling", {})
        self.tile_attention = tiling.get("tile_pooling") == "attention"
        dense_features = embedding_cache.get("dense_features", {})
        self.dense_features = bool(dense_features.get("enabled", False))
        self.dense_tokens_per_tile = int(dense_features.get("tokens_per_tile", 1))
        self.dense_streams = set(dense_features.get("streams", ["macro", "micro"] if self.dense_features else []))
        tile_counts = []
        tile_counts.extend(tiling.get("macro_tile_counts", {}).values())
        tile_counts.extend(tiling.get("micro_tile_counts", {}).values())
        self.max_tiles = max([int(x) for x in tile_counts], default=1)
        self.macro_embeddings = {
            _path_key(path): value
            for path, value in embedding_cache.get("macro", {}).items()
        }
        self.micro_embeddings = {
            _path_key(path): value
            for path, value in embedding_cache.get("micro", {}).items()
        }
        self.macro_aug_embeddings = {
            _path_key(path): [item for item in values]
            for path, values in embedding_cache.get("macro_aug", {}).items()
        }
        self.micro_aug_embeddings = {
            _path_key(path): [item for item in values]
            for path, values in embedding_cache.get("micro_aug", {}).items()
        }
        self.augmentation_views = int(embedding_cache.get("augmentation", {}).get("views", 0))
        self.use_augmentation = bool(use_augmentation and self.augmentation_views > 0)
        if self.use_augmentation:
            requested_multiplier = augmentation_multiplier if augmentation_multiplier is not None else self.augmentation_views + 1
            self.sample_multiplier = max(1, min(int(requested_multiplier), self.augmentation_views + 1))
        else:
            self.sample_multiplier = 1
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.normalize()
        self.df = self.df.sort_values(["station_year", "date"]).reset_index(drop=True)
        self.groups = {k: g.set_index("date") for k, g in self.df.groupby("station_year")}
        self.samples, self.sample_coverage, self.filter_summary = _build_coverage_filtered_samples(
            self.groups,
            self.config,
            self._has_required_stream,
            min_stage_support_days=min_stage_support_days,
            min_window_coverage_days=min_window_coverage_days,
        )
        if self.filter_summary["skipped_missing_target"]:
            print(
                f"Warning: skipped {self.filter_summary['skipped_missing_target']} target-day samples "
                "because required cached embeddings were missing"
            )

    def _has_required_stream(self, row: pd.Series) -> bool:
        def cached(path: object, table: Dict[str, torch.Tensor]) -> bool:
            key = _path_key(path)
            return key is not None and key in table

        if self.config.stream == "micro":
            return cached(row.get("micro_path"), self.micro_embeddings)
        if self.config.stream == "macro":
            return cached(row.get("macro_path"), self.macro_embeddings)
        return cached(row.get("macro_path"), self.macro_embeddings) and cached(row.get("micro_path"), self.micro_embeddings)

    def __len__(self) -> int:
        return len(self.samples) * self.sample_multiplier

    def _row_for_date(self, group: pd.DataFrame, date: pd.Timestamp) -> Tuple[Optional[pd.Series], float]:
        if date in group.index:
            row = group.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row, 1.0
        if not self.fallback_to_nearest or len(group.index) == 0:
            return None, 0.0
        nearest_pos = np.argmin(np.abs((group.index - date).days))
        nearest_date = group.index[int(nearest_pos)]
        row = group.loc[nearest_date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row, 0.0

    def _load_embedding(self, path: Optional[str], stream: str, aug_index: int = 0) -> torch.Tensor:
        key = _path_key(path)
        stream_uses_dense = self.dense_features and stream in self.dense_streams
        zero_shape = (
            (self.max_tiles, self.dense_tokens_per_tile, self.feature_dim)
            if self.tile_attention and stream_uses_dense
            else (self.max_tiles, self.feature_dim)
            if self.tile_attention
            else (self.feature_dim,)
        )
        if key is None:
            return torch.zeros(zero_shape)
        table = self.macro_embeddings if stream == "macro" else self.micro_embeddings
        aug_table = self.macro_aug_embeddings if stream == "macro" else self.micro_aug_embeddings
        value = None
        if aug_index > 0:
            aug_values = aug_table.get(key, [])
            if aug_index - 1 < len(aug_values):
                value = aug_values[aug_index - 1]
        if value is None:
            value = table.get(key)
        if value is None:
            return torch.zeros(zero_shape)
        value = value.float()
        if not self.tile_attention:
            while value.ndim > 1:
                value = value.mean(dim=0)
            return value
        if stream_uses_dense:
            if value.ndim == 2:
                value = value.unsqueeze(1)
            if value.ndim != 3 or value.shape[-2:] != (self.dense_tokens_per_tile, self.feature_dim):
                raise ValueError(
                    f"Dense cache tensor for {key} has shape {tuple(value.shape)}; expected "
                    f"[tiles, {self.dense_tokens_per_tile}, {self.feature_dim}]"
                )
            out = torch.zeros(zero_shape)
            n_tiles = min(value.shape[0], self.max_tiles)
            out[:n_tiles] = value[:n_tiles]
            return out
        if value.ndim == 1:
            value = value.unsqueeze(0)
        out = torch.zeros(self.max_tiles, self.feature_dim)
        n_tiles = min(value.shape[0], self.max_tiles)
        out[:n_tiles] = value[:n_tiles]
        return out

    def _tile_mask(self, path: Optional[str], stream: str) -> torch.Tensor:
        key = _path_key(path)
        table = self.macro_embeddings if stream == "macro" else self.micro_embeddings
        value = table.get(key)
        if not self.tile_attention:
            return torch.tensor([value is not None], dtype=torch.bool)
        mask = torch.zeros(self.max_tiles, dtype=torch.bool)
        if value is None:
            return mask
        n_tiles = value.shape[0] if value.ndim >= 2 else 1
        mask[: min(n_tiles, self.max_tiles)] = True
        return mask

    def __getitem__(self, idx: int):
        sample_idx = idx // self.sample_multiplier
        aug_index = idx % self.sample_multiplier
        station_year, center_date = self.samples[sample_idx]
        coverage = self.sample_coverage[(station_year, center_date)]
        group = self.groups[station_year]
        start = center_date - pd.Timedelta(days=self.config.center)
        planting_date = _planting_date_from_group(group)

        macro_features, micro_features, mask, temporal_features = [], [], [], []
        macro_tile_masks, micro_tile_masks = [], []
        for step in range(self.config.window_days):
            current_date = start + pd.Timedelta(days=step)
            row, present = self._row_for_date(group, current_date)
            macro_path = None if row is None else row.get("macro_path")
            micro_path = None if row is None else row.get("micro_path")
            macro_input_path = macro_path if self.config.stream in {"macro", "both"} else None
            micro_input_path = micro_path if self.config.stream in {"micro", "both"} else None
            temporal_features.append(
                _temporal_features_for_date(
                    current_date,
                    planting_date,
                    row=row,
                    feature_columns=self.config.temporal_feature_columns,
                )
            )
            macro_features.append(self._load_embedding(macro_input_path, "macro", aug_index=aug_index))
            micro_features.append(self._load_embedding(micro_input_path, "micro", aug_index=aug_index))
            macro_tile_mask = self._tile_mask(macro_input_path, "macro")
            micro_tile_mask = self._tile_mask(micro_input_path, "micro")
            macro_tile_masks.append(macro_tile_mask)
            micro_tile_masks.append(micro_tile_mask)
            macro_ok = bool(macro_tile_mask.any())
            micro_ok = bool(micro_tile_mask.any())
            if row is None:
                mask.append(0.0)
            elif self.config.stream == "micro":
                mask.append(float(present and micro_ok))
            elif self.config.stream == "macro":
                mask.append(float(present and macro_ok))
            else:
                mask.append(float(present and macro_ok and micro_ok))

        center_row = group.loc[center_date]
        if isinstance(center_row, pd.DataFrame):
            center_row = center_row.iloc[0]
        target = _target_to_tensor(center_row["target"])
        date_score = _date_score_to_tensor(center_row)
        label = int(torch.argmax(target).item())

        return {
            "macro": torch.stack(macro_features),
            "micro": torch.stack(micro_features),
            "temporal_features": torch.stack(temporal_features),
            "location_features": _location_features_from_row(
                center_row,
                self.config.location_feature_columns,
            ),
            "macro_tile_mask": torch.stack(macro_tile_masks),
            "micro_tile_mask": torch.stack(micro_tile_masks),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target": target,
            "date_score": date_score,
            "label": torch.tensor(label, dtype=torch.long),
            "station_year": station_year,
            "date": str(center_date.date()),
            "window_coverage_days": torch.tensor(coverage["window_coverage_days"], dtype=torch.long),
            "stage_support_days": torch.tensor(coverage["stage_support_days"], dtype=torch.long),
            "augmentation_index": torch.tensor(aug_index, dtype=torch.long),
        }


class SMOTEEmbeddingWindowDataset(Dataset):
    """Leakage-safe SMOTE for cached temporal embedding windows.

    Synthetic examples interpolate nearest training windows with the same hard
    stage label. SMOTE is applied to frozen embedding tensors, never pixels,
    and this wrapper must only be used for a training fold.
    """

    def __init__(
        self,
        base_dataset: MultiScaleEmbeddingWindowDataset,
        target_ratio: float = 1.0,
        k_neighbors: int = 5,
        max_synthetic_samples: int = 0,
        seed: int = 42,
    ):
        if not isinstance(base_dataset, MultiScaleEmbeddingWindowDataset):
            raise TypeError("SMOTEEmbeddingWindowDataset requires MultiScaleEmbeddingWindowDataset")
        if target_ratio <= 0:
            raise ValueError("SMOTE target_ratio must be greater than zero")
        if k_neighbors < 1:
            raise ValueError("SMOTE k_neighbors must be at least one")

        self.base_dataset = base_dataset
        self.target_ratio = float(target_ratio)
        self.k_neighbors = int(k_neighbors)
        self.max_synthetic_samples = int(max_synthetic_samples)
        self.seed = int(seed)
        self.sample_multiplier = getattr(base_dataset, "sample_multiplier", 1)
        self.synthetic_specs: List[Tuple[int, int, float]] = []
        self.summary = self._build_synthetic_specs()

    def _center_row(self, dataset_index: int) -> Tuple[pd.Series, int]:
        sample_idx = dataset_index // self.base_dataset.sample_multiplier
        station_year, center_date = self.base_dataset.samples[sample_idx]
        group = self.base_dataset.groups[station_year]
        row = group.loc[center_date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row, sample_idx

    def _label_for_index(self, dataset_index: int) -> int:
        row, _ = self._center_row(dataset_index)
        return int(torch.argmax(_target_to_tensor(row["target"])).item())

    def _descriptor_for_index(self, dataset_index: int) -> np.ndarray:
        row, _ = self._center_row(dataset_index)
        features = []
        if self.base_dataset.config.stream in {"macro", "both"}:
            value = self.base_dataset._load_embedding(row.get("macro_path"), "macro")
            features.append(value.mean(dim=0) if value.ndim == 2 else value)
        if self.base_dataset.config.stream in {"micro", "both"}:
            value = self.base_dataset._load_embedding(row.get("micro_path"), "micro")
            features.append(value.mean(dim=0) if value.ndim == 2 else value)
        if not features:
            raise RuntimeError("SMOTE could not build a descriptor because no image stream is active")
        return torch.cat(features).float().cpu().numpy()

    def _build_synthetic_specs(self) -> Dict[str, object]:
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:
            raise ImportError(
                "SMOTE requires scikit-learn. Install it with: python -m pip install scikit-learn"
            ) from exc

        rng = np.random.default_rng(self.seed)
        # Augmented cache views are alternate renderings of a real window. They
        # count toward balance but are not separate neighbour-search samples.
        source_indices = np.arange(len(self.base_dataset.samples), dtype=np.int64) * self.base_dataset.sample_multiplier
        labels = np.asarray([self._label_for_index(int(idx)) for idx in source_indices], dtype=np.int64)
        class_counts = {
            int(label): int((labels == label).sum()) * self.base_dataset.sample_multiplier
            for label in np.unique(labels)
        }
        largest_class_count = max(class_counts.values(), default=0)
        requested_by_class = {
            label: max(0, int(math.ceil(largest_class_count * self.target_ratio)) - count)
            for label, count in class_counts.items()
        }

        if self.max_synthetic_samples > 0:
            requested_total = sum(requested_by_class.values())
            if requested_total > self.max_synthetic_samples:
                scale = self.max_synthetic_samples / requested_total
                requested_by_class = {
                    label: int(math.floor(count * scale)) for label, count in requested_by_class.items()
                }
                remaining = self.max_synthetic_samples - sum(requested_by_class.values())
                for label in sorted(requested_by_class, key=lambda item: (-class_counts[item], item)):
                    if remaining <= 0:
                        break
                    maximum = max(0, int(math.ceil(largest_class_count * self.target_ratio)) - class_counts[label])
                    if requested_by_class[label] < maximum:
                        requested_by_class[label] += 1
                        remaining -= 1

        generated_by_class: Dict[int, int] = {}
        skipped_classes: Dict[int, str] = {}
        for label, requested_count in requested_by_class.items():
            if requested_count <= 0:
                continue
            member_indices = source_indices[labels == label]
            if len(member_indices) < 2:
                skipped_classes[label] = "fewer than two training windows"
                continue

            descriptors = np.stack([self._descriptor_for_index(int(idx)) for idx in member_indices])
            neighbors = NearestNeighbors(
                n_neighbors=min(self.k_neighbors + 1, len(member_indices)),
                metric="cosine",
            ).fit(descriptors)
            neighbor_positions = neighbors.kneighbors(descriptors, return_distance=False)

            for _ in range(requested_count):
                source_position = int(rng.integers(len(member_indices)))
                candidates = neighbor_positions[source_position]
                candidates = candidates[candidates != source_position]
                if len(candidates) == 0:
                    continue
                neighbor_position = int(rng.choice(candidates))
                self.synthetic_specs.append(
                    (
                        int(member_indices[source_position]),
                        int(member_indices[neighbor_position]),
                        float(rng.uniform(0.05, 0.95)),
                    )
                )
                generated_by_class[label] = generated_by_class.get(label, 0) + 1

        return {
            "original_samples": len(self.base_dataset),
            "synthetic_samples": len(self.synthetic_specs),
            "total_samples": len(self),
            "class_counts_before": class_counts,
            "requested_synthetic_by_class": requested_by_class,
            "generated_synthetic_by_class": generated_by_class,
            "skipped_classes": skipped_classes,
            "target_ratio": self.target_ratio,
            "k_neighbors": self.k_neighbors,
            "max_synthetic_samples": self.max_synthetic_samples,
            "seed": self.seed,
        }

    def __len__(self) -> int:
        return len(self.base_dataset) + len(self.synthetic_specs)

    @staticmethod
    def _blend_tensor(first: torch.Tensor, second: torch.Tensor, interpolation: float) -> torch.Tensor:
        return first + (second - first) * interpolation

    def __getitem__(self, idx: int):
        if idx < len(self.base_dataset):
            return self.base_dataset[idx]

        source_idx, neighbor_idx, interpolation = self.synthetic_specs[idx - len(self.base_dataset)]
        source = self.base_dataset[source_idx]
        neighbor = self.base_dataset[neighbor_idx]
        output = dict(source)
        for key in ("macro", "micro", "temporal_features", "location_features", "target", "date_score"):
            output[key] = self._blend_tensor(source[key], neighbor[key], interpolation)
        for key in ("mask", "macro_tile_mask", "micro_tile_mask"):
            if key in source:
                output[key] = torch.logical_or(source[key], neighbor[key])
        output["label"] = source["label"].clone()
        output["station_year"] = f"SMOTE:{source['station_year']}+{neighbor['station_year']}"
        output["date"] = f"SMOTE:{source['date']}+{neighbor['date']}"
        output["augmentation_index"] = torch.tensor(-1, dtype=torch.long)
        return output


class ViTFeatureEncoder(nn.Module):
    def __init__(self, backbone: str = DINO_DEFAULT_BACKBONE, pretrained: bool = True, out_dim: int = 512):
        super().__init__()
        if _is_huggingface_backbone(backbone):
            base = ViTBackboneFeatureExtractor(backbone, pretrained=pretrained)
            in_dim = base.out_dim
            self.backbone = base
        else:
            if tvm is None:
                raise ImportError("torchvision is required for torchvision ViTFeatureEncoder")
            weights = _torchvision_vit_weights(backbone, pretrained)
            base = getattr(tvm, backbone)(weights=weights)
            in_dim = base.heads.head.in_features
            base.heads.head = nn.Identity()
            self.backbone = base
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class ViTBackboneFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = DINO_DEFAULT_BACKBONE,
        pretrained: bool = True,
        local_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.backbone_name = _normalize_backbone_name(backbone)
        self.backbone_source = "huggingface" if _is_huggingface_backbone(self.backbone_name) else "torchvision"
        self.preprocess_image_size = 224
        self.preprocess_mean = [0.485, 0.456, 0.406]
        self.preprocess_std = [0.229, 0.224, 0.225]
        self.patch_size = 16
        self.num_register_tokens = 0

        if self.backbone_source == "huggingface":
            if AutoConfig is None or AutoImageProcessor is None or AutoModel is None:
                raise ImportError("transformers>=4.56 is required for DINOv3/Hugging Face backbones")
            if local_config is not None:
                try:
                    from transformers import DINOv3ViTConfig
                except ImportError as exc:
                    raise ImportError("This transformers version cannot construct DINOv3 locally") from exc
                config = DINOv3ViTConfig(**local_config)
                self.processor = None
                self.backbone = AutoModel.from_config(config)
                self.out_dim = int(config.hidden_size)
                self.patch_size = int(config.patch_size)
                self.num_register_tokens = int(config.num_register_tokens)
                return
            load_kwargs = {}
            if not os.path.isdir(self.backbone_name) and get_huggingface_token is not None:
                # Pass the active token explicitly. This works even when
                # HF_HUB_DISABLE_IMPLICIT_TOKEN is set and provides gated
                # DINOv3 access from non-interactive training scripts.
                token = get_huggingface_token()
                if token:
                    load_kwargs["token"] = token
            self.processor = AutoImageProcessor.from_pretrained(self.backbone_name, **load_kwargs)
            config = AutoConfig.from_pretrained(self.backbone_name, **load_kwargs)
            self.backbone = (
                AutoModel.from_pretrained(self.backbone_name, **load_kwargs)
                if pretrained
                else AutoModel.from_config(config)
            )
            self.out_dim = int(config.hidden_size)
            patch_size = getattr(config, "patch_size", 16)
            self.patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)
            self.num_register_tokens = int(getattr(config, "num_register_tokens", 0))
            self.preprocess_image_size = _processor_image_size(self.processor, default=224)
            self.preprocess_mean = list(getattr(self.processor, "image_mean", self.preprocess_mean))
            self.preprocess_std = list(getattr(self.processor, "image_std", self.preprocess_std))
        else:
            if tvm is None:
                raise ImportError("torchvision is required for torchvision ViTBackboneFeatureExtractor")
            weights = _torchvision_vit_weights(self.backbone_name, pretrained)
            base = getattr(tvm, self.backbone_name)(weights=weights)
            self.out_dim = base.heads.head.in_features
            base.heads.head = nn.Identity()
            self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_source == "huggingface":
            outputs = self.backbone(pixel_values=x)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            return outputs.last_hidden_state[:, 0]
        return self.backbone(x)

    def forward_dense(
        self,
        x: torch.Tensor,
        grid_size: int = 2,
        include_cls: bool = True,
    ) -> torch.Tensor:
        """Return compact dense descriptors as [batch, tokens, feature_dim]."""
        if self.backbone_source != "huggingface":
            raise ValueError("Dense patch extraction is supported only for Hugging Face ViT backbones")
        outputs = self.backbone(pixel_values=x)
        return compact_dense_tokens(
            outputs.last_hidden_state,
            pixel_height=x.shape[-2],
            pixel_width=x.shape[-1],
            patch_size=self.patch_size,
            num_register_tokens=self.num_register_tokens,
            grid_size=grid_size,
            include_cls=include_cls,
        )


def compact_dense_tokens(
    last_hidden_state: torch.Tensor,
    pixel_height: int,
    pixel_width: int,
    patch_size: int,
    num_register_tokens: int,
    grid_size: int = 2,
    include_cls: bool = True,
) -> torch.Tensor:
    """Remove register tokens and spatially pool patch tokens to a fixed grid."""
    if last_hidden_state.ndim != 3:
        raise ValueError(f"Expected [batch, tokens, dim], got {tuple(last_hidden_state.shape)}")
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")
    patch_rows = int(pixel_height) // int(patch_size)
    patch_cols = int(pixel_width) // int(patch_size)
    expected_patches = patch_rows * patch_cols
    patch_start = 1 + int(num_register_tokens)
    patch_end = patch_start + expected_patches
    if patch_rows < 1 or patch_cols < 1 or last_hidden_state.shape[1] < patch_end:
        raise ValueError(
            "Backbone token count is incompatible with the input/patch geometry: "
            f"tokens={last_hidden_state.shape[1]} expected_at_least={patch_end} "
            f"input={pixel_height}x{pixel_width} patch_size={patch_size}"
        )
    patches = last_hidden_state[:, patch_start:patch_end]
    patches = patches.reshape(patches.shape[0], patch_rows, patch_cols, patches.shape[-1])
    patches = patches.permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(patches, output_size=(grid_size, grid_size))
    pooled = pooled.flatten(2).transpose(1, 2)
    if include_cls:
        pooled = torch.cat([last_hidden_state[:, :1], pooled], dim=1)
    return pooled


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def _torchvision_vit_weights(backbone: str, pretrained: bool):
    if not pretrained:
        return None
    parts = backbone.split("_")
    weights_enum_name = "ViT_" + "_".join(part.upper() for part in parts[1:]) + "_Weights" if parts and parts[0].lower() == "vit" else f"{backbone}_Weights"
    weights_enum = getattr(tvm, weights_enum_name, None)
    if weights_enum is None:
        print(f"Warning: could not find weights enum {weights_enum_name} for {backbone}; using string fallback")
        return "IMAGENET1K_V1"
    return weights_enum.IMAGENET1K_V1


def _normalize_backbone_name(backbone: str) -> str:
    value = str(backbone or DINO_DEFAULT_BACKBONE)
    return DINO_BACKBONE_ALIASES.get(value.lower(), value)


def _is_huggingface_backbone(backbone: str) -> bool:
    value = _normalize_backbone_name(backbone)
    normalized = value.lower()
    # Fine-tuned Hugging Face checkpoints are ordinary local directories.
    # Detect both POSIX and Windows paths rather than relying on a model-ID slash.
    if os.path.isdir(value) and os.path.isfile(os.path.join(value, "config.json")):
        return True
    return "/" in normalized or "\\" in normalized or normalized.startswith("dinov3")


def _processor_image_size(processor, default: int = 224) -> int:
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        for key in ("height", "width", "shortest_edge"):
            value = size.get(key)
            if value is not None:
                return int(value)
    if isinstance(size, (list, tuple)) and size:
        return int(size[0])
    if isinstance(size, int):
        return int(size)
    return default


class TileAttentionPooler(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        max_rows: int = 64,
        max_cols: int = 64,
        grid_aspect_ratio: float = 4.0 / 3.0,
    ):
        super().__init__()
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.grid_aspect_ratio = grid_aspect_ratio
        self.row_pos = nn.Parameter(torch.zeros(1, 1, max_rows, feature_dim))
        self.col_pos = nn.Parameter(torch.zeros(1, 1, max_cols, feature_dim))
        nn.init.trunc_normal_(self.row_pos, std=0.02)
        nn.init.trunc_normal_(self.col_pos, std=0.02)
        self.score = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, tile_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features.ndim == 3:
            return features
        if features.ndim != 4:
            raise ValueError(f"Expected tile features with 3 or 4 dims, got shape {tuple(features.shape)}")

        tile_count = features.size(-2)
        cols = max(1, math.ceil(math.sqrt(tile_count * self.grid_aspect_ratio)))
        rows = math.ceil(tile_count / cols)
        if rows > self.max_rows or cols > self.max_cols:
            raise ValueError(
                f"Tile grid {rows}x{cols} exceeds positional capacity "
                f"{self.max_rows}x{self.max_cols}"
            )
        tile_index = torch.arange(tile_count, device=features.device)
        row_index = torch.div(tile_index, cols, rounding_mode="floor")
        col_index = tile_index.remainder(cols)
        tile_pos = self.row_pos[:, :, row_index, :] + self.col_pos[:, :, col_index, :]
        features = features + tile_pos
        # Keep attention masking/softmax in fp32 so AMP fp16 does not overflow
        # on large negative mask values.
        logits = self.score(features).squeeze(-1).float()
        if tile_mask is not None:
            valid = tile_mask.bool()
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
            empty = ~valid.any(dim=-1, keepdim=True)
            logits = torch.where(empty, torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=-1)
        if tile_mask is not None:
            weights = weights * tile_mask.bool().float()
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weights = weights / denom
        return (features * weights.unsqueeze(-1)).sum(dim=-2)


class HierarchicalDenseTilePooler(nn.Module):
    """Attend over DINOv3 dense descriptors inside tiles, then over image tiles."""

    def __init__(
        self,
        feature_dim: int,
        dense_tokens_per_tile: int,
        patch_hidden_dim: int = 128,
    ):
        super().__init__()
        if dense_tokens_per_tile < 1:
            raise ValueError("dense_tokens_per_tile must be at least 1")
        self.dense_tokens_per_tile = int(dense_tokens_per_tile)
        self.patch_pos = nn.Parameter(torch.zeros(1, 1, 1, self.dense_tokens_per_tile, feature_dim))
        nn.init.trunc_normal_(self.patch_pos, std=0.02)
        self.patch_score = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, patch_hidden_dim),
            nn.GELU(),
            nn.Linear(patch_hidden_dim, 1),
        )
        self.tile_pool = TileAttentionPooler(feature_dim)

    def forward(self, features: torch.Tensor, tile_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features.ndim != 5:
            return self.tile_pool(features, tile_mask)
        if features.shape[-2] != self.dense_tokens_per_tile:
            raise ValueError(
                f"Expected {self.dense_tokens_per_tile} dense tokens per tile, "
                f"got {features.shape[-2]}"
            )
        features = features + self.patch_pos
        patch_logits = self.patch_score(features).squeeze(-1).float()
        patch_weights = torch.softmax(patch_logits, dim=-1).to(features.dtype)
        tile_features = (features * patch_weights.unsqueeze(-1)).sum(dim=-2)
        return self.tile_pool(tile_features, tile_mask)


class TemporalAggregationMixin:
    def _init_temporal_aggregation(self, embed_dim: int, temporal_aggregation: str):
        if temporal_aggregation not in {"target", "mean", "cls"}:
            raise ValueError("temporal_aggregation must be one of: target, mean, cls")
        self.temporal_aggregation = temporal_aggregation
        if temporal_aggregation == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_temporal_features(
        self,
        embed_dim: int,
        temporal_feature_dim: int,
        dropout: float,
        temporal_feature_hidden_dim: int = 0,
        weather_feature_dim: int = 0,
        temporal_feature_fusion: str = "gated",
        temporal_feature_gate_init: float = 0.1,
        weather_feature_gate_init: float = 0.1,
    ):
        self.temporal_feature_dim = int(temporal_feature_dim)
        self.temporal_feature_hidden_dim = int(temporal_feature_hidden_dim)
        self.weather_feature_dim = int(weather_feature_dim)
        self.calendar_feature_dim = self.temporal_feature_dim - self.weather_feature_dim
        self.temporal_feature_fusion_type = str(temporal_feature_fusion)
        if self.temporal_feature_fusion_type not in {"gated", "legacy"}:
            raise ValueError("temporal_feature_fusion must be one of: gated, legacy")
        if self.weather_feature_dim < 0 or self.weather_feature_dim > self.temporal_feature_dim:
            raise ValueError("weather_feature_dim must be between zero and temporal_feature_dim")

        if self.temporal_feature_fusion_type == "legacy":
            if self.temporal_feature_dim <= 0:
                self.temporal_feature_fusion = nn.Identity()
                return
            input_dim = embed_dim + self.temporal_feature_dim
            if self.temporal_feature_hidden_dim > 0:
                self.temporal_feature_fusion = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, self.temporal_feature_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.temporal_feature_hidden_dim, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            else:
                self.temporal_feature_fusion = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            return

        if self.temporal_feature_dim <= 0:
            return
        for name, value in (
            ("temporal_feature_gate_init", temporal_feature_gate_init),
            ("weather_feature_gate_init", weather_feature_gate_init),
        ):
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")

        def metadata_mlp(feature_dim: int) -> nn.Sequential:
            if self.temporal_feature_hidden_dim > 0:
                return nn.Sequential(
                    nn.Linear(feature_dim, self.temporal_feature_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.temporal_feature_hidden_dim, embed_dim),
                )
            return nn.Sequential(
                nn.Linear(feature_dim, embed_dim),
            )

        if self.calendar_feature_dim > 0:
            self.temporal_feature_mlp = metadata_mlp(self.calendar_feature_dim)
            gate_logit = math.log(
                float(temporal_feature_gate_init) / (1.0 - float(temporal_feature_gate_init))
            )
            self.temporal_feature_gate_logit = nn.Parameter(
                torch.tensor(gate_logit, dtype=torch.float32)
            )
        if self.weather_feature_dim > 0:
            self.weather_feature_mlp = metadata_mlp(self.weather_feature_dim)
            gate_logit = math.log(
                float(weather_feature_gate_init) / (1.0 - float(weather_feature_gate_init))
            )
            self.weather_feature_gate_logit = nn.Parameter(
                torch.tensor(gate_logit, dtype=torch.float32)
            )

    def _add_temporal_features(self, x: torch.Tensor, temporal_features: Optional[torch.Tensor]):
        if getattr(self, "temporal_feature_dim", 0) <= 0:
            return x
        if temporal_features is None:
            temporal_features = torch.zeros(
                x.size(0),
                x.size(1),
                self.temporal_feature_dim,
                dtype=x.dtype,
                device=x.device,
            )
        else:
            temporal_features = temporal_features.to(device=x.device, dtype=x.dtype)
        if temporal_features.shape[:2] != x.shape[:2] or temporal_features.shape[-1] != self.temporal_feature_dim:
            raise ValueError(
                "temporal_features must have shape "
                f"[batch, days, {self.temporal_feature_dim}], got {tuple(temporal_features.shape)}"
            )
        if self.temporal_feature_fusion_type == "legacy":
            return self.temporal_feature_fusion(torch.cat([x, temporal_features], dim=-1))

        fused = x
        if self.calendar_feature_dim > 0:
            calendar_features = temporal_features[..., : self.calendar_feature_dim]
            calendar_embedding = self.temporal_feature_mlp(calendar_features)
            calendar_gate = torch.sigmoid(self.temporal_feature_gate_logit).to(dtype=x.dtype)
            fused = fused + calendar_gate * calendar_embedding
        if self.weather_feature_dim > 0:
            weather_features = temporal_features[..., self.calendar_feature_dim :]
            weather_embedding = self.weather_feature_mlp(weather_features)
            weather_gate = torch.sigmoid(self.weather_feature_gate_logit).to(dtype=x.dtype)
            fused = fused + weather_gate * weather_embedding
        return fused

    def _init_location_features(
        self,
        embed_dim: int,
        location_feature_dim: int,
        location_feature_hidden_dim: int,
        dropout: float,
        location_gate_init: float = 0.1,
    ):
        self.location_feature_dim = int(location_feature_dim)
        self.location_feature_hidden_dim = int(location_feature_hidden_dim)
        if self.location_feature_dim <= 0:
            return
        if self.location_feature_hidden_dim <= 0:
            raise ValueError("location_feature_hidden_dim must be positive when location metadata is enabled")
        if not 0.0 < float(location_gate_init) < 1.0:
            raise ValueError("location_gate_init must be strictly between 0 and 1")
        self.location_feature_mlp = nn.Sequential(
            nn.Linear(self.location_feature_dim, self.location_feature_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.location_feature_hidden_dim, embed_dim),
        )
        gate_logit = math.log(float(location_gate_init) / (1.0 - float(location_gate_init)))
        self.location_gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
        self.location_output_norm = nn.LayerNorm(embed_dim)

    def _fuse_location_features(
        self,
        representation: torch.Tensor,
        location_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if getattr(self, "location_feature_dim", 0) <= 0:
            return representation
        if location_features is None:
            location_features = torch.zeros(
                representation.size(0),
                self.location_feature_dim,
                dtype=representation.dtype,
                device=representation.device,
            )
        else:
            location_features = location_features.to(
                device=representation.device,
                dtype=representation.dtype,
            )
        if location_features.ndim != 2 or location_features.shape != (
            representation.size(0),
            self.location_feature_dim,
        ):
            raise ValueError(
                "location_features must have shape "
                f"[batch, {self.location_feature_dim}], got {tuple(location_features.shape)}"
            )
        location_embedding = self.location_feature_mlp(location_features)
        gate = torch.sigmoid(self.location_gate_logit).to(dtype=representation.dtype)
        return self.location_output_norm(representation + gate * location_embedding)

    def _prepare_temporal(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        if self.temporal_aggregation != "cls":
            return x, mask
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        if mask is not None:
            cls_mask = torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask.bool()], dim=1)
        return x, mask

    def _aggregate_temporal(self, x: torch.Tensor, mask: Optional[torch.Tensor], target_index: int):
        if self.temporal_aggregation == "cls":
            return x[:, 0]
        if self.temporal_aggregation == "mean":
            if mask is None:
                return x.mean(dim=1)
            weights = mask.bool().float().unsqueeze(-1)
            return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return x[:, target_index]

    def _init_temporal_model(
        self,
        embed_dim: int,
        temporal_layers: int,
        temporal_heads: int,
        dropout: float,
        temporal_model: str = "transformer",
        norm_first: bool = False,
        ffn_multiplier: float = 4.0,
    ):
        if temporal_model not in {"transformer", "lstm", "gru"}:
            raise ValueError("temporal_model must be one of: transformer, lstm, gru")
        self.temporal_model = temporal_model
        if temporal_model in {"lstm", "gru"} and self.temporal_aggregation == "cls":
            print(
                f"Warning: temporal_aggregation='cls' is Transformer-specific; using 'target' for {temporal_model}.",
                flush=True,
            )
            self.temporal_aggregation = "target"
        self.pos = PositionalEncoding(embed_dim)
        if temporal_model == "transformer":
            if ffn_multiplier <= 0:
                raise ValueError("ffn_multiplier must be positive")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=temporal_heads,
                dim_feedforward=max(embed_dim, int(round(embed_dim * ffn_multiplier))),
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=norm_first,
            )
            self.temporal = nn.TransformerEncoder(
                encoder_layer,
                num_layers=temporal_layers,
                enable_nested_tensor=not norm_first,
            )
            return
        rnn_cls = nn.LSTM if temporal_model == "lstm" else nn.GRU
        self.temporal = rnn_cls(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=temporal_layers,
            dropout=dropout if temporal_layers > 1 else 0.0,
            batch_first=True,
        )

    def _run_temporal_model(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.pos(x)
        x, mask = self._prepare_temporal(x, mask)
        if self.temporal_model == "transformer":
            key_padding_mask = None if mask is None else ~mask.bool()
            return self.temporal(x, src_key_padding_mask=key_padding_mask), mask

        if mask is not None:
            x = x * mask.bool().unsqueeze(-1).to(dtype=x.dtype)
        output, _ = self.temporal(x)
        return output, mask


class MultiScaleTemporalTransformer(nn.Module, TemporalAggregationMixin):
    def __init__(
        self,
        num_classes: int = len(BASE_CLASSES),
        image_backbone: str = DINO_DEFAULT_BACKBONE,
        embed_dim: int = 512,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        dropout: float = 0.1,
        pretrained: bool = True,
        target_index: Optional[int] = None,
        temporal_aggregation: str = "target",
        temporal_model: str = "transformer",
        temporal_feature_dim: int = 0,
        temporal_feature_hidden_dim: int = 0,
        weather_feature_dim: int = 0,
        temporal_feature_fusion: str = "gated",
        temporal_feature_gate_init: float = 0.1,
        weather_feature_gate_init: float = 0.1,
        location_feature_dim: int = 0,
        location_feature_hidden_dim: int = 16,
        location_gate_init: float = 0.1,
        temporal_norm_first: bool = False,
        temporal_ffn_multiplier: float = 4.0,
    ):
        super().__init__()
        self.target_index = target_index
        self._init_temporal_aggregation(embed_dim, temporal_aggregation)
        self._init_temporal_features(
            embed_dim,
            temporal_feature_dim,
            dropout,
            temporal_feature_hidden_dim,
            weather_feature_dim,
            temporal_feature_fusion,
            temporal_feature_gate_init,
            weather_feature_gate_init,
        )
        self._init_location_features(
            embed_dim,
            location_feature_dim,
            location_feature_hidden_dim,
            dropout,
            location_gate_init,
        )
        self.macro_encoder = ViTFeatureEncoder(image_backbone, pretrained=pretrained, out_dim=embed_dim)
        self.micro_encoder = ViTFeatureEncoder(image_backbone, pretrained=pretrained, out_dim=embed_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._init_temporal_model(embed_dim, temporal_layers, temporal_heads, dropout, temporal_model, temporal_norm_first, temporal_ffn_multiplier)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(
        self,
        macro: torch.Tensor,
        micro: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        location_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, days, channels, height, width = macro.shape
        macro_flat = macro.reshape(bsz * days, channels, height, width)
        micro_flat = micro.reshape(bsz * days, channels, height, width)
        macro_feat = self.macro_encoder(macro_flat).reshape(bsz, days, -1)
        micro_feat = self.micro_encoder(micro_flat).reshape(bsz, days, -1)
        x = self.fusion(torch.cat([macro_feat, micro_feat], dim=-1))
        x = self._add_temporal_features(x, temporal_features)
        x, mask = self._run_temporal_model(x, mask)
        target_index = days // 2 if self.target_index is None else self.target_index
        representation = self._aggregate_temporal(x, mask, target_index)
        representation = self._fuse_location_features(representation, location_features)
        return self.classifier(representation)


class SingleStreamTemporalTransformer(nn.Module, TemporalAggregationMixin):
    def __init__(
        self,
        stream: str = "micro",
        num_classes: int = len(BASE_CLASSES),
        image_backbone: str = DINO_DEFAULT_BACKBONE,
        embed_dim: int = 512,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        dropout: float = 0.1,
        pretrained: bool = True,
        target_index: Optional[int] = None,
        temporal_aggregation: str = "target",
        temporal_model: str = "transformer",
        temporal_feature_dim: int = 0,
        temporal_feature_hidden_dim: int = 0,
        weather_feature_dim: int = 0,
        temporal_feature_fusion: str = "gated",
        temporal_feature_gate_init: float = 0.1,
        weather_feature_gate_init: float = 0.1,
        location_feature_dim: int = 0,
        location_feature_hidden_dim: int = 16,
        location_gate_init: float = 0.1,
        temporal_norm_first: bool = False,
        temporal_ffn_multiplier: float = 4.0,
    ):
        super().__init__()
        if stream not in {"macro", "micro"}:
            raise ValueError("SingleStreamTemporalTransformer stream must be 'macro' or 'micro'")
        self.stream = stream
        self.target_index = target_index
        self._init_temporal_aggregation(embed_dim, temporal_aggregation)
        self._init_temporal_features(
            embed_dim,
            temporal_feature_dim,
            dropout,
            temporal_feature_hidden_dim,
            weather_feature_dim,
            temporal_feature_fusion,
            temporal_feature_gate_init,
            weather_feature_gate_init,
        )
        self._init_location_features(
            embed_dim,
            location_feature_dim,
            location_feature_hidden_dim,
            dropout,
            location_gate_init,
        )
        self.encoder = ViTFeatureEncoder(image_backbone, pretrained=pretrained, out_dim=embed_dim)
        self._init_temporal_model(embed_dim, temporal_layers, temporal_heads, dropout, temporal_model, temporal_norm_first, temporal_ffn_multiplier)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(
        self,
        macro: torch.Tensor,
        micro: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        location_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        frames = micro if self.stream == "micro" else macro
        bsz, days, channels, height, width = frames.shape
        flat = frames.reshape(bsz * days, channels, height, width)
        x = self.encoder(flat).reshape(bsz, days, -1)
        x = self._add_temporal_features(x, temporal_features)
        x, mask = self._run_temporal_model(x, mask)
        target_index = days // 2 if self.target_index is None else self.target_index
        representation = self._aggregate_temporal(x, mask, target_index)
        representation = self._fuse_location_features(representation, location_features)
        return self.classifier(representation)


class MultiScaleEmbeddingTemporalTransformer(nn.Module, TemporalAggregationMixin):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = len(BASE_CLASSES),
        embed_dim: int = 512,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        dropout: float = 0.1,
        target_index: Optional[int] = None,
        temporal_aggregation: str = "target",
        temporal_model: str = "transformer",
        temporal_feature_dim: int = 0,
        temporal_feature_hidden_dim: int = 0,
        weather_feature_dim: int = 0,
        temporal_feature_fusion: str = "gated",
        temporal_feature_gate_init: float = 0.1,
        weather_feature_gate_init: float = 0.1,
        location_feature_dim: int = 0,
        location_feature_hidden_dim: int = 16,
        location_gate_init: float = 0.1,
        temporal_norm_first: bool = False,
        temporal_ffn_multiplier: float = 4.0,
        dense_tokens_per_tile: int = 0,
    ):
        super().__init__()
        self.target_index = target_index
        self._init_temporal_aggregation(embed_dim, temporal_aggregation)
        self._init_temporal_features(
            embed_dim,
            temporal_feature_dim,
            dropout,
            temporal_feature_hidden_dim,
            weather_feature_dim,
            temporal_feature_fusion,
            temporal_feature_gate_init,
            weather_feature_gate_init,
        )
        self._init_location_features(
            embed_dim,
            location_feature_dim,
            location_feature_hidden_dim,
            dropout,
            location_gate_init,
        )
        self.tile_pool = (
            HierarchicalDenseTilePooler(feature_dim, dense_tokens_per_tile)
            if dense_tokens_per_tile > 0
            else TileAttentionPooler(feature_dim)
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._init_temporal_model(embed_dim, temporal_layers, temporal_heads, dropout, temporal_model, temporal_norm_first, temporal_ffn_multiplier)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(
        self,
        macro: torch.Tensor,
        micro: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        macro_tile_mask: Optional[torch.Tensor] = None,
        micro_tile_mask: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        location_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        macro = self.tile_pool(macro, macro_tile_mask)
        micro = self.tile_pool(micro, micro_tile_mask)
        x = self.fusion(torch.cat([macro, micro], dim=-1))
        x = self._add_temporal_features(x, temporal_features)
        x, mask = self._run_temporal_model(x, mask)
        target_index = x.size(1) // 2 if self.target_index is None else self.target_index
        representation = self._aggregate_temporal(x, mask, target_index)
        representation = self._fuse_location_features(representation, location_features)
        return self.classifier(representation)


class SingleStreamEmbeddingTemporalTransformer(nn.Module, TemporalAggregationMixin):
    def __init__(
        self,
        feature_dim: int,
        stream: str = "micro",
        num_classes: int = len(BASE_CLASSES),
        embed_dim: int = 512,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        dropout: float = 0.1,
        target_index: Optional[int] = None,
        temporal_aggregation: str = "target",
        temporal_model: str = "transformer",
        temporal_feature_dim: int = 0,
        temporal_feature_hidden_dim: int = 0,
        weather_feature_dim: int = 0,
        temporal_feature_fusion: str = "gated",
        temporal_feature_gate_init: float = 0.1,
        weather_feature_gate_init: float = 0.1,
        location_feature_dim: int = 0,
        location_feature_hidden_dim: int = 16,
        location_gate_init: float = 0.1,
        temporal_norm_first: bool = False,
        temporal_ffn_multiplier: float = 4.0,
        dense_tokens_per_tile: int = 0,
    ):
        super().__init__()
        if stream not in {"macro", "micro"}:
            raise ValueError("SingleStreamEmbeddingTemporalTransformer stream must be 'macro' or 'micro'")
        self.stream = stream
        self.target_index = target_index
        self._init_temporal_aggregation(embed_dim, temporal_aggregation)
        self._init_temporal_features(
            embed_dim,
            temporal_feature_dim,
            dropout,
            temporal_feature_hidden_dim,
            weather_feature_dim,
            temporal_feature_fusion,
            temporal_feature_gate_init,
            weather_feature_gate_init,
        )
        self._init_location_features(
            embed_dim,
            location_feature_dim,
            location_feature_hidden_dim,
            dropout,
            location_gate_init,
        )
        self.tile_pool = (
            HierarchicalDenseTilePooler(feature_dim, dense_tokens_per_tile)
            if dense_tokens_per_tile > 0
            else TileAttentionPooler(feature_dim)
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._init_temporal_model(embed_dim, temporal_layers, temporal_heads, dropout, temporal_model, temporal_norm_first, temporal_ffn_multiplier)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(
        self,
        macro: torch.Tensor,
        micro: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        macro_tile_mask: Optional[torch.Tensor] = None,
        micro_tile_mask: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        location_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = micro if self.stream == "micro" else macro
        tile_mask = micro_tile_mask if self.stream == "micro" else macro_tile_mask
        features = self.tile_pool(features, tile_mask)
        x = self.proj(features)
        x = self._add_temporal_features(x, temporal_features)
        x, mask = self._run_temporal_model(x, mask)
        target_index = x.size(1) // 2 if self.target_index is None else self.target_index
        representation = self._aggregate_temporal(x, mask, target_index)
        representation = self._fuse_location_features(representation, location_features)
        return self.classifier(representation)


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal CDF loss for ordered phenology classes.

    The model still predicts one logit per class. Comparing cumulative
    distributions penalizes far-away stage errors more than neighboring errors.
    """

    def __init__(self, power: int = 2):
        super().__init__()
        if power not in {1, 2}:
            raise ValueError("OrdinalRegressionLoss power must be 1 or 2")
        self.power = power

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        # There are C-1 meaningful ordinal thresholds for C classes. The final
        # CDF entry is always one and otherwise only dilutes the loss scale.
        pred_cdf = torch.cumsum(probs, dim=-1)[..., :-1]
        target_cdf = torch.cumsum(target, dim=-1)[..., :-1]
        diff = torch.abs(pred_cdf - target_cdf)
        if self.power == 2:
            diff = diff.square()
        return diff.mean()


class HybridOrdinalLoss(nn.Module):
    """Blend exact soft-label classification with ordinal distance."""

    def __init__(self, power: int = 2, cross_entropy_weight: float = 0.5):
        super().__init__()
        if not 0.0 <= cross_entropy_weight <= 1.0:
            raise ValueError("cross_entropy_weight must be in [0, 1]")
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.soft_ce = SoftTargetCrossEntropy()
        self.ordinal = OrdinalRegressionLoss(power=power)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.soft_ce(logits, target)
        ordinal = self.ordinal(logits, target)
        return self.cross_entropy_weight * ce + (1.0 - self.cross_entropy_weight) * ordinal
