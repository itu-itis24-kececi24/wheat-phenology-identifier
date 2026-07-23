import ast
import itertools
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover - imported in notebook/runtime environments
    T = None


DATE_RE = re.compile(r"(\d{4})[_-](\d{2})[_-](\d{2})|(\d{4})(\d{2})(\d{2})")
CURRENT_DATA_RE = re.compile(
    r"(?P<station>\d{2}[_.,]\d{2})-(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DAYS_SINCE_PLANTING_FEATURE = "days_since_planting"
DEFAULT_TEMPORAL_FEATURE_COLUMNS = (DAYS_SINCE_PLANTING_FEATURE,)
DAYS_SINCE_PLANTING_SCALE = 365.0

BASE_CLASSES = [
    "OffSeason",
    "PS1",
    "PS2",
    "PS3",
    "PS4",
    "PS5",
    "PS6",
    "PS7",
]

# Shifted labeling: keep 1-Ekim as planting metadata, but start visible
# phenology classes at 2-Cikis. Therefore:
# PS1 = 2-Cikis -> 3-Cimlenme, ..., PS7 = 8-Olgunlasma -> 9-Hasat.
PHENOLOGY_BOUNDARY_OFFSET = 1

STAGE_COLUMNS = [
    "1-Ekim",
    "2 - Çıkış",
    "3 - Çimlenme",
    "4 - Kardeşlenme",
    "5 - Sapa Kalkma",
    "6 - Başaklanma",
    "7 - Çiçeklenme",
    "8 - Olgunlaşma",
    "9 - Hasat",
]

STAGE_COLUMN_ALIASES = {
    "1-Ekim": ["1-Ekim"],
    "2 - Çıkış": ["2 - Cikis", "2 - Çıkış"],
    "3 - Çimlenme": ["3 - Cimlenme", "3 - Çimlenme"],
    "4 - Kardeşlenme": ["4 - Kardeslenme", "4 - Kardeşlenme"],
    "5 - Sapa Kalkma": ["5 - Sapa Kalkma"],
    "6 - Başaklanma": ["6 - Basaklanma", "6 - Başaklanma"],
    "7 - Çiçeklenme": ["7 - Ciceklenme", "7 - Çiçeklenme"],
    "8 - Olgunlaşma": ["8 - Olgunlasma", "8 - Olgunlaşma"],
    "9 - Hasat": ["9 - Hasat"],
}


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


def _local_status_path(path_from_csv: object, year_path: str) -> Optional[str]:
    if not isinstance(path_from_csv, str) or not path_from_csv:
        return None
    parts = re.split(r"[\\/]+", path_from_csv)
    for i, part in enumerate(parts):
        if re.fullmatch(r"K\d+", part, flags=re.IGNORECASE) and i + 2 < len(parts):
            candidate = os.path.join(year_path, *parts[i:])
            return candidate if os.path.isfile(candidate) else None
    candidate = os.path.join(year_path, os.path.basename(path_from_csv))
    return candidate if os.path.isfile(candidate) else None


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
    stage_labels = [label for label in class_to_idx if label.startswith("PS")]
    stage_labels = sorted(stage_labels, key=lambda label: int(label.replace("PS", "")))
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
        blend = 0.5 * (1.0 - delta / max(transition_days, 1))
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
    stage_labels = [label for label in class_to_idx if label.startswith("PS")]
    stage_labels = sorted(stage_labels, key=lambda label: int(label.replace("PS", "")))
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
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    groups = sorted(list(pd.unique(meta_df[group_col])))
    if n_train + n_val + n_test > len(groups):
        raise ValueError(
            f"Not enough groups ({len(groups)}) for "
            f"n_train={n_train} + n_val={n_val} + n_test={n_test}"
        )

    all_splits = []
    for test_groups in itertools.combinations(groups, n_test):
        remaining = [group for group in groups if group not in set(test_groups)]
        for val_groups in itertools.combinations(remaining, n_val):
            val_set = set(val_groups)
            train_groups = [group for group in remaining if group not in val_set]
            all_splits.append((train_groups, list(val_groups), list(test_groups)))

    if num_folds is not None and num_folds < len(all_splits):
        rng = random.Random(random_state)
        chosen = rng.sample(all_splits, num_folds)
    else:
        chosen = all_splits

    folds = []
    for train_groups, val_groups, test_groups in chosen:
        train_idx = meta_df.index[meta_df[group_col].isin(train_groups)].to_numpy()
        val_idx = meta_df.index[meta_df[group_col].isin(val_groups)].to_numpy()
        test_idx = meta_df.index[meta_df[group_col].isin(test_groups)].to_numpy()
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

        by_date: Dict[pd.Timestamp, Dict[str, str]] = {}
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
            by_date.setdefault(image_date, {})[scale] = image_path

        for date, paths in sorted(by_date.items()):
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


@dataclass
class WindowConfig:
    window_days: int = 31
    center_offset: Optional[int] = None
    require_center_image: bool = True
    classes: Tuple[str, ...] = tuple(BASE_CLASSES)
    stream: str = "micro"
    temporal_feature_columns: Tuple[str, ...] = ()

    @property
    def center(self) -> int:
        return self.window_days // 2 if self.center_offset is None else self.center_offset

    @property
    def temporal_feature_dim(self) -> int:
        return len(self.temporal_feature_columns)


def build_image_transform(
    image_size: int = 224,
    train: bool = False,
    augment: bool = False,
    crop_scale_min: float = 0.75,
    hflip_prob: float = 0.5,
    rotation_degrees: float = 5.0,
    color_jitter: float = 0.2,
    blur_prob: float = 0.1,
    erasing_prob: float = 0.1,
):
    if T is None:
        raise ImportError("torchvision is required for image transforms")

    image_size = int(image_size)
    if train and augment:
        crop_scale_min = min(max(float(crop_scale_min), 0.05), 1.0)
        pil_transforms = [
            T.RandomResizedCrop(
                (image_size, image_size),
                scale=(crop_scale_min, 1.0),
                ratio=(0.9, 1.1),
            )
        ]
        if hflip_prob > 0:
            pil_transforms.append(T.RandomHorizontalFlip(p=float(hflip_prob)))
        if rotation_degrees > 0:
            pil_transforms.append(T.RandomRotation(degrees=float(rotation_degrees)))
        if color_jitter > 0:
            jitter = float(color_jitter)
            pil_transforms.append(
                T.ColorJitter(
                    brightness=jitter,
                    contrast=jitter,
                    saturation=min(jitter, 0.4),
                    hue=min(jitter * 0.25, 0.05),
                )
            )
        if blur_prob > 0:
            pil_transforms.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                    p=float(blur_prob),
                )
            )
    else:
        pil_transforms = [T.Resize((image_size, image_size))]

    tensor_transforms = [
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if train and augment and erasing_prob > 0:
        tensor_transforms.append(
            T.RandomErasing(
                p=float(erasing_prob),
                scale=(0.01, 0.08),
                ratio=(0.3, 3.3),
                value="random",
            )
        )
    return T.Compose([*pil_transforms, *tensor_transforms])


class MultiScaleWindowDataset(Dataset):
    def __init__(
        self,
        daily_df: pd.DataFrame,
        config: WindowConfig,
        transform=None,
        image_size: int = 224,
        synchronized_transform: bool = False,
        fallback_to_nearest: bool = True,
    ):
        if T is None and transform is None:
            raise ImportError("torchvision is required when transform is not provided")
        self.df = daily_df.copy()
        self.config = config
        self.image_size = int(image_size)
        self.synchronized_transform = synchronized_transform
        self.fallback_to_nearest = fallback_to_nearest
        self.transform = transform or T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.empty_image = torch.zeros(3, self.image_size, self.image_size)
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.normalize()
        self.df = self.df.sort_values(["station_year", "date"]).reset_index(drop=True)
        self.groups = {k: g.set_index("date") for k, g in self.df.groupby("station_year")}
        self.samples = self._build_samples()

    def _has_required_stream(self, row: pd.Series) -> bool:
        if self.config.stream == "micro":
            return pd.notna(row.get("micro_path"))
        if self.config.stream == "macro":
            return pd.notna(row.get("macro_path"))
        return pd.notna(row.get("macro_path")) and pd.notna(row.get("micro_path"))

    def _build_samples(self) -> List[Tuple[str, pd.Timestamp]]:
        samples = []
        for station_year, group in self.groups.items():
            for date, row in group.iterrows():
                if self.config.require_center_image and not self._has_required_stream(row):
                    continue
                samples.append((station_year, pd.Timestamp(date)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Optional[str], transform_seed: Optional[int] = None) -> Tuple[torch.Tensor, bool]:
        if path is None or pd.isna(path):
            return self.empty_image.clone(), False
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if transform_seed is None:
                    return self.transform(img), True
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(transform_seed))
                    return self.transform(img), True
        except (OSError, UnidentifiedImageError) as exc:
            print(
                f"Warning: could not load image {path}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return self.empty_image.clone(), False

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
        group = self.groups[station_year]
        start = center_date - pd.Timedelta(days=self.config.center)
        planting_date = group.iloc[0].get("planting_date") if "planting_date" in group.columns else None
        transform_seed = (
            int(torch.randint(0, 2**31 - 1, (1,)).item())
            if self.synchronized_transform
            else None
        )

        macro_frames, micro_frames, mask, temporal_features = [], [], [], []
        for step in range(self.config.window_days):
            current_date = start + pd.Timedelta(days=step)
            row, present = self._row_for_date(group, current_date)
            macro_path = None if row is None else row.get("macro_path")
            micro_path = None if row is None else row.get("micro_path")
            macro_frame, macro_ok = self._load_image(
                macro_path if self.config.stream in {"macro", "both"} else None,
                transform_seed=transform_seed,
            )
            micro_frame, micro_ok = self._load_image(
                micro_path if self.config.stream in {"micro", "both"} else None,
                transform_seed=transform_seed,
            )
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
            if self.config.temporal_feature_columns:
                temporal_features.append(
                    _temporal_features_for_date(
                        current_date,
                        planting_date,
                        feature_columns=self.config.temporal_feature_columns,
                    )
                )

        center_row = group.loc[center_date]
        if isinstance(center_row, pd.DataFrame):
            center_row = center_row.iloc[0]
        target = torch.tensor(center_row["target"], dtype=torch.float32)
        date_score = _date_score_to_tensor(center_row)
        label = int(torch.argmax(target).item())

        return {
            "macro": torch.stack(macro_frames),
            "micro": torch.stack(micro_frames),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target": target,
            "date_score": date_score,
            "label": torch.tensor(label, dtype=torch.long),
            "station_year": station_year,
            "date": str(center_date.date()),
            "temporal_features": (
                torch.stack(temporal_features)
                if temporal_features
                else torch.zeros(self.config.window_days, 0, dtype=torch.float32)
            ),
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


def _days_since_planting_value(date: pd.Timestamp, planting_date: object) -> float:
    if planting_date is None or pd.isna(planting_date):
        return 0.0
    try:
        planting = pd.Timestamp(planting_date).normalize()
        current = pd.Timestamp(date).normalize()
    except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
        return 0.0
    return float((current - planting).days) / DAYS_SINCE_PLANTING_SCALE


def _temporal_features_for_date(
    date: pd.Timestamp,
    planting_date: object,
    feature_columns: Sequence[str] = DEFAULT_TEMPORAL_FEATURE_COLUMNS,
) -> torch.Tensor:
    values = []
    for column in feature_columns:
        if column == DAYS_SINCE_PLANTING_FEATURE:
            values.append(_days_since_planting_value(date, planting_date))
        else:
            values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)


class Conv3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 24,
        feature_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        layers = []
        current = in_channels
        for idx, out_channels in enumerate(channels):
            layers.extend(
                [
                    nn.Conv3d(current, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.GELU(),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.GELU(),
                ]
            )
            layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            current = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(current, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, frames: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"Expected frames with shape [B, D, C, H, W], got {tuple(frames.shape)}")
        if mask is not None:
            frames = frames * mask.bool().float().view(mask.size(0), mask.size(1), 1, 1, 1)
        x = frames.permute(0, 2, 1, 3, 4).contiguous()
        x = self.features(x)
        x = self.pool(x)
        return self.proj(x)


class TemporalMetadataMixin:
    def _init_temporal_metadata(
        self,
        visual_dim: int,
        temporal_feature_dim: int,
        hidden_dim: int,
        dropout: float,
        target_index: Optional[int],
    ) -> int:
        self.temporal_feature_dim = int(temporal_feature_dim)
        self.target_index = target_index
        if self.temporal_feature_dim <= 0:
            self.temporal_metadata = None
            return visual_dim
        hidden_dim = max(1, int(hidden_dim))
        self.temporal_metadata = nn.Sequential(
            nn.LayerNorm(self.temporal_feature_dim),
            nn.Linear(self.temporal_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        return visual_dim + hidden_dim

    def _fuse_temporal_metadata(
        self,
        visual_feature: torch.Tensor,
        temporal_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.temporal_metadata is None:
            return visual_feature
        if temporal_features is None:
            temporal_features = torch.zeros(
                visual_feature.size(0),
                1,
                self.temporal_feature_dim,
                device=visual_feature.device,
                dtype=visual_feature.dtype,
            )
        temporal_features = temporal_features.to(device=visual_feature.device, dtype=visual_feature.dtype)
        if temporal_features.ndim != 3 or temporal_features.size(-1) != self.temporal_feature_dim:
            raise ValueError(
                "temporal_features must have shape "
                f"[batch, days, {self.temporal_feature_dim}], got {tuple(temporal_features.shape)}"
            )
        target_index = temporal_features.size(1) // 2 if self.target_index is None else self.target_index
        target_index = max(0, min(int(target_index), temporal_features.size(1) - 1))
        metadata_feature = self.temporal_metadata(temporal_features[:, target_index])
        return torch.cat([visual_feature, metadata_feature], dim=-1)


class SingleStream3DCNN(nn.Module, TemporalMetadataMixin):
    def __init__(
        self,
        stream: str = "micro",
        num_classes: int = len(BASE_CLASSES),
        base_channels: int = 24,
        feature_dim: int = 256,
        dropout: float = 0.25,
        temporal_feature_dim: int = 0,
        temporal_feature_hidden_dim: int = 32,
        target_index: Optional[int] = None,
    ):
        super().__init__()
        if stream not in {"macro", "micro"}:
            raise ValueError("SingleStream3DCNN stream must be 'macro' or 'micro'")
        self.stream = stream
        self.encoder = Conv3DEncoder(base_channels=base_channels, feature_dim=feature_dim, dropout=dropout)
        classifier_dim = self._init_temporal_metadata(
            feature_dim,
            temporal_feature_dim,
            temporal_feature_hidden_dim,
            dropout,
            target_index,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, num_classes),
        )

    def forward(
        self,
        macro: torch.Tensor,
        micro: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        frames = micro if self.stream == "micro" else macro
        visual = self.encoder(frames, mask)
        return self.classifier(self._fuse_temporal_metadata(visual, temporal_features))


class MultiScale3DCNN(nn.Module, TemporalMetadataMixin):
    def __init__(
        self,
        num_classes: int = len(BASE_CLASSES),
        base_channels: int = 24,
        feature_dim: int = 256,
        dropout: float = 0.25,
        temporal_feature_dim: int = 0,
        temporal_feature_hidden_dim: int = 32,
        target_index: Optional[int] = None,
    ):
        super().__init__()
        self.macro_encoder = Conv3DEncoder(base_channels=base_channels, feature_dim=feature_dim, dropout=dropout)
        self.micro_encoder = Conv3DEncoder(base_channels=base_channels, feature_dim=feature_dim, dropout=dropout)
        classifier_dim = self._init_temporal_metadata(
            feature_dim * 2,
            temporal_feature_dim,
            temporal_feature_hidden_dim,
            dropout,
            target_index,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self,
        macro: torch.Tensor,
        micro: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        macro_feat = self.macro_encoder(macro, mask)
        micro_feat = self.micro_encoder(micro, mask)
        visual = torch.cat([macro_feat, micro_feat], dim=-1)
        return self.classifier(self._fuse_temporal_metadata(visual, temporal_features))


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()
