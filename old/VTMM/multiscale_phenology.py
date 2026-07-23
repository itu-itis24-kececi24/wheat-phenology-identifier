import ast
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

try:
    from framework import WheatFramework
except ModuleNotFoundError:  # Allows `from VTMM.multiscale_phenology import ...`
    from .framework import WheatFramework

try:
    import torchvision.models as tvm
    import torchvision.transforms as T
except Exception:  # pragma: no cover - imported in notebook/runtime environments
    tvm = None
    T = None


DATE_RE = re.compile(r"(\d{4})[_-](\d{2})[_-](\d{2})|(\d{4})(\d{2})(\d{2})")
CURRENT_DATA_RE = re.compile(
    r"(?P<station>\d{2}[_.,]\d{2})-(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

BASE_CLASSES = [
    "OffSeason",
    "PS0",
    "PS1",
    "PS2",
    "PS3",
    "PS4",
    "PS5",
    "PS6",
    "PS7",
]

STAGE_COLUMNS = [
    "1-Ekim",
    "2 - Cikis",
    "3 - Cimlenme",
    "4 - Kardeslenme",
    "5 - Sapa Kalkma",
    "6 - Basaklanma",
    "7 - Ciceklenme",
    "8 - Olgunlasma",
    "9 - Hasat",
]

STAGE_COLUMN_ALIASES = {
    "1-Ekim": ["1-Ekim"],
    "2 - Cikis": ["2 - Cikis", "2 - Çıkış"],
    "3 - Cimlenme": ["3 - Cimlenme", "3 - Çimlenme"],
    "4 - Kardeslenme": ["4 - Kardeslenme", "4 - Kardeşlenme"],
    "5 - Sapa Kalkma": ["5 - Sapa Kalkma"],
    "6 - Basaklanma": ["6 - Basaklanma", "6 - Başaklanma"],
    "7 - Ciceklenme": ["7 - Ciceklenme", "7 - Çiçeklenme"],
    "8 - Olgunlasma": ["8 - Olgunlasma", "8 - Olgunlaşma"],
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
        except Exception:
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
    except Exception:
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


def _station_folder_variants(station_raw: object) -> List[str]:
    try:
        numeric = f"{float(station_raw):05.2f}"
    except Exception:
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
    except Exception:
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
    preferred_camera: Optional[str] = "K1",
    use_status_csv: bool = True,
) -> Iterable[str]:
    """
    Iterate the repository's current layout:

    data/<station>/<year>/K1|K2/1X|10X/<station>-YYYY_MM_DD-HH_MM-K*-*X.jpeg

    If a day_image_status CSV exists, valid rows from it are preferred because it
    already excludes corrupt files. Otherwise, the function falls back to scanning
    the year folder.
    """
    camera_filter = preferred_camera.upper() if preferred_camera else None
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
            for status_file in status_files:
                status_df = pd.read_csv(status_file)
                for _, row in status_df.iterrows():
                    if str(row.get("status", "")).lower() != "valid":
                        continue
                    camera = str(row.get("camera", "")).upper()
                    if camera_filter and camera != camera_filter:
                        continue
                    local_path = _local_status_path(row.get("valid_file"), year_path)
                    if local_path is not None:
                        yield local_path
            continue

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
) -> np.ndarray:
    target = np.zeros(len(class_to_idx), dtype=np.float32)
    planting, harvest = boundaries[0], boundaries[-1]

    if date < planting:
        if "OffSeason" in class_to_idx:
            target[class_to_idx["OffSeason"]] = 1.0
        else:
            target[class_to_idx["Dormancy"]] = 1.0
        return target
    if date > harvest:
        if "OffSeason" in class_to_idx:
            target[class_to_idx["OffSeason"]] = 1.0
        else:
            target[class_to_idx["PostHarvest"]] = 1.0
        return target

    hard_idx = None
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= date < boundaries[i + 1]:
            hard_idx = class_to_idx[f"PS{i}"]
            break
    if hard_idx is None:
        hard_idx = class_to_idx["PS7"]

    target[hard_idx] = 1.0
    if transition_days <= 0:
        return target

    # Blend neighboring classes near biological transition boundaries.
    for boundary_i in range(1, len(boundaries) - 1):
        delta = abs((date - boundaries[boundary_i]).days)
        if delta > transition_days:
            continue
        left = class_to_idx[f"PS{boundary_i - 1}"]
        right = class_to_idx[f"PS{boundary_i}"]
        blend = 0.5 * (1.0 - delta / max(transition_days, 1))
        target[:] = 0.0
        if date < boundaries[boundary_i]:
            target[left] = 1.0 - blend
            target[right] = blend
        else:
            target[left] = blend
            target[right] = 1.0 - blend
        return target

    return target


def build_multiscale_daily_dataframe(
    excel_path: str,
    root_dir: str,
    include_preplant_days: int = 30,
    include_postharvest_days: int = 30,
    transition_days: int = 2,
    classes: Sequence[str] = BASE_CLASSES,
    preferred_camera: Optional[str] = "K1",
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
    framework = WheatFramework(excel_path=excel_path, root_dir=root_dir)
    class_to_idx = {name: i for i, name in enumerate(classes)}

    df = framework._get_dates_from_excel()

    rows = []
    for _, record in df.iterrows():
        station_path, station_folder = framework.find_station_path(record.get("Station Code"))
        if station_path is None:
            continue

        boundaries = framework.get_stage_boundaries(record)
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
            preferred_camera=preferred_camera,
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
            hard = classes[int(np.argmax(soft))]
            rows.append(
                {
                    "station_year": f"{station_folder}_{record.get('Year')}",
                    "group_id": record.get("ID"),
                    "station_code": record.get("Station Code"),
                    "year": record.get("Year"),
                    "date": date,
                    "macro_path": paths.get("macro"),
                    "micro_path": paths.get("micro"),
                    "label": hard,
                    "target": soft.tolist(),
                }
            )

    return pd.DataFrame(rows)


@dataclass
class WindowConfig:
    window_days: int = 31
    center_offset: Optional[int] = None
    require_center_image: bool = True
    classes: Tuple[str, ...] = tuple(BASE_CLASSES)

    @property
    def center(self) -> int:
        return self.window_days // 2 if self.center_offset is None else self.center_offset


class MultiScaleWindowDataset(Dataset):
    def __init__(
        self,
        daily_df: pd.DataFrame,
        config: WindowConfig,
        transform=None,
        fallback_to_nearest: bool = True,
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
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Tuple[str, pd.Timestamp]]:
        samples = []
        for station_year, group in self.groups.items():
            for date, row in group.iterrows():
                if self.config.require_center_image and pd.isna(row.get("macro_path")) and pd.isna(row.get("micro_path")):
                    continue
                samples.append((station_year, pd.Timestamp(date)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Optional[str]) -> torch.Tensor:
        if path is None or pd.isna(path):
            return torch.zeros(3, 224, 224)
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def _row_for_date(self, group: pd.DataFrame, date: pd.Timestamp) -> Tuple[Optional[pd.Series], float]:
        if date in group.index:
            return group.loc[date], 1.0
        if not self.fallback_to_nearest or len(group.index) == 0:
            return None, 0.0
        nearest_pos = np.argmin(np.abs((group.index - date).days))
        nearest_date = group.index[int(nearest_pos)]
        return group.loc[nearest_date], 0.0

    def __getitem__(self, idx: int):
        station_year, center_date = self.samples[idx]
        group = self.groups[station_year]
        start = center_date - pd.Timedelta(days=self.config.center)

        macro_frames, micro_frames, mask = [], [], []
        for step in range(self.config.window_days):
            row, present = self._row_for_date(group, start + pd.Timedelta(days=step))
            macro_frames.append(self._load_image(None if row is None else row.get("macro_path")))
            micro_frames.append(self._load_image(None if row is None else row.get("micro_path")))
            mask.append(present)

        center_row = group.loc[center_date]
        target = torch.tensor(center_row["target"], dtype=torch.float32)
        label = int(torch.argmax(target).item())

        return {
            "macro": torch.stack(macro_frames),
            "micro": torch.stack(micro_frames),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target": target,
            "label": torch.tensor(label, dtype=torch.long),
            "station_year": station_year,
            "date": str(center_date.date()),
        }


def _path_key(path: Optional[str]) -> Optional[str]:
    if path is None or pd.isna(path):
        return None
    return os.path.abspath(os.path.normpath(str(path)))


def _target_to_tensor(value: object) -> torch.Tensor:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return torch.tensor(value, dtype=torch.float32)


class MultiScaleEmbeddingWindowDataset(Dataset):
    def __init__(
        self,
        daily_df: pd.DataFrame,
        config: WindowConfig,
        embedding_cache: Dict,
        fallback_to_nearest: bool = True,
    ):
        self.df = daily_df.copy()
        self.config = config
        self.fallback_to_nearest = fallback_to_nearest
        self.feature_dim = int(embedding_cache["feature_dim"])
        self.macro_embeddings = {
            _path_key(path): value.float()
            for path, value in embedding_cache.get("macro", {}).items()
        }
        self.micro_embeddings = {
            _path_key(path): value.float()
            for path, value in embedding_cache.get("micro", {}).items()
        }
        self.df["date"] = pd.to_datetime(self.df["date"]).dt.normalize()
        self.df = self.df.sort_values(["station_year", "date"]).reset_index(drop=True)
        self.groups = {k: g.set_index("date") for k, g in self.df.groupby("station_year")}
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Tuple[str, pd.Timestamp]]:
        samples = []
        for station_year, group in self.groups.items():
            for date, row in group.iterrows():
                if self.config.require_center_image and pd.isna(row.get("macro_path")) and pd.isna(row.get("micro_path")):
                    continue
                samples.append((station_year, pd.Timestamp(date)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _row_for_date(self, group: pd.DataFrame, date: pd.Timestamp) -> Tuple[Optional[pd.Series], float]:
        if date in group.index:
            return group.loc[date], 1.0
        if not self.fallback_to_nearest or len(group.index) == 0:
            return None, 0.0
        nearest_pos = np.argmin(np.abs((group.index - date).days))
        nearest_date = group.index[int(nearest_pos)]
        return group.loc[nearest_date], 0.0

    def _load_embedding(self, path: Optional[str], stream: str) -> torch.Tensor:
        key = _path_key(path)
        if key is None:
            return torch.zeros(self.feature_dim)
        table = self.macro_embeddings if stream == "macro" else self.micro_embeddings
        return table.get(key, torch.zeros(self.feature_dim))

    def __getitem__(self, idx: int):
        station_year, center_date = self.samples[idx]
        group = self.groups[station_year]
        start = center_date - pd.Timedelta(days=self.config.center)

        macro_features, micro_features, mask = [], [], []
        for step in range(self.config.window_days):
            row, present = self._row_for_date(group, start + pd.Timedelta(days=step))
            macro_features.append(self._load_embedding(None if row is None else row.get("macro_path"), "macro"))
            micro_features.append(self._load_embedding(None if row is None else row.get("micro_path"), "micro"))
            mask.append(present)

        center_row = group.loc[center_date]
        target = _target_to_tensor(center_row["target"])
        label = int(torch.argmax(target).item())

        return {
            "macro": torch.stack(macro_features),
            "micro": torch.stack(micro_features),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target": target,
            "label": torch.tensor(label, dtype=torch.long),
            "station_year": station_year,
            "date": str(center_date.date()),
        }


class ViTFeatureEncoder(nn.Module):
    def __init__(self, backbone: str = "vit_b_16", pretrained: bool = True, out_dim: int = 512):
        super().__init__()
        if tvm is None:
            raise ImportError("torchvision is required for ViTFeatureEncoder")
        weights = None
        if pretrained:
            weights_enum_name = f"{backbone.upper()}_Weights"
            weights_enum_name = weights_enum_name.replace("VIT_", "ViT_")
            weights_enum = getattr(tvm, weights_enum_name, None)
            weights = weights_enum.IMAGENET1K_V1 if weights_enum is not None else "IMAGENET1K_V1"
        base = getattr(tvm, backbone)(weights=weights)
        in_dim = base.heads.head.in_features
        base.heads.head = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class ViTBackboneFeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "vit_b_16", pretrained: bool = True):
        super().__init__()
        if tvm is None:
            raise ImportError("torchvision is required for ViTBackboneFeatureExtractor")
        weights = None
        if pretrained:
            weights_enum_name = f"{backbone.upper()}_Weights"
            weights_enum_name = weights_enum_name.replace("VIT_", "ViT_")
            weights_enum = getattr(tvm, weights_enum_name, None)
            weights = weights_enum.IMAGENET1K_V1 if weights_enum is not None else "IMAGENET1K_V1"
        base = getattr(tvm, backbone)(weights=weights)
        self.out_dim = base.heads.head.in_features
        base.heads.head = nn.Identity()
        self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


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


class MultiScaleTemporalTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = len(BASE_CLASSES),
        image_backbone: str = "vit_b_16",
        embed_dim: int = 512,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.macro_encoder = ViTFeatureEncoder(image_backbone, pretrained=pretrained, out_dim=embed_dim)
        self.micro_encoder = ViTFeatureEncoder(image_backbone, pretrained=pretrained, out_dim=embed_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=temporal_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.pos = PositionalEncoding(embed_dim)
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, macro: torch.Tensor, micro: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, days, channels, height, width = macro.shape
        macro_flat = macro.reshape(bsz * days, channels, height, width)
        micro_flat = micro.reshape(bsz * days, channels, height, width)
        macro_feat = self.macro_encoder(macro_flat).reshape(bsz, days, -1)
        micro_feat = self.micro_encoder(micro_flat).reshape(bsz, days, -1)
        x = self.fusion(torch.cat([macro_feat, micro_feat], dim=-1))
        x = self.pos(x)
        key_padding_mask = None if mask is None else ~mask.bool()
        x = self.temporal(x, src_key_padding_mask=key_padding_mask)
        center = days // 2
        return self.classifier(x[:, center])


class MultiScaleEmbeddingTemporalTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = len(BASE_CLASSES),
        embed_dim: int = 512,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=temporal_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.pos = PositionalEncoding(embed_dim)
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, macro: torch.Tensor, micro: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.fusion(torch.cat([macro, micro], dim=-1))
        x = self.pos(x)
        key_padding_mask = None if mask is None else ~mask.bool()
        x = self.temporal(x, src_key_padding_mask=key_padding_mask)
        center = x.size(1) // 2
        return self.classifier(x[:, center])


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()
