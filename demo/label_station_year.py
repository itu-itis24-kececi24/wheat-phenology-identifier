#!/usr/bin/env python3
"""Propose BBCH milestone dates for one station-year with a trained DINOv3 model.

Each image is encoded once. Daily causal-window predictions are decoded into a
non-regressing stage sequence and converted to proposed milestone dates. Outputs
are review artifacts; canonical label files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DINO_DIR = PROJECT_ROOT / "DINOv3_BBCH"
for import_path in (str(DINO_DIR), str(PROJECT_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import infer_cached_window as inference
from backbone_weights import infer_dinov3_config, load_backbone_weights, read_backbone_checkpoint
from multiscale_phenology import (
    LOCATION_ELEVATION_CENTER,
    LOCATION_ELEVATION_SCALE,
    LOCATION_LATITUDE_CENTER,
    LOCATION_LATITUDE_SCALE,
    LOCATION_LONGITUDE_CENTER,
    LOCATION_LONGITUDE_SCALE,
    ViTBackboneFeatureExtractor,
    station_location_features,
)
from precompute_multiscale_embeddings import path_key


MILESTONE_TO_CLASS = (
    ("1-Sowing", "BBCH0"),
    ("2 - Emergence", "BBCH1"),
    ("3 - Tillering", "BBCH2"),
    ("4 - Stem Elongation", "BBCH3"),
    ("5 - Heading", "BBCH5"),
    ("6 - Flowering", "BBCH6_7"),
    ("7 - Maturity", "BBCH8"),
    ("8 - Harvest", None),
)
DAILY_BASE_COLUMNS = [
    "date",
    "image_path",
    "available_window_images",
    "raw_prediction",
    "raw_confidence",
    "decoded_prediction",
    "decoded_confidence",
    "eligible_for_milestones",
]


def parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None or not str(value).strip():
        return None
    return pd.Timestamp(value).normalize()


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_signature(checkpoint_path: Path, backbone_weights: Path, checkpoint: dict) -> str:
    payload = {
        "version": 1,
        "checkpoint": file_identity(checkpoint_path),
        "backbone_weights": file_identity(backbone_weights),
        "tiling": checkpoint.get("tiling_config", {}),
        "preprocess": checkpoint.get("preprocess_config", {}),
        "dense": checkpoint.get("dense_feature_config", {}),
        "feature_dim": checkpoint.get("embedding_feature_dim"),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_image_dir(
    station_dir: Optional[Path],
    data_path: Path,
    station_code: Optional[str],
    year: Optional[str],
    camera: str,
) -> Path:
    if station_dir is None:
        if not station_code or not year:
            raise ValueError("Provide --station-dir, or provide --data-path, --station-code, and --year.")
        base = data_path / station_code / str(year)
    else:
        base = station_dir
        if year and (base / str(year)).is_dir():
            base = base / str(year)
    base = base.resolve()
    if base.name.upper() == "10X" and base.is_dir():
        return base
    if (base / "10X").is_dir():
        return (base / "10X").resolve()
    cameras = [camera] if camera != "AUTO" else ["K1", "K2"]
    for candidate_camera in cameras:
        candidate = base / candidate_camera / "10X"
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find a 10X folder under {base} for camera {camera}.")


def infer_station_code(explicit: Optional[str], image_dir: Path) -> Optional[str]:
    if explicit:
        return explicit.replace("_", ".").replace(",", ".")
    for part in reversed(image_dir.parts):
        if len(part) == 5 and part[2] in "._," and part[:2].isdigit() and part[3:].isdigit():
            return part.replace("_", ".").replace(",", ".")
    return None


def monotonic_viterbi_decode(
    probabilities: Sequence[Sequence[float]],
    dates: Sequence[pd.Timestamp],
    advance_penalty: float = 0.0,
) -> list[int]:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != len(dates):
        raise ValueError("Probabilities must have shape [dates, classes].")
    if probs.shape[0] == 0:
        return []
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    scores = np.full_like(log_probs, -np.inf)
    backpointers = np.full(probs.shape, -1, dtype=np.int64)
    scores[0] = log_probs[0]
    for time_index in range(1, probs.shape[0]):
        day_gap = max(1, int((pd.Timestamp(dates[time_index]) - pd.Timestamp(dates[time_index - 1])).days))
        for current_stage in range(probs.shape[1]):
            candidates = []
            for previous_stage in range(max(0, current_stage - day_gap), current_stage + 1):
                score = scores[time_index - 1, previous_stage]
                score -= float(advance_penalty) * (current_stage - previous_stage)
                candidates.append((score, previous_stage))
            best_score, best_previous = max(candidates, key=lambda item: item[0])
            scores[time_index, current_stage] = best_score + log_probs[time_index, current_stage]
            backpointers[time_index, current_stage] = best_previous
    decoded = [int(np.argmax(scores[-1]))]
    for time_index in range(probs.shape[0] - 1, 0, -1):
        decoded.append(int(backpointers[time_index, decoded[-1]]))
    return list(reversed(decoded))


def decode_with_sowing_anchor(
    probabilities: Sequence[Sequence[float]],
    dates: Sequence[pd.Timestamp],
    classes: Sequence[str],
    sowing_date: Optional[pd.Timestamp],
    advance_penalty: float = 0.0,
) -> list[int]:
    decode_probabilities = [np.asarray(values, dtype=np.float64).copy() for values in probabilities]
    decode_dates = [pd.Timestamp(value).normalize() for value in dates]
    drop_anchor = False
    if sowing_date is not None:
        sowing_date = pd.Timestamp(sowing_date).normalize()
        bbch0_index = list(classes).index("BBCH0")
        anchor = np.zeros(len(classes), dtype=np.float64)
        anchor[bbch0_index] = 1.0
        if decode_dates[0] == sowing_date:
            decode_probabilities[0] = anchor
        elif sowing_date < decode_dates[0]:
            decode_dates.insert(0, sowing_date)
            decode_probabilities.insert(0, anchor)
            drop_anchor = True
    decoded = monotonic_viterbi_decode(decode_probabilities, decode_dates, advance_penalty)
    return decoded[1:] if drop_anchor else decoded


def propose_milestones(
    daily_rows: Sequence[dict[str, Any]],
    classes: Sequence[str],
    sowing_date: Optional[pd.Timestamp] = None,
    harvest_date: Optional[pd.Timestamp] = None,
) -> list[dict[str, Any]]:
    class_to_index = {label: index for index, label in enumerate(classes)}
    eligible = [row for row in daily_rows if row.get("eligible_for_milestones")]
    proposals = []
    for milestone, stage_class in MILESTONE_TO_CLASS:
        if milestone == "1-Sowing" and sowing_date is not None:
            proposals.append(
                {
                    "milestone": milestone,
                    "stage_class": stage_class,
                    "proposed_date": str(sowing_date.date()),
                    "confidence": "",
                    "source": "user_supplied",
                    "status": "provided",
                    "previous_observation": "",
                    "first_supporting_observation": "",
                    "supporting_observations": "",
                }
            )
            continue
        if milestone == "8 - Harvest":
            proposals.append(
                {
                    "milestone": milestone,
                    "stage_class": "",
                    "proposed_date": str(harvest_date.date()) if harvest_date is not None else "",
                    "confidence": "",
                    "source": "user_supplied" if harvest_date is not None else "not_estimable_by_checkpoint",
                    "status": "provided" if harvest_date is not None else "needs_review",
                    "previous_observation": "",
                    "first_supporting_observation": "",
                    "supporting_observations": "",
                }
            )
            continue
        matches = [row for row in eligible if row.get("decoded_prediction") == stage_class]
        if not matches:
            proposals.append(
                {
                    "milestone": milestone,
                    "stage_class": stage_class or "",
                    "proposed_date": "",
                    "confidence": "",
                    "source": "model_transition",
                    "status": "stage_not_observed",
                    "previous_observation": "",
                    "first_supporting_observation": "",
                    "supporting_observations": 0,
                }
            )
            continue
        first = matches[0]
        first_index = next(index for index, row in enumerate(eligible) if row is first)
        previous_date = eligible[first_index - 1]["date"] if first_index > 0 else ""
        class_index = class_to_index[stage_class]
        confidence_rows = matches[: min(3, len(matches))]
        confidence = float(np.mean([row[f"probability_{stage_class}"] for row in confidence_rows]))
        source = "model_first_observed_bbch0" if milestone == "1-Sowing" else "model_transition"
        status = "heuristic_endpoint" if milestone == "1-Sowing" else "proposed"
        proposals.append(
            {
                "milestone": milestone,
                "stage_class": stage_class,
                "proposed_date": first["date"],
                "confidence": round(confidence, 6),
                "source": source,
                "status": status,
                "previous_observation": previous_date,
                "first_supporting_observation": first["date"],
                "supporting_observations": len(matches),
            }
        )
    return proposals


def build_label_row(
    proposals: Sequence[dict[str, Any]], station_code: str, year: str, camera: str
) -> dict[str, Any]:
    dates = {row["milestone"]: row["proposed_date"] for row in proposals}
    row = {"Station Code": station_code, "Year": year}
    row.update({milestone: dates.get(milestone, "") for milestone, _ in MILESTONE_TO_CLASS})
    row["kamera"] = camera[1:] if camera.upper() in {"K1", "K2"} else camera
    row["proposal_status"] = "REVIEW_REQUIRED"
    return row


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def custom_location_features(args, station_code: str, expected_dim: int, device: torch.device) -> torch.Tensor:
    supplied = [args.latitude, args.longitude, args.elevation]
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise ValueError("Provide --latitude, --longitude, and --elevation together.")
        features = torch.tensor(
            [
                (args.latitude - LOCATION_LATITUDE_CENTER) / LOCATION_LATITUDE_SCALE,
                (args.longitude - LOCATION_LONGITUDE_CENTER) / LOCATION_LONGITUDE_SCALE,
                (args.elevation - LOCATION_ELEVATION_CENTER) / LOCATION_ELEVATION_SCALE,
            ],
            dtype=torch.float32,
        )
    else:
        features = station_location_features(station_code, strict=True)
    if features.numel() != expected_dim:
        raise ValueError(f"Checkpoint expects {expected_dim} location features, got {features.numel()}.")
    return features.unsqueeze(0).to(device)


def load_feature_cache(cache_path: Path, signature: str, rebuild: bool) -> dict[str, Any]:
    if rebuild or not cache_path.is_file():
        return {"version": 1, "signature": signature, "images": {}}
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        print("Embedding cache configuration changed; rebuilding it.", flush=True)
        return {"version": 1, "signature": signature, "images": {}}
    payload.setdefault("images", {})
    return payload


def cached_feature(payload: dict[str, Any], image_path: str) -> Optional[torch.Tensor]:
    key = path_key(image_path)
    item = payload.get("images", {}).get(key)
    if not item:
        return None
    try:
        stat = Path(image_path).stat()
    except OSError:
        return None
    if item.get("size") != stat.st_size or item.get("mtime_ns") != stat.st_mtime_ns:
        return None
    return item.get("feature")


def store_feature(payload: dict[str, Any], image_path: str, feature: torch.Tensor) -> None:
    stat = Path(image_path).stat()
    payload["images"][path_key(image_path)] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "feature": feature.detach().cpu().to(torch.float16),
    }


def save_feature_cache(payload: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)


def encode_images(
    image_paths: Sequence[str],
    checkpoint: dict,
    checkpoint_path: Path,
    backbone_weights: Path,
    cache_path: Path,
    device: torch.device,
    tile_batch_size: int,
    rebuild_cache: bool,
) -> dict[str, torch.Tensor]:
    signature = cache_signature(checkpoint_path, backbone_weights, checkpoint)
    cache_payload = load_feature_cache(cache_path, signature, rebuild_cache)
    encoded = {}
    missing = []
    for image_path in image_paths:
        feature = cached_feature(cache_payload, image_path)
        if feature is None:
            missing.append(image_path)
        else:
            encoded[path_key(image_path)] = feature
    print(f"Embedding cache: {len(encoded)} reused, {len(missing)} to encode.", flush=True)
    if not missing:
        return encoded

    backbone_payload, weights_backbone = read_backbone_checkpoint(str(backbone_weights))
    local_config = infer_dinov3_config(backbone_payload)
    backbone_name = weights_backbone or checkpoint.get("image_backbone")
    encoder = ViTBackboneFeatureExtractor(
        backbone_name,
        pretrained=False,
        local_config=local_config,
    ).to(device)
    load_backbone_weights(encoder, str(backbone_weights), payload=backbone_payload)
    encoder.eval()
    tiling = checkpoint.get("tiling_config", {})
    dense = checkpoint.get("dense_feature_config", {})
    preprocess = checkpoint.get("preprocess_config", {})
    for index, image_path in enumerate(missing, 1):
        feature = inference.encode_one_image(
            encoder,
            image_path,
            device,
            tile_size=int(tiling.get("tile_size", 224)),
            tile_stride=int(tiling.get("tile_stride", 224)),
            max_tiles=int(tiling.get("max_tiles", 0)),
            image_size=int(tiling.get("vit_image_size", 224)),
            tile_pooling=tiling.get("tile_pooling", "attention"),
            tile_batch_size=tile_batch_size,
            mean=preprocess.get("mean"),
            std=preprocess.get("std"),
            dense_feature_config=dense,
        )
        store_feature(cache_payload, image_path, feature)
        encoded[path_key(image_path)] = feature.to(torch.float16)
        if index % 10 == 0 or index == len(missing):
            print(f"Encoded {index}/{len(missing)} new images.", flush=True)
        if index % 50 == 0:
            save_feature_cache(cache_payload, cache_path)
    save_feature_cache(cache_payload, cache_path)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return encoded


def predict_dates(
    by_date: dict[pd.Timestamp, str],
    target_dates: Sequence[pd.Timestamp],
    encoded: dict[str, torch.Tensor],
    checkpoint: dict,
    station_code: str,
    sowing_date: Optional[pd.Timestamp],
    weather_cache: Optional[str],
    location_features: torch.Tensor,
    device: torch.device,
    minimum_coverage: int,
) -> list[dict[str, Any]]:
    classes = list(checkpoint["classes"])
    window_days = int(checkpoint.get("window_days", 21))
    stream = checkpoint.get("stream", "micro")
    feature_dim = int(checkpoint.get("embedding_feature_dim", 768))
    dense = checkpoint.get("dense_feature_config", {})
    tiling = checkpoint.get("tiling_config", {})
    temporal_columns = inference.temporal_feature_columns_from_checkpoint(checkpoint)
    model = inference.build_model(checkpoint, feature_dim, classes, stream, device, dense)
    rows = []
    for index, target_date in enumerate(target_dates, 1):
        start_date = target_date - pd.Timedelta(days=window_days - 1)
        window_paths = [by_date.get(start_date + pd.Timedelta(days=offset)) for offset in range(window_days)]
        target_path = by_date[target_date]
        target_feature = encoded[path_key(target_path)]
        features, mask = [], []
        for image_path in window_paths:
            if image_path is None:
                features.append(torch.zeros_like(target_feature))
                mask.append(False)
            else:
                features.append(encoded[path_key(image_path)])
                mask.append(True)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
        temporal_features = inference.build_temporal_features(
            target_date,
            str(sowing_date.date()) if sowing_date is not None else None,
            window_days,
            temporal_columns,
            weather_cache=weather_cache,
            station_code=station_code,
            gdd_base_temp=float(checkpoint.get("gdd_base_temp", 0.0)),
        ).unsqueeze(0).to(device)
        if tiling.get("tile_pooling", "attention") == "attention":
            feature_tensor, tile_mask = inference.pad_tile_features(features, feature_dim)
            feature_tensor = feature_tensor.unsqueeze(0).to(device)
            tile_mask = tile_mask.unsqueeze(0).to(device)
            zeros = torch.zeros_like(feature_tensor)
            zero_mask = torch.zeros_like(tile_mask)
            macro = feature_tensor if stream in {"macro", "both"} else zeros
            micro = feature_tensor if stream in {"micro", "both"} else zeros
            macro_mask = tile_mask if stream in {"macro", "both"} else zero_mask
            micro_mask = tile_mask if stream in {"micro", "both"} else zero_mask
            with torch.inference_mode():
                logits = model(
                    macro,
                    micro,
                    mask_tensor,
                    macro_mask,
                    micro_mask,
                    temporal_features,
                    location_features,
                )
        else:
            feature_tensor = torch.stack(
                [feature.float() if feature.ndim == 1 else feature.float().mean(dim=0) for feature in features]
            ).unsqueeze(0).to(device)
            zeros = torch.zeros_like(feature_tensor)
            macro = feature_tensor if stream in {"macro", "both"} else zeros
            micro = feature_tensor if stream in {"micro", "both"} else zeros
            with torch.inference_mode():
                logits = model(
                    macro,
                    micro,
                    mask_tensor,
                    temporal_features=temporal_features,
                    location_features=location_features,
                )
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        raw_index = int(np.argmax(probabilities))
        row = {
            "date": str(target_date.date()),
            "image_path": str(Path(target_path).resolve()),
            "available_window_images": int(sum(mask)),
            "raw_prediction": classes[raw_index],
            "raw_confidence": round(float(probabilities[raw_index]), 6),
            "eligible_for_milestones": int(sum(mask) >= minimum_coverage),
            "_probabilities": probabilities,
        }
        for label, probability in zip(classes, probabilities):
            row[f"probability_{label}"] = round(float(probability), 6)
        rows.append(row)
        if index % 25 == 0 or index == len(target_dates):
            print(f"Predicted {index}/{len(target_dates)} dates.", flush=True)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Propose station-year BBCH milestone dates using the trained demo model."
    )
    parser.add_argument("--station-dir", type=Path, default=None, help="Station, station-year, camera, or 10X directory.")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--station-code", default=None, help="For example 02.03; inferred from the path when possible.")
    parser.add_argument("--year", default=None, help="Season folder name, such as 2015.")
    parser.add_argument("--camera", choices=("AUTO", "K1", "K2"), default="AUTO")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "demo" / "best_model.pt")
    parser.add_argument("--image-backbone-weights", type=Path, default=PROJECT_ROOT / "demo" / "best_finetune.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sowing-date", default=None, help="Known sowing date YYYY-MM-DD; strongly recommended.")
    parser.add_argument("--harvest-date", default=None, help="Optional known harvest date YYYY-MM-DD.")
    parser.add_argument("--start-date", default=None, help="First target date; defaults to sowing date or first image.")
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last target date; defaults to harvest date when supplied, otherwise the last image.",
    )
    parser.add_argument("--minimum-window-images", type=int, default=6)
    parser.add_argument("--advance-penalty", type=float, default=0.0)
    parser.add_argument("--tile-batch-size", type=int, default=32)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--weather-cache", default=None)
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    backbone_weights = args.image_backbone_weights.resolve()
    if not checkpoint_path.is_file() or not backbone_weights.is_file():
        raise FileNotFoundError("The temporal checkpoint and image-backbone weights must both exist.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not checkpoint.get("uses_embedding_cache"):
        raise ValueError("This command currently requires a cached-embedding temporal checkpoint.")
    if checkpoint.get("stream", "micro") != "micro":
        raise ValueError("This station-year command currently supports checkpoints trained on the micro/10X stream.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    image_dir = resolve_image_dir(args.station_dir, args.data_path, args.station_code, args.year, args.camera)
    station_code = infer_station_code(args.station_code, image_dir)
    if not station_code:
        raise ValueError("Could not infer the station code; provide --station-code.")
    year = str(args.year or image_dir.parents[1].name)
    camera = image_dir.parent.name
    by_date = inference.find_images_by_date(str(image_dir), stream="micro")
    if not by_date:
        raise ValueError(f"No dated 10X images found in {image_dir}.")

    sowing_date = parse_date(args.sowing_date)
    harvest_date = parse_date(args.harvest_date)
    start_date = parse_date(args.start_date) or sowing_date or min(by_date)
    end_date = parse_date(args.end_date) or harvest_date or max(by_date)
    if end_date < start_date:
        raise ValueError("--end-date must not be earlier than --start-date.")
    target_dates = sorted(date for date in by_date if start_date <= date <= end_date)
    if not target_dates:
        raise ValueError("No image dates fall inside the requested date range.")
    if sowing_date is None:
        print(
            "Warning: no --sowing-date supplied. Days-since-sowing metadata will be zero and the "
            "sowing proposal will only be a first-observed-BBCH0 heuristic.",
            flush=True,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (args.embedding_cache or output_dir / "image_embeddings.pt").resolve()
    context_start = start_date - pd.Timedelta(days=int(checkpoint.get("window_days", 21)) - 1)
    all_needed_paths = sorted(
        {path for date, path in by_date.items() if context_start <= date <= end_date}
    )
    encoded = encode_images(
        all_needed_paths,
        checkpoint,
        checkpoint_path,
        backbone_weights,
        cache_path,
        device,
        args.tile_batch_size,
        args.rebuild_cache,
    )
    location_dim = int(checkpoint.get("location_feature_dim", 0))
    if location_dim:
        location_features = custom_location_features(args, station_code, location_dim, device)
    else:
        location_features = torch.zeros(1, 0, device=device)
    rows = predict_dates(
        by_date,
        target_dates,
        encoded,
        checkpoint,
        station_code,
        sowing_date,
        args.weather_cache,
        location_features,
        device,
        args.minimum_window_images,
    )
    eligible_indices = [
        index
        for index, row in enumerate(rows)
        if row["eligible_for_milestones"]
        and (sowing_date is None or pd.Timestamp(row["date"]) >= sowing_date)
    ]
    if not eligible_indices:
        raise ValueError(
            "No prediction has enough window coverage for milestone decoding; lower --minimum-window-images."
        )
    eligible_probabilities = [rows[index]["_probabilities"] for index in eligible_indices]
    eligible_dates = [pd.Timestamp(rows[index]["date"]) for index in eligible_indices]
    classes = list(checkpoint["classes"])
    decoded = decode_with_sowing_anchor(
        eligible_probabilities,
        eligible_dates,
        classes,
        sowing_date,
        args.advance_penalty,
    )
    for row in rows:
        row["decoded_prediction"] = ""
        row["decoded_confidence"] = ""
    for row_index, decoded_index in zip(eligible_indices, decoded):
        rows[row_index]["decoded_prediction"] = classes[decoded_index]
        rows[row_index]["decoded_confidence"] = round(
            float(rows[row_index]["_probabilities"][decoded_index]), 6
        )
    for row in rows:
        row.pop("_probabilities", None)

    proposals = propose_milestones(rows, classes, sowing_date, harvest_date)
    label_row = build_label_row(proposals, station_code, year, camera)
    probability_columns = [f"probability_{label}" for label in classes]
    daily_columns = DAILY_BASE_COLUMNS + probability_columns
    write_csv(output_dir / "daily_predictions.csv", rows, daily_columns)
    write_csv(output_dir / "milestone_proposals.csv", proposals)
    write_csv(output_dir / "label_row_proposal.csv", [label_row])
    summary = {
        "status": "REVIEW_REQUIRED",
        "station_code": station_code,
        "year": year,
        "camera": camera,
        "image_dir": str(image_dir),
        "checkpoint": str(checkpoint_path),
        "image_backbone_weights": str(backbone_weights),
        "embedding_cache": str(cache_path),
        "device": str(device),
        "image_dates": len(by_date),
        "predicted_dates": len(rows),
        "milestone_eligible_dates": len(eligible_indices),
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "sowing_date_supplied": sowing_date is not None,
        "harvest_date_supplied": harvest_date is not None,
        "limitations": [
            "Proposals require human review and are not written to canonical label files.",
            "Without --sowing-date, BBCH0 cannot distinguish true sowing from earlier bare-field images.",
            "Harvest is not estimable because the checkpoint has no post-harvest class.",
        ],
        "proposals": proposals,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nProposed milestones (review required):")
    for proposal in proposals:
        value = proposal["proposed_date"] or "NOT ESTIMATED"
        print(f"  {proposal['milestone']}: {value} [{proposal['status']}]")
    print(f"\nOutputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
