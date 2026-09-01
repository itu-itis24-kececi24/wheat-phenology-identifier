#!/usr/bin/env python3
"""Audit 1X/10X folders using filenames and classical image registration.

The command is deliberately read-only: it writes a CSV report and never renames,
moves, or deletes an image.  Visual evidence comes from same-scale feature
registration against nearby, trusted images from both magnification streams.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
STREAMS = ("1X", "10X")
QUARANTINE_DIRECTORIES = {"bad_images", "truncated_images", "corrupt_images"}
DISCREPANCY_STATUSES = {
    "LIKELY_MISFILED",
    "LIKELY_FILENAME_ERROR",
    "FILENAME_FOLDER_MISMATCH",
    "MISSING_FILENAME_LABEL",
}
TIMESTAMP_RE = re.compile(
    r"-(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})-"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})(?:_|-)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"(?:^|[-_ ])(10X|1X)(?=$|[-_ .])", re.IGNORECASE)


@dataclass(frozen=True)
class ImageRecord:
    path: str
    relative_path: str
    station: str
    year: str
    camera: str
    folder_label: str
    filename_label: str
    timestamp: Optional[datetime]

    @property
    def group_key(self) -> tuple[str, str, str]:
        return self.station, self.year, self.camera

    @property
    def trusted_label(self) -> str:
        if self.folder_label == self.filename_label and self.folder_label in STREAMS:
            return self.folder_label
        return ""


@dataclass
class RegistrationResult:
    reference_path: str = ""
    score: float = 0.0
    scale: float = math.nan
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = math.nan


@dataclass
class AuditRow:
    path: str
    relative_path: str
    station: str
    year: str
    camera: str
    timestamp: str
    folder_label: str
    filename_label: str
    visual_label: str
    visual_confidence: float
    score_1x: float
    score_10x: float
    reference_1x: str
    reference_10x: str
    scale_to_1x: float
    scale_to_10x: float
    inliers_1x: int
    inliers_10x: int
    status: str
    reason: str


def parse_filename_label(filename: str) -> str:
    """Return the last standalone 1X/10X token in a filename."""
    matches = TOKEN_RE.findall(Path(filename).stem)
    return matches[-1].upper() if matches else ""


def parse_timestamp(filename: str) -> Optional[datetime]:
    match = TIMESTAMP_RE.search(Path(filename).name)
    if not match:
        return None
    try:
        return datetime(**{key: int(value) for key, value in match.groupdict().items()})
    except ValueError:
        return None


def record_from_path(path: Path, data_root: Path) -> Optional[ImageRecord]:
    try:
        relative = path.resolve().relative_to(data_root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    stream_index = next((i for i, part in enumerate(parts[:-1]) if part.upper() in STREAMS), None)
    if stream_index is None or stream_index < 3:
        return None
    return ImageRecord(
        path=str(path.resolve()),
        relative_path=str(relative),
        station=parts[stream_index - 3],
        year=parts[stream_index - 2],
        camera=parts[stream_index - 1],
        folder_label=parts[stream_index].upper(),
        filename_label=parse_filename_label(path.name),
        timestamp=parse_timestamp(path.name),
    )


def discover_images(data_root: Path) -> list[ImageRecord]:
    records = []
    for root, directories, filenames in os.walk(data_root, onerror=lambda _error: None):
        directories[:] = [
            name for name in directories if name.lower() not in QUARANTINE_DIRECTORIES
        ]
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            record = record_from_path(path, data_root)
            if record is not None:
                records.append(record)
    records.sort(key=lambda item: (item.group_key, item.timestamp or datetime.max, item.path))
    return records


def nearest_references(
    query: ImageRecord,
    candidates: Sequence[ImageRecord],
    count: int,
    max_days: float,
) -> list[ImageRecord]:
    if query.timestamp is None:
        return []
    ranked = []
    for candidate in candidates:
        if candidate.path == query.path or candidate.timestamp is None:
            continue
        distance = abs((candidate.timestamp - query.timestamp).total_seconds()) / 86400.0
        if distance <= max_days:
            ranked.append((distance, candidate.path, candidate))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:count]]


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for visual auditing. Install project dependencies with "
            "'.venv\\Scripts\\python.exe -m pip install -r requirements.txt', or use "
            "--filename-only."
        ) from exc
    return cv2


class FeatureCache:
    def __init__(self, method: str = "sift", max_side: int = 960, capacity: int = 512):
        self.cv2 = _load_cv2()
        self.method = method
        self.max_side = max_side
        self.capacity = capacity
        self.cache: OrderedDict[str, tuple[tuple[int, int], list, Optional[np.ndarray]]] = OrderedDict()
        if method == "sift":
            if not hasattr(self.cv2, "SIFT_create"):
                raise RuntimeError("This OpenCV build does not provide SIFT; use --feature-method akaze.")
            self.detector = self.cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.03)
            self.norm = self.cv2.NORM_L2
        else:
            self.detector = self.cv2.AKAZE_create()
            self.norm = self.cv2.NORM_HAMMING

    def get(self, path: str):
        cached = self.cache.pop(path, None)
        if cached is not None:
            self.cache[path] = cached
            return cached
        raw = np.fromfile(path, dtype=np.uint8)
        gray = self.cv2.imdecode(raw, self.cv2.IMREAD_GRAYSCALE)
        if gray is None:
            result = ((0, 0), [], None)
        else:
            height, width = gray.shape
            scale = min(1.0, self.max_side / max(height, width))
            if scale < 1.0:
                gray = self.cv2.resize(
                    gray,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=self.cv2.INTER_AREA,
                )
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            result = (gray.shape, keypoints, descriptors)
        self.cache[path] = result
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return result


def _normalised_points(keypoints, indices: Iterable[int], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return np.float32(
        [[keypoints[index].pt[0] / width, keypoints[index].pt[1] / height] for index in indices]
    )


def homography_scale(homography: np.ndarray, point: tuple[float, float] = (0.5, 0.5)) -> float:
    """Return the local isotropic scale implied by a projective homography."""
    x, y = point
    h = homography / homography[2, 2]
    denominator = h[2, 0] * x + h[2, 1] * y + 1.0
    if abs(denominator) < 1e-9:
        return math.nan
    numerator_x = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    numerator_y = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    jacobian = np.array(
        [
            [
                (h[0, 0] * denominator - numerator_x * h[2, 0]) / denominator**2,
                (h[0, 1] * denominator - numerator_x * h[2, 1]) / denominator**2,
            ],
            [
                (h[1, 0] * denominator - numerator_y * h[2, 0]) / denominator**2,
                (h[1, 1] * denominator - numerator_y * h[2, 1]) / denominator**2,
            ],
        ],
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(jacobian))
    return math.sqrt(abs(determinant)) if np.isfinite(determinant) else math.nan


def register_pair(
    query_path: str,
    reference_path: str,
    cache: FeatureCache,
    ratio_threshold: float = 0.75,
    ransac_threshold: float = 0.012,
    minimum_matches: int = 10,
) -> RegistrationResult:
    cv2 = cache.cv2
    query_shape, query_keypoints, query_descriptors = cache.get(query_path)
    ref_shape, ref_keypoints, ref_descriptors = cache.get(reference_path)
    if query_descriptors is None or ref_descriptors is None:
        return RegistrationResult(reference_path=reference_path)
    matcher = cv2.BFMatcher(cache.norm)
    pairs = matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < ratio_threshold * second.distance]
    if len(good) < minimum_matches:
        return RegistrationResult(reference_path=reference_path, good_matches=len(good))
    source = _normalised_points(query_keypoints, (match.queryIdx for match in good), query_shape)
    target = _normalised_points(ref_keypoints, (match.trainIdx for match in good), ref_shape)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, ransac_threshold)
    if homography is None or mask is None:
        return RegistrationResult(reference_path=reference_path, good_matches=len(good))
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)
    scale = homography_scale(homography)
    if inliers:
        projected = cv2.perspectiveTransform(source[inlier_mask, None, :], homography)[:, 0, :]
        reprojection_error = float(np.median(np.linalg.norm(projected - target[inlier_mask], axis=1)))
    else:
        reprojection_error = math.nan
    if not np.isfinite(scale) or scale <= 0 or inliers < 8:
        score = 0.0
    else:
        scale_penalty = math.exp(-abs(math.log2(scale)) / 0.65)
        count_support = min(1.0, inliers / 40.0)
        error_support = math.exp(-reprojection_error / 0.015)
        score = count_support * inlier_ratio * error_support * scale_penalty
    return RegistrationResult(
        reference_path=reference_path,
        score=float(score),
        scale=float(scale),
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=float(inlier_ratio),
        reprojection_error=reprojection_error,
    )


def best_registration(
    query: ImageRecord,
    references: Sequence[ImageRecord],
    cache: FeatureCache,
) -> RegistrationResult:
    results = [register_pair(query.path, reference.path, cache) for reference in references]
    return max(results, key=lambda result: (result.score, result.inliers), default=RegistrationResult())


def classify_scores(
    score_1x: float,
    score_10x: float,
    minimum_score: float,
    minimum_confidence: float,
) -> tuple[str, float]:
    total = score_1x + score_10x
    if max(score_1x, score_10x) < minimum_score or total <= 0:
        return "", 0.0
    label = "1X" if score_1x > score_10x else "10X"
    confidence = max(score_1x, score_10x) / total
    if confidence < minimum_confidence:
        return "", confidence
    return label, confidence


def decide_status(record: ImageRecord, visual_label: str) -> tuple[str, str]:
    folder = record.folder_label
    filename = record.filename_label
    if filename and filename != folder:
        if visual_label == filename:
            return "LIKELY_MISFILED", "filename and visual evidence agree against folder"
        if visual_label == folder:
            return "LIKELY_FILENAME_ERROR", "visual evidence agrees with folder against filename"
        return "FILENAME_FOLDER_MISMATCH", "filename and folder disagree; visual evidence is inconclusive"
    if visual_label and visual_label != folder:
        return "LIKELY_MISFILED", "visual evidence disagrees with folder"
    if not filename:
        return "MISSING_FILENAME_LABEL", "filename has no standalone 1X/10X token"
    if not visual_label:
        return "VISUAL_INCONCLUSIVE", "insufficient or conflicting registration evidence"
    return "CONSISTENT", "folder, filename, and visual evidence agree"


def decide_filename_only_status(record: ImageRecord) -> tuple[str, str]:
    if not record.filename_label:
        return "MISSING_FILENAME_LABEL", "filename has no standalone 1X/10X token"
    if record.filename_label != record.folder_label:
        return "FILENAME_FOLDER_MISMATCH", "filename and folder disagree"
    return "FILENAME_FOLDER_CONSISTENT", "filename and folder agree; visual audit was disabled"


def _audit_group(payload) -> list[dict]:
    records, arguments = payload
    if not arguments["filename_only"]:
        try:
            cv2 = _load_cv2()
            cv2.setNumThreads(1)
            cache = FeatureCache(arguments["feature_method"], arguments["max_side"])
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize visual audit for {records[0].group_key}: {exc}") from exc
    else:
        cache = None
    trusted = {label: [record for record in records if record.trusted_label == label] for label in STREAMS}
    rows = []
    for record in records:
        registrations = {label: RegistrationResult() for label in STREAMS}
        if cache is not None:
            for label in STREAMS:
                references = nearest_references(
                    record,
                    trusted[label],
                    arguments["reference_count"],
                    arguments["max_days"],
                )
                registrations[label] = best_registration(record, references, cache)
        visual_label, confidence = classify_scores(
            registrations["1X"].score,
            registrations["10X"].score,
            arguments["minimum_score"],
            arguments["minimum_confidence"],
        )
        if arguments["filename_only"]:
            status, reason = decide_filename_only_status(record)
        else:
            status, reason = decide_status(record, visual_label)
        rows.append(
            asdict(
                AuditRow(
                    path=record.path,
                    relative_path=record.relative_path,
                    station=record.station,
                    year=record.year,
                    camera=record.camera,
                    timestamp=record.timestamp.isoformat(sep=" ") if record.timestamp else "",
                    folder_label=record.folder_label,
                    filename_label=record.filename_label,
                    visual_label=visual_label,
                    visual_confidence=round(confidence, 6),
                    score_1x=round(registrations["1X"].score, 6),
                    score_10x=round(registrations["10X"].score, 6),
                    reference_1x=registrations["1X"].reference_path,
                    reference_10x=registrations["10X"].reference_path,
                    scale_to_1x=registrations["1X"].scale,
                    scale_to_10x=registrations["10X"].scale,
                    inliers_1x=registrations["1X"].inliers,
                    inliers_10x=registrations["10X"].inliers,
                    status=status,
                    reason=reason,
                )
            )
        )
    return rows


def audit_records(records: Sequence[ImageRecord], arguments: dict, workers: int = 1) -> list[dict]:
    groups: dict[tuple[str, str, str], list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(record.group_key, []).append(record)
    payloads = [(group, arguments) for _, group in sorted(groups.items())]
    rows = []
    if workers <= 1:
        for payload in payloads:
            rows.extend(_audit_group(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_audit_group, payload) for payload in payloads]
            for future in as_completed(futures):
                rows.extend(future.result())
    rows.sort(key=lambda row: row["relative_path"])
    return rows


def write_report(rows: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(AuditRow)]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discrepancy_rows(rows: Sequence[dict]) -> list[dict]:
    return [row for row in rows if row["status"] in DISCREPANCY_STATUSES]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only 1X/10X audit using filename checks and classical feature registration."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("magnification_audit.csv"))
    parser.add_argument("--feature-method", choices=("sift", "akaze"), default="sift")
    parser.add_argument("--reference-count", type=int, default=2)
    parser.add_argument("--max-days", type=float, default=14.0)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--minimum-score", type=float, default=0.08)
    parser.add_argument("--minimum-confidence", type=float, default=0.70)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--station", action="append", help="Only audit this station (repeatable).")
    parser.add_argument("--year", action="append", help="Only audit this year (repeatable).")
    parser.add_argument("--limit", type=int, default=0, help="Development-only maximum image count.")
    parser.add_argument(
        "--filename-only",
        action="store_true",
        help="Skip OpenCV registration and report folder/filename disagreements only.",
    )
    parser.add_argument(
        "--only-discrepancies",
        action="store_true",
        help="Write only discrepancy rows to CSV; omit consistent and inconclusive rows.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = args.data_path.resolve()
    if not data_path.is_dir():
        print(f"Dataset directory does not exist: {data_path}", file=sys.stderr)
        return 2
    records = discover_images(data_path)
    if args.station:
        stations = set(args.station)
        records = [record for record in records if record.station in stations]
    if args.year:
        years = set(args.year)
        records = [record for record in records if record.year in years]
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        print("No images under 1X/10X folders were found.", file=sys.stderr)
        return 2
    settings = {
        "feature_method": args.feature_method,
        "reference_count": max(1, args.reference_count),
        "max_days": args.max_days,
        "max_side": args.max_side,
        "minimum_score": args.minimum_score,
        "minimum_confidence": args.minimum_confidence,
        "filename_only": args.filename_only,
    }
    print(f"Auditing {len(records)} images from {data_path}")
    rows = audit_records(records, settings, workers=args.workers)
    report_rows = discrepancy_rows(rows) if args.only_discrepancies else rows
    write_report(report_rows, args.output.resolve())
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Report written to {args.output.resolve()} ({len(report_rows)} rows)")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
