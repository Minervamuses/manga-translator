from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from manga_translator import detector as detector_module
from manga_translator.config import DetectionConfig, PostprocessConfig
from manga_translator.detector import DetectionResult, TextRegion, detect_text_regions
from manga_translator.image_io import read_image

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[2]


def _model_path() -> Path:
    configured = os.getenv("MANGA_TRANSLATOR_DETECTOR_MODEL", "").strip()
    candidate = Path(configured) if configured else ROOT / "models" / "comictextdetector.pt"
    if not candidate.is_file():
        pytest.skip(
            "target GPU parity requires MANGA_TRANSLATOR_DETECTOR_MODEL or "
            "models/comictextdetector.pt"
        )
    return candidate


def _corpus_paths() -> list[Path]:
    configured = os.getenv("MANGA_TRANSLATOR_PARITY_IMAGES", "").strip()
    if configured:
        candidate = Path(configured)
        if candidate.is_dir():
            paths = sorted(
                path
                for path in candidate.iterdir()
                if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}
            )
        else:
            paths = [Path(value) for value in configured.split(os.pathsep) if value]
    else:
        paths = []
        pages_dir = ROOT / "benchmarks" / "regression_v032" / "pages"
        for page_path in sorted(pages_dir.glob("*.json")):
            page = json.loads(page_path.read_text(encoding="utf-8"))
            source = ROOT / str(page["source_image"])
            if source.is_file():
                paths.append(source)
    return [path.resolve() for path in paths if path.is_file()]


def _box_iou(left: TextRegion, right: TextRegion) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.w, right.x + right.w)
    y2 = min(left.y + left.h, right.y + right.h)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def _best_match(region: TextRegion, candidates: list[TextRegion]) -> tuple[float, TextRegion | None]:
    if not candidates:
        return 0.0, None
    matched = max(candidates, key=lambda candidate: _box_iou(region, candidate))
    return _box_iou(region, matched), matched


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_on = left > 0
    right_on = right > 0
    union = int(np.count_nonzero(left_on | right_on))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(left_on & right_on)) / union


def _run_corpus(paths: list[Path], model_path: Path, *, half: bool) -> list[DetectionResult]:
    config = DetectionConfig(model_path=model_path, device="cuda", half=half)
    postprocess = PostprocessConfig()
    outputs: list[DetectionResult] = []
    for path in paths:
        image = read_image(path)
        assert image is not None, f"unreadable parity page: {path}"
        result = detect_text_regions(image, config, postprocess)
        assert result.issues == [], f"detector runtime fallback on {path}: {result.issues}"
        assert detector_module._detector.half is half
        outputs.append(result)
    return outputs


def test_fp16_matches_fp32_on_target_gpu_and_real_corpus() -> None:
    if not torch.cuda.is_available():
        pytest.skip("target CUDA GPU is unavailable")
    settings = json.loads(
        (ROOT / "benchmarks" / "detector_fp16_parity.json").read_text(encoding="utf-8")
    )
    thresholds = settings["thresholds"]
    paths = _corpus_paths()
    if len(paths) < settings["required_pages"]:
        pytest.skip(
            f"target parity requires {settings['required_pages']} real pages; found {len(paths)}"
        )

    model_path = _model_path()
    detector_module._detector = None
    detector_module._detector_key = None
    fp32_results = _run_corpus(paths, model_path, half=False)
    fp16_results = _run_corpus(paths, model_path, half=True)

    box_ious: list[float] = []
    score_errors: list[float] = []
    mask_ious: list[float] = []
    small_total = 0
    small_matched = 0
    fp32_count = 0
    fp16_count = 0
    for path, fp32, fp16 in zip(paths, fp32_results, fp16_results, strict=True):
        image = read_image(path)
        assert image is not None
        page_area = image.shape[0] * image.shape[1]
        fp32_count += len(fp32.regions_post)
        fp16_count += len(fp16.regions_post)
        mask_ious.append(_mask_iou(fp32.mask, fp16.mask))
        for region in fp32.regions_post:
            overlap, matched = _best_match(region, fp16.regions_post)
            box_ious.append(overlap)
            if matched is not None:
                score_errors.append(abs(region.confidence - matched.confidence))
            if region.area / page_area <= thresholds["small_text_max_area_ratio"]:
                small_total += 1
                small_matched += overlap >= 0.5

    count_ratio = min(fp32_count, fp16_count) / max(fp32_count, fp16_count, 1)
    box_mean_iou = float(np.mean(box_ious)) if box_ious else 1.0
    score_mae = float(np.mean(score_errors)) if score_errors else 0.0
    mask_mean_iou = float(np.mean(mask_ious))
    small_recall = small_matched / small_total if small_total else 1.0

    assert count_ratio >= thresholds["box_count_ratio_min"]
    assert box_mean_iou >= thresholds["box_mean_iou_min"]
    assert score_mae <= thresholds["matched_score_mae_max"]
    assert mask_mean_iou >= thresholds["mask_iou_min"]
    assert small_recall >= thresholds["small_text_recall_min"]
