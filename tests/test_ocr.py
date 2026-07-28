from __future__ import annotations

import numpy as np

from manga_translator.config import OCRConfig
from manga_translator.detector import TextGroup, TextRegion
from manga_translator import ocr as ocr_module
from manga_translator.ocr import (
    OCRCandidate,
    OCRResult,
    _combine_region_candidates,
    _crop_group_mask,
    _crop_region_mask,
    _select_best_candidate,
    assess_ocr_result,
    normalize_ocr_text,
    ocr_quality_score,
    sanitize_ocr_text,
)


def candidate(text: str, quality: float = 0.8, source: str = "test") -> OCRCandidate:
    return OCRCandidate(text, normalize_ocr_text(text), quality, source)


def test_sanitize_ocr_removes_control_and_replacement_chars() -> None:
    assert sanitize_ocr_text(" こ\u200bん\ufffdに\nちは ") == "こんにちは"
    assert sanitize_ocr_text("||丨‖") == ""


def test_japanese_text_scores_higher_than_noise() -> None:
    assert ocr_quality_score("今日はどうしたの？") > ocr_quality_score("◆◆◆◆◆◆")
    assert ocr_quality_score("今日はどうしたの？") >= OCRConfig().min_quality_score
    assert ocr_quality_score("漢") < OCRConfig().short_text_min_quality


def test_fallback_ocr_requires_multiple_japanese_chars_and_candidate_agreement() -> None:
    cfg = OCRConfig()
    weak = OCRResult(
        text="漢",
        normalized="漢",
        confidence=0.90,
        source="raw",
        candidates=[candidate("漢", 0.90, "raw")],
    )
    accepted, reason = assess_ocr_result(weak, cfg, fallback_only=True)
    assert not accepted
    assert reason.startswith("fallback_too_short")

    strong = OCRResult(
        text="大丈夫",
        normalized="大丈夫",
        confidence=0.90,
        source="raw",
        candidates=[
            candidate("大丈夫", 0.90, "raw"),
            candidate("大丈夫", 0.88, "mask"),
        ],
    )
    accepted, reason = assess_ocr_result(strong, cfg, fallback_only=True)
    assert accepted
    assert reason == ""


def test_select_best_candidate_prefers_more_complete_agreeing_text() -> None:
    best = _select_best_candidate(
        [
            candidate("今日は", 0.82, "raw"),
            candidate("今日はどうしたの", 0.79, "regions"),
            candidate("今日はどうした", 0.80, "threshold"),
        ]
    )
    assert best is not None
    assert best.text == "今日はどうしたの"


def test_region_combiner_keeps_legitimate_repeated_adjacent_lines() -> None:
    left = TextRegion(id="r0", x=10, y=10, w=20, h=30)
    right = TextRegion(id="r1", x=60, y=10, w=20, h=30)
    combined = _combine_region_candidates(
        [(left, candidate("うん")), (right, candidate("うん"))]
    )
    assert combined is not None
    assert combined.text == "うんうん"


def test_region_combiner_dedupes_overlapping_detector_passes() -> None:
    first = TextRegion(id="r0", x=10, y=10, w=40, h=50)
    second = TextRegion(id="r1", x=12, y=12, w=38, h=48)
    combined = _combine_region_candidates(
        [(first, candidate("大丈夫？")), (second, candidate("大丈夫?"))]
    )
    assert combined is not None
    assert normalize_ocr_text(combined.text, weak=True) in {"大丈夫", "大丈夫"}
    assert combined.text.count("大丈夫") == 1


def test_crop_group_mask_supports_local_masks_and_padding() -> None:
    local = np.zeros((20, 30), dtype=np.uint8)
    local[5:10, 8:14] = 255
    group = TextGroup(
        id="g0",
        region_ids=[],
        bbox=(40, 50, 30, 20),
        vertical=False,
        mask=local,
    )
    crop = _crop_group_mask(group, (35, 45, 40, 30), (200, 200))
    assert crop is not None
    assert crop.shape == (30, 40)
    assert int(np.count_nonzero(crop)) == 30
    assert np.all(crop[10:15, 13:19] == 255)


def test_crop_region_mask_does_not_allocate_a_full_page() -> None:
    local = np.zeros((10, 12), dtype=np.uint8)
    local[2:6, 3:8] = 255
    region = TextRegion(id="r0", x=100, y=200, w=12, h=10, local_mask=local)

    crop = _crop_region_mask(region, (96, 196, 20, 18))

    assert crop is not None
    assert crop.shape == (18, 20)
    assert int(np.count_nonzero(crop)) == 20
    assert np.all(crop[6:10, 7:12] == 255)


def test_adaptive_mode_compares_mask_even_when_raw_text_looks_valid(monkeypatch) -> None:
    image = np.full((100, 140, 3), 255, dtype=np.uint8)
    local_mask = np.full((50, 60), 255, dtype=np.uint8)
    region = TextRegion(id="r0", x=20, y=20, w=60, h=50, local_mask=local_mask)
    group = TextGroup(
        id="g0",
        region_ids=["r0"],
        bbox=region.bbox,
        vertical=False,
        mask=local_mask,
    )
    calls: list[str] = []

    def fake_make_candidate(_image, source: str, _cfg: OCRConfig) -> OCRCandidate:
        calls.append(source)
        if source.endswith(":raw"):
            return candidate("今日は", 0.90, source)
        if source.endswith(":mask"):
            return candidate("今日はどうしたの", 0.88, source)
        return candidate("今日はどうしたの", 0.86, source)

    monkeypatch.setattr(ocr_module, "_make_candidate", fake_make_candidate)
    result = ocr_module.ocr_group_detailed(
        image,
        group,
        {"r0": region},
        OCRConfig(
            ensemble_mode="adaptive",
            use_contrast_variant=False,
            use_threshold_variant=False,
            use_region_fallback=False,
        ),
    )

    assert any(source.endswith(":mask") for source in calls)
    assert result.text == "今日はどうしたの"


def test_region_combiner_treats_whole_sentence_and_column_fragments_as_alternatives() -> None:
    outer = TextRegion(id="outer", x=0, y=0, w=120, h=180)
    right = TextRegion(id="right", x=70, y=0, w=45, h=180)
    left = TextRegion(id="left", x=10, y=0, w=45, h=180)

    combined = _combine_region_candidates(
        [
            (outer, candidate("今日はどうしたの", 0.82, "outer")),
            (right, candidate("今日は", 0.80, "right")),
            (left, candidate("どうしたの", 0.80, "left")),
        ]
    )

    assert combined is not None
    assert combined.text == "今日はどうしたの"
    assert combined.text.count("今日は") == 1
    assert combined.text.count("どうしたの") == 1
