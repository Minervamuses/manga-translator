from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from manga_translator import pipeline as pipeline_module
from manga_translator.config import AppConfig, OpenRouterConfig, PathsConfig
from manga_translator.detector import DetectionResult, TextRegion
from manga_translator.stages.render import RenderProfile, RenderStageResult
from manga_translator.style.extract import extract_style_fingerprint
from manga_translator.typography.render import conservative_render_style


def _canvas(color=(255, 255, 255)) -> np.ndarray:
    image = np.empty((40, 40, 3), dtype=np.uint8)
    image[:] = color
    return image


def _rect_mask() -> np.ndarray:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[8:32, 12:28] = 255
    return mask


def test_black_and_colored_fill_are_recovered_from_original_pixels() -> None:
    mask = _rect_mask()
    black = _canvas()
    black[mask > 0] = (5, 5, 5)
    black_style = extract_style_fingerprint(black, mask, bbox=(0, 0, 40, 40))

    colored = _canvas((40, 40, 40))
    colored[mask > 0] = (20, 60, 220)
    color_style = extract_style_fingerprint(colored, mask, bbox=(0, 0, 40, 40))

    assert black_style.fill.value is not None
    assert max(black_style.fill.value) <= 8
    assert color_style.fill.value == pytest.approx((220, 60, 20), abs=3)
    assert color_style.fill.sample_count > 100
    assert color_style.ink_density.value == pytest.approx(np.count_nonzero(mask) / mask.size)


def test_white_fill_black_outline_has_distinct_stroke() -> None:
    mask = _rect_mask()
    image = _canvas((180, 180, 180))
    image[mask > 0] = (4, 4, 4)
    image[13:27, 17:23] = (248, 248, 248)
    style = extract_style_fingerprint(image, mask, bbox=(0, 0, 40, 40))

    assert style.fill.value is not None and min(style.fill.value) >= 240
    assert style.stroke.value is not None and max(style.stroke.value) <= 10
    assert style.stroke_width.status == "known"
    assert style.stroke_width.sample_count > 0


def test_light_dialogue_box_uses_black_fill_without_decoration() -> None:
    mask = _rect_mask()
    image = _canvas((248, 248, 248))
    image[mask > 0] = (4, 4, 4)
    style = extract_style_fingerprint(image, mask, bbox=(0, 0, 40, 40))

    rendered = conservative_render_style(style, maximum_stroke_width=2)

    assert style.background is not None and style.background.value is not None
    assert min(style.background.value) >= 240
    assert rendered.fill == (0, 0, 0, 255)
    assert rendered.stroke is None
    assert rendered.stroke_width == 0
    assert rendered.shadow is None


def test_contrast_guard_allows_only_legible_low_confidence_outline() -> None:
    mask = _rect_mask()
    image = _canvas((34, 34, 34))
    image[mask > 0] = (228, 228, 228)
    image[13:27, 17:23] = (34, 34, 34)
    style = extract_style_fingerprint(image, mask, bbox=(0, 0, 40, 40))
    assert style.stroke.value is not None

    accepted = style.renderer_values(
        min_confidence=0.99,
        default_fill=(34, 34, 34),
        default_stroke=None,
        stroke_min_confidence=0.0,
        minimum_stroke_contrast=96.0,
    )
    rejected = style.renderer_values(
        min_confidence=0.99,
        default_fill=(34, 34, 34),
        default_stroke=None,
        stroke_min_confidence=0.0,
        minimum_stroke_contrast=255.0,
    )

    assert accepted["stroke_rgb"] == style.stroke.value
    assert accepted["stroke_width"] != 0.0
    assert rejected["stroke_rgb"] is None
    assert rejected["stroke_width"] == 0.0


def test_gradient_background_does_not_override_colored_ink() -> None:
    mask = _rect_mask()
    gradient = np.linspace(15, 235, 40, dtype=np.uint8)
    image = np.repeat(gradient[:, None], 40, axis=1)
    image = np.repeat(image[..., None], 3, axis=2)
    image[mask > 0] = (30, 190, 70)
    style = extract_style_fingerprint(image, mask, bbox=(0, 0, 40, 40))

    assert style.fill.value == pytest.approx((70, 190, 30), abs=3)
    assert style.fill.confidence > 0.7


def test_hard_shadow_requires_contrast_offset_and_connectivity() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[8:28, 8:22] = 255
    image = _canvas()
    image[11:31, 12:26] = (90, 40, 20)
    image[mask > 0] = (10, 10, 10)
    style = extract_style_fingerprint(image, mask, bbox=(0, 0, 40, 40))

    assert style.shadow.status == "known"
    assert style.shadow.offset.value is not None
    assert style.shadow.offset.value[0] > 0
    assert style.shadow.offset.value[1] > 0

    no_shadow = _canvas()
    no_shadow[mask > 0] = (10, 10, 10)
    assert extract_style_fingerprint(
        no_shadow, mask, bbox=(0, 0, 40, 40)
    ).shadow.status == "unknown"


def test_unknown_style_uses_role_defaults_and_rejects_inpainted_source() -> None:
    empty = np.zeros((40, 40), dtype=np.uint8)
    style = extract_style_fingerprint(_canvas(), empty, bbox=(0, 0, 40, 40))
    rendered = style.renderer_values(
        min_confidence=0.7, default_fill=(1, 2, 3), default_stroke=None
    )
    assert rendered["fill_rgb"] == (1, 2, 3)
    assert rendered["shadow"] is None
    with pytest.raises(ValueError, match="original_image"):
        extract_style_fingerprint(
            _canvas(), empty, bbox=(0, 0, 40, 40), source="inpainted_image"
        )


def test_pipeline_extracts_style_before_inpaint(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    assert cv2.imwrite(str(image_path), _canvas())
    mask = _rect_mask()
    region = TextRegion(id="r0000", x=0, y=0, w=40, h=40, local_mask=mask)
    detection = DetectionResult(
        regions_raw=[region],
        regions_post=[region],
        groups=[],
        mask=mask,
    )
    events = []
    real_extract = extract_style_fingerprint

    def extract(*args, **kwargs):
        events.append("style")
        return real_extract(*args, **kwargs)

    def render_page(image, requests, *_args, **_kwargs):
        events.append("inpaint")
        return RenderStageResult(
            image.copy(),
            (),
            RenderProfile(1, int(image.nbytes), 0, 0, len(requests)),
        )

    monkeypatch.setattr(pipeline_module, "detect_text_regions", lambda *_args: detection)
    monkeypatch.setattr(pipeline_module, "extract_style_fingerprint", extract)
    monkeypatch.setattr(pipeline_module, "render_page_atomic", render_page)
    config = AppConfig(
        openrouter=OpenRouterConfig(api_key="test", model="test"),
        paths=PathsConfig(output_dir=tmp_path / "output"),
    )
    result = pipeline_module.process_single_page(image_path, config, {})

    assert events == ["style", "inpaint"]
    assert len(result.style_fingerprints) == 1
    assert next(iter(result.style_fingerprints.values())).source == "original_image"
