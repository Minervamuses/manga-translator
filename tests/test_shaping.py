from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from manga_translator.runtime.capabilities import (
    PillowCapabilities,
    ShapingCapabilityError,
    pillow_capabilities,
    require_raqm,
)
from manga_translator.typography.fonts import FontCatalog, FontResolver, FontRole
from manga_translator.typography.shaping import PillowRaqmEngine, RunShaper

ROOT = Path(__file__).parents[1]


class RecordingEngine:
    def __init__(self) -> None:
        self.measured = []

    def measure(self, run, **kwargs):
        self.measured.append((run.text, kwargs))
        length = len(run.text.encode("utf-8"))
        return (0.0, 0.0, float(length), 20.0), float(length)

    def render(self, image, run, **kwargs):
        raise AssertionError("not used")


@pytest.fixture(scope="module")
def resolver() -> FontResolver:
    catalog = FontCatalog.from_paths(
        [ROOT / "fonts/Iansui-Regular.ttf", ROOT / "fonts/NotoSansCJKtc-Regular.otf"]
    )
    return FontResolver(catalog, ROOT / "config/fonts.yaml")


def test_missing_raqm_is_blocked_without_codepoint_fallback() -> None:
    unavailable = PillowCapabilities("2.13", False, None, None, None)
    with pytest.raises(ShapingCapabilityError, match="RAQM"):
        require_raqm(unavailable)
    if not pillow_capabilities().production_shaping:
        with pytest.raises(ShapingCapabilityError):
            PillowRaqmEngine()


def test_vertical_column_is_measured_as_complete_font_runs(resolver: FontResolver) -> None:
    engine = RecordingEngine()
    runs = RunShaper(resolver, engine).shape(
        "（你好！）ー12丂",
        role=FontRole.HANDWRITTEN,
        size=32,
        direction="ttb",
        stroke_width=2,
    )

    assert [item[0] for item in engine.measured] == [run.text for run in runs]
    assert "".join(run.text for run in runs) == "（你好！）ー12丂"
    assert all(run.direction == "ttb" for run in runs)
    assert all(run.language == "zh-Hant-TW" for run in runs)
    assert all(run.features == () for run in runs)
    assert all("vert" not in run.features and "vrt2" not in run.features for run in runs)
    assert all(run.font_sha256 and run.glyph_coverage for run in runs)
    assert [run.anchor[1] for run in runs] == sorted(run.anchor[1] for run in runs)


def test_fallback_never_splits_an_extended_grapheme(resolver: FontResolver) -> None:
    engine = RecordingEngine()
    runs = RunShaper(resolver, engine).shape(
        "Ae\u0301B", role=FontRole.NEUTRAL_SANS, size=24, direction="ltr"
    )
    assert "".join(run.text for run in runs) == "Ae\u0301B"
    assert not any(run.text == "\u0301" for run in runs)


@pytest.mark.skipif(
    not pillow_capabilities().production_shaping,
    reason="target Pillow build has no RAQM/HarfBuzz/FriBiDi",
)
def test_raqm_measurement_matches_rendered_alpha_bbox(resolver: FontResolver) -> None:
    engine = PillowRaqmEngine()
    run = RunShaper(resolver, engine).shape(
        "（中文！？）ー12", role=FontRole.NEUTRAL_SANS, size=40, direction="ttb"
    )[0]
    image = Image.new("RGBA", (200, 600), (0, 0, 0, 0))
    engine.render(image, run, size=40, fill=(0, 0, 0, 255), stroke_width=0)
    alpha_bbox = image.getchannel("A").getbbox()
    assert alpha_bbox is not None
    measured_width = run.bbox[2] - run.bbox[0]
    measured_height = run.bbox[3] - run.bbox[1]
    assert abs((alpha_bbox[2] - alpha_bbox[0]) - measured_width) <= 3
    assert abs((alpha_bbox[3] - alpha_bbox[1]) - measured_height) <= 3
