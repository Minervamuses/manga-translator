from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_translator.typography.fonts import (
    FontCatalog,
    FontResolver,
    FontRole,
    MissingGlyphError,
    ResolverEvidence,
    font_has_glyph,
)

ROOT = Path(__file__).parents[1]
IANSUI = ROOT / "fonts" / "Iansui-Regular.ttf"
NOTO = ROOT / "fonts" / "NotoSansCJKtc-Regular.otf"


@pytest.fixture(scope="module")
def catalog() -> FontCatalog:
    return FontCatalog.from_paths([IANSUI, NOTO])


def test_bundled_catalog_matches_reviewed_manifest(catalog: FontCatalog) -> None:
    manifest = json.loads((ROOT / "assets/fonts/manifest.json").read_text(encoding="utf-8"))
    expected = {item["sha256"]: item for item in manifest["fonts"]}

    assert {record.sha256 for record in catalog.records} == set(expected)
    for record in catalog.records:
        item = expected[record.sha256]
        assert len(record.codepoints) == item["coverage_codepoints"]
        assert record.weight_class == item["weight_class"]
        assert record.width_class == item["width_class"]
        assert record.has_vertical_metrics
        assert record.has_vertical_features
        assert "locl" in record.gsub_features or record.family == "Iansui"
        assert "SIL Open Font License" in record.license_text


def test_cmap_is_authoritative_and_not_notdef_raster(catalog: FontCatalog) -> None:
    assert font_has_glyph(NOTO, "繁")
    assert not font_has_glyph(IANSUI, "丂")
    assert font_has_glyph(NOTO, "丂")
    assert not font_has_glyph(IANSUI, "\U0010ffff")


def test_missing_glyphs_form_minimal_fallback_runs_without_tofu(catalog: FontCatalog) -> None:
    resolver = FontResolver(catalog, ROOT / "config/fonts.yaml")
    runs = resolver.fallback_runs("中丂丂文", FontRole.HANDWRITTEN)

    assert [run.text for run in runs] == ["中", "丂丂", "文"]
    assert runs[0].font.family == "Iansui"
    assert runs[1].font.family == "Noto Sans CJK TC"
    assert all(run.font.covers(run.text) for run in runs)
    with pytest.raises(MissingGlyphError):
        resolver.fallback_runs("\U0010ffff", FontRole.NEUTRAL_SANS)


def test_low_confidence_page_dialogue_stays_on_neutral_role(catalog: FontCatalog) -> None:
    resolver = FontResolver(catalog, ROOT / "config/fonts.yaml")
    special = resolver.resolve_role(
        page_id="page",
        evidence=ResolverEvidence(
            confidence=0.95,
            same_glyph_scores=((FontRole.HANDWRITTEN, 0.9),),
        ),
    )
    low_confidence = resolver.resolve_role(
        page_id="other-page",
        evidence=ResolverEvidence(confidence=0.2),
    )
    consistent = resolver.resolve_role(
        page_id="page",
        evidence=ResolverEvidence(confidence=0.7, edge_roundness=0.9),
    )

    assert special is FontRole.HANDWRITTEN
    assert low_confidence is FontRole.NEUTRAL_SANS
    assert consistent is FontRole.HANDWRITTEN


def test_manual_and_page_overrides_take_priority(catalog: FontCatalog) -> None:
    resolver = FontResolver(catalog, ROOT / "config/fonts.yaml")
    evidence = ResolverEvidence(confidence=0.0)
    assert resolver.resolve_role(
        page_id="p1", evidence=evidence, page_override=FontRole.FORMAL_SERIF
    ) is FontRole.FORMAL_SERIF
    assert resolver.resolve_role(
        page_id="p2", evidence=evidence, manual_override=FontRole.ROUNDED
    ) is FontRole.ROUNDED
