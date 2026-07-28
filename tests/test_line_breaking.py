from __future__ import annotations

import pytest

from manga_translator.typography.breaking import (
    BreakStrength,
    analyze_line_breaks,
    balanced_legal_chunks,
    greedy_legal_wrap,
    validate_breaks,
)
from manga_translator.typography.vertical import VerticalOrientation, vertical_runs


@pytest.mark.parametrize(
    ("text", "legal"),
    [
        ("日文", (1, 2)),
        ("a b", (2, 3)),
        ("A\r\nB", (3, 4)),
        ("你好，世界", (1, 3, 4, 5)),
    ],
)
def test_unmodified_uax14_boundaries(text: str, legal: tuple[int, ...]) -> None:
    assert analyze_line_breaks(text).legal_indices == legal


def test_clreq_tailoring_rejects_forbidden_line_edges() -> None:
    text = "他說「你好」，真的。"
    analysis = analyze_line_breaks(text)
    rules = {item.rule for item in analysis.violations}

    assert "clreq_no_opening_punctuation_at_line_end" in rules
    assert "clreq_no_closing_punctuation_at_line_start" in rules
    assert validate_breaks(text, analysis.legal_indices) == ()


@pytest.mark.parametrize("text", ["王小明", "2026", "……", "——", "&amp;"])
def test_names_numbers_pairs_ellipsis_dash_and_entities_have_no_hard_violation(text: str) -> None:
    analysis = analyze_line_breaks(text, atomic_spans=((0, len(text)),))
    assert validate_breaks(text, analysis.legal_indices, atomic_spans=((0, len(text)),)) == ()
    assert all(not (0 < index < len(text)) for index in analysis.legal_indices)


def test_explicit_newline_is_preserved_as_mandatory_break() -> None:
    text = "第一行\r\n第二行"
    analysis = analyze_line_breaks(text)
    newline = next(item for item in analysis.opportunities if item.index == 5)

    assert newline.strength is BreakStrength.MANDATORY
    assert newline.reasons == ("explicit_newline",)
    assert greedy_legal_wrap(text, len, 100) == ("第一行", "第二行")


def test_greedy_and_balanced_wrapping_only_use_legal_boundaries() -> None:
    text = "他說「你好」，王小明回答。"
    horizontal = greedy_legal_wrap(text, len, 6)
    vertical = balanced_legal_chunks(text, 3)

    for chunks in (horizontal, vertical):
        indices = []
        cursor = 0
        for chunk in chunks[:-1]:
            cursor = text.index(chunk, cursor) + len(chunk)
            indices.append(cursor)
        assert validate_breaks(text, indices) == ()


def test_uax50_vertical_runs_keep_cjk_upright_and_rotate_latin_runs() -> None:
    runs = vertical_runs("中文ABCD英文ABCDE")

    assert [(run.text, run.orientation) for run in runs] == [
        ("中文", VerticalOrientation.UPRIGHT),
        ("ABCD", VerticalOrientation.TATE_CHU_YOKO),
        ("英文", VerticalOrientation.UPRIGHT),
        ("ABCDE", VerticalOrientation.ROTATED),
    ]


def test_uax50_short_digits_are_single_tate_chu_yoko_candidate() -> None:
    runs = vertical_runs("第2026年")

    assert [(run.text, run.orientation) for run in runs] == [
        ("第", VerticalOrientation.UPRIGHT),
        ("2026", VerticalOrientation.TATE_CHU_YOKO),
        ("年", VerticalOrientation.UPRIGHT),
    ]
