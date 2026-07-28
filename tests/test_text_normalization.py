from __future__ import annotations

from manga_translator.text import grapheme_clusters, normalize_text
from manga_translator.typesetter import _balanced_chunks, _sanitize_render_text, _visible_length


def test_crlf_controls_and_outer_whitespace_are_auditable_and_reversible() -> None:
    raw = " \r\n『猫…』\x00\r\n——！？ \t"
    variants = normalize_text(raw)

    assert variants.raw == raw
    assert variants.reconstruct_raw() == raw
    assert variants.nfc_display == "『猫…』\n——！？"
    assert [item.rule_id for item in variants.transformations] == [
        "line_endings.lf",
        "controls.remove_explicit",
        "outer_whitespace.strip",
    ]
    assert variants.transformations[0].before == raw
    assert variants.transformations[-1].after == variants.nfc_display


def test_nfkc_is_comparison_only_and_never_overwrites_display() -> None:
    variants = normalize_text("ＡＢＣ！？ ①")
    assert variants.nfc_display == "ＡＢＣ！？ ①"
    assert variants.nfkc_comparison_key == "abc!? 1"


def test_uax29_keeps_emoji_combining_marks_and_variation_selectors_together() -> None:
    text = "👨‍👩‍👧‍👦e\u0301✈️"
    assert grapheme_clusters(text) == ("👨‍👩‍👧‍👦", "e\u0301", "✈️")
    assert _visible_length(text) == 3
    assert _balanced_chunks(text, 2) == ("👨‍👩‍👧‍👦e\u0301", "✈️")


def test_explicit_newlines_and_preferred_break_markers_are_preserved() -> None:
    variants = normalize_text("第一行\r\n第二行｜第三行", preferred_break_markers=("｜",))
    clusters = grapheme_clusters(variants.nfc_display)
    assert "\n" in clusters
    assert "｜" in clusters
    assert variants.preferred_breaks == (
        clusters.index("\n") + 1,
        clusters.index("｜") + 1,
    )
    assert _sanitize_render_text(variants.nfc_display) == variants.nfc_display


def test_paired_punctuation_is_not_deleted_or_ascii_folded_for_display() -> None:
    text = "「真的嗎？」『……』——！！"
    variants = normalize_text(text)
    assert variants.nfc_display == text
    assert "「" in variants.nfc_display and "……" in variants.nfc_display
