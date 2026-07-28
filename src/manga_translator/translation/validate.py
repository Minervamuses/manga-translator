"""Non-destructive deterministic validation for translation revisions."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


def normalize_display_text(raw_text: str) -> str:
    """Only normalize NFC and safe whitespace/control characters; preserve punctuation."""

    text = unicodedata.normalize("NFC", str(raw_text))
    safe: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category in {"Cs", "Co", "Cn"}:
            continue
        if category in {"Cc", "Cf"}:
            if character in {"\n", "\r", "\t"}:
                safe.append(" ")
            continue
        safe.append(character)
    return re.sub(r"\s+", " ", "".join(safe)).strip()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def _is_kana(character: str) -> bool:
    code = ord(character)
    return 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF


def _is_cjk(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x323AF
    )


def _has_adjacent_repeat(text: str, minimum: int = 4) -> bool:
    compact = _compact(text)
    for length in range(len(compact) // 2, minimum - 1, -1):
        for start in range(len(compact) - length * 2 + 1):
            if compact[start : start + length] == compact[start + length : start + length * 2]:
                return True
    return False


@dataclass(frozen=True, slots=True)
class TranslationInput:
    unit_id: str
    source: str
    raw_translation: str
    entity_refs: tuple[str, ...] = ()

    @property
    def display_translation(self) -> str:
        return normalize_display_text(self.raw_translation)


@dataclass(frozen=True, slots=True)
class TranslationIssue:
    code: str
    unit_id: str | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TranslationValidationResult:
    inputs: tuple[TranslationInput, ...]
    issues: tuple[TranslationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def issues_for(self, unit_id: str) -> tuple[TranslationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.unit_id == unit_id)


def validate_translation_batch(
    inputs: tuple[TranslationInput, ...],
    *,
    expected_ids: tuple[str, ...] | None = None,
    approved_entities: dict[str, str] | None = None,
    maximum_length_ratio: float = 4.0,
) -> TranslationValidationResult:
    """Report issues without ever mutating ``raw_translation``."""

    issues: list[TranslationIssue] = []
    approved = approved_entities or {}
    counts = Counter(item.unit_id for item in inputs)
    for unit_id, count in counts.items():
        if count > 1:
            issues.append(
                TranslationIssue(
                    "illegal_id", unit_id, "duplicate translation unit ID", {"count": count}
                )
            )
    if expected_ids is not None:
        expected = set(expected_ids)
        actual = set(counts)
        for unit_id in sorted(actual - expected):
            issues.append(TranslationIssue("illegal_id", unit_id, "unknown translation unit ID"))
        for unit_id in sorted(expected - actual):
            issues.append(TranslationIssue("illegal_id", unit_id, "missing translation unit ID"))

    displays: dict[str, list[TranslationInput]] = defaultdict(list)
    for item in inputs:
        display = item.display_translation
        compact_source = _compact(item.source)
        compact_display = _compact(display)
        if "\ufffd" in item.raw_translation or any(
            token in item.raw_translation for token in ("Ã", "â€", "ðŸ")
        ):
            issues.append(
                TranslationIssue(
                    "encoding_anomaly", item.unit_id, "translation contains encoding artifacts"
                )
            )
        if not display:
            issues.append(TranslationIssue("empty", item.unit_id, "translation is empty"))
            continue
        if compact_source == compact_display and any(_is_kana(char) for char in compact_source):
            issues.append(
                TranslationIssue("source_echo", item.unit_id, "translation echoes source")
            )

        meaningful = [
            char for char in compact_display if char.isalnum() or _is_cjk(char) or _is_kana(char)
        ]
        kana = sum(_is_kana(char) for char in meaningful)
        if len(meaningful) >= 4 and kana / len(meaningful) > 0.35:
            issues.append(
                TranslationIssue(
                    "untranslated_japanese_ratio",
                    item.unit_id,
                    "translation retains too much Japanese kana",
                    {"ratio": kana / len(meaningful)},
                )
            )
        limit = max(24, int(max(1, len(compact_source)) * maximum_length_ratio + 12))
        if len(compact_display) > limit:
            issues.append(
                TranslationIssue(
                    "length_anomaly",
                    item.unit_id,
                    "translation is implausibly long",
                    {"actual": len(compact_display), "maximum": limit},
                )
            )
        if _has_adjacent_repeat(display) and not _has_adjacent_repeat(item.source, minimum=2):
            issues.append(
                TranslationIssue("repeated_content", item.unit_id, "adjacent content is repeated")
            )

        for required in item.entity_refs:
            if normalize_display_text(required) not in display:
                issues.append(
                    TranslationIssue(
                        "missing_entity",
                        item.unit_id,
                        "required entity is absent",
                        {"entity": required},
                    )
                )
        for source_name, approved_target in approved.items():
            if source_name in item.source and approved_target not in display:
                issues.append(
                    TranslationIssue(
                        "approved_glossary_mismatch",
                        item.unit_id,
                        "approved glossary translation is absent",
                        {"source": source_name, "approved_target": approved_target},
                    )
                )
        displays[compact_display].append(item)

    for duplicate, members in displays.items():
        if duplicate and len(members) > 1 and len({item.source for item in members}) > 1:
            for item in members:
                issues.append(
                    TranslationIssue(
                        "cross_unit_duplication",
                        item.unit_id,
                        "different source units share identical output",
                        {"other_ids": sorted(x.unit_id for x in members if x is not item)},
                    )
                )
    return TranslationValidationResult(inputs=inputs, issues=tuple(issues))
