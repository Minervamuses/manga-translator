"""Non-destructive deterministic validation for translation revisions."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any


def normalize_display_text(raw_text: str) -> str:
    """Only normalize NFC and safe whitespace/control characters; preserve punctuation."""

    if not isinstance(raw_text, str):
        raise TypeError("translation text must be a string")
    text = unicodedata.normalize("NFC", raw_text)
    safe: list[str] = []
    for character in text:
        category = unicodedata.category(character)
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
    if len(compact) > 256:
        return False
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
    approved_entities: Mapping[str, str] | None = None,
    maximum_length_ratio: float = 4.0,
) -> TranslationValidationResult:
    """Report issues without ever mutating ``raw_translation``."""

    if (
        isinstance(maximum_length_ratio, bool)
        or not isinstance(maximum_length_ratio, (int, float))
        or not isfinite(maximum_length_ratio)
        or maximum_length_ratio <= 0
    ):
        raise ValueError("maximum_length_ratio must be finite and positive")
    for item in inputs:
        if not isinstance(item, TranslationInput):
            raise TypeError("inputs must contain TranslationInput values")
        if not isinstance(item.unit_id, str) or not item.unit_id.strip():
            raise ValueError("translation unit IDs must not be empty")
        if not isinstance(item.source, str):
            raise TypeError("translation sources must be strings")
        if not isinstance(item.raw_translation, str):
            raise TypeError("raw translations must be strings")
        if any(
            not isinstance(ref, str) or not normalize_display_text(ref)
            for ref in item.entity_refs
        ):
            raise ValueError("entity references must be non-empty strings")
    if expected_ids is not None:
        if any(not isinstance(unit_id, str) or not unit_id.strip() for unit_id in expected_ids):
            raise ValueError("expected translation unit IDs must not be empty")
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("expected translation unit IDs must be unique")
    if approved_entities is not None and not isinstance(approved_entities, Mapping):
        raise TypeError("approved_entities must be a mapping")
    approved: dict[str, str] = {}
    for source_name, target in (approved_entities or {}).items():
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError("approved entity source names must not be empty")
        if not isinstance(target, str) or not normalize_display_text(target):
            raise ValueError("approved entity targets must not be empty")
        approved[unicodedata.normalize("NFC", source_name)] = normalize_display_text(target)

    issues: list[TranslationIssue] = []
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
        unsupported = sorted(
            {
                f"U+{ord(character):04X}"
                for character in item.raw_translation
                if unicodedata.category(character) in {"Cs", "Co", "Cn"}
            }
        )
        if unsupported:
            issues.append(
                TranslationIssue(
                    "unsupported_unicode",
                    item.unit_id,
                    "translation contains unsupported Unicode code points",
                    {"code_points": unsupported},
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
        normalized_source = unicodedata.normalize("NFC", item.source)
        for source_name, approved_target in approved.items():
            if source_name in normalized_source and approved_target not in display:
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
