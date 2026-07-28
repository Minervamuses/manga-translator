"""Bounded exact-ID targeted and layout-overflow repair coordination."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..contracts.mapping import MappingContractError, request_map_from_ids, source_sha256
from ..typography.layout import LayoutOverflow, LayoutResult
from .validate import (
    TranslationInput,
    TranslationIssue,
    normalize_display_text,
    validate_translation_batch,
)

RepairKind = Literal["targeted", "compact"]


@dataclass(frozen=True, slots=True)
class RepairUnit:
    unit_id: str
    source: str
    original_text: str
    issues: tuple[TranslationIssue, ...] = ()
    must_preserve_entities: tuple[str, ...] = ()
    must_preserve_facts: tuple[str, ...] = ()
    approved_entities: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class RepairPromptItem:
    id: str
    source: str
    current_translation: str
    repair_reasons: tuple[str, ...]
    kind: RepairKind
    must_preserve_entities: tuple[str, ...]
    must_preserve_facts: tuple[str, ...]
    max_graphemes: int | None = None
    max_lines: int | None = None


@dataclass(frozen=True, slots=True)
class TranslationRevision:
    unit_id: str
    revision_number: int
    kind: RepairKind
    original_text: str
    previous_text: str
    raw_repair_text: str
    display_text: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    unit_id: str
    original_text: str
    final_text: str
    kept_original: bool
    revision: TranslationRevision | None = None
    issues: tuple[TranslationIssue, ...] = ()


RepairProvider = Callable[[tuple[RepairPromptItem, ...]], Awaitable[Sequence[Mapping[str, Any]]]]


def _preservation_issues(unit: RepairUnit, candidate: str) -> tuple[TranslationIssue, ...]:
    display = normalize_display_text(candidate)
    issues: list[TranslationIssue] = []
    for entity in unit.must_preserve_entities:
        if entity not in display:
            issues.append(
                TranslationIssue(
                    "repair_dropped_entity",
                    unit.unit_id,
                    "repair dropped a required entity",
                    {"entity": entity},
                )
            )
    for fact in unit.must_preserve_facts:
        if fact not in display:
            issues.append(
                TranslationIssue(
                    "repair_dropped_fact",
                    unit.unit_id,
                    "repair dropped a required fact",
                    {"fact": fact},
                )
            )
    negations = {
        token for token in ("不", "沒", "無", "別", "勿", "未") if token in unit.original_text
    }
    if negations and not any(token in display for token in negations):
        issues.append(
            TranslationIssue(
                "repair_dropped_negation",
                unit.unit_id,
                "repair dropped a critical negation",
                {"expected_any": sorted(negations)},
            )
        )
    return tuple(issues)


class RepairCoordinator:
    def __init__(self, provider: RepairProvider, *, max_revisions_per_unit: int = 2) -> None:
        if max_revisions_per_unit < 1:
            raise ValueError("max_revisions_per_unit must be positive")
        self.provider = provider
        self.max_revisions_per_unit = max_revisions_per_unit
        self._revision_counts: dict[str, int] = {}
        self._attempt_counts: dict[str, int] = {}

    def _unchanged(
        self, unit: RepairUnit, issues: tuple[TranslationIssue, ...] = ()
    ) -> RepairOutcome:
        return RepairOutcome(
            unit.unit_id, unit.original_text, unit.original_text, True, issues=issues
        )

    async def _call_exact(self, items: tuple[RepairPromptItem, ...]) -> dict[str, str]:
        raw = await self.provider(items)
        item_ids = [item.id for item in items]
        hashes = [source_sha256(item.source) for item in items]
        request = request_map_from_ids(item_ids, hashes, request_id="repair")
        enriched = []
        for response_item in raw:
            item_id = response_item.get("id")
            expected = request.by_item_id.get(item_id) if isinstance(item_id, str) else None
            enriched.append(
                {
                    **response_item,
                    "source_sha256": expected.source_sha256 if expected is not None else "",
                }
            )
        from ..contracts.mapping import validate_response_items

        batch = validate_response_items(request, enriched)
        return {item.item_id: item.translation for item in batch.responses}

    def _accept_revision(
        self,
        unit: RepairUnit,
        raw_candidate: str,
        *,
        kind: RepairKind,
        reason_codes: tuple[str, ...],
    ) -> RepairOutcome:
        display = normalize_display_text(raw_candidate)
        validation = validate_translation_batch(
            (
                TranslationInput(
                    unit.unit_id,
                    unit.source,
                    raw_candidate,
                    entity_refs=unit.must_preserve_entities,
                ),
            ),
            expected_ids=(unit.unit_id,),
            approved_entities=dict(unit.approved_entities or {}),
        )
        issues = (*validation.issues, *_preservation_issues(unit, raw_candidate))
        if issues:
            return self._unchanged(unit, tuple(issues))
        revision_number = self._revision_counts.get(unit.unit_id, 0) + 1
        self._revision_counts[unit.unit_id] = revision_number
        revision = TranslationRevision(
            unit_id=unit.unit_id,
            revision_number=revision_number,
            kind=kind,
            original_text=unit.original_text,
            previous_text=unit.original_text,
            raw_repair_text=raw_candidate,
            display_text=display,
            reason_codes=reason_codes,
        )
        return RepairOutcome(
            unit.unit_id,
            unit.original_text,
            display,
            False,
            revision=revision,
        )

    async def targeted_repair(self, units: Sequence[RepairUnit]) -> dict[str, RepairOutcome]:
        outcomes = {unit.unit_id: self._unchanged(unit) for unit in units}
        targets = tuple(
            unit
            for unit in units
            if unit.issues
            and self._attempt_counts.get(unit.unit_id, 0) < self.max_revisions_per_unit
        )
        if not targets:
            return outcomes
        prompt_items = tuple(
            RepairPromptItem(
                id=unit.unit_id,
                source=unit.source,
                current_translation=unit.original_text,
                repair_reasons=tuple(issue.code for issue in unit.issues),
                kind="targeted",
                must_preserve_entities=unit.must_preserve_entities,
                must_preserve_facts=unit.must_preserve_facts,
            )
            for unit in targets
        )
        for unit in targets:
            self._attempt_counts[unit.unit_id] = self._attempt_counts.get(unit.unit_id, 0) + 1
        try:
            repaired = await self._call_exact(prompt_items)
        except (MappingContractError, RuntimeError, TypeError, ValueError) as error:
            contract_issue = TranslationIssue(
                "repair_contract_failed", None, "repair response rejected", {"error": str(error)}
            )
            return {
                unit.unit_id: self._unchanged(unit, (contract_issue,) if unit in targets else ())
                for unit in units
            }
        for unit in targets:
            outcomes[unit.unit_id] = self._accept_revision(
                unit,
                repaired[unit.unit_id],
                kind="targeted",
                reason_codes=tuple(issue.code for issue in unit.issues),
            )
        return outcomes

    async def compact_repair(
        self,
        unit: RepairUnit,
        layout_result: LayoutResult,
        *,
        layout_check: Callable[[str], LayoutResult],
    ) -> RepairOutcome:
        """Compact only after LayoutOverflow; semantic and layout gates must both pass."""

        if not isinstance(layout_result, LayoutOverflow):
            return self._unchanged(unit)
        if self._attempt_counts.get(unit.unit_id, 0) >= self.max_revisions_per_unit:
            return self._unchanged(
                unit,
                (
                    TranslationIssue(
                        "repair_limit_reached", unit.unit_id, "repair revision limit reached"
                    ),
                ),
            )
        item = RepairPromptItem(
            id=unit.unit_id,
            source=unit.source,
            current_translation=unit.original_text,
            repair_reasons=("layout_overflow",),
            kind="compact",
            must_preserve_entities=unit.must_preserve_entities,
            must_preserve_facts=unit.must_preserve_facts,
            max_graphemes=layout_result.suggested_max_graphemes,
            max_lines=layout_result.suggested_max_lines,
        )
        self._attempt_counts[unit.unit_id] = self._attempt_counts.get(unit.unit_id, 0) + 1
        try:
            repaired = await self._call_exact((item,))
        except (MappingContractError, RuntimeError, TypeError, ValueError) as error:
            return self._unchanged(
                unit,
                (
                    TranslationIssue(
                        "repair_contract_failed",
                        unit.unit_id,
                        "compact repair response rejected",
                        {"error": str(error)},
                    ),
                ),
            )
        outcome = self._accept_revision(
            unit, repaired[unit.unit_id], kind="compact", reason_codes=("layout_overflow",)
        )
        if outcome.kept_original:
            return outcome
        new_layout = layout_check(outcome.final_text)
        if isinstance(new_layout, LayoutOverflow):
            return self._unchanged(
                unit,
                (
                    TranslationIssue(
                        "compact_layout_failed",
                        unit.unit_id,
                        "compact repair still overflows",
                    ),
                ),
            )
        return outcome
