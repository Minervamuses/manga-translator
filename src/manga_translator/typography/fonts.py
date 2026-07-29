"""cmap-authoritative font catalog and stable role/fallback resolution."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import yaml
from fontTools.ttLib import TTFont

from ..text import grapheme_clusters


class FontRole(StrEnum):
    NEUTRAL_SANS = "neutral_sans"
    FORMAL_SERIF = "formal_serif"
    HANDWRITTEN = "handwritten"
    ROUNDED = "rounded"


@dataclass(frozen=True)
class FontRecord:
    path: Path
    sha256: str
    family: str
    subfamily: str
    full_name: str
    weight_class: int
    width_class: int
    codepoints: frozenset[int]
    variable_axes: tuple[tuple[str, float, float, float], ...]
    gsub_features: tuple[str, ...]
    gpos_features: tuple[str, ...]
    has_vertical_metrics: bool
    license_text: str
    license_url: str

    @property
    def has_vertical_features(self) -> bool:
        return bool({"vert", "vrt2"} & set(self.gsub_features))

    def covers(self, text: str) -> bool:
        return all(char.isspace() or ord(char) in self.codepoints for char in text)


def _features(font: TTFont, table_name: str) -> tuple[str, ...]:
    if table_name not in font:
        return ()
    feature_list = getattr(font[table_name].table, "FeatureList", None)
    if feature_list is None:
        return ()
    return tuple(sorted({record.FeatureTag for record in feature_list.FeatureRecord}))


@lru_cache(maxsize=64)
def inspect_font(path: str | Path) -> FontRecord:
    resolved = Path(path).expanduser().resolve()
    sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    with TTFont(resolved, lazy=True) as font:
        name = font["name"]
        os2 = font["OS/2"]
        cmap = font.getBestCmap() or {}
        axes = ()
        if "fvar" in font:
            axes = tuple(
                sorted(
                    (
                        axis.axisTag,
                        float(axis.minValue),
                        float(axis.defaultValue),
                        float(axis.maxValue),
                    )
                    for axis in font["fvar"].axes
                )
            )
        return FontRecord(
            path=resolved,
            sha256=sha256,
            family=name.getDebugName(1) or "unknown",
            subfamily=name.getDebugName(2) or "unknown",
            full_name=name.getDebugName(4) or "unknown",
            weight_class=int(os2.usWeightClass),
            width_class=int(os2.usWidthClass),
            codepoints=frozenset(cmap),
            variable_axes=axes,
            gsub_features=_features(font, "GSUB"),
            gpos_features=_features(font, "GPOS"),
            has_vertical_metrics="vhea" in font and "vmtx" in font,
            license_text=name.getDebugName(13) or "unknown",
            license_url=name.getDebugName(14) or "unknown",
        )


def font_has_glyph(path: str | Path, char: str) -> bool:
    return bool(char) and (char.isspace() or ord(char) in inspect_font(path).codepoints)


class FontCatalog:
    def __init__(self, records: list[FontRecord]) -> None:
        by_hash = {record.sha256: record for record in records}
        if len(by_hash) != len(records):
            raise ValueError("font catalog contains duplicate font bytes")
        self.records = tuple(records)
        self.by_path = {record.path: record for record in records}

    @classmethod
    def from_paths(cls, paths: list[str | Path]) -> FontCatalog:
        return cls([inspect_font(path) for path in paths])

    def record(self, path: str | Path) -> FontRecord:
        return self.by_path[Path(path).expanduser().resolve()]


@dataclass(frozen=True)
class FontRun:
    font: FontRecord
    text: str


@dataclass(frozen=True)
class ResolverEvidence:
    confidence: float
    ink_density: float | None = None
    width_height_ratio: float | None = None
    edge_roundness: float | None = None
    same_glyph_scores: tuple[tuple[FontRole, float], ...] = ()


class MissingGlyphError(ValueError):
    pass


class FontResolver:
    def __init__(self, catalog: FontCatalog, config_path: str | Path) -> None:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        self.catalog = catalog
        self.roles: dict[FontRole, tuple[FontRecord, ...]] = {}
        for role in FontRole:
            paths = raw.get("roles", {}).get(role.value, [])
            if not paths:
                raise ValueError(f"font role has no configured fonts: {role.value}")
            base = Path(config_path).resolve().parent.parent
            self.roles[role] = tuple(catalog.record(base / path) for path in paths)
        neutral = self.roles[FontRole.NEUTRAL_SANS][0]
        if not neutral.has_vertical_metrics or not neutral.has_vertical_features:
            raise ValueError("neutral_sans must provide vertical metrics and vert/vrt2")
        self._page_roles: dict[str, Counter[FontRole]] = {}

    @classmethod
    def from_paths(
        cls, primary: str | Path, fallback: str | Path | None = None
    ) -> FontResolver:
        """Build the production role map from configured, fingerprinted fonts."""

        paths = [Path(primary).expanduser().resolve()]
        if fallback is not None:
            resolved_fallback = Path(fallback).expanduser().resolve()
            if resolved_fallback not in paths:
                paths.append(resolved_fallback)
        catalog = FontCatalog.from_paths(paths)
        vertical = [
            record
            for record in catalog.records
            if record.has_vertical_metrics and record.has_vertical_features
        ]
        if not vertical:
            raise ValueError("configured fonts provide no vertical RAQM font")
        neutral = tuple(vertical + [record for record in catalog.records if record not in vertical])
        instance = cls.__new__(cls)
        instance.catalog = catalog
        instance.roles = {
            FontRole.NEUTRAL_SANS: neutral,
            FontRole.FORMAL_SERIF: neutral,
            FontRole.HANDWRITTEN: tuple(catalog.records),
            FontRole.ROUNDED: neutral,
        }
        instance._page_roles = {}
        return instance

    def resolve_role(
        self,
        *,
        page_id: str,
        evidence: ResolverEvidence,
        manual_override: FontRole | None = None,
        page_override: FontRole | None = None,
    ) -> FontRole:
        if manual_override is not None:
            chosen = manual_override
        elif page_override is not None:
            chosen = page_override
        elif evidence.confidence < 0.65:
            chosen = FontRole.NEUTRAL_SANS
        else:
            scores = Counter(dict(evidence.same_glyph_scores))
            if evidence.edge_roundness is not None and evidence.edge_roundness > 0.55:
                scores[FontRole.ROUNDED] += 0.25
            if evidence.ink_density is not None and evidence.ink_density > 0.48:
                scores[FontRole.FORMAL_SERIF] += 0.15
            chosen = scores.most_common(1)[0][0] if scores else FontRole.NEUTRAL_SANS
        history = self._page_roles.setdefault(page_id, Counter())
        if evidence.confidence < 0.8 and history:
            chosen = history.most_common(1)[0][0]
        history[chosen] += 1
        return chosen

    def fallback_runs(self, text: str, role: FontRole) -> tuple[FontRun, ...]:
        fonts = self.roles[role]
        runs: list[FontRun] = []
        for cluster in grapheme_clusters(text):
            font = next((candidate for candidate in fonts if candidate.covers(cluster)), None)
            if font is None:
                neutral = self.roles[FontRole.NEUTRAL_SANS]
                font = next((candidate for candidate in neutral if candidate.covers(cluster)), None)
            if font is None:
                codepoints = ",".join(f"U+{ord(char):04X}" for char in cluster)
                raise MissingGlyphError(f"no catalog font covers grapheme {codepoints}")
            if runs and runs[-1].font.sha256 == font.sha256:
                runs[-1] = FontRun(font, runs[-1].text + cluster)
            else:
                runs.append(FontRun(font, cluster))
        return tuple(runs)
