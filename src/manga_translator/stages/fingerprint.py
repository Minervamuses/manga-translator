"""Canonical, exhaustive stage fingerprint construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .base import StageSpec


def select_relevant_config(config: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in sorted(keys):
        value: Any = config
        present = True
        for component in key.split("."):
            if not isinstance(value, Mapping) or component not in value:
                value = None
                present = False
                break
            value = value[component]
        selected[key] = {"present": present, "value": value}
    return selected


def stage_fingerprint(
    spec: StageSpec,
    *,
    upstream_output_hashes: Sequence[str],
    config: Mapping[str, Any],
) -> str:
    dependencies = spec.fingerprint_dependencies
    payload = {
        "code_revision": spec.code_revision,
        "config": select_relevant_config(config, spec.config_keys),
        "dependency_versions": dict(sorted(dependencies.dependency_versions.items())),
        "entity_revision": dependencies.entity_revision,
        "font_hashes": sorted(dependencies.font_hashes),
        "glossary_revision": dependencies.glossary_revision,
        "model_hashes": sorted(dependencies.model_hashes),
        "preprocess_revision": dependencies.preprocess_revision,
        "prompt_revision": dependencies.prompt_revision,
        "schema_revision": dependencies.schema_revision,
        "stage": spec.name.value,
        "upstream_output_hashes": list(upstream_output_hashes),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
