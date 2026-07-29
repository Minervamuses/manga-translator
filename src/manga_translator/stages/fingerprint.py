"""Canonical, exhaustive stage fingerprint construction."""

from __future__ import annotations

import hashlib
import inspect
import json
import types
from collections.abc import Mapping, Sequence
from typing import Any

from .base import StageSpec


def _code_constant(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return _code_payload(value)
    if isinstance(value, tuple):
        return ["tuple", *(_code_constant(item) for item in value)]
    if isinstance(value, frozenset):
        encoded = [_code_constant(item) for item in value]
        return ["frozenset", *sorted(encoded, key=lambda item: repr(item))]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _code_payload(code: types.CodeType) -> dict[str, Any]:
    """Serialize executable semantics without path or source-line noise."""

    return {
        "argcount": code.co_argcount,
        "cellvars": code.co_cellvars,
        "code": code.co_code.hex(),
        "consts": [_code_constant(value) for value in code.co_consts],
        "flags": code.co_flags,
        "freevars": code.co_freevars,
        "kwonlyargcount": code.co_kwonlyargcount,
        "names": code.co_names,
        "posonlyargcount": code.co_posonlyargcount,
        "varnames": code.co_varnames,
    }


def callable_code_revision(callback: Any) -> str:
    """Hash a stage callback and the local callables it actually references.

    Runtime configuration captured by closures is intentionally excluded; it is
    already fingerprinted through ``config_keys``. Callable closure values (such
    as profiled stage wrappers) are included so their wrapped implementation is
    not hidden behind a fixed wrapper hash.
    """

    seen: set[int] = set()

    def describe(value: Any) -> dict[str, Any]:
        target = inspect.unwrap(value)
        if inspect.ismethod(target):
            target = target.__func__
        code = getattr(target, "__code__", None)
        if not isinstance(code, types.CodeType):
            raise TypeError("stage callback must expose Python executable code")
        marker = id(target)
        if marker in seen:
            return {"recursive": f"{target.__module__}.{target.__qualname__}"}
        seen.add(marker)
        children: list[dict[str, Any]] = []
        globals_map = getattr(target, "__globals__", {})
        for name in sorted(set(code.co_names)):
            child = globals_map.get(name)
            if inspect.isfunction(child) and child.__module__.startswith("manga_translator"):
                children.append(describe(child))
        closure = getattr(target, "__closure__", None) or ()
        for cell in closure:
            try:
                child = cell.cell_contents
            except ValueError:
                continue
            if inspect.isfunction(child):
                children.append(describe(child))
        return {
            "callable": f"{target.__module__}.{target.__qualname__}",
            "children": children,
            "code": _code_payload(code),
        }

    encoded = json.dumps(
        describe(callback),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
