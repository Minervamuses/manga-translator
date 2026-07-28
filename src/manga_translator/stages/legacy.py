"""Behavior-preserving stage adapters around the v0.3.2 page pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import cv2

from ..config import AppConfig
from ..domain.issues import StageName
from ..result import PageResult
from ..storage.job_store import JobStore
from .base import (
    ArtifactPayload,
    FingerprintDependencies,
    StageInputs,
    StageOutputs,
    StageSpec,
)
from .runner import STAGE_DAG

LegacyPageProcessor = Callable[..., PageResult]


def _file_fingerprint(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        return hashlib.sha256(f"missing:{path.resolve()}".encode()).hexdigest()


def stage_config(config: AppConfig, glossary_revision: str) -> dict[str, Any]:
    """Return fingerprintable configuration without persisting credentials."""

    return {
        "detection": config.detection.model_dump(mode="json"),
        "inpainting": config.inpainting.model_dump(mode="json"),
        "ocr": config.ocr.model_dump(mode="json"),
        "openrouter": config.openrouter.model_dump(mode="json", exclude={"api_key"}),
        "postprocess": config.postprocess.model_dump(mode="json"),
        "typesetting": config.typesetting.model_dump(mode="json"),
        "glossary_revision": glossary_revision,
    }


def build_legacy_stage_specs(
    *,
    image_path: Path,
    config: AppConfig,
    glossary: dict[str, str],
    source_bytes: bytes,
    store: JobStore,
    process_page: LegacyPageProcessor,
    result_holder: dict[str, PageResult],
    debug: bool,
    save_intermediate: bool,
    prep_manual: bool,
) -> dict[StageName, StageSpec]:
    glossary_revision = hashlib.sha256(
        json.dumps(glossary, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    model_hash = _file_fingerprint(config.detection.model_path)
    font_hashes = (_file_fingerprint(config.paths.font), _file_fingerprint(config.paths.font_fallback))

    def marker(stage: StageName, inputs: StageInputs) -> StageOutputs:
        payload = {
            "adapter": "v0.3.2",
            "stage": stage.value,
            "upstream": {
                name.value: [artifact.sha256 for artifact in artifacts]
                for name, artifacts in inputs.upstream.items()
            },
        }
        return StageOutputs(
            (
                ArtifactPayload(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
                    "application/vnd.manga-translator.stage+json",
                    "adapter_state",
                ),
            )
        )

    def source_stage(_context, _inputs) -> StageOutputs:
        return StageOutputs((ArtifactPayload(source_bytes, "application/octet-stream", "source"),))

    def legacy_render(_context, _inputs) -> StageOutputs:
        page = process_page(
            image_path,
            config,
            glossary,
            debug=debug,
            dump_json=False,
            save_intermediate=save_intermediate,
            prep_manual=prep_manual,
        )
        result_holder["page"] = page
        image = page.image
        if image is None:
            data = source_bytes
            media_type = "application/octet-stream"
        else:
            encoded, buffer = cv2.imencode(".png", image)
            if not encoded:
                raise RuntimeError("legacy adapter could not encode rendered page")
            data = buffer.tobytes()
            media_type = "image/png"
        return StageOutputs((ArtifactPayload(data, media_type, "rendered_page"),))

    def encode_stage(_context, inputs: StageInputs) -> StageOutputs:
        artifact = inputs.upstream[StageName.INPAINT_RENDER][0]
        return StageOutputs(
            (
                ArtifactPayload(
                    store.artifacts.read_bytes(artifact.sha256),
                    artifact.media_type,
                    "encoded_page",
                ),
            )
        )

    runners = {
        StageName.SOURCE: source_stage,
        StageName.INPAINT_RENDER: legacy_render,
        StageName.ENCODE: encode_stage,
    }
    config_keys: Mapping[StageName, tuple[str, ...]] = {
        StageName.DETECT: ("detection", "postprocess"),
        StageName.STYLE: (),
        StageName.SAFE_REGION: ("inpainting",),
        StageName.OCR: ("ocr",),
        StageName.ORDER: ("postprocess",),
        StageName.TRANSLATE: ("openrouter", "glossary_revision"),
        StageName.LAYOUT: ("typesetting",),
        StageName.INPAINT_RENDER: ("inpainting",),
    }
    specs: dict[StageName, StageSpec] = {}
    for stage, dependencies in STAGE_DAG.items():
        run = runners.get(stage)
        if run is None:
            run = lambda _context, inputs, current=stage: marker(current, inputs)
        specs[stage] = StageSpec(
            name=stage,
            dependencies=dependencies,
            run=run,
            code_revision="manga-translator-v0.3.2-p1-adapter.1",
            config_keys=config_keys.get(stage, ()),
            fingerprint_dependencies=FingerprintDependencies(
                model_hashes=(model_hash,) if stage in {StageName.DETECT, StageName.OCR} else (),
                font_hashes=font_hashes if stage is StageName.LAYOUT else (),
                dependency_versions={"adapter": "1"},
                preprocess_revision="v0.3.2" if stage is StageName.OCR else "",
                prompt_revision="v0.3.2" if stage is StageName.TRANSLATE else "",
                glossary_revision=glossary_revision if stage is StageName.TRANSLATE else "",
            ),
        )
    return specs
