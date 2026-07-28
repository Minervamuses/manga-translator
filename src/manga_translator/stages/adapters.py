"""Typed v0.3.2 component adapters for the persistent stage runner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..domain.issues import StageName
from ..manga_ocr_runtime import DEFAULT_MODEL_ID
from .base import (
    ArtifactContract,
    ArtifactSetContract,
    FingerprintDependencies,
    StageFunction,
    StageSpec,
)
from .runner import STAGE_DAG
from .state import MASK_MEDIA_TYPE, STATE_MEDIA_TYPE

SOURCE_MEDIA_TYPE = "application/octet-stream"
ENCODED_MEDIA_TYPE = "image/png"

SOURCE_CONTRACT = ArtifactSetContract(
    required=(ArtifactContract("source", SOURCE_MEDIA_TYPE),)
)
STATE_CONTRACT = ArtifactSetContract(
    required=(ArtifactContract("state", STATE_MEDIA_TYPE),),
    additional_media_types=(MASK_MEDIA_TYPE,),
)
RENDER_CONTRACT = ArtifactSetContract(
    required=(ArtifactContract("state", STATE_MEDIA_TYPE),),
    additional_media_types=(MASK_MEDIA_TYPE, ENCODED_MEDIA_TYPE),
)
ENCODE_CONTRACT = ArtifactSetContract(
    required=(ArtifactContract("encoded_page", ENCODED_MEDIA_TYPE),)
)


def _file_fingerprint(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        return hashlib.sha256(f"missing:{path.resolve()}".encode()).hexdigest()


def _identity_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _package_versions(*names: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "missing"
    return result


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


def glossary_fingerprint(glossary: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(glossary),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_pipeline_stage_specs(
    *,
    config: AppConfig,
    glossary_revision: str,
    runners: Mapping[StageName, StageFunction],
) -> dict[StageName, StageSpec]:
    """Build the fixed DAG and reject missing component adapters.

    A missing callback is a programming error.  In particular, it must never be
    replaced by a marker artifact that makes an unexecuted stage look successful.
    """

    missing = set(STAGE_DAG) - set(runners)
    extra = set(runners) - set(STAGE_DAG)
    if missing or extra:
        raise ValueError(
            "stage callbacks must exactly match the fixed DAG: "
            f"missing={sorted(item.value for item in missing)}, "
            f"extra={sorted(item.value for item in extra)}"
        )

    detector_hash = _file_fingerprint(config.detection.model_path)
    ocr_identity = _identity_fingerprint(f"{DEFAULT_MODEL_ID}:revision-unpinned")
    font_hashes = (
        _file_fingerprint(config.paths.font),
        _file_fingerprint(config.paths.font_fallback),
    )
    shared_versions = _package_versions("numpy", "opencv-python")
    dependency_versions: Mapping[StageName, dict[str, str]] = {
        StageName.DETECT: {**shared_versions, **_package_versions("torch")},
        StageName.OCR: {
            **shared_versions,
            **_package_versions("manga-ocr", "transformers", "torch"),
        },
        StageName.TRANSLATE: _package_versions("httpx"),
        StageName.LAYOUT: _package_versions("Pillow"),
        StageName.INPAINT_RENDER: {**shared_versions, **_package_versions("Pillow")},
        StageName.ENCODE: dict(shared_versions),
    }
    config_keys: Mapping[StageName, tuple[str, ...]] = {
        StageName.DETECT: ("detection", "postprocess"),
        StageName.STYLE: (),
        StageName.SAFE_REGION: ("inpainting",),
        StageName.OCR: ("ocr",),
        StageName.ORDER: ("postprocess.reading_order",),
        StageName.TRANSLATE: ("openrouter", "postprocess", "glossary_revision"),
        StageName.LAYOUT: ("typesetting", "postprocess"),
        StageName.INPAINT_RENDER: ("inpainting",),
    }
    output_contracts: Mapping[StageName, ArtifactSetContract] = {
        StageName.SOURCE: SOURCE_CONTRACT,
        StageName.DETECT: STATE_CONTRACT,
        StageName.STYLE: STATE_CONTRACT,
        StageName.SAFE_REGION: STATE_CONTRACT,
        StageName.OCR: STATE_CONTRACT,
        StageName.ORDER: STATE_CONTRACT,
        StageName.TRANSLATE: STATE_CONTRACT,
        StageName.LAYOUT: STATE_CONTRACT,
        StageName.INPAINT_RENDER: RENDER_CONTRACT,
        StageName.ENCODE: ENCODE_CONTRACT,
    }

    specs: dict[StageName, StageSpec] = {}
    for stage, dependencies in STAGE_DAG.items():
        specs[stage] = StageSpec(
            name=stage,
            dependencies=dependencies,
            run=runners[stage],
            code_revision=f"manga-translator-v0.3.2-p1-{stage.value}.2",
            config_keys=config_keys.get(stage, ()),
            input_contracts={
                dependency: output_contracts[dependency] for dependency in dependencies
            },
            output_contract=output_contracts[stage],
            fingerprint_dependencies=FingerprintDependencies(
                model_hashes=(
                    (detector_hash,)
                    if stage is StageName.DETECT
                    else (ocr_identity,)
                    if stage is StageName.OCR
                    else ()
                ),
                font_hashes=font_hashes if stage is StageName.LAYOUT else (),
                dependency_versions=dependency_versions.get(stage, {}),
                preprocess_revision="v0.3.2-ensemble.1" if stage is StageName.OCR else "",
                prompt_revision="v0.3.2-mapped.1" if stage is StageName.TRANSLATE else "",
                glossary_revision=(
                    glossary_revision if stage is StageName.TRANSLATE else ""
                ),
            ),
        )
    return specs
