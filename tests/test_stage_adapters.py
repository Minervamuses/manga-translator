from __future__ import annotations

from pathlib import Path

import pytest

from manga_translator.config import AppConfig, OpenRouterConfig, PathsConfig
from manga_translator.domain.issues import StageName
from manga_translator.stages.adapters import build_pipeline_stage_specs
from manga_translator.stages.base import ArtifactPayload, StageOutputs
from manga_translator.stages.runner import STAGE_DAG


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        openrouter=OpenRouterConfig(api_key="unused", model="test/model"),
        paths=PathsConfig(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            glossary=tmp_path / "glossary.json",
            font=tmp_path / "font.ttf",
            font_fallback=tmp_path / "fallback.ttf",
        ),
    )


def _callback(_context, _inputs) -> StageOutputs:
    return StageOutputs((ArtifactPayload(b"unused", "application/test", "unused"),))


def test_pipeline_stage_specs_reject_missing_callbacks(tmp_path: Path) -> None:
    callbacks = {stage: _callback for stage in STAGE_DAG if stage is not StageName.OCR}

    with pytest.raises(ValueError, match=r"missing=\['ocr'\]"):
        build_pipeline_stage_specs(
            config=_config(tmp_path),
            glossary_revision="g" * 64,
            runners=callbacks,
        )


def test_pipeline_stage_specs_declare_every_input_and_output_contract(
    tmp_path: Path,
) -> None:
    callbacks = {stage: _callback for stage in STAGE_DAG}

    specs = build_pipeline_stage_specs(
        config=_config(tmp_path),
        glossary_revision="g" * 64,
        runners=callbacks,
    )

    assert set(specs) == set(STAGE_DAG)
    for stage, dependencies in STAGE_DAG.items():
        assert specs[stage].run is _callback
        assert set(specs[stage].input_contracts) == set(dependencies)
        assert specs[stage].output_contract is not None
