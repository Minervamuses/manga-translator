"""Subprocess worker for real kill/restart/replay acceptance tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import cv2
import numpy as np
from click.testing import CliRunner

from manga_translator import pipeline as pipeline_module
from manga_translator.cli import cli
from manga_translator.config import AppConfig, OpenRouterConfig, PathsConfig
from manga_translator.contracts.mapping import (
    RawResponseRef,
    ResponseItem,
    bind_validated_responses,
)
from manga_translator.detector import DetectionResult, MaskSource, TextGroup, TextRegion
from manga_translator.domain.issues import StageName
from manga_translator.domain.serialization import canonical_document_bytes
from manga_translator.ocr import OCRCandidate, OCRResult
from manga_translator.stages.render import RenderProfile, RenderStageResult
from manga_translator.storage import ArtifactStore, JobStore
from manga_translator.translator import TranslationValidation
from manga_translator.typography.fonts import FontRole
from manga_translator.typography.layout import (
    AcceptedLayout,
    FontChoice,
    LayoutCandidate,
    LayoutDirection,
)
from manga_translator.typography.render import AtomicRenderOutcome
from manga_translator.typography.shaping import ShapedFontRun
from manga_translator.typography.solver import layout_plan_hash


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--kill-point", default="none")
    parser.add_argument("--mode", choices=("run", "replay"), default="run")
    return parser.parse_args()


def _counter_path(state: Path) -> Path:
    return state / "call-counts.json"


def _increment(state: Path, name: str) -> None:
    path = _counter_path(state)
    payload = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    payload[name] = int(payload.get(name, 0)) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _kill_once(state: Path, configured: str, actual: str) -> None:
    if configured != actual:
        return
    marker = state / f"killed-{actual}"
    if marker.exists():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(actual, encoding="utf-8")
    os._exit(91)


def _source(shared: Path) -> Path:
    path = shared / "input" / "page.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        assert cv2.imwrite(str(path), np.full((48, 64, 3), 255, dtype=np.uint8))
    glossary = shared / "glossary.json"
    if not glossary.exists():
        glossary.write_text('{"テスト":"測試"}', encoding="utf-8")
    return path


def _config(shared: Path, state: Path) -> AppConfig:
    return AppConfig(
        openrouter=OpenRouterConfig(api_key="local-counted-provider", model="test/model"),
        paths=PathsConfig(
            input_dir=shared / "input",
            output_dir=state / "output",
            glossary=shared / "glossary.json",
            font=shared / "missing-font.ttf",
            font_fallback=shared / "missing-fallback.ttf",
        ),
    )


def _detection() -> DetectionResult:
    region = TextRegion(
        id="r1",
        x=8,
        y=8,
        w=20,
        h=24,
        confidence=0.95,
        page_bbox=(8.0, 8.0, 20.0, 24.0),
        local_mask=np.full((24, 20), 255, dtype=np.uint8),
        mask_source=MaskSource(
            detector_pass=0,
            detection_input_size=1024,
            raw_index=0,
            source="fixture",
            source_region_id="r1",
            page_to_local_affine=(1.0, 0.0, -8.0, 0.0, 1.0, -8.0),
        ),
    )
    group = TextGroup(
        id="detected-1",
        region_ids=["r1"],
        bbox=(8, 8, 20, 24),
        vertical=True,
        mask=np.full((24, 20), 255, dtype=np.uint8),
    )
    return DetectionResult(
        regions_raw=[region],
        regions_post=[region],
        groups=[group],
        mask=np.full((48, 64), 255, dtype=np.uint8),
        raw_mask=np.full((48, 64), 255, dtype=np.uint8),
    )


def _install_fakes(state: Path, config: AppConfig, kill_point: str) -> None:
    def detect(*_args, **_kwargs) -> DetectionResult:
        _increment(state, "detector_load")
        _increment(state, "detector_forward")
        return _detection()

    def initialize() -> None:
        _increment(state, "ocr_load")

    def ocr(*_args, **_kwargs) -> OCRResult:
        _increment(state, "ocr_forward")
        return OCRResult(
            text="テスト",
            normalized="テスト",
            confidence=0.99,
            source="fixture",
            candidates=[OCRCandidate("テスト", "テスト", 0.99, "fixture")],
        )

    def request(groups, page_id, _config, _glossary):
        _increment(state, "provider_request")
        _ordered, _texts, request_map = pipeline_module._build_translation_request(
            groups, page_id
        )
        raw = json.dumps(
            {
                "request_id": request_map.request_id,
                "translations": ["測試翻譯" for _item in request_map.items],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        relative = Path("translation-responses") / f"{hashlib.sha256(raw).hexdigest()}.json"
        path = config.paths.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        reference = RawResponseRef.from_bytes(
            raw,
            media_type="application/json",
            relative_path=relative.as_posix(),
        )
        return bind_validated_responses(
            request_map,
            [
                ResponseItem(
                    item_id=item.item_id,
                    source_sha256=item.source_sha256,
                    translation="測試翻譯",
                    response_index=index,
                    raw_response_ref=reference,
                )
                for index, item in enumerate(request_map.items)
            ],
        )

    def layout(_original, groups, _regions, _config, safe_regions, _store):
        def accepted(group: TextGroup) -> AcceptedLayout:
            entry = pipeline_module._safe_region_entry(group, safe_regions)
            assert entry is not None
            roi_bbox = tuple(
                int(value) for value in entry["roi_bbox"]
            )
            _left, _top, width, height = roi_bbox
            alpha = np.zeros((height, width), dtype=np.uint8)
            alpha[height // 2, width // 2] = 255
            candidate = LayoutCandidate(
                font=FontChoice(FontRole.NEUTRAL_SANS),
                font_size=12,
                direction=LayoutDirection.VERTICAL,
                chunks=(group.translation,),
                break_indices=(),
                line_gap_em=1.0,
                tracking_em=0.0,
                anchor=(width / 2.0, height / 2.0),
                rotation_degrees=0.0,
            )
            run = ShapedFontRun(
                text=group.translation,
                font_sha256="a" * 64,
                font_path="fixture-font.ttf",
                glyph_coverage=tuple(ord(character) for character in group.translation),
                direction="ttb",
                language="zh-Hant",
                features=("vert", "vrt2"),
                bbox=(0.0, 0.0, 1.0, 1.0),
                advance=1.0,
                anchor=(width / 2.0, height / 2.0),
            )
            shaped_runs = (run,)
            return AcceptedLayout(
                candidate,
                alpha,
                1.0,
                0.0,
                layout_plan_hash(candidate, alpha, shaped_runs),
                shaped_runs,
            )

        return {
            group.id: accepted(group)
            for group in groups
            if group.translation_valid
        }

    def render_page(original, requests, _config):
        _kill_once(state, kill_point, "render-before")
        rendered = original.copy()
        outcomes = []
        for request in requests:
            x, y, _width, _height = request.roi_bbox
            rendered[y, x] = (0, 0, 0)
            outcomes.append(AtomicRenderOutcome(True, request.roi_bbox, 1, 0))
        return RenderStageResult(
            rendered,
            tuple(outcomes),
            RenderProfile(1, int(original.nbytes), 0, len(requests), 0),
        )

    real_apply = pipeline_module._apply_translation_batch

    def apply_translation(*args, **kwargs):
        _kill_once(state, kill_point, "provider-response-after")
        return real_apply(*args, **kwargs)

    real_finish = JobStore.finish_stage

    def finish_stage(self, **kwargs):
        result = real_finish(self, **kwargs)
        stage = str(kwargs["stage"])
        if stage == StageName.DETECT.value:
            _kill_once(state, kill_point, "detect-after")
        if stage == StageName.OCR.value:
            _kill_once(state, kill_point, "ocr-after")
        return result

    real_stage_runner = pipeline_module.StageRunner

    def short_lease_runner(**kwargs):
        return real_stage_runner(
            **kwargs,
            page_run_lease_seconds=0.25,
            page_run_poll_seconds=0.01,
            provider_response_lease_seconds=0.25,
            provider_response_poll_seconds=0.01,
        )

    pipeline_module.detect_text_regions = detect
    pipeline_module.initialize_ocr_model = initialize
    pipeline_module.ocr_group_detailed = ocr
    pipeline_module.assess_ocr_result = lambda result, *_args, **_kwargs: (
        bool(result.text),
        "" if result.text else "empty",
    )
    pipeline_module._request_translations = request
    pipeline_module._apply_translation_batch = apply_translation
    pipeline_module._preflight_raqm_layout_plans = layout
    pipeline_module.render_page_atomic = render_page
    pipeline_module.validate_translation = (
        lambda *_args, **_kwargs: TranslationValidation(valid=True)
    )
    pipeline_module.StageRunner = short_lease_runner
    JobStore.finish_stage = finish_stage


def _run(shared: Path, state: Path, kill_point: str) -> int:
    _source(shared)
    config = _config(shared, state)
    _install_fakes(state, config, kill_point)
    result = pipeline_module.run_pipeline(
        config,
        job_id="p1-restart",
        state_dir=state,
        resume=True,
    )
    page_id = result.pages[0].page_id
    with JobStore(state / "jobs.sqlite3", ArtifactStore(state / "artifacts")) as store:
        document = store.load_page_document(job_id="p1-restart", page_id=page_id)
    assert document is not None
    payload = {
        "status": result.status,
        "page_id": page_id,
        "document_sha256": hashlib.sha256(canonical_document_bytes(document)).hexdigest(),
        "counts": json.loads(_counter_path(state).read_text("utf-8")),
    }
    (state / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return 0


def _replay(shared: Path, state: Path) -> int:
    result = json.loads((state / "result.json").read_text("utf-8"))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline replay attempted model or network access")

    socket.socket = forbidden
    pipeline_module.detect_text_regions = forbidden
    pipeline_module.initialize_ocr_model = forbidden
    pipeline_module.ocr_group_detailed = forbidden
    pipeline_module._request_translations = forbidden
    manifest = state / "replayed.json"
    image = state / "replayed.png"
    invocation = CliRunner().invoke(
        cli,
        [
            "replay",
            "--config",
            str(shared / "offline-config.yaml"),
            "--state-dir",
            str(state),
            "--job",
            "p1-restart",
            "--page",
            result["page_id"],
            "--output",
            str(manifest),
            "--output-image",
            str(image),
        ],
    )
    if invocation.exit_code != 0:
        raise RuntimeError(invocation.output) from invocation.exception
    replay_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if replay_sha256 != result["document_sha256"]:
        raise AssertionError("offline replay document hash differs from durable truth")
    (state / "replay-result.json").write_text(
        json.dumps(
            {
                "document_sha256": replay_sha256,
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "network_access": 0,
                "model_loads": 0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


def main() -> int:
    args = _arguments()
    if args.mode == "replay":
        return _replay(args.shared, args.state)
    return _run(args.shared, args.state, args.kill_point)


if __name__ == "__main__":
    raise SystemExit(main())
