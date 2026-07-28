from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from manga_translator.config import OCRConfig
from manga_translator.manga_ocr_runtime import GenerationTokenMetrics, OCRBatchResult
from manga_translator.ocr import _make_candidate, _reset_ocr_state_for_tests, ocr_regions_batch
from manga_translator.stages.ocr import (
    DurableOCRViewCache,
    PageOCRStager,
    RegionOCRViews,
    ocr_view_cache_key,
)
from manga_translator.storage import ArtifactStore, JobStore


def _result(text: str, score: float = -0.05, margin: float = 0.9) -> OCRBatchResult:
    return OCRBatchResult(
        text=text,
        sequence=text,
        metrics=GenerationTokenMetrics((3, 2), (score, score), score, 0.1, margin, False),
        model_id="fake/model",
        model_revision="abc1234",
        generation_config={"max_length": 20},
        actual_batch_size=1,
    )


class FakeRuntime:
    model_id = "fake/model"
    revision = "abc1234"

    def __init__(self, outputs: dict[int, OCRBatchResult] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[list[int]] = []
        self.device = "cpu"
        self.generation_config = {"max_length": 20}

    def recognize_batch(self, images: list[Image.Image], *, batch_size=None):
        del batch_size
        identifiers = [int(np.asarray(image)[0, 0, 0]) for image in images]
        self.calls.append(identifiers)
        return tuple(
            replace(self.outputs.get(identifier, _result(str(identifier))), actual_batch_size=len(images))
            for identifier in identifiers
        )


def _image(identifier: int) -> np.ndarray:
    return np.full((8, 8, 3), identifier, dtype=np.uint8)


def test_ocr_regions_batch_calls_model_once_and_preserves_order(monkeypatch) -> None:
    from manga_translator import ocr as ocr_module

    runtime = FakeRuntime()
    _reset_ocr_state_for_tests()
    monkeypatch.setattr(ocr_module, "_get_model", lambda: runtime)

    output = ocr_regions_batch([_image(3), _image(1), _image(3), _image(2)])

    assert output == ["3", "1", "3", "2"]
    assert runtime.calls == [[3, 1, 2]]
    _reset_ocr_state_for_tests()


def test_page_staging_batches_initial_views_and_only_expands_uncertain_regions(tmp_path: Path) -> None:
    outputs = {
        10: _result("高信心"),
        11: _result("高信心"),
        20: _result("甲", score=-2.0, margin=0.1),
        21: _result("乙", score=-2.0, margin=0.1),
        22: _result("改善", score=-1.5, margin=0.2),
        23: _result("改善", score=-1.5, margin=0.2),
        24: _result("最終", score=-0.02, margin=0.95),
    }
    runtime = FakeRuntime(outputs)
    with JobStore(tmp_path / "jobs.sqlite3", ArtifactStore(tmp_path / "artifacts")) as store:
        stager = PageOCRStager(
            runtime,
            DurableOCRViewCache(store),
            preprocess_version="v1",
            batch_size=8,
        )
        regions = (
            RegionOCRViews("high", _image(10), mask_isolated=_image(11)),
            RegionOCRViews(
                "low",
                _image(20),
                mask_isolated=_image(21),
                contrast=_image(22),
                threshold=_image(23),
                constituents=(_image(24),),
            ),
        )

        staged = stager.run_page(regions)

    assert runtime.calls == [[10, 20, 11, 21], [22, 23], [24]]
    assert [view.view_type for view in staged[0].views] == ["raw", "mask"]
    assert [view.view_type for view in staged[1].views] == [
        "raw",
        "mask",
        "contrast",
        "threshold",
        "constituent:0",
    ]
    assert staged[1].selected.text == "最終"
    assert not staged[1].derived_views_are_independent_votes


def test_durable_view_cache_survives_runtime_and_stager_restart(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    database = tmp_path / "jobs.sqlite3"
    region = RegionOCRViews("r0", _image(7), mask_isolated=_image(8))
    first_runtime = FakeRuntime()
    with JobStore(database, artifacts) as store:
        first = PageOCRStager(
            first_runtime, DurableOCRViewCache(store), preprocess_version="v1", batch_size=4
        )
        first.run_page((region,))
    assert first_runtime.calls == [[7, 8]]

    second_runtime = FakeRuntime()
    with JobStore(database, artifacts) as store:
        second = PageOCRStager(
            second_runtime, DurableOCRViewCache(store), preprocess_version="v1", batch_size=4
        )
        staged = second.run_page((region,))

    assert second_runtime.calls == []
    assert all(view.cache_hit for view in staged[0].views)


def test_same_model_views_do_not_inflate_provisional_score_as_independent_votes(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime({10: _result("同文"), 11: _result("同文")})
    with JobStore(tmp_path / "jobs.sqlite3", ArtifactStore(tmp_path / "artifacts")) as store:
        stager = PageOCRStager(
            runtime, DurableOCRViewCache(store), preprocess_version="v1", batch_size=4
        )
        staged = stager.run_page(
            (
                RegionOCRViews("single", _image(10)),
                RegionOCRViews("derived", _image(10), mask_isolated=_image(11)),
            )
        )

    assert staged[0].provisional_score == staged[1].provisional_score
    assert not staged[1].derived_views_are_independent_votes


def test_view_cache_key_covers_mask_revision_preprocess_view_and_generation() -> None:
    image = _image(1)
    base = {
        "mask": np.zeros((8, 8), dtype=np.uint8),
        "model_revision": "abc1234",
        "preprocess_version": "v1",
        "view_type": "raw",
        "generation_config": {"max_length": 20},
    }
    original = ocr_view_cache_key(image, **base)
    variants = [
        {**base, "mask": np.ones((8, 8), dtype=np.uint8)},
        {**base, "model_revision": "def5678"},
        {**base, "preprocess_version": "v2"},
        {**base, "view_type": "mask"},
        {**base, "generation_config": {"max_length": 21}},
    ]
    assert all(ocr_view_cache_key(image, **variant) != original for variant in variants)


def test_default_preprocess_avoids_redundant_lanczos_upscale(monkeypatch) -> None:
    from manga_translator import ocr as ocr_module

    monkeypatch.setattr(
        ocr_module,
        "_upscale_for_ocr",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected pre-upscale")),
    )
    monkeypatch.setattr(ocr_module, "_ocr_image", lambda image, cache_key: "日本")

    candidate = _make_candidate(_image(5), "raw", OCRConfig())

    assert candidate.text == "日本"
    assert not OCRConfig().pre_upscale
