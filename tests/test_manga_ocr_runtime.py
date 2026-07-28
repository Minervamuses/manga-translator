from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from manga_translator import ocr as ocr_module
from manga_translator.manga_ocr_runtime import (
    DEFAULT_MODEL_REVISION,
    MangaOcrRuntime,
    _post_process,
    _score_generation,
)
from manga_translator.ocr import OCRInitializationError


def test_post_process_normalizes_spacing_width_and_ellipsis() -> None:
    assert _post_process(" ﾃ ｽ ﾄ … ・・ ") == "テスト....."


def _install_fake_transformers(monkeypatch, *, oom_above: int | None = None):
    calls: dict[str, object] = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls["processor_model"] = model_id
            calls["processor_kwargs"] = kwargs
            return cls()

        def __call__(self, *, images, return_tensors: str):
            assert all(isinstance(image, Image.Image) for image in images)
            assert return_tensors == "pt"
            ids = [int(np.asarray(image)[0, 0, 0]) % 10 for image in images]
            return SimpleNamespace(pixel_values=torch.tensor(ids).reshape(-1, 1).float())

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls["tokenizer_model"] = model_id
            calls["tokenizer_kwargs"] = kwargs
            return cls()

        def batch_decode(self, token_ids, skip_special_tokens: bool):
            assert skip_special_tokens is True
            calls["decoded_ids"] = token_ids.tolist()
            return [f" {int(row[1])} … " for row in token_ids]

    class FakeVisionEncoderDecoderModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls["model_id"] = model_id
            calls["model_kwargs"] = kwargs
            return cls()

        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def generate(self, pixel_values, **kwargs):
            size = int(pixel_values.shape[0])
            calls.setdefault("batch_attempts", []).append(size)
            if oom_above is not None and size > oom_above:
                raise torch.OutOfMemoryError("simulated out of memory")
            calls["generation_kwargs"] = kwargs
            ids = pixel_values[:, 0].long() + 3
            sequences = torch.stack((torch.ones_like(ids), ids, torch.full_like(ids, 2)), dim=1)
            first = torch.full((size, 16), -4.0)
            first.scatter_(1, ids[:, None], 4.0)
            second = torch.full((size, 16), -4.0)
            second[:, 2] = 4.0
            return SimpleNamespace(sequences=sequences, scores=(first, second))

    fake_transformers = ModuleType("transformers")
    fake_transformers.__version__ = "5.14.1"
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.ViTImageProcessor = FakeProcessor
    fake_transformers.VisionEncoderDecoderModel = FakeVisionEncoderDecoderModel

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return calls


def test_runtime_uses_explicit_vit_processor_and_japanese_tokenizer(monkeypatch) -> None:
    calls = _install_fake_transformers(monkeypatch)

    runtime = MangaOcrRuntime(
        "example/model",
        revision=DEFAULT_MODEL_REVISION,
        force_cpu=True,
        max_length=123,
    )
    result = runtime(Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8)))

    assert calls["processor_model"] == "example/model"
    assert calls["processor_kwargs"] == {"revision": DEFAULT_MODEL_REVISION}
    assert calls["tokenizer_model"] == "example/model"
    assert calls["tokenizer_kwargs"] == {
        "revision": DEFAULT_MODEL_REVISION,
        "tokenizer_type": "bert-japanese",
        "use_fast": False,
    }
    assert calls["model_kwargs"] == {"revision": DEFAULT_MODEL_REVISION}
    assert str(calls["model_device"]) == "cpu"
    assert calls["generation_kwargs"]["max_length"] == 123
    assert calls["generation_kwargs"]["return_dict_in_generate"] is True
    assert calls["generation_kwargs"]["output_scores"] is True
    assert calls["eval"] is True
    assert result == "8..."


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_true_batches_preserve_input_order(monkeypatch, batch_size: int) -> None:
    _calls = _install_fake_transformers(monkeypatch)
    runtime = MangaOcrRuntime(batch_size=batch_size, force_cpu=True)
    images = [Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)) for index in range(8)]

    results = runtime.recognize_batch(images, batch_size=batch_size)

    assert [result.text for result in results] == [f"{index + 3}..." for index in range(8)]
    assert all(result.model_revision == DEFAULT_MODEL_REVISION for result in results)


def test_token_scores_handle_empty_eos_padding_and_max_length_truncation() -> None:
    sequences = torch.tensor(
        [
            [1, 3, 2, 0],
            [1, 0, 0, 0],
            [1, 5, 6, 7],
        ]
    )
    scores = []
    for step, ids in enumerate(((3, 0, 5), (2, 0, 6), (0, 0, 7))):
        logits = torch.full((3, 10), -3.0)
        for row, token_id in enumerate(ids):
            logits[row, token_id] = 3.0 + step
        scores.append(logits)

    eos, empty, truncated = _score_generation(
        sequences,
        tuple(scores),
        pad_token_id=0,
        eos_token_id=2,
        max_length=4,
        torch_module=torch,
    )

    assert eos.token_ids == (3, 2)
    assert not eos.truncated
    assert eos.length_normalized_transition_logprob == pytest.approx(
        sum(eos.token_logprobs) / 2
    )
    assert empty.token_ids == ()
    assert empty.length_normalized_transition_logprob is None
    assert not empty.truncated
    assert truncated.token_ids == (5, 6, 7)
    assert truncated.truncated
    assert truncated.mean_entropy is not None and truncated.mean_entropy >= 0
    assert truncated.mean_margin is not None and 0 <= truncated.mean_margin <= 1


def test_oom_halves_only_current_batch_then_restores_requested_size(monkeypatch) -> None:
    calls = _install_fake_transformers(monkeypatch, oom_above=2)
    runtime = MangaOcrRuntime(batch_size=4, force_cpu=True)
    images = [Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)) for index in range(6)]

    results = runtime.recognize_batch(images)

    assert len(results) == 6
    assert calls["batch_attempts"] == [4, 2, 4, 2, 2]
    assert [(event.attempted_size, event.retry_size) for event in runtime.last_batch_retries] == [
        (4, 2),
        (4, 2),
    ]
    assert str(runtime.device) == "cpu"


def test_runtime_rejects_moving_or_empty_revision_before_loading(monkeypatch) -> None:
    with pytest.raises(ValueError, match="immutable 40-character commit hash"):
        MangaOcrRuntime(revision="main")


@pytest.mark.parametrize("revision", ["", "main", "revision-unpinned", "a" * 39])
def test_runtime_rejects_moving_or_invalid_revisions(revision: str) -> None:
    with pytest.raises(ValueError, match="immutable 40-character commit hash"):
        MangaOcrRuntime(revision=revision)


def test_ocr_model_initializes_only_once(monkeypatch) -> None:
    ocr_module._reset_ocr_state_for_tests()
    calls = 0
    sentinel = object()

    def build_runtime():
        nonlocal calls
        calls += 1
        return sentinel

    monkeypatch.setattr(ocr_module, "MangaOcrRuntime", build_runtime)

    assert ocr_module._get_model() is sentinel
    assert ocr_module._get_model() is sentinel
    assert calls == 1
    ocr_module._reset_ocr_state_for_tests()


def test_ocr_initialization_failure_is_cached_not_retried_per_region(monkeypatch) -> None:
    ocr_module._reset_ocr_state_for_tests()
    calls = 0

    def fail_runtime():
        nonlocal calls
        calls += 1
        raise ValueError("backend incompatible")

    monkeypatch.setattr(ocr_module, "MangaOcrRuntime", fail_runtime)

    with pytest.raises(OCRInitializationError) as first:
        ocr_module._get_model()
    with pytest.raises(OCRInitializationError) as second:
        ocr_module._get_model()

    assert calls == 1
    assert first.value is second.value
    assert "backend incompatible" in str(first.value)
    ocr_module._reset_ocr_state_for_tests()
