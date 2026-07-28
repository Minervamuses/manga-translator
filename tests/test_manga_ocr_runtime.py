from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from manga_translator import ocr as ocr_module
from manga_translator.manga_ocr_runtime import MangaOcrRuntime, _post_process
from manga_translator.ocr import OCRInitializationError


def test_post_process_normalizes_spacing_width_and_ellipsis() -> None:
    assert _post_process(" ﾃ ｽ ﾄ … ・・ ") == "テスト....."


def test_runtime_uses_explicit_vit_processor_and_japanese_tokenizer(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeTensor:
        def to(self, device):
            calls["tensor_device"] = device
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id: str):
            calls["processor_model"] = model_id
            return cls()

        def __call__(self, image, return_tensors: str):
            assert isinstance(image, Image.Image)
            assert return_tensors == "pt"
            return SimpleNamespace(pixel_values=FakeTensor())

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls["tokenizer_model"] = model_id
            calls["tokenizer_kwargs"] = kwargs
            return cls()

        def decode(self, token_ids, skip_special_tokens: bool):
            assert skip_special_tokens is True
            calls["decoded_ids"] = token_ids
            return " 日 本 … "

    class FakeGenerationMixin:
        pass

    class FakeVisionEncoderDecoderModel:
        @classmethod
        def from_pretrained(cls, model_id: str):
            calls["model_class"] = cls
            calls["model_id"] = model_id
            return cls()

        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def generate(self, pixel_values, max_length: int):
            calls["generate_input"] = pixel_values
            calls["max_length"] = max_length
            return [FakeTensor()]

    fake_transformers = ModuleType("transformers")
    fake_transformers.__version__ = "5.test"
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.ViTImageProcessor = FakeProcessor
    fake_transformers.VisionEncoderDecoderModel = FakeVisionEncoderDecoderModel
    fake_transformers.GenerationMixin = FakeGenerationMixin

    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "2.test"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.device = lambda name: f"device:{name}"
    fake_torch.inference_mode = nullcontext

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    runtime = MangaOcrRuntime("example/model", max_length=123)
    result = runtime(Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8)))

    assert calls["processor_model"] == "example/model"
    assert calls["tokenizer_model"] == "example/model"
    assert calls["tokenizer_kwargs"] == {
        "tokenizer_type": "bert-japanese",
        "use_fast": False,
    }
    assert issubclass(calls["model_class"], FakeGenerationMixin)
    assert calls["model_device"] == "device:cpu"
    assert calls["tensor_device"] == "device:cpu"
    assert calls["max_length"] == 123
    assert calls["eval"] is True
    assert result == "日本..."


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
