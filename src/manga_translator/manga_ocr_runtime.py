"""Transformers-compatible runtime for kha-white/manga-ocr-base.

The PyPI ``manga-ocr==0.1.11`` loader uses ``AutoFeatureExtractor``.  Newer
Transformers releases no longer route ViT image processors through that audio-
focused auto class, so the old loader raises before the model is created.

This module deliberately loads the same public model with explicit classes:
``ViTImageProcessor`` + Japanese BERT tokenizer +
``VisionEncoderDecoderModel``.  Imports stay lazy so detection-only commands do
not pay the Transformers import cost.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_MODEL_ID = "kha-white/manga-ocr-base"


class MangaOcrRuntime:
    """Small, dependency-stable inference wrapper for manga-ocr-base."""

    def __init__(
        self,
        pretrained_model_name_or_path: str | Path = DEFAULT_MODEL_ID,
        *,
        force_cpu: bool = False,
        max_length: int = 300,
    ) -> None:
        self.model_id = str(pretrained_model_name_or_path)
        self.max_length = int(max_length)

        try:
            import torch
            from transformers import (
                AutoTokenizer,
                VisionEncoderDecoderModel,
                ViTImageProcessor,
            )
            try:
                from transformers import GenerationMixin
            except ImportError:  # pragma: no cover - compatibility with older 4.x
                from transformers.generation.utils import GenerationMixin
        except Exception as error:  # pragma: no cover - exercised through caller tests
            raise RuntimeError(
                "OCR 執行元件無法匯入；需要 transformers、torch、fugashi 與 unidic-lite。"
            ) from error

        self._torch = torch
        model_class: type[Any]
        if issubclass(VisionEncoderDecoderModel, GenerationMixin):
            model_class = VisionEncoderDecoderModel
        else:
            # Transformers 新版不再保證 PreTrainedModel 自帶 GenerationMixin。
            # 與 manga-ocr 上游目前的相容作法一致，明確補回 generate()。
            model_class = type(
                "CompatibleMangaOcrModel",
                (VisionEncoderDecoderModel, GenerationMixin),
                {},
            )

        self.processor = ViTImageProcessor.from_pretrained(self.model_id)
        self.tokenizer = self._load_tokenizer(AutoTokenizer)
        self.model = model_class.from_pretrained(self.model_id)
        self.device = self._select_device(force_cpu=force_cpu)
        self.model.to(self.device)
        self.model.eval()

    def _load_tokenizer(self, auto_tokenizer: type[Any]) -> Any:
        """Force the slow Japanese BERT tokenizer on Transformers 5.x.

        Starting with recent Transformers versions, auto-detection may choose an
        incompatible fast-only tokenizer for a VisionEncoderDecoder config.  The
        model is fixed and known to use the Japanese BERT tokenizer, so making
        that choice explicit is safer than relying on config inference.
        """

        kwargs = {
            "tokenizer_type": "bert-japanese",
            "use_fast": False,
        }
        try:
            return auto_tokenizer.from_pretrained(self.model_id, **kwargs)
        except TypeError as error:
            # Very old supported Transformers builds may not accept
            # ``tokenizer_type``.  Keep a narrow fallback without hiding model or
            # network errors from the user.
            if "tokenizer_type" not in str(error):
                raise
            return auto_tokenizer.from_pretrained(self.model_id, use_fast=False)

    def _select_device(self, *, force_cpu: bool) -> Any:
        torch = self._torch
        if not force_cpu and torch.cuda.is_available():
            return torch.device("cuda")

        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        if not force_cpu and mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def __call__(self, img_or_path: str | Path | Image.Image) -> str:
        if isinstance(img_or_path, (str, Path)):
            with Image.open(img_or_path) as opened:
                image = opened.convert("L").convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert("L").convert("RGB")
        else:
            raise TypeError(
                "img_or_path 必須是圖片路徑或 PIL.Image，"
                f"實際收到：{type(img_or_path).__name__}"
            )

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with self._torch.inference_mode():
            generated = self.model.generate(pixel_values, max_length=self.max_length)

        token_ids = generated[0].detach().cpu()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return _post_process(text)


def _post_process(text: str) -> str:
    """Apply manga-ocr's lightweight output cleanup without importing it."""

    text = "".join(str(text or "").split())
    text = text.replace("…", "...")
    text = re.sub(r"[・.]{2,}", lambda match: "." * len(match.group(0)), text)
    # Normalizes half-width kana and width variants.  The project's outer OCR
    # sanitizer performs the final punctuation/noise cleanup.
    return unicodedata.normalize("NFKC", text)


def check_runtime_dependencies() -> dict[str, str]:
    """Import-only health check used by the CLI doctor command.

    This intentionally does not download or initialize the ~400 MB OCR model.
    """

    versions: dict[str, str] = {}
    try:
        import fugashi
        import torch
        import transformers
        import unidic_lite
    except Exception as error:
        raise RuntimeError(
            "OCR 執行元件不完整；需要 transformers、torch、fugashi 與 unidic-lite。"
        ) from error

    versions["torch"] = str(getattr(torch, "__version__", "unknown"))
    versions["transformers"] = str(getattr(transformers, "__version__", "unknown"))
    versions["fugashi"] = str(getattr(fugashi, "__version__", "installed"))
    versions["unidic_lite"] = str(getattr(unidic_lite, "__version__", "installed"))
    return versions
