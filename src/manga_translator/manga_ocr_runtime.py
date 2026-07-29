"""Pinned, batched Transformers runtime for kha-white/manga-ocr-base."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_MODEL_ID = "kha-white/manga-ocr-base"
DEFAULT_MODEL_REVISION = "aa6573bd10b0d446cbf622e29c3e084914df9741"
IMMUTABLE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
TESTED_TRANSFORMERS_VERSION = "5.14.1"


@dataclass(frozen=True)
class GenerationTokenMetrics:
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    length_normalized_transition_logprob: float | None
    mean_entropy: float | None
    mean_margin: float | None
    truncated: bool


@dataclass(frozen=True)
class OCRBatchResult:
    text: str
    sequence: str
    metrics: GenerationTokenMetrics
    model_id: str
    model_revision: str
    generation_config: dict[str, Any]
    actual_batch_size: int


@dataclass(frozen=True)
class BatchRetryEvent:
    offset: int
    attempted_size: int
    retry_size: int
    reason: str


def _score_generation(
    sequences: Any,
    scores: tuple[Any, ...],
    *,
    pad_token_id: int | None,
    eos_token_id: int | tuple[int, ...] | list[int] | None,
    max_length: int,
    torch_module: Any,
) -> tuple[GenerationTokenMetrics, ...]:
    """Score generated tokens, including EOS and excluding padding."""

    batch_size = int(sequences.shape[0])
    step_count = len(scores)
    eos_ids = (
        set(eos_token_id)
        if isinstance(eos_token_id, (tuple, list))
        else ({eos_token_id} if eos_token_id is not None else set())
    )
    if step_count == 0:
        return tuple(
            GenerationTokenMetrics((), (), None, None, None, False)
            for _ in range(batch_size)
        )

    generated_ids = sequences[:, -step_count:]
    prompt_width = int(sequences.shape[1]) - step_count
    log_probabilities = tuple(torch_module.log_softmax(step.float(), dim=-1) for step in scores)
    probabilities = tuple(step.exp() for step in log_probabilities)
    results: list[GenerationTokenMetrics] = []
    for batch_index in range(batch_size):
        token_ids: list[int] = []
        token_logprobs: list[float] = []
        entropies: list[float] = []
        margins: list[float] = []
        saw_eos = False
        for step_index, (log_probs, probs) in enumerate(zip(log_probabilities, probabilities)):
            token_id = int(generated_ids[batch_index, step_index].item())
            is_eos = token_id in eos_ids
            if not is_eos and pad_token_id is not None and token_id == pad_token_id:
                continue
            row_log_probs = log_probs[batch_index]
            row_probs = probs[batch_index]
            token_ids.append(token_id)
            token_logprobs.append(float(row_log_probs[token_id].item()))
            entropy_terms = torch_module.nan_to_num(
                -(row_probs * row_log_probs), nan=0.0, posinf=0.0, neginf=0.0
            )
            entropies.append(float(entropy_terms.sum().item()))
            top = torch_module.topk(row_probs, k=min(2, int(row_probs.shape[-1]))).values
            margins.append(float((top[0] - top[1]).item()) if len(top) > 1 else float(top[0].item()))
            if is_eos:
                saw_eos = True
                break
        count = len(token_logprobs)
        results.append(
            GenerationTokenMetrics(
                token_ids=tuple(token_ids),
                token_logprobs=tuple(token_logprobs),
                length_normalized_transition_logprob=(sum(token_logprobs) / count if count else None),
                mean_entropy=(sum(entropies) / count if count else None),
                mean_margin=(sum(margins) / count if count else None),
                truncated=bool(
                    count and not saw_eos and step_count and prompt_width + step_count >= max_length
                ),
            )
        )
    return tuple(results)


class MangaOcrRuntime:
    """Direct, revision-pinned runtime with order-preserving micro-batches."""

    def __init__(
        self,
        pretrained_model_name_or_path: str | Path = DEFAULT_MODEL_ID,
        *,
        revision: str = DEFAULT_MODEL_REVISION,
        force_cpu: bool = False,
        max_length: int = 300,
        batch_size: int = 4,
    ) -> None:
        self.model_id = str(pretrained_model_name_or_path)
        self.revision = str(revision).strip().lower()
        if IMMUTABLE_REVISION_PATTERN.fullmatch(self.revision) is None:
            raise ValueError("OCR model revision must be an immutable 40-character commit hash")
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self.last_batch_retries: tuple[BatchRetryEvent, ...] = ()

        try:
            import torch
            import transformers
            from transformers import (
                AutoTokenizer,
                VisionEncoderDecoderModel,
                ViTImageProcessor,
            )
        except Exception as error:  # pragma: no cover - exercised through caller tests
            raise RuntimeError(
                "OCR 執行元件無法匯入；需要 transformers、torch、fugashi 與 unidic-lite。"
            ) from error
        if str(transformers.__version__) != TESTED_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "OCR runtime 僅支援 lockfile 驗證版本："
                f"transformers=={TESTED_TRANSFORMERS_VERSION}，目前為 {transformers.__version__}"
            )

        self._torch = torch
        loader = {"revision": self.revision}
        self.processor = ViTImageProcessor.from_pretrained(self.model_id, **loader)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            tokenizer_type="bert-japanese",
            use_fast=False,
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id, **loader)
        self.device = self._select_device(force_cpu=force_cpu)
        self.model.to(self.device)
        self.model.eval()
        model_config = getattr(self.model, "config", None)
        model_generation = getattr(self.model, "generation_config", None)
        self.pad_token_id = next(
            (
                value
                for value in (
                    getattr(self.tokenizer, "pad_token_id", None),
                    getattr(model_config, "pad_token_id", None),
                    getattr(model_generation, "pad_token_id", None),
                )
                if value is not None
            ),
            None,
        )
        self.eos_token_id = next(
            (
                value
                for value in (
                    getattr(self.tokenizer, "eos_token_id", None),
                    getattr(model_config, "eos_token_id", None),
                    getattr(model_generation, "eos_token_id", None),
                )
                if value is not None
            ),
            None,
        )
        self.generation_config = {
            "max_length": self.max_length,
            "num_beams": 1,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

    def _select_device(self, *, force_cpu: bool) -> Any:
        torch = self._torch
        if not force_cpu and torch.cuda.is_available():
            return torch.device("cuda")
        if not force_cpu and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _prepare_image(img_or_path: str | Path | Image.Image) -> Image.Image:
        if isinstance(img_or_path, (str, Path)):
            with Image.open(img_or_path) as opened:
                return opened.convert("L").convert("RGB")
        if isinstance(img_or_path, Image.Image):
            return img_or_path.convert("L").convert("RGB")
        raise TypeError(
            "img_or_path 必須是圖片路徑或 PIL.Image，"
            f"實際收到：{type(img_or_path).__name__}"
        )

    def _infer_batch(self, images: list[Image.Image]) -> tuple[OCRBatchResult, ...]:
        encoded = self.processor(images=images, return_tensors="pt")
        pixel_values = encoded.pixel_values.to(self.device)
        with self._torch.inference_mode():
            generated = self.model.generate(pixel_values, **self.generation_config)
        sequences = generated.sequences.detach().cpu()
        scores = tuple(score.detach().cpu() for score in generated.scores)
        metrics = _score_generation(
            sequences,
            scores,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            max_length=self.max_length,
            torch_module=self._torch,
        )
        decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        actual_size = len(images)
        return tuple(
            OCRBatchResult(
                text=_post_process(sequence),
                sequence=sequence,
                metrics=token_metrics,
                model_id=self.model_id,
                model_revision=self.revision,
                generation_config=dict(self.generation_config),
                actual_batch_size=actual_size,
            )
            for sequence, token_metrics in zip(decoded, metrics)
        )

    def _is_oom(self, error: BaseException) -> bool:
        oom_type = getattr(self._torch, "OutOfMemoryError", ())
        return isinstance(error, oom_type) or "out of memory" in str(error).lower()

    def recognize_batch(
        self,
        images: list[str | Path | Image.Image],
        *,
        batch_size: int | None = None,
    ) -> tuple[OCRBatchResult, ...]:
        prepared = [self._prepare_image(image) for image in images]
        requested_size = max(1, int(batch_size or self.batch_size))
        results: list[OCRBatchResult] = []
        retries: list[BatchRetryEvent] = []
        offset = 0
        while offset < len(prepared):
            attempt_size = min(requested_size, len(prepared) - offset)
            while True:
                try:
                    batch = prepared[offset : offset + attempt_size]
                    results.extend(self._infer_batch(batch))
                    offset += attempt_size
                    break
                except RuntimeError as error:
                    if not self._is_oom(error) or attempt_size <= 1:
                        raise
                    retry_size = max(1, attempt_size // 2)
                    retries.append(
                        BatchRetryEvent(offset, attempt_size, retry_size, type(error).__name__)
                    )
                    attempt_size = retry_size
                    cuda = getattr(self._torch, "cuda", None)
                    if cuda is not None and cuda.is_available():
                        cuda.empty_cache()
        self.last_batch_retries = tuple(retries)
        return tuple(results)

    def __call__(self, img_or_path: str | Path | Image.Image) -> str:
        return self.recognize_batch([img_or_path], batch_size=1)[0].text


def _post_process(text: str) -> str:
    """Apply manga-ocr's lightweight output cleanup without importing it."""

    text = "".join(str(text or "").split())
    text = text.replace("…", "...")
    text = re.sub(r"[・.]{2,}", lambda match: "." * len(match.group(0)), text)
    return unicodedata.normalize("NFKC", text)


def check_runtime_dependencies() -> dict[str, str]:
    """Import-only health check used by the CLI doctor command."""

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
    if str(transformers.__version__) != TESTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers 必須鎖定 {TESTED_TRANSFORMERS_VERSION}，目前為 {transformers.__version__}"
        )
    versions["torch"] = str(getattr(torch, "__version__", "unknown"))
    versions["transformers"] = str(transformers.__version__)
    versions["fugashi"] = str(getattr(fugashi, "__version__", "installed"))
    versions["unidic_lite"] = str(getattr(unidic_lite, "__version__", "installed"))
    return versions
