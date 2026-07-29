"""Page-level staged OCR with durable per-view cache entries."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from ..manga_ocr_runtime import GenerationTokenMetrics, OCRBatchResult
from ..storage.job_store import JobStore


class BatchOCRRuntime(Protocol):
    model_id: str
    revision: str
    generation_config: dict[str, object]

    def recognize_batch(
        self, images: list[Image.Image], *, batch_size: int | None = None
    ) -> tuple[OCRBatchResult, ...]: ...


@dataclass(frozen=True)
class RegionOCRViews:
    region_id: str
    raw: np.ndarray
    mask: np.ndarray | None = None
    mask_isolated: np.ndarray | None = None
    contrast: np.ndarray | None = None
    threshold: np.ndarray | None = None
    constituents: tuple[np.ndarray, ...] = ()
    geometry_complex: bool = False


@dataclass(frozen=True)
class ViewOCRResult:
    region_id: str
    view_type: str
    result: OCRBatchResult
    cache_key: str
    cache_hit: bool


@dataclass(frozen=True)
class StagedRegionOCR:
    region_id: str
    selected: OCRBatchResult
    provisional_score: float
    disagreement: float
    views: tuple[ViewOCRResult, ...]
    derived_views_are_independent_votes: bool = False


def ocr_view_cache_key(
    image: np.ndarray,
    *,
    mask: np.ndarray | None,
    model_id: str,
    model_revision: str,
    preprocess_version: str,
    view_type: str,
    generation_config: dict[str, object],
) -> str:
    contiguous = np.ascontiguousarray(image)
    mask_bytes = b"" if mask is None else memoryview(np.ascontiguousarray(mask)).tobytes()
    metadata = {
        "shape": contiguous.shape,
        "dtype": str(contiguous.dtype),
        "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "model_id": model_id,
        "model_revision": model_revision,
        "preprocess_version": preprocess_version,
        "view_type": view_type,
        "generation_config": generation_config,
    }
    digest = hashlib.sha256(
        json.dumps(
            metadata, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    digest.update(memoryview(contiguous))
    return digest.hexdigest()


def _to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(image[:, :, ::-1]).convert("RGB")


def _decode_result(data: bytes) -> OCRBatchResult:
    payload = json.loads(data)
    metric_payload = payload.pop("metrics")
    metric_payload["token_ids"] = tuple(metric_payload["token_ids"])
    metric_payload["token_logprobs"] = tuple(metric_payload["token_logprobs"])
    metrics = GenerationTokenMetrics(**metric_payload)
    return OCRBatchResult(metrics=metrics, **payload)


class DurableOCRViewCache:
    def __init__(self, store: JobStore) -> None:
        self.store = store

    def get(self, key: str) -> OCRBatchResult | None:
        artifact = self.store.find_artifact(owner_type="ocr_view_cache", owner_id=key)
        return None if artifact is None else _decode_result(self.store.artifacts.read_bytes(artifact.sha256))

    def put(self, key: str, result: OCRBatchResult) -> None:
        payload = json.dumps(
            asdict(result),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        artifact = self.store.store_artifact(
            payload,
            media_type="application/vnd.manga-translator.ocr-view+json",
            owner_type="ocr_view_cache",
            owner_id=key,
        )
        with self.store.transaction() as connection:
            connection.execute(
                """
                DELETE FROM artifact_references
                WHERE owner_type='ocr_view_cache' AND owner_id=? AND sha256<>?
                """,
                (key, artifact.sha256),
            )


def _model_score(result: OCRBatchResult) -> float:
    metrics = result.metrics
    if not result.text or metrics.length_normalized_transition_logprob is None:
        return 0.0
    likelihood = math.exp(max(-20.0, min(0.0, metrics.length_normalized_transition_logprob)))
    margin = metrics.mean_margin or 0.0
    entropy_penalty = min(1.0, (metrics.mean_entropy or 0.0) / 8.0)
    score = max(0.0, min(1.0, 0.72 * likelihood + 0.28 * margin - 0.12 * entropy_penalty))
    if metrics.truncated:
        score *= 0.5
    return float(score)


def _text_disagreement(results: list[ViewOCRResult]) -> float:
    texts = [item.result.text for item in results if item.result.text]
    if len(texts) < 2:
        return 0.0
    longest = max(len(text) for text in texts)
    shortest = min(len(text) for text in texts)
    return 0.0 if len(set(texts)) == 1 else max(0.25, (longest - shortest) / max(1, longest))


class PageOCRStager:
    def __init__(
        self,
        runtime: BatchOCRRuntime,
        cache: DurableOCRViewCache,
        *,
        preprocess_version: str,
        batch_size: int,
        provisional_threshold: float = 0.62,
    ) -> None:
        if not preprocess_version.strip():
            raise ValueError("preprocess_version must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= provisional_threshold <= 1.0:
            raise ValueError("provisional_threshold must be between zero and one")
        self.runtime = runtime
        self.cache = cache
        self.preprocess_version = preprocess_version
        self.batch_size = batch_size
        self.provisional_threshold = provisional_threshold

    def _resolve(
        self,
        requests: list[tuple[str, str, np.ndarray, np.ndarray | None]],
    ) -> list[ViewOCRResult]:
        resolved: list[ViewOCRResult | None] = [None] * len(requests)
        misses: dict[str, tuple[np.ndarray, list[tuple[int, str, str]]]] = {}
        for index, (region_id, view_type, image, mask) in enumerate(requests):
            key = ocr_view_cache_key(
                image,
                mask=mask,
                model_id=self.runtime.model_id,
                model_revision=self.runtime.revision,
                preprocess_version=self.preprocess_version,
                view_type=view_type,
                generation_config=self.runtime.generation_config,
            )
            cached = self.cache.get(key)
            if cached is not None:
                resolved[index] = ViewOCRResult(region_id, view_type, cached, key, True)
            else:
                if key not in misses:
                    misses[key] = (image, [])
                misses[key][1].append((index, region_id, view_type))
        if misses:
            generated = self.runtime.recognize_batch(
                [_to_pil(image) for image, _destinations in misses.values()],
                batch_size=self.batch_size,
            )
            if len(generated) != len(misses):
                raise RuntimeError("staged OCR output count does not match requested views")
            for (key, (_image, destinations)), result in zip(misses.items(), generated):
                self.cache.put(key, result)
                for index, region_id, view_type in destinations:
                    resolved[index] = ViewOCRResult(region_id, view_type, result, key, False)
        if any(item is None for item in resolved):
            raise RuntimeError("staged OCR left unresolved view positions")
        return [item for item in resolved if item is not None]

    @staticmethod
    def _group(results: list[ViewOCRResult]) -> dict[str, list[ViewOCRResult]]:
        grouped: dict[str, list[ViewOCRResult]] = {}
        for result in results:
            grouped.setdefault(result.region_id, []).append(result)
        return grouped

    def run_page(self, regions: tuple[RegionOCRViews, ...]) -> tuple[StagedRegionOCR, ...]:
        region_ids = tuple(region.region_id for region in regions)
        if any(not region_id for region_id in region_ids):
            raise ValueError("region_id must not be empty")
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("region_id must be unique within a page")
        initial_requests = [
            (region.region_id, "raw", region.raw, region.mask) for region in regions
        ]
        initial_requests.extend(
            (region.region_id, "mask", region.mask_isolated, region.mask)
            for region in regions
            if region.mask_isolated is not None
        )
        all_results = self._resolve(initial_requests)
        grouped = self._group(all_results)

        uncertain: set[str] = set()
        for region in regions:
            candidates = grouped[region.region_id]
            score = max((_model_score(item.result) for item in candidates), default=0.0)
            disagreement = _text_disagreement(candidates)
            if score < self.provisional_threshold or disagreement > 0.0 or region.geometry_complex:
                uncertain.add(region.region_id)

        enhanced_requests: list[tuple[str, str, np.ndarray, np.ndarray | None]] = []
        for region in regions:
            if region.region_id not in uncertain:
                continue
            if region.contrast is not None:
                enhanced_requests.append((region.region_id, "contrast", region.contrast, region.mask))
            if region.threshold is not None:
                enhanced_requests.append((region.region_id, "threshold", region.threshold, region.mask))
        enhanced = self._resolve(enhanced_requests)
        all_results.extend(enhanced)
        grouped = self._group(all_results)

        fallback_requests: list[tuple[str, str, np.ndarray, np.ndarray | None]] = []
        for region in regions:
            candidates = grouped[region.region_id]
            score = max((_model_score(item.result) for item in candidates), default=0.0)
            disagreement = _text_disagreement(candidates)
            still_uncertain = (
                score < self.provisional_threshold
                or disagreement > 0.0
                or region.geometry_complex
            )
            if region.region_id in uncertain and still_uncertain:
                fallback_requests.extend(
                    (region.region_id, f"constituent:{index}", image, None)
                    for index, image in enumerate(region.constituents)
                )
        fallback = self._resolve(fallback_requests)
        all_results.extend(fallback)
        grouped = self._group(all_results)

        staged: list[StagedRegionOCR] = []
        for region in regions:
            candidates = grouped[region.region_id]
            selected_view = max(
                candidates,
                key=lambda item: (_model_score(item.result), -len(item.view_type), item.view_type),
            )
            disagreement = _text_disagreement(candidates)
            score = max(0.0, _model_score(selected_view.result) - 0.2 * disagreement)
            staged.append(
                StagedRegionOCR(
                    region_id=region.region_id,
                    selected=selected_view.result,
                    provisional_score=score,
                    disagreement=disagreement,
                    views=tuple(candidates),
                )
            )
        return tuple(staged)
