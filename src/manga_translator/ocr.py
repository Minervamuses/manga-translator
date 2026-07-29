"""manga-ocr 封裝：多視圖 OCR、品質評分、局部 fallback 與內容式 cache。"""

from __future__ import annotations

import hashlib
import itertools
import math
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import cv2
import numpy as np
from PIL import Image
from rich.console import Console

from .config import OCRConfig
from .detector import TextGroup, TextRegion
from .geometry import containment_ratio, iom
from .manga_ocr_runtime import MangaOcrRuntime
from .profiling import profile_span

console = Console()


class OCRInitializationError(RuntimeError):
    """Raised once when the local OCR runtime cannot be initialized."""


_model: MangaOcrRuntime | None = None
_model_init_error: OCRInitializationError | None = None
_ocr_cache: OrderedDict[str, str] = OrderedDict()
_CACHE_MAX_ITEMS = 128

_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
_SEPARATOR_NOISE = {"|", "｜", "¦", "‖", "∥", "￤", "丨"}
_JP_PUNCT_REPLACEMENTS = {
    "．．．": "…",
    "・・・": "…",
    "...": "…",
    "。。": "。",
}


@dataclass(frozen=True)
class OCRCandidate:
    text: str
    normalized: str
    quality: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "normalized": self.normalized,
            "quality": round(float(self.quality), 4),
            "source": self.source,
        }


@dataclass
class OCRResult:
    text: str
    normalized: str
    confidence: float
    source: str
    candidates: list[OCRCandidate] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return bool(self.normalized) and self.confidence > 0.0


def _get_model() -> MangaOcrRuntime:
    global _model, _model_init_error
    if _model is not None:
        return _model
    if _model_init_error is not None:
        raise _model_init_error

    console.print("[bold cyan]載入 manga-ocr 模型中...[/]")
    try:
        model = MangaOcrRuntime()
    except Exception as error:
        wrapped = OCRInitializationError(
            "manga-ocr 模型初始化失敗。專案已改用新版 Transformers 相容載入器，"
            f"但目前環境仍無法建立模型：{error}"
        )
        _model_init_error = wrapped
        raise wrapped from error

    _model = model
    console.print("[bold green]manga-ocr 模型載入完成[/]")
    return model


def initialize_ocr_model() -> None:
    """Initialize OCR exactly once before a page/batch starts."""

    _get_model()


def _reset_ocr_state_for_tests() -> None:
    global _model, _model_init_error
    _model = None
    _model_init_error = None
    _ocr_cache.clear()


def clear_ocr_result_cache() -> None:
    """Discard pixel-result reuse while keeping the initialized model warm."""

    _ocr_cache.clear()


def sanitize_ocr_text(text: str) -> str:
    """移除 OCR 常見控制字元、replacement char 與不可見雜訊。"""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    chars: list[str] = []
    for char in text:
        if char in _ZERO_WIDTH or char == "\ufffd":
            continue
        if char in _SEPARATOR_NOISE:
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Cs", "Co", "Cn"}:
            continue
        chars.append(char)
    cleaned = "".join(chars).strip()

    for old, new in _JP_PUNCT_REPLACEMENTS.items():
        cleaned = cleaned.replace(old, new)

    # manga-ocr 偶爾會在每個字之間插入空白；漫畫對白不需要保留排版空白。
    return "".join(cleaned.split())


def normalize_ocr_text(text: str, weak: bool = False) -> str:
    """正規化 OCR 文字；weak 模式額外忽略大部分標點，供去重比較。"""
    normalized = sanitize_ocr_text(text)
    if not weak:
        return normalized

    chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith(("P", "Z")):
            continue
        # 裝飾符號不影響是否為同一句，但保留日文長音符號與迭字符。
        if category.startswith("S") and char not in {"々", "〆", "ヶ", "ー"}:
            continue
        chars.append(char)
    return "".join(chars)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x323AF
    )


def _is_kana(char: str) -> bool:
    code = ord(char)
    return 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF


def _max_run_length(text: str) -> int:
    if not text:
        return 0
    longest = current = 1
    for previous, current_char in itertools.pairwise(text):
        if current_char == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def ocr_quality_score(text: str) -> float:
    """估計 OCR 結果是否像可翻譯的日文；不是語言模型信心值。"""
    cleaned = sanitize_ocr_text(text)
    if not cleaned:
        return 0.0

    visible = [char for char in cleaned if not char.isspace()]
    if not visible:
        return 0.0

    kana = sum(_is_kana(char) for char in visible)
    cjk = sum(_is_cjk(char) for char in visible)
    latin = sum(char.isascii() and char.isalpha() for char in visible)
    digits = sum(char.isdigit() for char in visible)
    punctuation = sum(unicodedata.category(char).startswith(("P", "S")) for char in visible)
    private_or_unknown = sum(
        unicodedata.category(char) in {"Co", "Cn", "Cs", "Cc"} for char in visible
    )

    length = len(visible)
    japanese_ratio = (kana + cjk) / length
    useful_ratio = (kana + cjk + latin + digits + punctuation) / length

    score = 0.18
    score += 0.52 * japanese_ratio
    score += 0.18 * min(1.0, useful_ratio)
    score += min(0.10, math.log2(length + 1) * 0.025)

    if kana + cjk == 0 and latin > 0:
        # 英文擬聲詞或標語可以是合法漫畫文字，但信心不應高於日文。
        score = max(score, 0.38 if latin / length >= 0.6 else score)
    if punctuation == length:
        return 0.0
    if length == 1:
        if kana == 1:
            score = min(score, 0.72)
        elif cjk == 1:
            # 單一漢字很容易是衣服紋樣、髮絲或背景招牌誤辨；保留可能性，
            # 但不給足以通過 fallback 嚴格門檻的高分。
            score = min(score, 0.56)
        else:
            score = min(score, 0.42)
    elif length == 2 and kana + cjk == 1:
        score = min(score, 0.58)
    if private_or_unknown:
        score -= min(0.5, private_or_unknown / length)
    if _max_run_length(cleaned) >= 6:
        score -= 0.18
    if useful_ratio < 0.55:
        score -= 0.25

    return float(max(0.0, min(1.0, score)))


def _meaningful_japanese_count(text: str) -> int:
    cleaned = sanitize_ocr_text(text)
    return sum(_is_kana(char) or _is_cjk(char) for char in cleaned)


def _candidate_agreement_with_best(result: OCRResult) -> float:
    if not result.normalized:
        return 0.0
    best = OCRCandidate(
        text=result.text,
        normalized=result.normalized,
        quality=result.confidence,
        source=result.source,
    )
    agreements = [
        _candidate_similarity(best, candidate)
        for candidate in result.candidates
        if candidate.normalized and candidate.source != result.source
    ]
    return max(agreements, default=0.0)


def assess_ocr_result(
    result: OCRResult,
    cfg: OCRConfig,
    *,
    fallback_only: bool = False,
) -> tuple[bool, str]:
    """把 OCR 的語言品質與偵測來源一起判斷，避免線稿幻覺進入翻譯。

    一般 CTD 框允許「え」「嗯」之類極短台詞；純 mask fallback 因誤抓線稿風險
    高，必須同時具備較高分數、至少兩個日文字元，以及不同 preprocessing 視圖
    對結果的基本一致性。
    """
    normalized = normalize_ocr_text(result.text, weak=False)
    if not normalized:
        return False, "empty_ocr"

    visible = [char for char in normalized if not char.isspace()]
    if not visible:
        return False, "empty_ocr"
    if cfg.reject_symbol_only and not any(char.isalnum() for char in visible):
        return False, "symbol_only_ocr"

    quality = float(result.confidence)
    if cfg.reject_non_japanese_noise and quality < cfg.min_quality_score:
        return False, f"low_ocr_quality:{quality:.3f}"

    jp_count = _meaningful_japanese_count(normalized)
    if len(visible) <= 2 and quality < cfg.short_text_min_quality:
        return False, f"weak_short_ocr:{quality:.3f}"

    if fallback_only:
        if quality < cfg.fallback_min_quality_score:
            return False, f"fallback_low_quality:{quality:.3f}"
        if jp_count < cfg.fallback_min_japanese_chars:
            return False, f"fallback_too_short:{jp_count}"
        agreement = _candidate_agreement_with_best(result)
        if agreement < cfg.fallback_min_candidate_agreement:
            return False, f"fallback_disagreement:{agreement:.3f}"

    return True, ""


def _image_cache_key(image: np.ndarray, namespace: str = "") -> str:
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(namespace.encode("utf-8", errors="ignore"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(memoryview(contiguous))
    return digest.hexdigest()


def ocr_cache_key(
    image_path: str,
    bbox: tuple[int, int, int, int],
    text_hash_source: str = "",
) -> str:
    """保留舊 API；新流程實際以 crop bytes 作 cache key，避免同路徑舊結果污染。"""
    payload = f"{image_path}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}|{text_hash_source}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _remember_cache(key: str, text: str) -> None:
    _ocr_cache.pop(key, None)
    _ocr_cache[key] = text
    if len(_ocr_cache) > _CACHE_MAX_ITEMS:
        _ocr_cache.popitem(last=False)


def _pil_image(region_image: np.ndarray) -> Image.Image:
    if region_image.ndim == 2:
        rgb = cv2.cvtColor(region_image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _ocr_image(region_image: np.ndarray, cache_key: str | None = None) -> str:
    if region_image.size == 0:
        return ""

    key = cache_key or _image_cache_key(region_image)
    if key in _ocr_cache:
        return _ocr_cache[key]

    model = _get_model()
    device = str(getattr(model, "device", "cpu"))
    with profile_span("ocr_forward", gpu=device.startswith("cuda"), device=device):
        text = sanitize_ocr_text(
            model.recognize_batch([_pil_image(region_image)], batch_size=1)[0].text
        )
    _remember_cache(key, text)
    return text


def ocr_region(region_image: np.ndarray, cache_key: str | None = None) -> str:
    return _ocr_image(region_image, cache_key=cache_key)


def ocr_regions_batch(
    region_images: list[np.ndarray],
    cache_prefix: str = "",
) -> list[str]:
    del cache_prefix  # cache 以像素內容為準；路徑或批次位置不應製造重複項目。
    if not region_images:
        return []
    keys = [_image_cache_key(image) for image in region_images]
    results: list[str | None] = [None] * len(region_images)
    misses: dict[str, tuple[np.ndarray, list[int]]] = {}
    for index, (key, image) in enumerate(zip(keys, region_images)):
        cached = _ocr_cache.get(key)
        if cached is not None:
            _ocr_cache.move_to_end(key)
            results[index] = cached
            continue
        if key not in misses:
            misses[key] = (image, [])
        misses[key][1].append(index)
    if misses:
        model = _get_model()
        device = str(getattr(model, "device", "cpu"))
        images = [_pil_image(image) for image, _indices in misses.values()]
        with profile_span("ocr_forward", gpu=device.startswith("cuda"), device=device):
            generated = model.recognize_batch(images)
        if len(generated) != len(images):
            raise RuntimeError("OCR batch output count does not match input count")
        for (key, (_image, indices)), item in zip(misses.items(), generated):
            text = sanitize_ocr_text(item.text)
            _remember_cache(key, text)
            for index in indices:
                results[index] = text
    if any(result is None for result in results):
        raise RuntimeError("OCR batch left unresolved input positions")
    return [str(result) for result in results]


def _expanded_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    cfg: OCRConfig,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    image_h, image_w = image_shape
    pad = max(cfg.crop_padding_min_px, round(max(w, h) * cfg.crop_padding_ratio))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_w, x + w + pad)
    y2 = min(image_h, y + h + pad)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _crop_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return image[y : y + h, x : x + w].copy()


def _crop_full_mask(mask: np.ndarray | None, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    if mask is None or mask.size == 0:
        return None
    x, y, w, h = bbox
    crop = mask[y : y + h, x : x + w]
    if crop.shape[:2] != (h, w):
        return None
    return crop.copy()


def _crop_group_mask(
    group: TextGroup,
    crop_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    """取得與 OCR crop 對齊的 mask，並相容舊版整頁 group mask。"""
    mask = group.mask
    if mask is None or mask.size == 0:
        return None

    crop_x, crop_y, crop_w, crop_h = crop_bbox
    image_h, image_w = image_shape
    if mask.shape[:2] == (image_h, image_w):
        return _crop_full_mask(mask, crop_bbox)

    group_x, group_y, group_w, group_h = group.bbox
    if mask.shape[:2] != (group_h, group_w):
        mask = cv2.resize(mask, (group_w, group_h), interpolation=cv2.INTER_NEAREST)

    output = np.zeros((crop_h, crop_w), dtype=np.uint8)
    overlap_x1 = max(crop_x, group_x)
    overlap_y1 = max(crop_y, group_y)
    overlap_x2 = min(crop_x + crop_w, group_x + group_w)
    overlap_y2 = min(crop_y + crop_h, group_y + group_h)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return output

    source_x1 = overlap_x1 - group_x
    source_y1 = overlap_y1 - group_y
    source_x2 = source_x1 + (overlap_x2 - overlap_x1)
    source_y2 = source_y1 + (overlap_y2 - overlap_y1)
    target_x1 = overlap_x1 - crop_x
    target_y1 = overlap_y1 - crop_y
    target_x2 = target_x1 + (overlap_x2 - overlap_x1)
    target_y2 = target_y1 + (overlap_y2 - overlap_y1)
    output[target_y1:target_y2, target_x1:target_x2] = mask[
        source_y1:source_y2, source_x1:source_x2
    ]
    return output


def _upscale_for_ocr(image: np.ndarray, cfg: OCRConfig) -> np.ndarray:
    h, w = image.shape[:2]
    min_side = max(1, min(h, w))
    factor = min(cfg.upscale_max_factor, max(1.0, cfg.upscale_min_side / min_side))
    if factor <= 1.02:
        return image
    new_w = max(1, round(w * factor))
    new_h = max(1, round(h * factor))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def _resize_mask_like(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    if mask.shape[:2] == image.shape[:2]:
        return mask
    return cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)


def _normalize_text_polarity(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary_mask = mask > 0
    if not np.any(binary_mask):
        return gray

    inside = gray[binary_mask]
    outside = gray[~binary_mask]
    inside_median = float(np.median(inside)) if inside.size else 127.0
    outside_median = float(np.median(outside)) if outside.size else 255.0

    # manga-ocr 對「深色字／淺色底」最穩；白字黑底先反相。
    if inside_median > outside_median:
        return 255 - gray
    return gray


def _mask_isolated_variant(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = _resize_mask_like(mask, image)
    binary = (mask > 20).astype(np.uint8) * 255
    if not np.any(binary):
        return image

    binary = cv2.dilate(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = _normalize_text_polarity(gray, binary)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    isolated = np.full_like(gray, 255)
    isolated[binary > 0] = gray[binary > 0]
    return cv2.cvtColor(isolated, cv2.COLOR_GRAY2BGR)


def _threshold_variant(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if mask is not None and np.any(mask):
        resized_mask = _resize_mask_like(mask, image)
        gray = _normalize_text_polarity(gray, resized_mask)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _threshold, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # 讓背景維持白色；如果整張偏黑就反相。
    if float(binary.mean()) < 127:
        binary = 255 - binary
    if mask is not None and np.any(mask):
        resized_mask = _resize_mask_like(mask, image)
        expanded = cv2.dilate((resized_mask > 0).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        binary[expanded == 0] = 255
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _contrast_variant(image: np.ndarray) -> np.ndarray:
    """提升低對比、描邊或網點背景上的文字，同時避免過度銳化。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def _make_candidate(image: np.ndarray, source: str, cfg: OCRConfig) -> OCRCandidate:
    with profile_span("ocr_view", source=source):
        prepared = _upscale_for_ocr(image, cfg) if cfg.pre_upscale else image
        # 相同像素在不同 detector pass／group 中應共用 OCR 結果；source 只供 debug。
        key = _image_cache_key(prepared)
        text = _ocr_image(prepared, cache_key=key)
        normalized = normalize_ocr_text(text, weak=False)
    return OCRCandidate(
        text=text,
        normalized=normalized,
        quality=ocr_quality_score(text),
        source=source,
    )


def _make_candidates_batch(
    requests: list[tuple[np.ndarray, str]],
    cfg: OCRConfig,
) -> list[OCRCandidate]:
    if not requests:
        return []
    prepared = [
        _upscale_for_ocr(image, cfg) if cfg.pre_upscale else image
        for image, _source in requests
    ]
    with profile_span("ocr_view_batch", view_count=len(prepared)):
        texts = ocr_regions_batch(prepared)
    return [
        OCRCandidate(
            text=text,
            normalized=normalize_ocr_text(text, weak=False),
            quality=ocr_quality_score(text),
            source=source,
        )
        for text, (_image, source) in zip(texts, requests, strict=True)
    ]


def _candidate_similarity(a: OCRCandidate, b: OCRCandidate) -> float:
    a_norm = normalize_ocr_text(a.text, weak=True)
    b_norm = normalize_ocr_text(b.text, weak=True)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if a_norm in b_norm or b_norm in a_norm:
        return min(len(a_norm), len(b_norm)) / max(len(a_norm), len(b_norm))
    return SequenceMatcher(None, a_norm, b_norm, autojunk=False).ratio()


def _dedupe_candidates(candidates: list[OCRCandidate]) -> list[OCRCandidate]:
    # 同一文字若由 raw／mask／threshold 等不同視圖各自讀出，這正是重要的
    # 一致性證據，不能只因 normalized 相同就折疊成一筆。
    by_text_and_source: dict[tuple[str, str], OCRCandidate] = {}
    for candidate in candidates:
        normalized = candidate.normalized
        if not normalized:
            continue
        key = (normalized, candidate.source)
        previous = by_text_and_source.get(key)
        if previous is None or candidate.quality > previous.quality:
            by_text_and_source[key] = candidate
    return list(by_text_and_source.values())


def _select_best_candidate(candidates: list[OCRCandidate]) -> OCRCandidate | None:
    candidates = _dedupe_candidates(candidates)
    if not candidates:
        return None

    lengths = sorted(len(candidate.normalized) for candidate in candidates if candidate.normalized)
    median_length = lengths[len(lengths) // 2] if lengths else 1

    best: OCRCandidate | None = None
    best_score = -1.0
    for candidate in candidates:
        similarities = [
            _candidate_similarity(candidate, other)
            for other in candidates
            if other is not candidate
        ]
        agreement = max(similarities, default=0.0)
        score = candidate.quality * 0.72 + agreement * 0.28

        # 當另一候選只是本候選的缺字版本時，偏向較完整的文字。
        for other in candidates:
            if other is candidate:
                continue
            small = normalize_ocr_text(other.text, weak=True)
            large = normalize_ocr_text(candidate.text, weak=True)
            if small and small in large and len(large) > len(small):
                score += min(0.12, (len(large) - len(small)) * 0.025)

        # 避免單一 preprocessing 幻覺出遠長於其他候選的字串。
        if median_length > 0 and len(candidate.normalized) > median_length * 2.8 + 6:
            score -= 0.18

        if score > best_score:
            best = candidate
            best_score = score
    return best


def _ordered_group_regions(
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
) -> list[TextRegion]:
    regions = [regions_by_id[rid] for rid in group.region_ids if rid in regions_by_id]
    if group.vertical:
        return sorted(regions, key=lambda r: (-(r.x + r.w / 2.0), r.y + r.h / 2.0))
    return sorted(regions, key=lambda r: (r.y + r.h / 2.0, r.x + r.w / 2.0))


def _crop_region_mask(
    region: TextRegion,
    crop_bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    local = region.local_mask
    if local is None or local.size == 0 or not np.any(local):
        return None
    if local.shape[:2] != (region.h, region.w):
        local = cv2.resize(local, (region.w, region.h), interpolation=cv2.INTER_NEAREST)

    crop_x, crop_y, crop_w, crop_h = crop_bbox
    output = np.zeros((crop_h, crop_w), dtype=np.uint8)
    overlap_x1 = max(crop_x, region.x)
    overlap_y1 = max(crop_y, region.y)
    overlap_x2 = min(crop_x + crop_w, region.x + region.w)
    overlap_y2 = min(crop_y + crop_h, region.y + region.h)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return output

    source_x1 = overlap_x1 - region.x
    source_y1 = overlap_y1 - region.y
    source_x2 = source_x1 + (overlap_x2 - overlap_x1)
    source_y2 = source_y1 + (overlap_y2 - overlap_y1)
    target_x1 = overlap_x1 - crop_x
    target_y1 = overlap_y1 - crop_y
    target_x2 = target_x1 + (overlap_x2 - overlap_x1)
    target_y2 = target_y1 + (overlap_y2 - overlap_y1)
    output[target_y1:target_y2, target_x1:target_x2] = local[
        source_y1:source_y2,
        source_x1:source_x2,
    ]
    return output


def _ocr_single_region(
    image: np.ndarray,
    region: TextRegion,
    cfg: OCRConfig,
    namespace: str,
) -> OCRCandidate | None:
    bbox = _expanded_bbox(region.bbox, image.shape[:2], cfg)
    crop = _crop_bbox(image, bbox)
    requests = [(crop, f"{namespace}:raw")]

    crop_mask = _crop_region_mask(region, bbox)
    if crop_mask is not None and cfg.use_mask_isolation:
        requests.append(
            (
                _mask_isolated_variant(crop, crop_mask),
                f"{namespace}:mask",
            )
        )
    if cfg.use_contrast_variant:
        requests.append(
            (
                _contrast_variant(crop),
                f"{namespace}:contrast",
            )
        )
    if cfg.use_threshold_variant:
        requests.append(
            (
                _threshold_variant(crop, crop_mask),
                f"{namespace}:threshold",
            )
        )
    candidates = _make_candidates_batch(requests, cfg)
    return _select_best_candidate(candidates)


def _regions_are_duplicate(a: TextRegion, b: TextRegion) -> bool:
    """只合併幾何上確實指向同一塊文字的 region OCR。

    單靠文字相同會誤刪漫畫中合法的重複台詞，例如相鄰兩人都說「嗯」。
    """
    return (
        iom(a, b) >= 0.55
        or containment_ratio(a, b) >= 0.82
        or containment_ratio(b, a) >= 0.82
    )


def _candidate_overlap_coverage(a: OCRCandidate, b: OCRCandidate) -> float:
    """How much of the shorter OCR string is explained by the longer one."""

    a_norm = normalize_ocr_text(a.text, weak=True)
    b_norm = normalize_ocr_text(b.text, weak=True)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm in b_norm or b_norm in a_norm:
        return 1.0
    shorter, longer = sorted((a_norm, b_norm), key=len)
    match = SequenceMatcher(None, shorter, longer, autojunk=False).find_longest_match(
        0,
        len(shorter),
        0,
        len(longer),
    )
    return match.size / max(1, len(shorter))


def _nested_candidate_is_same_text(
    region: TextRegion,
    candidate: OCRCandidate,
    existing_region: TextRegion,
    existing: OCRCandidate,
) -> bool:
    if not _regions_are_duplicate(region, existing_region):
        return False

    similarity = _candidate_similarity(candidate, existing)
    overlap_coverage = _candidate_overlap_coverage(candidate, existing)
    if similarity >= 0.72 or overlap_coverage >= 0.68:
        return True

    # The wide multi-column OCR may differ by one or two characters from an
    # individual column.  Never concatenate that full sentence with its nested
    # fragment merely because the OCR typo lowered string similarity.
    small_area = min(region.area, existing_region.area)
    large_area = max(region.area, existing_region.area, 1)
    area_ratio = small_area / large_area
    longer, shorter = sorted((candidate, existing), key=lambda item: len(item.normalized))
    return (
        area_ratio <= 0.42
        and len(longer.normalized) >= len(shorter.normalized) + 2
        and longer.quality >= shorter.quality - 0.12
    )


def _combine_region_candidates(
    candidates: list[tuple[TextRegion, OCRCandidate]],
) -> OCRCandidate | None:
    """Combine constituent OCR without appending a whole sentence to its columns.

    Multi-scale CTD commonly returns both an outer, multi-column region and one
    region per column.  Treating every region as an independent reading unit
    creates ``whole sentence + first half + second half``.  Here the outer OCR and
    the ordered leaf-column OCR are alternative hypotheses, never additive parts.
    """
    if not candidates:
        return None

    # First collapse only near-identical detector passes.  A much smaller nested
    # column is deliberately *not* collapsed here; it is handled as a leaf below.
    collapsed: list[tuple[TextRegion, OCRCandidate]] = []
    for region, candidate in candidates:
        duplicate_index: int | None = None
        for index, (existing_region, existing) in enumerate(collapsed):
            small_area = min(region.area, existing_region.area)
            large_area = max(region.area, existing_region.area, 1)
            area_ratio = small_area / large_area
            same_footprint = iom(region, existing_region) >= 0.62 and area_ratio >= 0.55
            if not same_footprint:
                continue
            similarity = _candidate_similarity(candidate, existing)
            coverage = _candidate_overlap_coverage(candidate, existing)
            if similarity >= 0.66 or coverage >= 0.78:
                duplicate_index = index
                break

        if duplicate_index is None:
            collapsed.append((region, candidate))
            continue

        existing_region, existing = collapsed[duplicate_index]
        winner_region, winner = max(
            ((existing_region, existing), (region, candidate)),
            key=lambda pair: (
                pair[1].quality + min(0.10, len(pair[1].normalized) * 0.006),
                len(pair[1].normalized),
                pair[0].area,
            ),
        )
        collapsed[duplicate_index] = (winner_region, winner)

    if not collapsed:
        return None

    # A candidate that geometrically contains a clearly smaller candidate is an
    # aggregate hypothesis.  Only candidates with no children become leaf columns.
    aggregate_indices: set[int] = set()
    for outer_index, (outer_region, _outer_candidate) in enumerate(collapsed):
        for inner_index, (inner_region, _inner_candidate) in enumerate(collapsed):
            if inner_index == outer_index:
                continue
            size_ratio = inner_region.area / max(1, outer_region.area)
            if (
                size_ratio <= 0.78
                and containment_ratio(inner_region, outer_region) >= 0.88
            ):
                aggregate_indices.add(outer_index)
                break

    leaves = [
        pair for index, pair in enumerate(collapsed) if index not in aggregate_indices
    ]
    alternatives: list[OCRCandidate] = [
        candidate for index, (_region, candidate) in enumerate(collapsed)
        if index in aggregate_indices
    ]

    if leaves:
        leaf_text = "".join(candidate.text for _region, candidate in leaves)
        leaf_quality = sum(candidate.quality for _region, candidate in leaves) / len(leaves)
        alternatives.append(
            OCRCandidate(
                text=leaf_text,
                normalized=normalize_ocr_text(leaf_text),
                quality=min(1.0, leaf_quality + (0.03 if len(leaves) > 1 else 0.0)),
                source="regions:leaves" if aggregate_indices else "regions:combined",
            )
        )

    alternatives = [candidate for candidate in alternatives if candidate.normalized]
    if not alternatives:
        return None

    # Prefer a more complete alternative when it explains most of a shorter one
    # and is not materially less reliable.  This also handles a one-glyph typo in
    # the aggregate OCR without concatenating both hypotheses.
    winner = _select_best_candidate(alternatives)
    for candidate in alternatives:
        if winner is None or candidate is winner:
            continue
        longer, shorter = sorted((winner, candidate), key=lambda item: len(item.normalized), reverse=True)
        if (
            len(longer.normalized) >= len(shorter.normalized) + 2
            and _candidate_overlap_coverage(longer, shorter) >= 0.62
            and longer.quality >= shorter.quality - 0.16
        ):
            winner = longer
    return winner


def ocr_group_detailed(
    image: np.ndarray,
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    cfg: OCRConfig | None = None,
    image_key: str = "",
) -> OCRResult:
    """對群組跑多視圖 OCR，並在需要時比較 constituent-region OCR。"""
    del image_key  # cache 由 crop bytes 決定，刻意不使用可能過期的檔案路徑。
    cfg = cfg or OCRConfig()
    bbox = _expanded_bbox(group.bbox, image.shape[:2], cfg)
    crop = _crop_bbox(image, bbox)
    crop_mask = _crop_group_mask(group, bbox, image.shape[:2])

    regions = _ordered_group_regions(group, regions_by_id)
    has_fallback_region = any(region.source == "mask_fallback" for region in regions)
    has_duplicate_region = any(region.candidate_duplicate for region in regions)

    # adaptive 也固定比較一次 mask-isolated OCR。單靠「像不像日文」的分數
    # 無法辨識「結果很合理、但只讀到半句」；raw 與 mask 互相比較才有機會發現。
    compare_mask = (
        cfg.ensemble_mode in {"adaptive", "always"}
        and crop_mask is not None
        and np.any(crop_mask)
        and cfg.use_mask_isolation
    )
    initial_requests = [(crop, f"group:{group.id}:raw")]
    if compare_mask:
        initial_requests.append(
            (
                _mask_isolated_variant(crop, crop_mask),
                f"group:{group.id}:mask",
            )
        )
    candidates = _make_candidates_batch(initial_requests, cfg)
    raw_candidate = candidates[0]
    mask_candidate = candidates[1] if compare_mask else None

    raw_mask_similarity = (
        _candidate_similarity(raw_candidate, mask_candidate)
        if mask_candidate is not None
        else 1.0
    )
    raw_mask_length_gap = (
        abs(len(raw_candidate.normalized) - len(mask_candidate.normalized))
        if mask_candidate is not None
        else 0
    )
    adaptive_disagreement = (
        mask_candidate is not None
        and bool(raw_candidate.normalized or mask_candidate.normalized)
        and (raw_mask_similarity < 0.88 or raw_mask_length_gap >= 2)
    )
    needs_enhanced_ensemble = (
        cfg.ensemble_mode == "always"
        or (
            cfg.ensemble_mode == "adaptive"
            and (
                raw_candidate.quality < 0.76
                or len(regions) > 1
                or has_fallback_region
                or has_duplicate_region
                or adaptive_disagreement
            )
        )
    )

    enhanced_requests: list[tuple[np.ndarray, str]] = []
    if needs_enhanced_ensemble and cfg.use_contrast_variant:
        enhanced_requests.append(
            (
                _contrast_variant(crop),
                f"group:{group.id}:contrast",
            )
        )
    if needs_enhanced_ensemble and cfg.use_threshold_variant:
        enhanced_requests.append(
            (
                _threshold_variant(crop, crop_mask),
                f"group:{group.id}:threshold",
            )
        )
    candidates.extend(_make_candidates_batch(enhanced_requests, cfg))

    primary_best = _select_best_candidate(candidates)
    primary_quality = primary_best.quality if primary_best is not None else 0.0
    needs_region_fallback = cfg.use_region_fallback and (
        len(regions) > 1
        or primary_quality < max(0.66, cfg.min_quality_score)
        or has_fallback_region
        or has_duplicate_region
        or adaptive_disagreement
    )
    if needs_region_fallback:
        region_candidates: list[tuple[TextRegion, OCRCandidate]] = []
        for index, region in enumerate(regions):
            candidate = _ocr_single_region(
                image,
                region,
                cfg,
                namespace=f"region:{group.id}:{index}",
            )
            if candidate is not None and candidate.normalized:
                region_candidates.append((region, candidate))
        combined = _combine_region_candidates(region_candidates)
        if combined is not None:
            candidates.append(combined)

    best = _select_best_candidate(candidates)
    if best is None:
        return OCRResult("", "", 0.0, "none", candidates=[])

    confidence = best.quality
    if cfg.reject_non_japanese_noise and confidence < cfg.min_quality_score:
        return OCRResult(
            text=best.text,
            normalized=best.normalized,
            confidence=confidence,
            source=best.source,
            candidates=_dedupe_candidates(candidates),
        )

    return OCRResult(
        text=best.text,
        normalized=best.normalized,
        confidence=confidence,
        source=best.source,
        candidates=_dedupe_candidates(candidates),
    )


def ocr_group(
    image: np.ndarray,
    group: TextGroup,
    regions_by_id: dict[str, TextRegion],
    image_key: str = "",
    cfg: OCRConfig | None = None,
) -> str:
    """向後相容 wrapper；新流程請用 :func:`ocr_group_detailed`。"""
    return ocr_group_detailed(
        image=image,
        group=group,
        regions_by_id=regions_by_id,
        cfg=cfg,
        image_key=image_key,
    ).text
