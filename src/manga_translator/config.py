"""設定檔載入、路徑解析與驗證。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OpenRouterConfig(BaseModel):
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    batch_size: int = Field(default=20, ge=1, le=200)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    retry_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    translation_mode: Literal["batch", "context"] = "context"
    context_size: int = Field(default=5, ge=0, le=20)
    page_context_mode: Literal["window", "page"] = "page"
    request_timeout_sec: float = Field(default=90.0, ge=10.0, le=600.0)
    content_retries: int = Field(default=2, ge=0, le=5)
    validate_translation: bool = True
    max_output_length_ratio: float = Field(default=4.0, ge=1.5, le=20.0)

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v: object) -> str:
        cfg_key = str(v or "").strip()
        if cfg_key and cfg_key != "YOUR_OPENROUTER_API_KEY":
            return cfg_key

        env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        return env_key or cfg_key


class PathsConfig(BaseModel):
    input_dir: Path = Path("./input")
    output_dir: Path = Path("./output")
    glossary: Path = Path("./glossary.json")
    font: Path = Path("./fonts/Iansui-Regular.ttf")
    font_fallback: Path = Path("./fonts/NotoSansCJKtc-Regular.otf")


class DetectionConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: Path = Path("./models/comictextdetector.pt")
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    input_size: int = Field(default=1024, ge=320, le=3072)
    # 額外尺寸會和主偵測結果合併。空陣列代表只跑一次。
    additional_input_sizes: list[int] = Field(default_factory=list)
    half: bool = False
    nms_thresh: float = Field(default=0.35, ge=0.05, le=0.95)
    conf_thresh: float = Field(default=0.30, ge=0.01, le=0.99)
    mask_thresh: float = Field(default=0.30, ge=0.01, le=0.99)

    # refined mask 只保留同時受到 raw segmentation 支持的像素，避免人物線稿、
    # 玻璃反光與網點被 annotation refinement 一起擴進擦除範圍。
    keep_undetected_mask: bool = False
    raw_support_threshold: int = Field(default=30, ge=1, le=254)
    raw_support_dilate: int = Field(default=2, ge=0, le=32)

    # YOLO 沒形成文字框時，從 segmentation mask 回收候選區域。
    # 這條路徑對漫畫線稿非常敏感，因此預設關閉；開啟後仍只使用候選自身的
    # 像素 mask，絕不退回整個矩形框擦除。
    mask_fallback_enabled: bool = False
    mask_fallback_primary_only: bool = True
    mask_fallback_threshold: int = Field(default=80, ge=1, le=254)
    mask_fallback_min_area: int = Field(default=48, ge=1)
    mask_fallback_max_area_ratio: float = Field(default=0.20, gt=0.0, le=1.0)
    mask_fallback_padding: int = Field(default=4, ge=0, le=128)
    mask_fallback_min_density: float = Field(default=0.012, ge=0.0, le=1.0)
    mask_fallback_min_components: int = Field(default=2, ge=1, le=100)

    @field_validator("additional_input_sizes")
    @classmethod
    def validate_additional_sizes(cls, values: list[int]) -> list[int]:
        out: list[int] = []
        for value in values:
            size = int(value)
            if size < 320 or size > 3072:
                raise ValueError("additional_input_sizes 必須介於 320 到 3072")
            if size not in out:
                out.append(size)
        return out


class PostprocessConfig(BaseModel):
    min_region_area: int = Field(default=36, ge=1)
    drop_thin_ratio: float = Field(default=0.015, ge=0.0, le=1.0)
    same_text_iom_thresh: float = Field(default=0.55, ge=0.0, le=1.0)
    containment_ratio_thresh: float = Field(default=0.82, ge=0.0, le=1.0)
    substring_match_enabled: bool = True
    fuzzy_text_similarity_thresh: float = Field(default=0.78, ge=0.0, le=1.0)
    duplicate_near_gap_ratio: float = Field(default=0.22, ge=0.0, le=2.0)
    group_center_dist_ratio: float = Field(default=1.2, ge=0.0, le=10.0)
    group_iom_thresh: float = Field(default=0.15, ge=0.0, le=1.0)
    reading_order: Literal["jp_vertical", "auto"] = "jp_vertical"
    enable_ocr_dedup: bool = True
    enable_grouping: bool = True
    enable_group_translate: bool = True
    enable_render_collision_filter: bool = True
    render_collision_mask_iou: float = Field(default=0.48, ge=0.0, le=1.0)
    # 小框若實際字形大多落在大框字形內，即使面積差很多，也視為同一句的
    # 欄位碎片。這是多解析度偵測最常見的重疊來源。
    render_collision_mask_containment: float = Field(default=0.72, ge=0.0, le=1.0)
    render_collision_iom: float = Field(default=0.78, ge=0.0, le=1.0)
    render_collision_containment: float = Field(default=0.88, ge=0.0, le=1.0)
    nested_fragment_containment: float = Field(default=0.92, ge=0.0, le=1.0)
    nested_fragment_text_coverage: float = Field(default=0.62, ge=0.0, le=1.0)


class OCRConfig(BaseModel):
    model_id: str = Field(default="kha-white/manga-ocr-base", min_length=1)
    revision: str = Field(
        default="aa6573bd10b0d446cbf622e29c3e084914df9741",
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-f]{40}$",
    )
    batch_size: int = Field(default=4, ge=1, le=64)
    max_length: int = Field(default=300, ge=2, le=1024)
    pre_upscale: bool = False
    # OCR crop 會向外擴張，避免字體描邊或小假名被 bbox 切掉。
    crop_padding_ratio: float = Field(default=0.08, ge=0.0, le=0.5)
    crop_padding_min_px: int = Field(default=4, ge=0, le=128)
    upscale_min_side: int = Field(default=160, ge=32, le=2048)
    upscale_max_factor: float = Field(default=3.0, ge=1.0, le=8.0)
    ensemble_mode: Literal["off", "adaptive", "always"] = "adaptive"
    use_mask_isolation: bool = True
    use_contrast_variant: bool = True
    use_threshold_variant: bool = True
    use_region_fallback: bool = True
    min_quality_score: float = Field(default=0.46, ge=0.0, le=1.0)
    short_text_min_quality: float = Field(default=0.66, ge=0.0, le=1.0)
    fallback_min_quality_score: float = Field(default=0.74, ge=0.0, le=1.0)
    fallback_min_japanese_chars: int = Field(default=2, ge=1, le=20)
    fallback_min_candidate_agreement: float = Field(default=0.70, ge=0.0, le=1.0)
    reject_non_japanese_noise: bool = True
    reject_symbol_only: bool = True


class TypesettingConfig(BaseModel):
    direction: Literal["vertical", "horizontal", "auto"] = "auto"
    font_size_min: int = Field(default=10, ge=4, le=300)
    # 80 px 會直接把 2K/4K 漫畫原本約 90–130 px 的字壓小。此值只是絕對
    # 安全上限，實際字級仍由原字形估計與可用空間共同決定。
    font_size_max: int = Field(default=180, ge=4, le=500)
    text_color: str | tuple[int, int, int] = "auto"
    line_spacing: float = Field(default=1.02, gt=0.1, le=5.0)
    render_scope: Literal["region", "group_mask", "group_bbox"] = "group_mask"
    layout_from_mask: bool = True
    layout_mask_dilate: int = Field(default=2, ge=0, le=64)
    layout_padding_px: int = Field(default=2, ge=0, le=64)
    layout_mode: Literal["preserve", "tight", "group"] = "preserve"
    layout_padding_ratio: float = Field(default=0.18, ge=0.0, le=1.5)
    font_size_scale: float = Field(default=1.0, gt=0.1, le=4.0)
    max_font_growth_ratio: float = Field(default=1.15, ge=0.5, le=4.0)
    # 先擴張至對話框安全內部、增加欄／行，再考慮縮字。一般情況不低於
    # 原字級的 85%；若 reject_unreadable_layout=true，放不下就保留原文。
    min_font_scale: float = Field(default=0.85, ge=0.3, le=1.5)
    hard_min_font_scale: float = Field(default=0.62, ge=0.2, le=1.5)
    # 只要原字級附近有任何可行排版，就禁止成本函式為了「看起來更滿」
    # 而選到明顯更小的字。只有原尺寸附近完全放不下，才進入有限縮字階段。
    font_preserve_floor_scale: float = Field(default=0.92, ge=0.5, le=1.5)
    adaptive_bubble_layout: bool = True
    bubble_search_expand_ratio: float = Field(default=0.72, ge=0.0, le=4.0)
    bubble_search_max_px: int = Field(default=720, ge=32, le=4096)
    bubble_inner_margin_ratio: float = Field(default=0.07, ge=0.0, le=0.5)
    vertical_char_spacing: float = Field(default=1.03, ge=0.75, le=2.0)
    min_char_spacing_ratio: float = Field(default=0.88, ge=0.5, le=2.0)
    max_char_spacing_ratio: float = Field(default=1.32, ge=0.7, le=3.0)
    min_column_spacing_ratio: float = Field(default=0.92, ge=0.5, le=3.0)
    max_column_spacing_ratio: float = Field(default=2.35, ge=0.8, le=5.0)
    min_chars_per_column: int = Field(default=2, ge=1, le=20)
    reject_unreadable_layout: bool = True
    outline_width: int = Field(default=0, ge=0, le=20)
    outline_color: tuple[int, int, int] | str = "auto"
    inner_padding: int = Field(default=2, ge=0, le=100)
    replace_unsupported_glyphs: bool = True

    @model_validator(mode="after")
    def validate_font_sizes(self) -> TypesettingConfig:
        if self.font_size_max < self.font_size_min:
            raise ValueError("font_size_max 不可小於 font_size_min")
        if self.hard_min_font_scale > self.min_font_scale:
            raise ValueError("hard_min_font_scale 不可大於 min_font_scale")
        if self.font_preserve_floor_scale < self.min_font_scale:
            raise ValueError("font_preserve_floor_scale 不可小於 min_font_scale")
        return self


class InpaintingConfig(BaseModel):
    method: Literal["white", "telea", "navier_stokes", "hybrid"] = "hybrid"
    mask_dilate: int = Field(default=1, ge=0, le=128)
    use_group_union_mask: bool = True
    extra_mask_dilate: int = Field(default=0, ge=0, le=128)
    inpaint_radius: float = Field(default=2.0, ge=0.5, le=20.0)
    allow_bbox_fallback: bool = False
    hybrid_ring_radius: int = Field(default=5, ge=1, le=64)
    hybrid_flat_std_threshold: float = Field(default=18.0, ge=0.0, le=128.0)
    hybrid_dominant_color_tolerance: int = Field(default=18, ge=1, le=128)
    hybrid_dominant_color_ratio: float = Field(default=0.78, ge=0.0, le=1.0)
    hybrid_min_ring_pixels: int = Field(default=24, ge=1)
    # 平坦對話框裡，detector mask 常只覆蓋黑色筆畫核心。只在已確認背景
    # 近似純色時，向外吸收與背景有明顯色差的反鋸齒邊緣，避免留下淡灰原文。
    hybrid_flat_edge_expand: int = Field(default=3, ge=0, le=16)
    hybrid_flat_edge_contrast: int = Field(default=10, ge=1, le=128)
    hybrid_flat_edge_max_growth: float = Field(default=3.2, ge=1.0, le=12.0)
    # OCR 或翻譯失敗時保留原文，避免產生空白對話框。
    only_translated_groups: bool = True


class AppConfig(BaseModel):
    openrouter: OpenRouterConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    typesetting: TypesettingConfig = Field(default_factory=TypesettingConfig)
    inpainting: InpaintingConfig = Field(default_factory=InpaintingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        config_path = Path(path).expanduser().resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = cls(**data)

        # 所有相對路徑都以 config.yaml 所在目錄為基準，而不是目前 shell cwd。
        base_dir = config_path.parent

        def resolve(value: Path) -> Path:
            value = value.expanduser()
            return value.resolve() if value.is_absolute() else (base_dir / value).resolve()

        config.paths = config.paths.model_copy(
            update={
                "input_dir": resolve(config.paths.input_dir),
                "output_dir": resolve(config.paths.output_dir),
                "glossary": resolve(config.paths.glossary),
                "font": resolve(config.paths.font),
                "font_fallback": resolve(config.paths.font_fallback),
            }
        )
        config.detection = config.detection.model_copy(
            update={"model_path": resolve(config.detection.model_path)}
        )
        return config
