from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from manga_translator.manga_ocr_runtime import (
    DEFAULT_MODEL_REVISION,
    MangaOcrRuntime,
)


@pytest.mark.model_integration
def test_real_model_batch_one_equivalence_and_batch_order() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "samples" / "before_fix" / "0188_ive_hwa002.jpg"
    if not source.is_file():
        pytest.skip("verified source page is unavailable")
    with Image.open(source) as page:
        crops = (
            page.crop((1150, 50, 1260, 250)).convert("RGB"),
            page.crop((920, 180, 1030, 390)).convert("RGB"),
        )
    runtime = MangaOcrRuntime(
        revision=DEFAULT_MODEL_REVISION,
        max_length=80,
        batch_size=2,
    )

    singles = tuple(runtime(image) for image in crops)
    batched = runtime.recognize_batch(list(crops), batch_size=2)

    assert tuple(item.text for item in batched) == singles
    assert all(item.model_revision == DEFAULT_MODEL_REVISION for item in batched)
    assert all(item.metrics.token_ids for item in batched)
