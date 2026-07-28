from __future__ import annotations

from pathlib import Path

import numpy as np

from manga_translator.image_io import read_image, write_image


def test_unicode_path_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "漫畫測試頁.png"
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[2:8, 3:10] = (10, 120, 240)
    assert write_image(path, image)
    loaded = read_image(path)
    assert loaded is not None
    assert np.array_equal(loaded, image)
