"""OpenCV 對 Windows 非 ASCII 路徑的安全讀寫。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path: str | Path, image: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    if suffix == ".jpeg":
        suffix = ".jpg"
    try:
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            return False
        encoded.tofile(str(path))
        return True
    except (OSError, cv2.error):
        return False
