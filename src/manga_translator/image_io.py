"""OpenCV 對 Windows 非 ASCII 路徑的安全讀寫。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageEncodeError(RuntimeError):
    """OpenCV could not encode an output image."""


class ImageWriteError(OSError):
    """Encoded image bytes could not be persisted."""


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
    try:
        write_image_or_raise(path, image)
        return True
    except (ImageEncodeError, ImageWriteError):
        return False


def write_image_or_raise(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    suffix = path.suffix.lower() or ".png"
    if suffix == ".jpeg":
        suffix = ".jpg"
    try:
        ok, encoded = cv2.imencode(suffix, image)
    except cv2.error as error:
        raise ImageEncodeError(f"無法編碼圖片：{path}") from error
    if not ok:
        raise ImageEncodeError(f"無法編碼圖片：{path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded.tofile(str(path))
    except OSError as error:
        raise ImageWriteError(f"無法寫入圖片：{path}") from error
