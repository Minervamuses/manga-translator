from __future__ import annotations

import numpy as np

from manga_translator.ctd.utils import db_utils


def test_polygon_offset_works_without_optional_pyclipper(monkeypatch) -> None:
    monkeypatch.setattr(db_utils, "pyclipper", None)
    box = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)

    expanded = db_utils._offset_polygon(box, 5)
    shrunk = db_utils._offset_polygon(box, -3)

    assert expanded.shape[1] == 2
    assert expanded[:, 0].min() <= 5.1
    assert expanded[:, 0].max() >= 34.9
    assert shrunk.shape == (4, 2)
    assert np.allclose(shrunk.min(axis=0), (13, 13), atol=0.1)
    assert np.allclose(shrunk.max(axis=0), (27, 27), atol=0.1)
