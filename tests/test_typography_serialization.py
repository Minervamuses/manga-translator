from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from manga_translator.typography.fonts import FontRole
from manga_translator.typography.layout import (
    AcceptedLayout,
    FontChoice,
    LayoutCandidate,
    LayoutDirection,
)
from manga_translator.typography.serialization import (
    decode_layout_bundle,
    encode_layout_bundle,
)
from manga_translator.typography.shaping import ShapedFontRun
from manga_translator.typography.solver import layout_plan_hash


def _accepted() -> AcceptedLayout:
    alpha = np.zeros((24, 20), dtype=np.uint8)
    alpha[4:20, 7:13] = 255
    candidate = LayoutCandidate(
        FontChoice(FontRole.NEUTRAL_SANS),
        14,
        LayoutDirection.VERTICAL,
        ("測試", "翻譯"),
        (2,),
        1.0,
        0.1,
        (10.0, 12.0),
        0.0,
    )
    runs = (
        ShapedFontRun(
            text="測試翻譯",
            font_sha256="b" * 64,
            font_path="fixture.ttf",
            glyph_coverage=tuple(map(ord, "測試翻譯")),
            direction="ttb",
            language="zh-Hant",
            features=("vert", "vrt2"),
            bbox=(7.0, 4.0, 13.0, 20.0),
            advance=16.0,
            anchor=(10.0, 12.0),
        ),
    )
    return AcceptedLayout(
        candidate,
        alpha,
        1.0,
        0.25,
        layout_plan_hash(candidate, alpha, runs),
        runs,
    )


def test_layout_bundle_round_trip_is_canonical_and_retains_shaping() -> None:
    encoded = encode_layout_bundle({"g001": _accepted()})
    decoded = decode_layout_bundle(encoded)

    assert encode_layout_bundle(decoded) == encoded
    assert decoded["g001"].shaped_runs[0].font_sha256 == _accepted().shaped_runs[0].font_sha256
    assert decoded["g001"].shaped_runs[0].glyph_coverage == _accepted().shaped_runs[0].glyph_coverage
    assert np.array_equal(decoded["g001"].alpha, _accepted().alpha)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: replace(plan, shaped_runs=()),
        lambda plan: replace(plan, containment=0.994),
        lambda plan: replace(plan, plan_hash="tampered"),
    ],
)
def test_encoder_rejects_layouts_that_are_not_hard_valid(mutation) -> None:
    with pytest.raises(ValueError):
        encode_layout_bundle({"g001": mutation(_accepted())})


def test_decoder_rejects_tampered_plan_payload() -> None:
    payload = json.loads(encode_layout_bundle({"g001": _accepted()}))
    payload["plans"]["g001"]["candidate"]["font_size"] += 1
    tampered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(ValueError, match="plan hash"):
        decode_layout_bundle(tampered)
