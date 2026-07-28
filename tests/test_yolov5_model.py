from __future__ import annotations

from manga_translator.ctd.models.yolov5.yolo import Model, parse_model


def test_parse_model_scales_channels_with_yolov5_divisibility_rule() -> None:
    model, saved = parse_model(
        {
            "anchors": 3,
            "nc": 1,
            "depth_multiple": 1.0,
            "width_multiple": 0.5,
            "backbone": [[-1, 1, "Conv", [15, 3, 1]]],
            "head": [],
        },
        ch=[3],
    )

    assert model[0].conv.in_channels == 3
    assert model[0].conv.out_channels == 8
    assert saved == []


def test_model_initialization_has_all_required_yolov5_helpers() -> None:
    model = Model(
        {
            "anchors": [[10, 13, 16, 30, 33, 23]],
            "nc": 1,
            "depth_multiple": 1.0,
            "width_multiple": 1.0,
            "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
            "head": [[[-1], 1, "Detect", [1, [[10, 13, 16, 30, 33, 23]]]]],
        }
    )

    assert model.model[-1].stride.tolist() == [2.0]
    assert not hasattr(model.fuse().model[0], "bn")
