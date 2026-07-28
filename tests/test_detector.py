from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from manga_translator import detector as detector_module
from manga_translator.config import DetectionConfig, PostprocessConfig
from manga_translator.ctd.basemodel import (
    DetectorRuntimeContractError,
    assert_detector_runtime_contract,
)
from manga_translator.ctd.inference import TextDetector, UnsupportedDetectorBackendError
from manga_translator.detector import (
    TextRegion,
    _classify_cuda_error,
    _conservative_text_mask,
    _extract_mask_fallback_regions,
    _get_detector,
    _resolve_detector_runtime,
    detect_text_regions,
    postprocess_regions,
)


def test_text_detector_passes_resolved_half_mode_to_base_model(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTextDetBase:
        def __init__(self, _model_path, *, device, half, act):
            captured.update(device=device, half=half, act=act)

    monkeypatch.setattr("manga_translator.ctd.inference.TextDetBase", FakeTextDetBase)

    detector = TextDetector("detector.pt", device="cpu", half=True)

    assert captured["half"] is False
    assert detector.half is False
    assert detector.runtime_issues[0]["code"] == "detector_fp16_downgraded"

    detector = TextDetector("detector.pt", device="cuda", half=True)

    assert captured["half"] is True
    assert detector.half is True
    assert detector.runtime_issues == []


def test_detector_runtime_contract_rejects_dtype_mismatch() -> None:
    model = nn.Linear(4, 2).float()
    half_input = torch.ones((1, 4), dtype=torch.float16)

    with pytest.raises(DetectorRuntimeContractError, match="model/input contract mismatch"):
        assert_detector_runtime_contract((model,), half_input)


def test_detector_runtime_contract_rejects_device_mismatch() -> None:
    model = nn.Linear(4, 2, device="meta")
    cpu_input = torch.ones((1, 4), dtype=torch.float32)

    with pytest.raises(DetectorRuntimeContractError, match=r"model devices=\['meta'\]"):
        assert_detector_runtime_contract((model,), cpu_input)


def test_onnx_backend_is_rejected_before_loading(tmp_path) -> None:
    model_path = tmp_path / "detector.onnx"

    with pytest.raises(UnsupportedDetectorBackendError, match="unsupported_detector_backend"):
        TextDetector(model_path)
    with pytest.raises(UnsupportedDetectorBackendError, match="only .pt") as captured:
        _get_detector(DetectionConfig(model_path=model_path, device="cpu"))

    assert captured.value.code == "unsupported_detector_backend"


@pytest.mark.parametrize(
    ("message", "classification"),
    [
        ("CUDA out of memory", "oom"),
        ("no kernel image is available", "unsupported_kernel"),
        ("CUDA error: device-side assert triggered", "device_lost"),
        ("cuDNN execution failed", "other"),
        ("ordinary CPU failure", None),
    ],
)
def test_cuda_errors_have_stable_classification(message, classification) -> None:
    assert _classify_cuda_error(RuntimeError(message)) == classification


def test_cpu_half_request_is_explicitly_downgraded() -> None:
    runtime = _resolve_detector_runtime(DetectionConfig(device="cpu", half=True))

    assert runtime.device == "cpu"
    assert runtime.half is False
    assert [issue.code for issue in runtime.issues] == ["detector_fp16_downgraded"]


def test_cuda_page_fallback_does_not_permanently_force_process_to_cpu(monkeypatch) -> None:
    detector_loads: list[str] = []
    failed_once = False

    def fake_get_detector(_cfg, *, runtime=None):
        assert runtime is not None
        detector_loads.append(runtime.device)
        return runtime.device

    def fake_pass(detector, image, _input_size, _keep_mask):
        nonlocal failed_once
        if detector == "cuda" and not failed_once:
            failed_once = True
            raise RuntimeError("CUDA out of memory")
        empty = np.zeros(image.shape[:2], dtype=np.uint8)
        return empty, empty, []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(detector_module, "_get_detector", fake_get_detector)
    monkeypatch.setattr(detector_module, "_run_detector_pass", fake_pass)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    config = DetectionConfig(device="cuda", half=True)

    first = detect_text_regions(image, config, PostprocessConfig())
    second = detect_text_regions(image, config, PostprocessConfig())

    assert detector_loads == ["cuda", "cpu", "cuda"]
    assert first.issues[0].code == "detector_cuda_oom"
    assert second.issues == []


def test_cuda_initialization_failure_retries_only_current_page_on_cpu(monkeypatch) -> None:
    detector_loads: list[str] = []
    failed_once = False

    def fake_get_detector(_cfg, *, runtime=None):
        nonlocal failed_once
        assert runtime is not None
        detector_loads.append(runtime.device)
        if runtime.device == "cuda" and not failed_once:
            failed_once = True
            raise RuntimeError("CUDA error: invalid device function")
        return runtime.device

    def fake_pass(_detector, image, _input_size, _keep_mask):
        empty = np.zeros(image.shape[:2], dtype=np.uint8)
        return empty, empty, []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(detector_module, "_get_detector", fake_get_detector)
    monkeypatch.setattr(detector_module, "_run_detector_pass", fake_pass)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    config = DetectionConfig(device="cuda", half=True)

    first = detect_text_regions(image, config, PostprocessConfig())
    second = detect_text_regions(image, config, PostprocessConfig())

    assert detector_loads == ["cuda", "cpu", "cuda"]
    assert first.issues[0].code == "detector_cuda_unsupported_kernel"
    assert first.issues[0].details["stage"] == "initialization"
    assert second.issues == []


def test_runtime_downgrade_is_retained_on_detection_result(monkeypatch) -> None:
    resolved_half: list[bool] = []

    def fake_get_detector(_cfg, *, runtime=None):
        assert runtime is not None
        resolved_half.append(runtime.half)
        return object()

    def fake_pass(_detector, image, _input_size, _keep_mask):
        empty = np.zeros(image.shape[:2], dtype=np.uint8)
        return empty, empty, []

    monkeypatch.setattr(detector_module, "_get_detector", fake_get_detector)
    monkeypatch.setattr(detector_module, "_run_detector_pass", fake_pass)

    result = detect_text_regions(
        np.zeros((12, 12, 3), dtype=np.uint8),
        DetectionConfig(device="cpu", half=True),
        PostprocessConfig(),
    )

    assert resolved_half == [False]
    assert [issue.code for issue in result.issues] == ["detector_fp16_downgraded"]


def test_group_masks_are_local_and_preserve_pixel_mask() -> None:
    refined = np.zeros((100, 120), dtype=np.uint8)
    refined[20:30, 40:50] = 255
    region = TextRegion(id="", x=38, y=18, w=16, h=16, vertical=False)

    regions, groups = postprocess_regions(
        [region],
        (100, 120),
        PostprocessConfig(enable_grouping=False, min_region_area=1),
        refined_mask=refined,
    )

    assert len(regions) == 1
    assert len(groups) == 1
    group = groups[0]
    assert group.mask is not None
    assert group.mask.shape == (group.h, group.w)
    assert int(np.count_nonzero(group.mask)) == 100


def test_mask_fallback_recovers_unboxed_text_like_cluster() -> None:
    mask = np.zeros((180, 180), dtype=np.uint8)
    # 模擬數個相鄰直排字的 segmentation pixels。
    for y in (35, 55, 75, 95):
        mask[y : y + 10, 92:100] = 255
        mask[y + 3 : y + 7, 88:104] = 255

    cfg = DetectionConfig(
        device="cpu",
        mask_fallback_enabled=True,
        mask_fallback_threshold=20,
        mask_fallback_min_area=12,
        mask_fallback_padding=3,
    )
    candidates = _extract_mask_fallback_regions(mask, [], cfg)

    assert candidates
    assert any(candidate.source == "mask_fallback" for candidate in candidates)
    assert any(candidate.vertical for candidate in candidates)
    assert all(candidate.local_mask is not None for candidate in candidates)
    assert all(np.any(candidate.local_mask) for candidate in candidates)


def test_mask_fallback_does_not_duplicate_covered_detector_box() -> None:
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[30:70, 50:65] = 255
    existing = [TextRegion(id="", x=45, y=25, w=28, h=55)]
    cfg = DetectionConfig(
        device="cpu",
        mask_fallback_enabled=True,
        mask_fallback_threshold=20,
        mask_fallback_min_area=8,
        mask_fallback_padding=4,
    )

    assert _extract_mask_fallback_regions(mask, existing, cfg) == []


def test_empty_refined_mask_does_not_fall_back_to_region_rectangle() -> None:
    refined = np.zeros((60, 80), dtype=np.uint8)
    region = TextRegion(id="", x=10, y=12, w=20, h=16)

    _regions, groups = postprocess_regions(
        [region],
        (60, 80),
        PostprocessConfig(enable_grouping=False, min_region_area=1),
        refined_mask=refined,
    )

    assert groups[0].mask is not None
    assert int(np.count_nonzero(groups[0].mask)) == 0


def test_conservative_mask_rejects_refined_pixels_without_raw_support() -> None:
    raw = np.zeros((40, 50), dtype=np.uint8)
    refined = np.zeros_like(raw)
    raw[10:14, 12:16] = 255
    refined[8:18, 10:20] = 255
    refined[25:35, 30:40] = 255  # 模擬人物線稿被 refinement 誤納入

    safe = _conservative_text_mask(
        refined,
        raw,
        DetectionConfig(
            device="cpu",
            raw_support_threshold=30,
            raw_support_dilate=1,
        ),
    )

    assert np.any(safe[9:17, 11:17])
    assert not np.any(safe[25:35, 30:40])


def test_huge_container_does_not_group_separate_real_regions() -> None:
    huge = TextRegion(id="", x=0, y=0, w=1000, h=1000, vertical=False)
    first = TextRegion(id="", x=40, y=50, w=80, h=120, vertical=True)
    second = TextRegion(id="", x=700, y=700, w=80, h=120, vertical=True)

    _regions, groups = postprocess_regions(
        [huge, first, second],
        (1000, 1000),
        PostprocessConfig(min_region_area=1, enable_grouping=True),
        refined_mask=None,
    )

    assert len(groups) == 3
