"""Single-working-image orchestration for atomic ROI render operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import InpaintingConfig
from ..typography.render import (
    AtomicRenderOutcome,
    AtomicRoiRequest,
    LayerRenderer,
    atomic_inpaint_render,
    render_layout_layer,
)


@dataclass(frozen=True)
class RenderProfile:
    page_copies: int
    page_bytes_copied: int
    roi_bytes_copied: int
    committed_rois: int
    rolled_back_rois: int


@dataclass(frozen=True)
class RenderStageResult:
    image: np.ndarray
    outcomes: tuple[AtomicRenderOutcome, ...]
    profile: RenderProfile


def render_page_atomic(
    original: np.ndarray,
    requests: tuple[AtomicRoiRequest, ...],
    inpainting: InpaintingConfig | None = None,
    *,
    renderer: LayerRenderer = render_layout_layer,
) -> RenderStageResult:
    """Use one mutable page and commit or roll back each ROI independently."""

    working = original.copy()
    outcomes = tuple(
        atomic_inpaint_render(working, request, inpainting, renderer=renderer)
        for request in requests
    )
    committed = sum(outcome.committed for outcome in outcomes)
    profile = RenderProfile(
        page_copies=1,
        page_bytes_copied=int(original.nbytes),
        roi_bytes_copied=sum(outcome.roi_bytes_copied for outcome in outcomes),
        committed_rois=committed,
        rolled_back_rois=len(outcomes) - committed,
    )
    return RenderStageResult(working, outcomes, profile)
