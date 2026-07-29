"""Registry and shared geometric frame for live DualFlowTransformers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import torch

from .direction import Direction
from .mathematics import expected_random_degree, threshold_for_expected_random_degree

if TYPE_CHECKING:
    from .core import DualFlowTransformer


class FlowRegistry:
    def __init__(self, frame_seed: Optional[int] = None) -> None:
        self._members: Dict[str, "DualFlowTransformer"] = {}
        self._direction_frames: Dict[int, torch.Tensor] = {}
        self._frame_generator: Optional[torch.Generator] = None
        if frame_seed is not None:
            self._frame_generator = torch.Generator().manual_seed(frame_seed)

    def direction_frame(self, d_model: int) -> torch.Tensor:
        """Return the shared projection frame for a given hidden width."""
        if d_model not in self._direction_frames:
            self._direction_frames[d_model] = Direction.random_projection(
                d_model, generator=self._frame_generator
            )
        return self._direction_frames[d_model].clone()

    def register(self, transformer: "DualFlowTransformer") -> None:
        if transformer.flow_id in self._members:
            raise ValueError(f"flow_id {transformer.flow_id!r} already registered")
        self._members[transformer.flow_id] = transformer

    def deregister(self, flow_id: str) -> None:
        self._members.pop(flow_id, None)

    def get(self, flow_id: str) -> Optional["DualFlowTransformer"]:
        return self._members.get(flow_id)

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator["DualFlowTransformer"]:
        return iter(self._members.values())

    def ids(self) -> List[str]:
        return list(self._members.keys())

    def null_expected_degree(self, threshold: float) -> float:
        return expected_random_degree(len(self), threshold)

    def threshold_for_null_degree(self, expected_degree: float) -> float:
        return threshold_for_expected_random_degree(len(self), expected_degree)

    def query_aligned(
        self,
        direction: torch.Tensor,
        threshold: float = 0.7,
        exclude: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple["DualFlowTransformer", float]]:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [-1, 1]")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")
        matches: List[Tuple["DualFlowTransformer", float]] = []
        for flow_id, transformer in self._members.items():
            if flow_id == exclude:
                continue
            cosine = float(Direction.cosine(direction, transformer.direction))
            if cosine >= threshold:
                matches.append((transformer, cosine))
        matches.sort(key=lambda pair: pair[1], reverse=True)
        return matches if top_k is None else matches[:top_k]
