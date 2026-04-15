"""Global registry of live DualFlowTransformers.

The registry is the substrate the network 'scans' to find aligned peers. It is
intentionally simple — a dict keyed by stable id, plus a brute-force cosine
search. For large swarms a kd-tree or HNSW index could be substituted; the
public API would not change.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, List, Tuple

import torch

from .direction import Direction

if TYPE_CHECKING:  # pragma: no cover
    from .core import DualFlowTransformer


class FlowRegistry:
    """A thread-unsafe in-process registry of live transformers.

    Multiple transformers sharing one registry can discover each other via
    `query_aligned`. A transformer never appears as a candidate match for
    itself.
    """

    def __init__(self) -> None:
        self._members: Dict[str, "DualFlowTransformer"] = {}

    # ------------------------------------------------------------------ membership

    def register(self, transformer: "DualFlowTransformer") -> None:
        if transformer.flow_id in self._members:
            raise ValueError(f"flow_id {transformer.flow_id!r} already registered")
        self._members[transformer.flow_id] = transformer

    def deregister(self, flow_id: str) -> None:
        self._members.pop(flow_id, None)

    def get(self, flow_id: str) -> "DualFlowTransformer | None":
        return self._members.get(flow_id)

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator["DualFlowTransformer"]:
        return iter(self._members.values())

    def ids(self) -> List[str]:
        return list(self._members.keys())

    # ------------------------------------------------------------------ search

    def query_aligned(
        self,
        direction: torch.Tensor,
        threshold: float = 0.7,
        exclude: str | None = None,
        top_k: int | None = None,
    ) -> List[Tuple["DualFlowTransformer", float]]:
        """Return live transformers whose direction has cosine ≥ threshold.

        Parameters
        ----------
        direction : (3,) tensor
            The query direction (assumed unit-norm).
        threshold : float
            Cosine cutoff for the alignment cone (1.0 = identical, 0.0 = orthogonal).
        exclude : str | None
            Optional flow_id to skip (typically the querying transformer's own id).
        top_k : int | None
            If given, return only the k most aligned matches.

        Returns
        -------
        List of (transformer, cosine) pairs, sorted by descending cosine.
        """
        matches: List[Tuple["DualFlowTransformer", float]] = []
        for fid, t in self._members.items():
            if fid == exclude:
                continue
            cos = float(Direction.cosine(direction, t.direction))
            if cos >= threshold:
                matches.append((t, cos))
        matches.sort(key=lambda pair: pair[1], reverse=True)
        if top_k is not None:
            matches = matches[:top_k]
        return matches
