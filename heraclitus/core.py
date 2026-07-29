"""The DualFlowTransformer research prototype."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from .attention import DirectionModulatedAttention
from .connections import FlowConnection
from .counter_flow import CounterFlow, CounterFlowSnapshot
from .direction import Direction
from .forward_learner import ForwardLearner
from .network import FlowRegistry
from .utils import new_flow_id


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DirectionModulatedAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.register_buffer("mlp_ema", torch.zeros(d_model))

    def forward(self, x: Tensor, direction: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), direction)
        return x + self.mlp(self.norm2(x))


class DualFlowTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        registry: Optional[FlowRegistry] = None,
        flow_id: Optional[str] = None,
        lr: float = 1e-3,
        direction_momentum: float = 0.9,
        keep_counter_flows: int = 8,
        direction_projection: Optional[Tensor] = None,
    ):
        super().__init__()
        if not 0.0 <= direction_momentum <= 1.0:
            raise ValueError("direction_momentum must lie in [0, 1]")
        if keep_counter_flows < 0:
            raise ValueError("keep_counter_flows must be non-negative")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.flow_id = flow_id or new_flow_id()
        self.direction_momentum = direction_momentum
        self.keep_counter_flows = keep_counter_flows

        self.blocks = nn.ModuleList(
            [_Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)

        if direction_projection is None:
            projection = (
                registry.direction_frame(d_model)
                if registry is not None
                else Direction.random_projection(d_model)
            )
        else:
            projection = direction_projection.detach().clone()
        if projection.shape != (d_model, 3):
            raise ValueError(f"direction_projection must have shape ({d_model}, 3)")
        self.register_buffer("direction_projection", projection)
        self.register_buffer("direction", Direction.random())

        self.learner = ForwardLearner(lr=lr)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.connections: Dict[str, FlowConnection] = {}
        self.counter_flows: List[CounterFlow] = []
        self.registry = registry
        if registry is not None:
            registry.register(self)

    @property
    def counter_flow(self) -> Optional[CounterFlow]:
        return self.counter_flows[-1] if self.counter_flows else None

    def forward(self, x: Tensor, learn: bool = False) -> Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {x.shape[-1]}")
        if learn:
            self._snapshot_counter_flow()

        h = x
        for block in self.blocks:
            pre_norm = block.norm1(h)
            attn_result = block.attn(pre_norm, self.direction, return_context=True)
            attn_out, attn_context = attn_result
            h_after_attn = h + attn_out

            mlp_in = block.norm2(h_after_attn)
            mlp_hidden = block.mlp[0](mlp_in)
            mlp_act = block.mlp[1](mlp_hidden)
            mlp_out = block.mlp[2](mlp_act)
            h = h_after_attn + mlp_out

            if learn:
                self.learner.hebbian_update(block.attn.out, attn_context, attn_out)
                block.mlp_ema = self.learner.predictive_update(
                    block.mlp[2].bias, mlp_out, block.mlp_ema
                )

        out = self.norm_f(h)
        if learn:
            self._update_direction(out)
            self.step += 1
        return out

    @torch.no_grad()
    def _update_direction(self, features: Tensor) -> None:
        target = Direction.from_features(features, self.direction_projection)
        momentum = self.direction_momentum
        self.direction = Direction.normalize(
            momentum * self.direction + (1.0 - momentum) * target
        )

    def scan_and_connect(
        self,
        threshold: float = 0.7,
        top_k: Optional[int] = None,
        replace: bool = False,
    ) -> List[FlowConnection]:
        if self.registry is None:
            return []
        if replace:
            self.connections.clear()
        candidates = self.registry.query_aligned(
            self.direction, threshold=threshold, exclude=self.flow_id, top_k=top_k
        )
        new_edges: List[FlowConnection] = []
        for peer, cosine in candidates:
            edge = FlowConnection(
                source_id=self.flow_id,
                target_id=peer.flow_id,
                weight=cosine,
                forged_at_step=int(self.step.item()),
                direction_at_forge=self.direction.detach().clone(),
            )
            self.connections[edge.id] = edge
            new_edges.append(edge)
        return new_edges

    def _frozen_clone(self) -> "DualFlowTransformer":
        clone = DualFlowTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            registry=None,
            flow_id=f"{self.flow_id}-counter",
            lr=self.learner.lr,
            direction_momentum=self.direction_momentum,
            keep_counter_flows=0,
            direction_projection=self.direction_projection,
        )
        clone.load_state_dict(self.state_dict())
        clone.connections = {}
        clone.counter_flows = []
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        clone.eval()
        return clone

    def _snapshot_counter_flow(self) -> CounterFlow:
        snapshot = CounterFlowSnapshot(
            step=int(self.step.item()),
            direction=self.direction.detach().clone(),
            state_dict={key: value.detach().clone() for key, value in self.state_dict().items()},
            connection_ids=list(self.connections.keys()),
            metadata={"n_connections": len(self.connections)},
        )
        counter_flow = CounterFlow(
            parent_id=self.flow_id,
            snapshot=snapshot,
            frozen_module=self._frozen_clone(),
        )
        self.counter_flows.append(counter_flow)
        if len(self.counter_flows) > self.keep_counter_flows:
            self.counter_flows.pop(0)
        return counter_flow

    def extra_repr(self) -> str:
        return (
            f"flow_id={self.flow_id} d_model={self.d_model} "
            f"n_heads={self.n_heads} n_layers={self.n_layers} step={int(self.step.item())} "
            f"connections={len(self.connections)} counter_flows={len(self.counter_flows)}"
        )
