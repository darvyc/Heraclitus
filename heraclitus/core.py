"""DualFlowTransformer — the headline module.

Each `DualFlowTransformer`:

  * runs a stack of direction-modulated transformer blocks,
  * applies forward-pass learning during `forward(..., learn=True)`,
  * maintains a learned 3D direction `d ∈ S²`,
  * snapshots its prior state as a `CounterFlow` before any update,
  * and, on demand, scans a shared `FlowRegistry` to forge `FlowConnection`s
    with peers whose direction now aligns with its own.

Forward-pass learning is gradient-free, so a `DualFlowTransformer` can be
embedded inside a normal autograd graph without interfering with it. If
`learn=False`, the module behaves like a vanilla transformer.
"""
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
    """One transformer block: dir-modulated attention + MLP, both pre-normed."""

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
        # Persistent EMA for predictive-coding update on the MLP output.
        self.register_buffer("mlp_ema", torch.zeros(d_model))

    def forward(self, x: Tensor, direction: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), direction)
        x = x + self.mlp(self.norm2(x))
        return x


class DualFlowTransformer(nn.Module):
    """The 3D Dual-Flow Transformer.

    Parameters
    ----------
    d_model, n_heads, n_layers : int
        Standard transformer hyperparameters.
    registry : FlowRegistry | None
        Shared registry. If given, this transformer auto-registers on init and
        can scan the registry for aligned peers.
    flow_id : str | None
        Stable id; auto-generated if not supplied.
    lr : float
        Learning rate for forward-pass updates.
    direction_momentum : float
        EMA momentum for the live S^2 direction (1.0 = never moves).
    keep_counter_flows : int
        Maximum number of past Counter-Flows to retain. Older ones are dropped.
    """

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
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.flow_id = flow_id or new_flow_id()
        self.direction_momentum = direction_momentum
        self.keep_counter_flows = keep_counter_flows

        self.blocks = nn.ModuleList(
            [_Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)

        # Projection from d_model features to a 3D direction.
        self.dir_proj = nn.Linear(d_model, 3, bias=False)

        # Live direction buffer (unit-norm). Starts random on S^2.
        self.register_buffer("direction", Direction.random())

        # Forward-pass learner.
        self.learner = ForwardLearner(lr=lr)

        # Update step counter.
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

        # Connection bookkeeping (id -> FlowConnection).
        self.connections: Dict[str, FlowConnection] = {}

        # Counter-Flow chain (most recent last).
        self.counter_flows: List[CounterFlow] = []

        # Registry hookup.
        self.registry: Optional[FlowRegistry] = registry
        if registry is not None:
            registry.register(self)

    # ------------------------------------------------------------------ properties

    @property
    def counter_flow(self) -> Optional[CounterFlow]:
        """The most recent Counter-Flow, or None if none yet exists."""
        return self.counter_flows[-1] if self.counter_flows else None

    # ------------------------------------------------------------------ forward

    def forward(self, x: Tensor, learn: bool = False) -> Tensor:
        """Run the stack.

        Parameters
        ----------
        x : (B, T, d_model) tensor
        learn : bool
            If True, snapshot a Counter-Flow, run forward-pass learning rules,
            and update the live 3D direction.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {x.shape[-1]}")

        if learn:
            self._snapshot_counter_flow()

        h = x
        for block in self.blocks:
            pre_norm = block.norm1(h)
            attn_out = block.attn(pre_norm, self.direction)
            h_after_attn = h + attn_out

            mlp_in = block.norm2(h_after_attn)
            mlp_hidden = block.mlp[0](mlp_in)            # (B, T, mlp)
            mlp_act = block.mlp[1](mlp_hidden)
            mlp_out = block.mlp[2](mlp_act)              # (B, T, d_model)
            h = h_after_attn + mlp_out

            if learn:
                # Hebbian on the value projection's bookend: pre = pre_norm,
                # post = attn_out — both naturally aligned in d_model.
                self.learner.hebbian_update(block.attn.out, pre_norm, attn_out)
                # Predictive-coding update on the MLP's output projection bias.
                if block.mlp[2].bias is None:
                    # Materialise a bias parameter on first learning pass.
                    bias = nn.Parameter(torch.zeros(self.d_model, device=h.device))
                    block.mlp[2].bias = bias
                block.mlp_ema = self.learner.predictive_update(
                    block.mlp[2].bias, mlp_out, block.mlp_ema
                )

        out = self.norm_f(h)

        if learn:
            self._update_direction(out)
            self.step += 1

        return out

    # ------------------------------------------------------------------ direction

    @torch.no_grad()
    def _update_direction(self, features: Tensor) -> None:
        """Slide the live direction toward the projection of the latest features."""
        target = Direction.from_features(features, self.dir_proj.weight.t())  # (3,)
        m = self.direction_momentum
        new = m * self.direction + (1.0 - m) * target
        self.direction = Direction.normalize(new)

    # ------------------------------------------------------------------ scanning

    def scan_and_connect(
        self,
        threshold: float = 0.7,
        top_k: Optional[int] = None,
        replace: bool = False,
    ) -> List[FlowConnection]:
        """Find peers in the registry whose direction aligns with our own.

        Parameters
        ----------
        threshold : float
            Cosine cutoff for the alignment cone.
        top_k : int | None
            If given, keep only the k most aligned new edges.
        replace : bool
            If True, drop all existing connections before forging new ones.

        Returns
        -------
        The newly forged FlowConnections (does not include pre-existing ones).
        """
        if self.registry is None:
            return []

        if replace:
            self.connections.clear()

        candidates = self.registry.query_aligned(
            self.direction, threshold=threshold, exclude=self.flow_id, top_k=top_k
        )

        new_edges: List[FlowConnection] = []
        for peer, cos in candidates:
            edge = FlowConnection(
                source_id=self.flow_id,
                target_id=peer.flow_id,
                weight=cos,
                forged_at_step=int(self.step.item()),
                direction_at_forge=self.direction.detach().clone(),
            )
            self.connections[edge.id] = edge
            new_edges.append(edge)
        return new_edges

    # ------------------------------------------------------------------ counter-flow

    def _snapshot_counter_flow(self) -> CounterFlow:
        """Freeze the current state as a new Counter-Flow."""
        snapshot = CounterFlowSnapshot(
            step=int(self.step.item()),
            direction=self.direction.detach().clone(),
            state_dict={k: v.detach().clone() for k, v in self.state_dict().items()},
            connection_ids=list(self.connections.keys()),
            metadata={"n_connections": len(self.connections)},
        )
        cf = CounterFlow(parent_id=self.flow_id, snapshot=snapshot, parent_module=self)
        self.counter_flows.append(cf)
        if len(self.counter_flows) > self.keep_counter_flows:
            self.counter_flows.pop(0)
        return cf

    # ------------------------------------------------------------------ misc

    def extra_repr(self) -> str:
        return (
            f"flow_id={self.flow_id} d_model={self.d_model} "
            f"n_heads={self.n_heads} n_layers={self.n_layers} step={int(self.step.item())} "
            f"connections={len(self.connections)} counter_flows={len(self.counter_flows)}"
        )
