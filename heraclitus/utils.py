"""Small helpers."""
from __future__ import annotations

import uuid


def new_flow_id(prefix: str = "flow") -> str:
    """Stable, collision-resistant id for a transformer in the registry."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
