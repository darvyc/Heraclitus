"""Basic usage of a single DualFlowTransformer."""
import torch
from heraclitus import DualFlowTransformer

torch.manual_seed(0)

net = DualFlowTransformer(d_model=32, n_heads=4, n_layers=2)
print(net)
print("initial direction:", net.direction.tolist())

x = torch.randn(2, 8, 32)
for i in range(3):
    y = net(x, learn=True)
    cf = net.counter_flow
    print(
        f"step={int(net.step)} "
        f"direction={[round(v, 3) for v in net.direction.tolist()]} "
        f"opposition={float(cf.opposition(net.direction)):.3f} rad"
    )

print("\nMost recent Counter-Flow:")
print(net.counter_flow.summary())
