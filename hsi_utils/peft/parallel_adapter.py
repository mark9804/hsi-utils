import torch.nn as nn


class ParallelAdapter(nn.Module):
    """Parallel adapter: output = frozen_original(x) + adapter(x).

    Wraps a Conv2d layer (typically depthwise) with a lightweight bottleneck
    adapter in parallel. The original layer is frozen; only the adapter trains.

    Note: assumes in_channels == out_channels on the original layer
    (true for depthwise convolutions).

    Args:
        original: The Conv2d layer to wrap (will be frozen).
        bottleneck: Channel reduction factor for the adapter.
    """

    def __init__(self, original: nn.Conv2d, bottleneck: int = 4):
        super().__init__()
        ch = original.out_channels
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Conv2d(ch, ch // bottleneck, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // bottleneck, ch, 1, bias=False),
        )
        # Zero init so adapter output starts at zero
        nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, x):
        return self.original(x) + self.adapter(x)
