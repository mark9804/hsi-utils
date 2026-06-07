import math

import torch
import torch.nn as nn


class MaskedLoRA(nn.Module):
    """LoRA for merged linear/conv layers with per-split adaptation control.

    Splits the output dimension into len(adapt_mask) equal parts.
    Only splits where adapt_mask[i] is True get trainable LoRA paths;
    the rest contribute structural zeros (zero gradient guaranteed).

    Works with both nn.Linear and nn.Conv2d (1x1) layers.

    Args:
        orig_layer: The frozen original layer to wrap.
        adapt_mask: Boolean list, length = number of output splits.
            True at index i means the i-th split gets a LoRA path.
        rank: LoRA bottleneck rank.
        alpha: LoRA scaling factor (effective scale = alpha / rank).
    """

    def __init__(
        self,
        orig_layer: nn.Module,
        adapt_mask: list[bool],
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        if not any(adapt_mask):
            raise ValueError("adapt_mask must have at least one True entry")

        self.orig_layer = orig_layer
        for p in self.orig_layer.parameters():
            p.requires_grad = False

        self.scaling = alpha / rank
        self.n_splits = len(adapt_mask)

        is_conv = isinstance(orig_layer, nn.Conv2d)
        self._is_conv = is_conv

        if is_conv:
            in_dim = orig_layer.in_channels
            self.split_dim = orig_layer.out_channels // self.n_splits
        else:
            in_dim = orig_layer.in_features
            self.split_dim = orig_layer.out_features // self.n_splits

        # Build LoRA A/B pairs only for adapted splits
        self.lora_pairs = nn.ModuleDict()
        for i, adapt in enumerate(adapt_mask):
            if adapt:
                if is_conv:
                    A = nn.Conv2d(in_dim, rank, 1, bias=False)
                    B = nn.Conv2d(rank, self.split_dim, 1, bias=False)
                else:
                    A = nn.Linear(in_dim, rank, bias=False)
                    B = nn.Linear(rank, self.split_dim, bias=False)
                nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
                nn.init.zeros_(B.weight)
                self.lora_pairs[str(i)] = nn.ModuleList([A, B])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.orig_layer(x)

        parts = []
        for i in range(self.n_splits):
            key = str(i)
            if key in self.lora_pairs:
                A, B = self.lora_pairs[key]
                parts.append(B(A(x)) * self.scaling)
            else:
                # Structural zero -- no trainable params for this split
                if self._is_conv:
                    parts.append(
                        x.new_zeros(x.shape[0], self.split_dim, x.shape[2], x.shape[3])
                    )
                else:
                    parts.append(x.new_zeros(*x.shape[:-1], self.split_dim))

        cat_dim = 1 if self._is_conv else -1
        return out + torch.cat(parts, dim=cat_dim)


def inject_masked_lora(
    model: nn.Module,
    module_type: type,
    attr_name: str,
    adapt_mask: list[bool],
    rank: int = 8,
    alpha: float = 16.0,
    labels: list[str] | None = None,
) -> int:
    """Replace attr_name on all module_type instances with MaskedLoRA wrappers.

    Architecture-agnostic: the caller provides the concrete module type and
    attribute name, so this function has no dependency on any model package.

    Args:
        model: The full model to scan.
        module_type: The nn.Module subclass to match (e.g. HS_MSA).
        attr_name: The attribute name on matched modules to replace (e.g. "to_kv").
        adapt_mask: Passed to MaskedLoRA -- see its docstring.
        rank: LoRA bottleneck rank.
        alpha: LoRA scaling factor.
        labels: Human-readable names for each split (for logging only).

    Returns:
        Number of layers replaced.
    """
    count = 0
    device = next(model.parameters()).device
    for _name, module in model.named_modules():
        if isinstance(module, module_type) and hasattr(module, attr_name):
            orig = getattr(module, attr_name)
            wrapper = MaskedLoRA(orig, adapt_mask, rank, alpha).to(device)
            setattr(module, attr_name, wrapper)
            count += 1

    return count
