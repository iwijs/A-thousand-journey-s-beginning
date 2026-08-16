"""Low-rank adapters for selected Linear layers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target: str = "attention"

    def __post_init__(self):
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.target not in {"attention", "all-linear"}:
            raise ValueError("target must be 'attention' or 'all-linear'")

    def to_dict(self) -> dict:
        return asdict(self)


class LoRALinear(nn.Module):
    """Frozen base linear plus scaled trainable BA low-rank update."""

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base = base
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Linear(
            base.in_features, config.rank, bias=False, **factory_kwargs
        )
        self.lora_b = nn.Linear(
            config.rank, base.out_features, bias=False, **factory_kwargs
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling
        return self.base(inputs) + update

    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            update = self.lora_b.weight @ self.lora_a.weight
            merged.weight.copy_(self.base.weight + self.scaling * update)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def _replace(module: nn.Module, config: LoRAConfig, prefix: str = "") -> list[str]:
    replaced = []
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        is_attention = full_name.endswith("attention.qkv") or full_name.endswith("attention.projection")
        is_feed_forward = ".feed_forward.layers." in full_name and isinstance(child, nn.Linear)
        should_replace = isinstance(child, nn.Linear) and (
            is_attention or (config.target == "all-linear" and is_feed_forward)
        )
        if should_replace:
            setattr(module, name, LoRALinear(child, config))
            replaced.append(full_name)
        else:
            replaced.extend(_replace(child, config, full_name))
    return replaced


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced = _replace(model, config)
    if not replaced:
        raise ValueError("no target Linear layers were found")
    return replaced


def merge_lora(module: nn.Module) -> list[str]:
    merged = []
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged_linear())
            merged.append(name)
        else:
            merged.extend(f"{name}.{item}" for item in merge_lora(child))
    return merged


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total
