"""A from-scratch ResNet-18 adapted to 32x32 CIFAR images."""

from __future__ import annotations

import torch
from torch import nn


def _normalization(channels: int, use_batch_norm: bool) -> nn.Module:
    return nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with an optional projected residual shortcut."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_residual: bool = True,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_batch_norm,
        )
        self.norm1 = _normalization(out_channels, use_batch_norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batch_norm,
        )
        self.norm2 = _normalization(out_channels, use_batch_norm)

        self.shortcut: nn.Module = nn.Identity()
        if use_residual and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=not use_batch_norm,
                ),
                _normalization(out_channels, use_batch_norm),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.relu(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        if self.use_residual:
            outputs = outputs + self.shortcut(inputs)
        return self.relu(outputs)


class CifarResNet18(nn.Module):
    """ResNet-18 with a CIFAR stem: 3x3 conv and no initial max pooling."""

    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 64,
        use_residual: bool = True,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.in_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                base_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batch_norm,
            ),
            _normalization(base_channels, use_batch_norm),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(base_channels, blocks=2, stride=1)
        self.stage2 = self._make_stage(base_channels * 2, blocks=2, stride=2)
        self.stage3 = self._make_stage(base_channels * 4, blocks=2, stride=2)
        self.stage4 = self._make_stage(base_channels * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_channels * 8, num_classes)
        self._initialize_parameters()

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [
            BasicBlock(
                self.in_channels,
                out_channels,
                stride=stride,
                use_residual=self.use_residual,
                use_batch_norm=self.use_batch_norm,
            )
        ]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.in_channels,
                    out_channels,
                    use_residual=self.use_residual,
                    use_batch_norm=self.use_batch_norm,
                )
            )
        return nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = self.stage1(features)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        return self.classifier(torch.flatten(self.pool(features), 1))


def build_model(
    name: str = "resnet18",
    base_channels: int = 64,
    use_batch_norm: bool = True,
) -> CifarResNet18:
    if name == "resnet18":
        use_residual = True
    elif name == "plain18":
        use_residual = False
    else:
        raise ValueError(f"unknown model {name!r}; choose 'resnet18' or 'plain18'")
    return CifarResNet18(
        base_channels=base_channels,
        use_residual=use_residual,
        use_batch_norm=use_batch_norm,
    )
