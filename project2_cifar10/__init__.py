"""CIFAR-10 CNN training project with validation-only model selection."""

from .models import BasicCNN, build_model

__all__ = ["BasicCNN", "build_model"]
