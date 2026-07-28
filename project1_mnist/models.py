"""Model definitions for the MNIST comparison experiment."""

from torch import nn


class MLP(nn.Module):
    """A baseline that treats a 28x28 image as a 784-dimensional vector."""

    def __init__(self, hidden_features: int = 256, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, images):
        return self.network(images)


class SimpleCNN(nn.Module):
    """A small CNN that preserves and exploits image spatial structure."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def build_model(name: str) -> nn.Module:
    """Build a named model so training and evaluation share one definition."""

    normalized_name = name.lower()
    if normalized_name == "mlp":
        return MLP()
    if normalized_name == "cnn":
        return SimpleCNN()
    raise ValueError(f"unknown model {name!r}; choose 'mlp' or 'cnn'")
