"""PyTorch model definitions that mirror netkit tutorial architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TutorialMlp784(nn.Module):
    """784 -> 128 ReLU -> 10 linear (softmax applied at export to match netkit)."""

    def __init__(self, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TutorialCnn28(nn.Module):
    """Conv32/ReLU/Pool -> Conv64/ReLU/Pool -> Flatten -> Dense128/ReLU -> Dense10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        for layer in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # netkit flatten follows NHWC memory order (spatial, then channels)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SpeechKwsCnn(nn.Module):
    """16x10 MFCC-like map -> depthwise-separable conv stack -> 12 keyword logits."""

    def __init__(self, num_classes: int = 12) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.dw1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8)
        self.pw1 = nn.Conv2d(8, 16, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dw2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 24, kernel_size=1, stride=1)
        self.fc = nn.Linear(24 * 4 * 2, num_classes)
        for layer in (self.stem, self.dw1, self.pw1, self.dw2, self.pw2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        x = F.relu(self.pw1(F.relu(self.dw1(x))))
        x = self.pool(x)
        x = F.relu(self.pw2(F.relu(self.dw2(x))))
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.flatten(x, 1)
        return self.fc(x)
