# Pipeline step 6: classifier backbone (used by 7_train and 8_eval_run).
from __future__ import annotations

import torch
import torch.nn as nn


def build_resnet18_grayscale(num_classes: int = 5) -> nn.Module:
    from torchvision import models

    try:
        from torchvision.models import ResNet18_Weights

        m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        m = models.resnet18(pretrained=True)
    old = m.conv1
    m.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        m.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
