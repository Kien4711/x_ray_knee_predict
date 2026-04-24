# Pipeline step 6: classifier backbones (used by 7_train, 8_eval_run, 10_compare).
from __future__ import annotations

import torch
import torch.nn as nn

# Public slugs for CLI and train_config / checkpoints
ARCHITECTURES: tuple[str, ...] = ("resnet50", "efficientnet_b0", "mobilenet_v2")


def normalize_architecture(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    aliases = {
        "mobilenetv2": "mobilenet_v2",
        "efficientnetb0": "efficientnet_b0",
    }
    n = aliases.get(n, n)
    if n not in ARCHITECTURES:
        raise ValueError(f"Unknown model {name!r}. Choose one of: {', '.join(ARCHITECTURES)}")
    return n


def _rgb_to_gray_weights(old: nn.Conv2d) -> nn.Conv2d:
    new = nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


def build_resnet50_grayscale(num_classes: int = 5) -> nn.Module:
    return build_knee_classifier("resnet50", num_classes=num_classes)


def build_knee_classifier(architecture: str, *, num_classes: int = 5) -> nn.Module:
    arch = normalize_architecture(architecture)
    from torchvision import models

    if arch == "resnet50":
        try:
            from torchvision.models import ResNet50_Weights

            m = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            m = models.resnet50(weights="IMAGENET1K_V1")
        old = m.conv1
        m.conv1 = _rgb_to_gray_weights(old)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m

    if arch == "efficientnet_b0":
        try:
            from torchvision.models import EfficientNet_B0_Weights

            m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            m = models.efficientnet_b0(weights="IMAGENET1K_V1")
        stem = m.features[0]
        stem[0] = _rgb_to_gray_weights(stem[0])
        in_f = m.classifier[1].in_features
        drop_p = m.classifier[0].p if hasattr(m.classifier[0], "p") else 0.2
        m.classifier = nn.Sequential(nn.Dropout(p=drop_p), nn.Linear(in_f, num_classes))
        return m

    if arch == "mobilenet_v2":
        try:
            from torchvision.models import MobileNet_V2_Weights

            m = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception:
            m = models.mobilenet_v2(weights="IMAGENET1K_V1")
        first_block = m.features[0][0]
        m.features[0][0] = _rgb_to_gray_weights(first_block)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(f"Unhandled architecture {arch!r}")
