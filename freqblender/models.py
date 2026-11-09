import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class Detector(nn.Module):
    """Two logits: index 0 real, index 1 fake; includes input normalization."""
    def __init__(self, architecture="efficientnet_b4", pretrained=True):
        super().__init__()
        options = {
            "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        }
        if architecture not in options:
            raise ValueError(f"Unknown architecture {architecture}")
        factory, weights = options[architecture]
        self.architecture = architecture
        self.network = factory(weights=weights if pretrained else None)
        if architecture == "resnet34":
            self.network.fc = nn.Linear(self.network.fc.in_features, 2)
        else:
            self.network.classifier[-1] = nn.Linear(self.network.classifier[-1].in_features, 2)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.network((x - self.mean) / self.std)


def freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class IdentityEncoder(nn.Module):
    """Pretrained FaceNet substitute OR user-provided MobileNet-ArcFace TorchScript.

    TorchScript contract: RGB [-1,1], input_size square -> [B,D] embedding.
    Backbone is frozen, but gradients through the input are deliberately retained.
    """
    def __init__(self, checkpoint=None, input_size=112):
        super().__init__()
        if checkpoint:
            self.backbone = torch.jit.load(str(checkpoint), map_location="cpu")
            self.input_size = input_size
            self.kind = "external_arcface"
        else:
            from facenet_pytorch import InceptionResnetV1
            self.backbone = InceptionResnetV1(pretrained="vggface2")
            self.input_size = 160
            self.kind = "facenet_vggface2_substitute"
        freeze(self.backbone)

    def forward(self, x):
        x = F.interpolate(x, (self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return F.normalize(self.backbone(2*x - 1), dim=-1)


def binary_margin(logits):
    return logits[:, 1] - logits[:, 0]
