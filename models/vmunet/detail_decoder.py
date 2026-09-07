"""Multi-scale semantic decoding with a separate image-detail path."""
import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.GELU(),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            conv_block(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.gelu(x + self.body(x))


class DetailDecoder(nn.Module):
    """Replace the original Mamba decoder, additive skips and final patch expansion.

    Inputs use the existing encoder's NHWC feature contract. The shallowest
    skip is the patch embedding; deeper skips have passed preceding VSS stages.
    """
    def __init__(self, encoder_channels, input_channels, num_classes):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(channels, 96, 1) for channels in encoder_channels
        ])
        self.gates = nn.ModuleList([nn.Conv2d(192, 96, 1) for _ in range(3)])
        self.fusions = nn.ModuleList([ResidualBlock(96) for _ in range(3)])
        self.scale_fusion = nn.Sequential(nn.Conv2d(384, 96, 1), ResidualBlock(96))
        self.detail_half = nn.Sequential(conv_block(input_channels, 32, stride=2), ResidualBlock(32))
        self.detail_full = conv_block(input_channels, 16)
        self.semantic_half = nn.Conv2d(96, 32, 1)
        self.half_fusion = nn.Sequential(conv_block(64, 32), ResidualBlock(32))
        self.full_fusion = conv_block(48, 16)
        self.classifier = nn.Conv2d(16, num_classes, 1)

    @staticmethod
    def resize(x, reference):
        return F.interpolate(x, size=reference.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, image, bottleneck, skips):
        # Replace the pre-final-stage skip with the processed bottleneck.
        features = list(skips[:3]) + [bottleneck]
        pyramid = [project(feature.permute(0, 3, 1, 2).contiguous())
                   for project, feature in zip(self.lateral, features)]
        for level in (2, 1, 0):
            context = self.resize(pyramid[level + 1], pyramid[level])
            gate = torch.sigmoid(self.gates[level](torch.cat([pyramid[level], context], dim=1)))
            pyramid[level] = self.fusions[level](gate * pyramid[level] + (1 - gate) * context)
        semantic = self.scale_fusion(torch.cat(
            [pyramid[0]] + [self.resize(x, pyramid[0]) for x in pyramid[1:]], dim=1))
        # Existing full-supervision preprocessing supplies images in [0, 255].
        detail_input = image / 255.0
        half = self.detail_half(detail_input)
        half = self.half_fusion(torch.cat([self.resize(self.semantic_half(semantic), half), half], dim=1))
        full = self.detail_full(detail_input)
        full = self.full_fusion(torch.cat([self.resize(half, full), full], dim=1))
        return self.classifier(full)
