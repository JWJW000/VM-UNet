"""R3: joint encoder context and progressive logit recovery around the intact decoder."""
import torch
from torch import nn
import torch.nn.functional as F

from .detail_decoder import conv_block, ResidualBlock


class ContextAdapter(nn.Module):
    def __init__(self, dims, num_classes):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, 32, 1) for c in dims])
        self.context = nn.Sequential(conv_block(128, 64), ResidualBlock(64))
        self.skip_residuals = nn.ModuleList([nn.Conv2d(64, c, 1) for c in dims[:0:-1]])
        self.recovery = nn.Sequential(
            conv_block(dims[0] + 64, 64), ResidualBlock(64),
            nn.Conv2d(64, 32 * 4, 3, padding=1), nn.PixelShuffle(2), nn.GELU(),
            nn.Conv2d(32, 16 * 4, 3, padding=1), nn.PixelShuffle(2), nn.GELU(),
            nn.Conv2d(16, num_classes, 1),
        )
        # Start from the original prediction; inner branches receive gradients
        # after their zero-initialized output projections take the first update.
        for layer in list(self.skip_residuals) + [self.recovery[-1]]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode(self, bottleneck, skips):
        features = list(skips[:3]) + [bottleneck]
        size = skips[0].shape[1:3]
        projected = [F.interpolate(layer(x.permute(0, 3, 1, 2)), size=size,
                                   mode='bilinear', align_corners=False)
                     for layer, x in zip(self.projections, features)]
        return self.context(torch.cat(projected, dim=1))

    def augment_skips(self, context, skips):
        result = list(skips)
        for index, layer in enumerate(self.skip_residuals, 1):
            residual = layer(F.interpolate(context, size=skips[-index].shape[1:3],
                                           mode='bilinear', align_corners=False))
            result[-index] = skips[-index] + residual.permute(0, 2, 3, 1)
        return result

    def forward(self, decoded, context):
        return self.recovery(torch.cat([decoded.permute(0, 3, 1, 2), context], dim=1))
