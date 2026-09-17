import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import APConv3d


class AffineGuidanceBlock(nn.Module):
    """Modulate a 3D cost volume with a 2D monocular feature map."""

    def __init__(self, mono_channels, target_channels):
        super().__init__()
        self.proj = nn.Conv2d(mono_channels, target_channels * 2, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias[:target_channels])
        nn.init.zeros_(self.proj.bias[target_channels:])

    def forward(self, x3d, feature2d):
        params = self.proj(feature2d)
        if params.shape[2:] != x3d.shape[3:]:
            params = F.interpolate(params, size=x3d.shape[3:], mode="bilinear", align_corners=False)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return x3d * gamma.unsqueeze(2) + beta.unsqueeze(2)


class CostProcessHourglass3dAF(nn.Module):
    """APConv3d hourglass with monocular guidance at the bottleneck."""

    def __init__(
        self,
        in_channels=256,
        out_channels=1,
        features=64,
        kernel_size_spatial=(1, 3, 3),
        kernel_size_disp=(3, 1, 1),
        dilation=(1, 1, 1),
        groups=16,
        norm="bn",
        norm_params=None,
        mono_channels=32,
    ):
        super().__init__()
        del out_channels
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        elif len(dilation) == 2:
            dilation = (1, dilation[0], dilation[1])

        def conv(in_ch, spatial_ch, disparity_ch, stride=(1, 1, 1), conv_groups=1):
            return APConv3d(
                in_ch,
                spatial_ch,
                disparity_ch,
                kernel_size_spatial,
                kernel_size_disp,
                stride,
                dilation,
                groups=conv_groups,
                norm=norm,
                norm_params=norm_params,
            )

        self.head = nn.Sequential(
            conv(in_channels, features, features, conv_groups=groups),
            conv(features, features, features, conv_groups=groups),
        )
        self.down1 = nn.Sequential(
            conv(features, features * 2, features * 2, stride=(1, 2, 2)),
            conv(features * 2, features * 2, features * 2),
        )
        self.down2 = nn.Sequential(
            conv(features * 2, features * 4, features * 4, stride=(1, 2, 2)),
            conv(features * 4, features * 4, features * 4),
        )
        self.down3 = nn.Sequential(
            conv(features * 4, features * 8, features * 8, stride=(1, 2, 2)),
            conv(features * 8, features * 8, features * 8),
            conv(features * 8, features * 8, features * 8),
        )
        self.up3 = nn.Sequential(
            conv(features * 12, features * 4, features * 4),
            conv(features * 4, features * 4, features * 4),
        )
        self.up2 = nn.Sequential(
            conv(features * 6, features * 2, features * 2),
            conv(features * 2, features * 2, features * 2),
        )
        self.up1 = nn.Sequential(
            conv(features * 3, features, features),
            conv(features, features, features),
        )
        self.tail = nn.Sequential(
            conv(features, features // 2, features),
            conv(features // 2, features // 2, features // 2),
            APConv3d(
                features // 2,
                1,
                features // 2,
                7,
                7,
                (1, 1, 1),
                dilation,
                groups=1,
                norm=norm,
                norm_params=norm_params,
            ),
        )
        self.guide_bottleneck = AffineGuidanceBlock(mono_channels, features * 8)

    def forward(self, cost, mono_feature, return_feature=False):
        x1 = self.head(cost)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.guide_bottleneck(self.down3(x3), mono_feature)

        x3_dec = self.up3(torch.cat([F.interpolate(x4, size=x3.shape[2:], mode="trilinear", align_corners=True), x3], dim=1))
        x2_dec = self.up2(torch.cat([F.interpolate(x3_dec, size=x2.shape[2:], mode="trilinear", align_corners=True), x2], dim=1))
        feature = self.up1(torch.cat([F.interpolate(x2_dec, size=x1.shape[2:], mode="trilinear", align_corners=True), x1], dim=1))
        output = self.tail(feature).squeeze(1)
        return (output, feature) if return_feature else output
