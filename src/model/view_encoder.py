import torch
import torch.nn as nn

from .basic import BasicConv2d, ResBlock


class FeatureExtraction(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        dilation=(1, 1),
        norm="bn",
        norm_params=None,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                4,
                kernel_size,
                stride=1,
                padding=(
                    dilation[0] * (kernel_size[0] - 1) // 2,
                    dilation[1] * (kernel_size[1] - 1) // 2,
                ),
                dilation=dilation,
            ),
            nn.LeakyReLU(0.2, True),
        )

        self.layer1 = self._make_layer(
            4, 4, 2, kernel_size, 1, dilation, norm, norm_params
        )
        self.layer2 = self._make_layer(
            4, 8, 2, kernel_size, 1, dilation, norm, norm_params
        )
        self.layer3 = self._make_layer(
            8, 16, 2, kernel_size, 1, dilation, norm, norm_params
        )

        self.branch1 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            BasicConv2d(
                16, 4, (1, 1), 1, mode=f"conv_{norm}_relu", norm_params=norm_params
            ),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(4, 4),
            BasicConv2d(
                16, 4, (1, 1), 1, mode=f"conv_{norm}_relu", norm_params=norm_params
            ),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(8, 8),
            BasicConv2d(
                16, 4, (1, 1), 1, mode=f"conv_{norm}_relu", norm_params=norm_params
            ),
            nn.UpsamplingBilinear2d(scale_factor=8),
        )

        self.lastconv = nn.Sequential(
            BasicConv2d(
                28,
                16,
                kernel_size,
                1,
                dilation,
                mode=f"conv_{norm}_relu",
                norm_params=norm_params,
            ),
            nn.Conv2d(16, out_channels, 1, 1),
        )

    @staticmethod
    def _make_layer(
        in_channels,
        out_channels,
        blocks,
        kernel_size,
        stride,
        dilation=(1, 1),
        norm="bn",
        norm_params=None,
    ):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = BasicConv2d(in_channels, out_channels, (1, 1), 1, mode="conv")

        layers = [
            ResBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                dilation=dilation,
                downsample=downsample,
                norm=norm,
                norm_params=norm_params,
            )
        ]
        for _ in range(1, blocks):
            layers.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    dilation=dilation,
                    norm=norm,
                    norm_params=norm_params,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        x = torch.cat(
            [layer3, self.branch1(layer3), self.branch2(layer3), self.branch3(layer3)],
            dim=1,
        )
        return self.lastconv(x)
