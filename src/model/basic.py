import torch.nn as nn


def _activation(name, params=None):
    if name == "leaky_relu":
        return nn.LeakyReLU((params or {}).get("alpha", 0.2), inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        dilation=(1, 1),
        groups=1,
        bias=False,
        mode="conv_bn",
        activation="leaky_relu",
        activation_params=None,
        norm_params=None,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        padding = tuple(d * (k - 1) // 2 for d, k in zip(dilation, kernel_size))
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if mode in ("conv_bn", "conv_bn_relu"):
            modules.append(nn.BatchNorm2d(out_channels))
        elif mode in ("conv_gn", "conv_gn_relu"):
            num_groups = (norm_params or {}).get("num_groups", out_channels // 2)
            modules.append(nn.GroupNorm(max(num_groups, 1), out_channels))
        if mode in ("conv_bn_relu", "conv_gn_relu", "conv_relu"):
            modules.append(_activation(activation, activation_params))
        self.conv = nn.Sequential(*modules)

    def forward(self, value):
        return self.conv(value)


class BasicConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        groups=1,
        bias=False,
        mode="conv_bn",
        activation="leaky_relu",
        activation_params=None,
        norm_params=None,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(dilation, int):
            dilation = (dilation,) * 3
        padding = tuple(d * (k - 1) // 2 for d, k in zip(dilation, kernel_size))
        modules = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if mode in ("conv_bn", "conv_bn_relu"):
            modules.append(nn.BatchNorm3d(out_channels))
        elif mode in ("conv_gn", "conv_gn_relu"):
            num_groups = (norm_params or {}).get("num_groups", out_channels // 2)
            modules.append(nn.GroupNorm(max(num_groups, 1), out_channels))
        if mode in ("conv_bn_relu", "conv_gn_relu"):
            modules.append(_activation(activation, activation_params))
        self.conv = nn.Sequential(*modules)

    def forward(self, value):
        return self.conv(value)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        dilation=(1, 1),
        groups=1,
        bias=False,
        downsample=None,
        norm="bn",
        norm_params=None,
        activation="relu",
        activation_params=None,
    ):
        super().__init__()
        self.downsample = downsample
        self.conv = nn.Sequential(
            BasicConv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                mode=f"conv_{norm}_relu",
                activation=activation,
                activation_params=activation_params,
                norm_params=norm_params,
            ),
            BasicConv2d(
                out_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                groups=groups,
                bias=bias,
                mode=f"conv_{norm}",
                norm_params=norm_params,
            ),
        )
        self.act = _activation(activation, activation_params) if activation != "identity" else nn.Identity()

    def forward(self, value):
        skip = self.downsample(value) if self.downsample is not None else value
        return self.act(self.conv(value) + skip)


class APConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        kernel_size_spatial=(1, 3, 3),
        kernel_size_disp=(3, 1, 1),
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        groups=1,
        bias=False,
        norm="bn",
        norm_params=None,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        if isinstance(kernel_size_spatial, int):
            kernel_size_spatial = (1, kernel_size_spatial, kernel_size_spatial)
        if isinstance(kernel_size_disp, int):
            kernel_size_disp = (kernel_size_disp, 1, 1)
        self.conv_spatial = BasicConv3d(
            in_channels,
            mid_channels,
            kernel_size_spatial,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            mode=f"conv_{norm}_relu",
            norm_params=norm_params,
        )
        self.conv_disp = BasicConv3d(
            mid_channels,
            out_channels,
            kernel_size_disp,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            mode=f"conv_{norm}_relu",
            norm_params=norm_params,
        )

    def forward(self, value):
        return self.conv_disp(self.conv_spatial(value))
