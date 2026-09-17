# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Modifications Copyright (c) 2026 SynLF Authors.
# Derived from NVlabs/FoundationStereo core/update.py.
# See LICENSES/FoundationStereo-LICENSE.txt for the applicable license.


import torch
import torch.nn as nn
import torch.nn.functional as F


class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)

class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels=4, corr_radius=4, ngroup=8):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2*corr_radius + 1) * ngroup
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 16, 7, padding=3)
        self.convd2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv = nn.Conv2d(64+16, 32-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, x, hx):
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h


class SimpleConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super(SimpleConvGRU, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim+hidden_dim, input_dim+hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = RaftConvGRU(hidden_dim, input_dim, kernel_size)

    def forward(self, h, *x):
        x = torch.cat(x, dim=1)
        x = self.conv0(x)
        hx = torch.cat([x, h], dim=1)
        hx = self.conv1(hx)
        h = self.gru(h, x, hx)
        return h


class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, corr_levels=4, corr_radius=4, n_gru_layers=3, hidden_dim=128, ngroup=1, upsample_factor=4, out_mask=True):
        super().__init__()
        self.n_gru_layers = n_gru_layers
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius, ngroup)

        if n_gru_layers == 3:
            self.gru16 = SimpleConvGRU(hidden_dim, hidden_dim * 2)
        if n_gru_layers >= 2:
            self.gru08 = SimpleConvGRU(hidden_dim, hidden_dim * (n_gru_layers == 3) + hidden_dim * 2)
        self.gru04 = SimpleConvGRU(hidden_dim, hidden_dim * (n_gru_layers > 1) + hidden_dim * 2)
        self.disp_head = DispHead(hidden_dim, hidden_dim*2)
        self.upsample_factor = upsample_factor
        if upsample_factor > 1:
            if out_mask:
                self.mask = nn.Sequential(
                    nn.Conv2d(hidden_dim, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, (upsample_factor**2)*9, 3, padding=1),
                    # nn.ReLU(inplace=True), # 直接是权重的话不应该加relu
                    )
            else:
                self.mask = nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1),
                    nn.ReLU(inplace=True),
                    )
    def forward(self, net, inp, corr, disp):
        if self.n_gru_layers == 3:
            net[2] = self.gru16(net[2], inp[2], pool2x(net[1]))
        if self.n_gru_layers >= 2:
            if self.n_gru_layers > 2:
                net[1] = self.gru08(net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], inp[1], pool2x(net[0]))
        motion_features = self.encoder(disp, corr)
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        if self.n_gru_layers > 1:
            net[0] = self.gru04(net[0], motion_features, interp(net[1], net[0]))

        delta_disp = self.disp_head(net[0])
        if self.upsample_factor > 1:
            mask = .25 * self.mask(net[0])
            return net, mask, delta_disp
        else:
            mask = None
            return net, delta_disp
