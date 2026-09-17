import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import BasicConv2d, ResBlock
from .cost import CostProcessHourglass3dAF
from .view_encoder import FeatureExtraction
from utils.cost_volume import get_costvolume
from utils.model_ops import freeze_model, upsample_disp
from .depth_anything_v2.dpt import DepthAnythingV2
from .update import BasicSelectiveMultiUpdateBlock
from .corr import GeoEncodingVolume
from .triton_vis_splat import soft_visibility_forward_splat_triton


class DisparityRegression(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, maxdisp, mindisp):
        if x.dim() > 4:
            x = x.squeeze(1)
        assert x.dim() == 4
        x = F.softmax(x, dim=1)

        disp = torch.arange(mindisp, maxdisp + 1, device=x.device, dtype=x.dtype)
        disp = disp.view(1, x.size(1), 1, 1)

        pred = torch.sum(x * disp, 1, keepdim=True)
        return pred


class ContextFeatureFusion(nn.Module):
    def __init__(self, in_channels, vit_channels, out_channels, norm="gn", norm_params=None) -> None:
        super().__init__()
        self.conv = BasicConv2d(in_channels + vit_channels, out_channels, kernel_size=3, mode="conv")
        downsample = BasicConv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            mode=f"conv_{norm}",
            norm_params=norm_params,
        )
        self.down1 = nn.Sequential(
            ResBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                downsample=downsample,
                norm=norm,
                norm_params=norm_params,
            ),
            ResBlock(out_channels, out_channels, kernel_size=3, stride=1, norm=norm, norm_params=norm_params),
        )
        downsample = BasicConv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            mode=f"conv_{norm}",
            norm_params=norm_params,
        )
        self.down2 = nn.Sequential(
            ResBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                downsample=downsample,
                norm=norm,
                norm_params=norm_params,
            ),
            ResBlock(out_channels, out_channels, kernel_size=3, stride=1, norm=norm, norm_params=norm_params),
        )
        output_list = []
        for _ in range(2):
            conv_out = nn.Sequential(
                ResBlock(out_channels, out_channels, kernel_size=3, norm=norm, norm_params=norm_params, stride=1),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            )
            output_list.append(conv_out)
        self.outputs01 = nn.ModuleList(output_list)
        output_list = []
        for _ in range(2):
            conv_out = nn.Sequential(
                ResBlock(out_channels, out_channels, kernel_size=3, norm=norm, norm_params=norm_params, stride=1),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            )
            output_list.append(conv_out)
        self.outputs02 = nn.ModuleList(output_list)
        output_list = []
        for _ in range(2):
            conv_out = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            output_list.append(conv_out)
        self.outputs04 = nn.ModuleList(output_list)

    def forward(self, x, vit_feature):
        x = torch.cat([x, vit_feature], dim=1)
        x = self.conv(x)
        outputs01 = [f(x) for f in self.outputs01]

        y = self.down1(x)
        outputs02 = [f(y) for f in self.outputs02]

        z = self.down2(y)
        outputs04 = [f(z) for f in self.outputs04]

        return (outputs01, outputs02, outputs04)


class VisDepth(nn.Module):
    def __init__(
        self,
        maxdisp,
        mindisp,
        disp_upfactor,
        cost_downfactor,
        viewspos,
        in_dim,
        out_dim,
        enc_ksize=3,
        enc_dila=1,
        enc_norm="gn",
        enc_nparams=None,
        cost_dim=8,
        cost_ksize_xy=3,
        cost_ksize_d=3,
        cost_dila=1,
        cost_norm="gn",
        cost_nparams=None,
        ctx_dim=32,
        ctx_norm="gn",
        ctx_nparams=None,
        corr_levels=2,
        corr_radius=1,
        cost_dim_keep=False,
        gru_hidden_dim=32,
        gru_n_layers=3,
        gru_iters=12,
        cost_interp_method="naive",
        ncc3D="cosine",
        ncc3D_args={"reduce": False},
        ncc3D_normalize=True,
        vit_type="vits",
        vit_ckpt=None,
        load_pretrained=True,
    ):
        super().__init__()
        self.maxdisp = maxdisp
        self.mindisp = mindisp
        self.displevels = (maxdisp - mindisp) * disp_upfactor + 1
        self.disp_upfactor = disp_upfactor
        self.viewspos = torch.tensor(viewspos, dtype=torch.float32)
        self.centerview_index = torch.where(self.viewspos.sum(-1) == 0)[0].item()
        self.vnum = len(self.viewspos)
        self.src_vnum = self.vnum - 1
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.cost_interp_method = cost_interp_method
        self.cost_dim_keep = cost_dim_keep
        self.ncc3D = ncc3D
        self.ncc3D_args = dict(ncc3D_args)
        self.ncc3D_normalize = ncc3D_normalize
        self.eps = 1e-6

        self.encoder = FeatureExtraction(
            in_dim,
            out_dim,
            enc_ksize,
            enc_dila,
            norm=enc_norm,
            norm_params=enc_nparams
        )
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        vit_feature = model_configs[vit_type]["features"] // 2
        depth_anything = DepthAnythingV2(**model_configs[vit_type])
        if load_pretrained:
            if vit_ckpt is None:
                raise ValueError("vit_ckpt must point to the directory containing the Depth Anything V2 weights")
            depth_anything.load_state_dict(
                torch.load(f"{vit_ckpt}/depth_anything_v2_{vit_type}.pth", map_location="cpu", weights_only=False)
            )
        self.vit = freeze_model(depth_anything)

        if ncc3D != "GWC":
            cost_in_channels = out_dim * (self.vnum - 1) if ncc3D_args["reduce"] is False else (self.vnum - 1)
        else:
            cost_in_channels = (self.vnum - 1) * ncc3D_args["num_groups"]

        if cost_in_channels % self.src_vnum != 0:
            raise ValueError(
                f"cost_in_channels ({cost_in_channels}) must be divisible by src_vnum ({self.src_vnum})"
            )

        self.raw_cost_channels_per_view = cost_in_channels // self.src_vnum
        self.geo_channels = cost_dim if cost_dim_keep else 1

        self.costprocess_af = CostProcessHourglass3dAF(
            in_channels=cost_in_channels,
            out_channels=cost_dim,
            features=cost_dim,
            kernel_size_spatial=cost_ksize_xy,
            kernel_size_disp=cost_ksize_d,
            dilation=cost_dila,
            groups=1,
            norm=cost_norm,
            norm_params=cost_nparams,
            mono_channels=vit_feature,
        )

        self.raw_cost_inject_fuse3d = nn.Sequential(
            nn.Conv3d(self.geo_channels + self.raw_cost_channels_per_view, self.geo_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.geo_channels, self.geo_channels, kernel_size=3, padding=1),
        )

        self.cnet = ContextFeatureFusion(out_dim, vit_feature, ctx_dim, norm=ctx_norm, norm_params=ctx_nparams)
        self.disparityregression = DisparityRegression()

        self.update_block = BasicSelectiveMultiUpdateBlock(
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            n_gru_layers=gru_n_layers,
            hidden_dim=gru_hidden_dim,
            ngroup=self.geo_channels,
            upsample_factor=cost_downfactor,
        )

        self.cost_downfactor = cost_downfactor
        self.gru_iters = gru_iters

    def _reshape_raw_cost(self, cost_raw):
        b, c, d, h, w = cost_raw.shape
        if c % self.src_vnum != 0:
            raise ValueError(f"Raw cost channels ({c}) must be divisible by src views ({self.src_vnum})")
        c_per_view = c // self.src_vnum
        return cost_raw.view(b, self.src_vnum, c_per_view, d, h, w)

    def _aggregate_occ_cost(self, raw_cost, vis):
        num = (raw_cost * vis.unsqueeze(2)).sum(dim=1)
        den = vis.sum(dim=1) + self.eps
        return num / den.unsqueeze(1)

    def forward(self, x, viewspos=None):
        iters = self.gru_iters
        if viewspos is None:
            viewspos = self.viewspos[None].repeat(x.shape[0], 1, 1)
        viewspos = viewspos.to(device=x.device, dtype=x.dtype)

        b, n, c, h, w = x.shape
        if n != self.vnum:
            raise ValueError(f"Expected {self.vnum} views, got {n}")

        x_reshaped = x.view(b * n, c, h, w)
        features = self.encoder(x_reshaped)
        feature_list = features.view(b, n, -1, h, w)

        views_feature = torch.cat(
            (feature_list[:, : self.centerview_index], feature_list[:, self.centerview_index + 1 :]),
            dim=1,
        )
        ref_feature = feature_list[:, self.centerview_index]
        viewspos_src = torch.cat(
            (viewspos[:, : self.centerview_index], viewspos[:, self.centerview_index + 1 :]),
            dim=1,
        )

        interp_method = self.cost_interp_method
        if interp_method == "naive_triton" and (not x.is_cuda):
            interp_method = "naive"

        cost_raw = get_costvolume(
            views_feature,
            ref_feature,
            viewspos_src,
            self.maxdisp,
            self.mindisp,
            self.disp_upfactor,
            self.cost_downfactor,
            batch_process=False,
            interp_method=interp_method,
            keep_3d=True,
            ncc3D=self.ncc3D,
            ncc3D_args=dict(self.ncc3D_args),
            normalize=self.ncc3D_normalize,
        )

        raw_cost_viewed = self._reshape_raw_cost(cost_raw)

        r = self.corr_radius
        dx = torch.linspace(-r, r, 2 * r + 1, requires_grad=False, device=cost_raw.device, dtype=cost_raw.dtype).reshape(
            1, 1, 2 * r + 1, 1
        )

        with torch.no_grad():
            mono_disp, mono_feature, ori_feature = self.vit.infer(x[:, self.centerview_index], return_feature=True)
            mono_disp = -mono_disp

        if self.cost_dim_keep:
            cost, cost_feature = self.costprocess_af(
                cost_raw,
                mono_feature,
                return_feature=True,
            )
        else:
            cost = self.costprocess_af(cost_raw, mono_feature)
            cost_feature = None

        disp_raw = (
            self.disparityregression(cost, self.maxdisp * self.disp_upfactor, self.mindisp * self.disp_upfactor)
            / self.disp_upfactor
        )

        if self.cost_downfactor > 1:
            ref_feature_ctx = F.interpolate(
                ref_feature, scale_factor=1 / self.cost_downfactor, mode="bilinear", align_corners=True
            )
            mono_feature_ctx = F.interpolate(
                mono_feature, scale_factor=1 / self.cost_downfactor, mode="bilinear", align_corners=True
            )
        else:
            ref_feature_ctx = ref_feature
            mono_feature_ctx = mono_feature

        cnet_list = list(self.cnet(ref_feature_ctx, vit_feature=mono_feature_ctx))
        net_list = [torch.tanh(v[0]) for v in cnet_list]
        inp_list = [torch.relu(v[1]) for v in cnet_list]

        b, _, hc, wc = disp_raw.shape
        coords = (
            torch.arange(wc, dtype=disp_raw.dtype, device=disp_raw.device).reshape(1, 1, wc, 1).repeat(b, hc, 1, 1)
        )
        disp = disp_raw.float()
        disp_preds = []

        for _ in range(iters):
            disp = disp.detach()

            vis = soft_visibility_forward_splat_triton(disp / self.cost_downfactor, viewspos_src)
            raw_cost_occ = self._aggregate_occ_cost(raw_cost_viewed, vis)

            geo_volume_base = cost_feature if self.cost_dim_keep else cost.unsqueeze(1)
            geo_volume_fused = self.raw_cost_inject_fuse3d(torch.cat([geo_volume_base, raw_cost_occ], dim=1))

            geo_fn = GeoEncodingVolume(geo_volume_fused, num_levels=self.corr_levels, dx=dx)
            geo_feat = geo_fn((disp - self.mindisp) * self.disp_upfactor, coords)

            if self.cost_downfactor > 1:
                net_list, mask, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp)
                disp = disp + delta_disp.float()
                disp_up = upsample_disp(disp.float(), mask.float(), self.cost_downfactor)
                disp_preds.append(disp_up)
            else:
                net_list, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp)
                disp = disp + delta_disp.float()
                disp_preds.append(F.interpolate(disp, scale_factor=self.cost_downfactor, mode="bilinear", align_corners=True))

        disp_raw_up = F.interpolate(disp_raw, scale_factor=self.cost_downfactor, mode="bilinear", align_corners=True)
        if disp_preds:
            disp_final = disp_preds[-1]
            disp_mid = disp_preds[:-1]
        else:
            disp_final = disp_raw_up
            disp_mid = []

        return {
            "disp_raw": disp_raw_up,
            "disp": disp_final,
            "disp_preds": disp_mid,
            "ori_feature": F.normalize(ori_feature, dim=1),
            "ref_feature": F.normalize(
                F.adaptive_avg_pool2d(feature_list[:, self.centerview_index], ori_feature.shape[-2:]),
                dim=1,
            ),
        }

    def get_group_parameters(self):
        params = list(self.named_parameters())
        param_group = [
            {"params": [p for n, p in params if n.startswith("encoder") and p.requires_grad]},
            {"params": [p for n, p in params if not n.startswith("encoder") and p.requires_grad]},
        ]
        return param_group
