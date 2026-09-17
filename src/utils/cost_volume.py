import torch
import torch.nn.functional as F
import numpy as np

from model.mvs_warp_triton import lf_shift_gwc


class CostDropout(torch.nn.Module):
    def __init__(self, p=0.2, mode='zero'):
        """
        Args:
            p (float): 丢弃概率
            mode (str): 'zero' (置0, 用于拼接/相关性),
                        'max' (置最大值, 用于距离代价),
                        'noise' (置噪声)
        """
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, cost_volume):
        # 训练阶段才进行 Dropout
        if not self.training or self.p == 0:
            return cost_volume

        # 生成与 cost_volume 形状相同的 mask (1保留, 0丢弃)
        # 假设 cost_volume shape: [B, C, D, H, W]
        mask = torch.bernoulli(torch.full_like(cost_volume, 1 - self.p))

        if self.mode == 'zero':
            # 标准做法：置0，并按比例放大剩余部分以保持期望一致
            # PyTorch 的 nn.Dropout 自动做了 / (1-p)，这里手动模拟
            return (cost_volume * mask) / (1 - self.p)

        elif self.mode == 'max':
            # 针对距离 Cost：被丢弃的地方设为一个大数
            max_val = cost_volume.detach().max()
            # mask 为 0 的地方（被丢弃），填入 max_val
            # 注意：这种非线性操作通常不需要像 Dropout 那样做 scaling
            return cost_volume * mask + (1 - mask) * max_val

        elif self.mode == 'noise':
            # 噪声填充
            noise = torch.randn_like(cost_volume) * cost_volume.std() + cost_volume.mean()
            return cost_volume * mask + (1 - mask) * noise

        return cost_volume

def cost_SAD(x1, x2, cost_downfactor, reduce=True):
    '''
    x1: (B, C, H, W)
    '''
    # 注意：作为 Similarity，我们取负数；如果你的 Loss 越小越好，则去掉负号
    cost = -torch.abs(x1 - x2)
    if reduce:
        cost = torch.sum(cost, dim=1, keepdim=True)
    return F.avg_pool2d(cost, kernel_size=cost_downfactor, stride=cost_downfactor)

def cost_SSD(x1, x2, cost_downfactor, reduce=True):
    cost = -(x1 - x2) ** 2
    if reduce:
        cost = torch.sum(cost, dim=1, keepdim=True)
    return F.avg_pool2d(cost, kernel_size=cost_downfactor, stride=cost_downfactor)

def cost_cosine(x1, x2, cost_downfactor, normalize=True, reduce=True):
    '''
    x1: (B, C, H, W)
    x2: (B, C, H, W)
    cost_downfactor: int
    reduce: bool
    return: (B, 1, H', W') if reduce else (B, C, H', W')
    '''
    if normalize:
        if reduce:
            sim = F.cosine_similarity(x1, x2, dim=1).unsqueeze(1)
        else:
            sim = F.normalize(x1, p=2, dim=1) * F.normalize(x2, p=2, dim=1)
    else:
        sim = x1 * x2
        if reduce:
            sim = sim.sum(dim=1, keepdim=True)
    return F.avg_pool2d(sim, kernel_size=cost_downfactor, stride=cost_downfactor)

def cost_group_correlation(x1, x2, cost_downfactor, normalize=True, num_groups=8):
    '''
    分组相关性 (Group-wise Correlation)
    输出: (B, num_groups, H', W')
    '''
    B, C, H, W = x1.shape
    assert C % num_groups == 0, f"Channels {C} must be divisible by num_groups {num_groups}"
    channels_per_group = C // num_groups
    x1_g = x1.view(B, num_groups, channels_per_group, H, W)
    x2_g = x2.view(B, num_groups, channels_per_group, H, W)
    if normalize:
        group_sim = torch.sum(F.normalize(x1_g, p=2, dim=2) * F.normalize(x2_g, p=2, dim=2), dim=2)
    else:
        group_sim = torch.sum(x1_g * x2_g, dim=2)
    return F.avg_pool2d(group_sim, kernel_size=cost_downfactor, stride=cost_downfactor)

def get_costvolume_naive(
    views_norm, ref_view_norm, viewspos,
    maxdisp, mindisp, scalefactor, cost_downfactor,
    batch_process=False, keep_3d=False, ncc3D=None
):
    B, N, C, H, W = views_norm.shape
    device = views_norm.device
    viewspos = viewspos.to(device)
    disparity_costs = []
    if ncc3D is None:
        ncc3D = lambda x1, x2: cost_cosine(x1, x2, cost_downfactor, reduce=True)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    base_grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2)

    if batch_process:
        base_grid_bn = base_grid.view(1, 1, H, W, 2).expand(B, N, H, W, 2)

        for disp in range(mindisp * scalefactor, maxdisp * scalefactor + 1):
            if disp == 0:
                warped = views_norm
            else:
                shift_x = -disp / scalefactor * viewspos[..., 1] / (W - 1) * 2
                shift_y = -disp / scalefactor * viewspos[..., 0] / (H - 1) * 2
                grid = base_grid_bn.clone()
                grid[..., 0] += shift_x.view(B, N, 1, 1)
                grid[..., 1] += shift_y.view(B, N, 1, 1)
                grid = grid.detach()
                v = views_norm.reshape(B * N, C, H, W)
                g = grid.reshape(B * N, H, W, 2)
                warped = F.grid_sample(v, g, align_corners=True).reshape(B, N, C, H, W)

            costs = [ncc3D(ref_view_norm, warped[:, i]) for i in range(N)]
            disparity_costs.append(torch.cat(costs, dim=1))
    else:
        base_grid_b = base_grid.view(1, H, W, 2).expand(B, H, W, 2)

        for disp in range(mindisp * scalefactor, maxdisp * scalefactor + 1):
            warped_list = []
            costs = []
            for i in range(N):
                if disp == 0:
                    warped = views_norm[:, i]
                else:
                    shift_x = -disp / scalefactor * viewspos[:, i, 1] / (W - 1) * 2
                    shift_y = -disp / scalefactor * viewspos[:, i, 0] / (H - 1) * 2
                    grid = base_grid_b.clone()
                    grid[..., 0] += shift_x.view(B, 1, 1)
                    grid[..., 1] += shift_y.view(B, 1, 1)
                    grid = grid.detach()
                    warped = F.grid_sample(views_norm[:, i], grid, align_corners=True)
                    warped_list.append(warped)

                costs.append(ncc3D(ref_view_norm, warped))
            disparity_costs.append(torch.cat(costs, dim=1))

    return torch.stack(disparity_costs, dim=2) if keep_3d else torch.cat(disparity_costs, dim=1)


def get_costvolume_naive_triton(
    views_norm, ref_view_norm, viewspos,
    maxdisp, mindisp, scalefactor, cost_downfactor,
    keep_3d=False, num_groups=1
):
    device = views_norm.device
    viewspos = viewspos.to(device)
    disparity_costs = []

    for disp in range(mindisp * scalefactor, maxdisp * scalefactor + 1):
        cost = lf_shift_gwc(views_norm, ref_view_norm, viewspos, disp / scalefactor, num_groups, cost_downfactor)
        disparity_costs.append(cost)
    cost_volume = torch.stack(disparity_costs, dim=3).flatten(1,2)

    return cost_volume if keep_3d else cost_volume.flatten(1,2)


def get_costvolume_fft(
    views_norm, ref_view_norm, viewspos,
    maxdisp, mindisp, scalefactor, cost_downfactor,
    batch_process=False, keep_3d=False, ncc3D=None
):
    if ncc3D is None:
        ncc3D = lambda x1, x2: cost_cosine(x1, x2, cost_downfactor, reduce=True)
    B, N, C, H, W = views_norm.shape
    disparity_costs = []

    max_shift_y = int(np.ceil(maxdisp / scalefactor * viewspos[..., 1].abs().max().item()))
    max_shift_x = int(np.ceil(maxdisp / scalefactor * viewspos[..., 0].abs().max().item()))
    pad_y = max(max_shift_y * 2, 20)
    pad_x = max(max_shift_x * 2, 20)
    H_pad, W_pad = H + 2 * pad_y, W + 2 * pad_x

    freq_y = torch.fft.fftfreq(H_pad, device=views_norm.device).view(1, 1, -1, 1)
    freq_x = torch.fft.fftfreq(W_pad, device=views_norm.device).view(1, 1, 1, -1)

    if batch_process:
        views_pad = F.pad(
            views_norm.reshape(B * N, C, H, W),
            (pad_x, pad_x, pad_y, pad_y)
        ).reshape(B, N, C, H_pad, W_pad)
        fft_views = torch.fft.fft2(views_pad)

        for disp in range(mindisp * scalefactor, maxdisp * scalefactor + 1):
            if disp == 0:
                warped = views_norm
            else:
                px = -(disp / scalefactor * viewspos[..., 1])
                py = -(disp / scalefactor * viewspos[..., 0])
                phase = torch.exp(
                    2j * torch.pi * (
                        px.view(B, N, 1, 1, 1) * freq_x.unsqueeze(1) +
                        py.view(B, N, 1, 1, 1) * freq_y.unsqueeze(1)
                    )
                )
                warped = torch.fft.ifft2(fft_views * phase).real
                warped = warped[..., pad_y:pad_y + H, pad_x:pad_x + W]

            costs = [ncc3D(ref_view_norm, warped[:, i]) for i in range(N)]
            disparity_costs.append(torch.cat(costs, dim=1))
    else:
        fft_views = [
            torch.fft.fft2(F.pad(views_norm[:, i], (pad_x, pad_x, pad_y, pad_y)))
            for i in range(N)
        ]

        for disp in range(mindisp * scalefactor, maxdisp * scalefactor + 1):
            costs = []
            for i in range(N):
                if disp == 0:
                    warped = views_norm[:, i]
                else:
                    px = -(disp / scalefactor * viewspos[:, i, 1])
                    py = -(disp / scalefactor * viewspos[:, i, 0])
                    phase = torch.exp(
                        2j * torch.pi * (
                            px.view(B, 1, 1, 1) * freq_x +
                            py.view(B, 1, 1, 1) * freq_y
                        )
                    )
                    warped = torch.fft.ifft2(fft_views[i] * phase).real
                    warped = warped[..., pad_y:pad_y + H, pad_x:pad_x + W]
                costs.append(ncc3D(ref_view_norm, warped))
            disparity_costs.append(torch.cat(costs, dim=1))

    return torch.stack(disparity_costs, dim=2) if keep_3d else torch.cat(disparity_costs, dim=1)


def get_costvolume(
    views_norm, ref_view_norm, viewspos,
    maxdisp, mindisp, scalefactor, cost_downfactor,
    batch_process=False, interp_method='naive', keep_3d=False,
    ncc3D='cosine', ncc3D_args={'reduce': False}, normalize=True
):
    if normalize:
        views_norm = F.normalize(views_norm, dim=2, p=2)
        ref_view_norm = F.normalize(ref_view_norm, dim=1, p=2)
    if interp_method == 'naive_triton':
        return get_costvolume_naive_triton(
            views_norm, ref_view_norm, viewspos,
            maxdisp, mindisp, scalefactor, cost_downfactor,
            keep_3d, ncc3D_args['num_groups']
        )
    if ncc3D == 'cosine':
        if 'num_groups' in ncc3D_args:
            ncc3D_args.pop('num_groups', None)
        ncc3D = lambda x1, x2: cost_cosine(x1, x2, cost_downfactor, normalize=not normalize, **ncc3D_args)
    elif ncc3D == 'SAD':
        if 'num_groups' in ncc3D_args:
            ncc3D_args.pop('num_groups', None)
        ncc3D = lambda x1, x2: cost_SAD(x1, x2, cost_downfactor, **ncc3D_args)
    elif ncc3D == 'SSD':
        if 'num_groups' in ncc3D_args:
            ncc3D_args.pop('num_groups', None)
        ncc3D = lambda x1, x2: cost_SSD(x1, x2, cost_downfactor, **ncc3D_args)
    elif ncc3D == 'GWC':
        if 'reduce' in ncc3D_args:
            ncc3D_args.pop('reduce', None)
        ncc3D = lambda x1, x2: cost_group_correlation(x1, x2, cost_downfactor, normalize=not normalize, **ncc3D_args)
    if interp_method == 'fft':
        return get_costvolume_fft(
            views_norm, ref_view_norm, viewspos,
            maxdisp, mindisp, scalefactor, cost_downfactor,
            batch_process, keep_3d, ncc3D
        )
    else:
        return get_costvolume_naive(
            views_norm, ref_view_norm, viewspos,
            maxdisp, mindisp, scalefactor, cost_downfactor,
            batch_process, keep_3d, ncc3D
        )
