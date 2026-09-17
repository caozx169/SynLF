import torch
import numpy as np
from torch.nn.functional import conv2d
from matplotlib import colormaps


def rescale(v, min_v, max_v):
    original_min = v.min()
    original_max = v.max()
    return (v - original_min) / (original_max - original_min) * (max_v - min_v) + min_v


def compute_lf_disparity(values, valid_mask, input_type, stats,
                          median_thresholds, median_multipliers,
                          beta=-9.0, p_scale=0.9):
    """
    统一将 depth 或 raw_disp 映射到光场 disparity。

    Args:
        values: (H, W) np.ndarray, 原始 depth 或 raw disparity
        valid_mask: (H, W) bool np.ndarray, 有效像素 mask
        input_type: "depth" 或 "raw_disp"
        stats: dict, 场景统计信息，包含 'p10', 'p90', 'median'
        median_thresholds: list[float], 升序的深度中位数分段阈值
        median_multipliers: list[float], 长度为 len(median_thresholds) + 1 的倍数列表
        beta: float, disparity 偏移量（通常为 mindisp）
        p_scale: float, 对统计量的缩放系数。
                 depth 类型: alpha = p10 * p_scale * multiplier
                 raw_disp 类型: normalized = values / (p90 * p_scale), 然后 multiplier * normalized + beta

    Returns:
        disparity: (H, W) np.ndarray
    """
    assert len(median_multipliers) == len(median_thresholds) + 1, \
        f"median_multipliers 长度应为 {len(median_thresholds) + 1}，实际为 {len(median_multipliers)}"

    median = stats['median']

    # 根据 median 找到对应的 multiplier
    multiplier_idx = len(median_thresholds)  # 默认取最后一个
    for i, threshold in enumerate(median_thresholds):
        if median < threshold:
            multiplier_idx = i
            break
    multiplier = median_multipliers[multiplier_idx]

    disparity = np.full_like(values, beta)

    if input_type == "depth":
        # depth → disparity: alpha / depth + beta
        p10 = stats['p10']
        alpha = p10 * p_scale * multiplier
        disparity[valid_mask] = alpha / values[valid_mask] + beta
    elif input_type == "raw_disp":
        # raw_disp → disparity: multiplier * (disp / (p90 * p_scale)) + beta
        p90 = stats['p90']
        normalized = values[valid_mask] / (p90 * p_scale)
        disparity[valid_mask] = multiplier * normalized + beta
    else:
        raise ValueError(f"Unknown input_type: {input_type}, expected 'depth' or 'raw_disp'")

    disparity[~valid_mask] = beta
    return disparity


def depth2normal(depth, K=None, auto_limit=True):
    """
    输入:
        depth: (H, W) 或 (B, H, W) 或 (B, C, H, W) 的深度图，值为浮点数，表示深度值
        K: 相机内参矩阵 (3x3)，如果提供则使用精确的几何方法计算法线
        auto_limit: 当 K=None 时，是否根据图像大小自动计算合适的深度范围（用于 rescale）
    输出:
        normal_map: (H, W, 3) 的法线图, RGB通道对应法线的 x, y, z 分量
    """
    # 保存原始形状以便后续恢复
    numdims = len(depth.shape)
    original_depth = depth

    # 扩展维度以进行卷积操作
    if numdims == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    elif numdims == 3:
        depth = depth.unsqueeze(1)

    if K is None and auto_limit:
        # 获取图像尺寸
        if len(depth.shape) == 4:  # (B, C, H, W)
            H, W = depth.shape[-2:]
        else:  # (B, H, W) 或其他
            H, W = depth.shape[-2:]

        # 计算基于图像大小的 limit
        # 使用图像对角线长度作为参考，归一化到 1080p 基准 (1920x1080)
        base_resolution = 768 * 1024  # 1080p 的像素数
        current_resolution = H * W
        resolution_scale = (current_resolution / base_resolution) ** 0.5  # 开方，因为 limit 是线性尺度

        base_limit = 200.0
        limit = base_limit * resolution_scale
        depth = rescale(depth, 0, limit)

    if K is None:
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)

        # 使用 Sobel 核计算梯度
        grad_x = conv2d(depth, sobel_x, padding=1)
        grad_y = conv2d(depth, sobel_y, padding=1)

        # 初始化法线图
        normal_map = torch.cat((-grad_x, -grad_y, torch.ones_like(depth)), dim=1)
        normal_map = normal_map / normal_map.norm(dim=1, keepdim=True)
    else:
        B, C, H, W = depth.shape
        x, y = torch.meshgrid([torch.arange(0, depth.shape[-2], dtype=torch.float32, device=depth.device),
                            torch.arange(0, depth.shape[-1], dtype=torch.float32, device=depth.device)], indexing='ij')
        xyz = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(1, 3,-1)
        xyz = torch.matmul(K.inverse(), xyz) * depth.reshape(B, 1, -1)
        xyz = xyz.view(depth.shape[0], 3, H, W)
        du = xyz[:, :, 1:, :] - xyz[:, :, :-1, :]
        du = torch.nn.functional.pad(du, (0, 0, 1, 0), mode='replicate')
        dv = xyz[:, :, :, 1:] - xyz[:, :, :, :-1]
        dv = torch.nn.functional.pad(dv, (1, 0, 0, 0), mode='replicate')
        normal_map = torch.cross(du, dv, dim=1)
        normal_map = normal_map / torch.norm(normal_map, dim=1, keepdim=True)

    if numdims == 2:
        normal_map = normal_map.squeeze()
    return normal_map


def colorize(value, vmin=None, vmax=None, cmap='Spectral'):
    istensor = isinstance(value, torch.Tensor)
    if istensor:
        value = value.detach().cpu().numpy()
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.

    cmapper = colormaps[cmap]
    cmapper.set_bad(color='black')
    cmapper.set_under(color='black')
    cmapper.set_over(color='black')
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1)) if not istensor else torch.tensor(img).permute([2, 0, 1])


def disp2depth(disp, coef):
    return coef[0] / (disp - coef[2]) - coef[1]
