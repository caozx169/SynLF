import torch
import numpy as np


def randomcrop(cropsize, *args):
    assert len(args) > 0
    if isinstance(args[0], torch.Tensor):
        h, w = cropsize
        H, W = args[0].shape[-2:]
        # 如果图像小于 cropsize，先 pad 到至少 cropsize 大小
        if H < h or W < w:
            pad_bottom = max(0, h - H)
            pad_right = max(0, w - W)
            args = [torch.nn.functional.pad(arg, (0, pad_right, 0, pad_bottom), mode='constant', value=0) for arg in args]
            H, W = args[0].shape[-2:]
        if H == h and W == w:
            return args
        i = torch.randint(0, H - h + 1, (1,))[0]
        j = torch.randint(0, W - w + 1, (1,))[0]
        return [arg[..., i:i+h, j:j+w] for arg in args]
    elif isinstance(args[0], np.ndarray):
        h, w = cropsize
        H, W = args[0].shape[:2]
        # 如果图像小于 cropsize，先 pad 到至少 cropsize 大小
        if H < h or W < w:
            pad_bottom = max(0, h - H)
            pad_right = max(0, w - W)
            padded = []
            for arg in args:
                if arg.ndim == 2:
                    pad_width = ((0, pad_bottom), (0, pad_right))
                else:
                    pad_width = ((0, pad_bottom), (0, pad_right)) + tuple((0, 0) for _ in range(arg.ndim - 2))
                padded.append(np.pad(arg, pad_width, mode='constant', constant_values=0))
            args = padded
            H, W = args[0].shape[:2]
        if H == h and W == w:
            return args
        i = np.random.randint(0, H - h + 1)
        j = np.random.randint(0, W - w + 1)
        return [arg[i:i+h, j:j+w, ...] for arg in args]
    else:
        raise ValueError(f'Invalid input type: {type(args[0])}')


def normalize_cropsize(cropsize, default_hw):
    """
    统一处理 cropsize 参数，返回 (cropsize_tuple, do_randomcrop)。

    Args:
        cropsize: None, int, tuple, list, 或 np.ndarray
        default_hw: 默认的 (H, W) 尺寸，当 cropsize 为 None 时使用
    Returns:
        (cropsize_tuple, do_randomcrop): 归一化后的 cropsize 元组和是否需要随机裁剪
    """
    if cropsize is None:
        return tuple(default_hw), False
    if isinstance(cropsize, int):
        return (cropsize, cropsize), True
    return tuple(cropsize), True


def mask_guided_randomcrop(cropsize, mask, *args):
    """
    Mask-guided random crop: 优先裁剪包含mask非零区域的patch。

    Args:
        cropsize: (h, w)
        mask: torch.Tensor 或 np.ndarray, 形状与 args 相同空间尺寸
        *args: 任意数量的图像或特征图（与 mask 尺寸对应）
    Returns:
        [cropped tensors/arrays ...]
    """
    assert len(args) > 0, "At least one tensor/array is required"
    h, w = cropsize

    # Torch 版本
    if isinstance(mask, torch.Tensor):
        H, W = mask.shape[-2:]
        if H <= h or W <= w:
            return args

        # 找到mask非零区域
        ys, xs = torch.nonzero(mask > 0, as_tuple=True)
        if len(ys) > 0:
            # 随机选一个非零点为中心
            yc, xc = ys[torch.randint(0, len(ys), (1,))], xs[torch.randint(0, len(xs), (1,))]
            # 计算裁剪窗口左上角
            i = int(torch.clamp(yc - h // 2, 0, H - h))
            j = int(torch.clamp(xc - w // 2, 0, W - w))
        else:
            # mask全零，退化为普通随机裁剪
            i = torch.randint(0, H - h + 1, (1,))[0]
            j = torch.randint(0, W - w + 1, (1,))[0]

        return [arg[..., i:i+h, j:j+w] for arg in args]

    # Numpy 版本
    elif isinstance(mask, np.ndarray):
        H, W = mask.shape[:2]
        if H <= h or W <= w:
            return args

        ys, xs = np.nonzero(mask > 0)
        if len(ys) > 0:
            idx = np.random.randint(0, len(ys))
            yc, xc = ys[idx], xs[idx]
            i = np.clip(yc - h // 2, 0, H - h)
            j = np.clip(xc - w // 2, 0, W - w)
        else:
            i = np.random.randint(0, H - h + 1)
            j = np.random.randint(0, W - w + 1)

        return [arg[i:i+h, j:j+w, ...] for arg in args]

    else:
        raise ValueError(f"Unsupported input type: {type(mask)}")
