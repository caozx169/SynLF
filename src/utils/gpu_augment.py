"""
GPU 端数据增强模块。
将原本在 CPU DataLoader worker 中执行的 _post_augment 操作
搬到 GPU 上执行，消除多进程竞争瓶颈。
"""

import torch
import torch.nn as nn
from torch.fft import fft2, ifft2
from torch.nn.functional import conv2d
from math import floor


# =========================================================
# Lp-Norm PSF 模型参数 (RSS 拟合)
# =========================================================
PSF_PARAMS = {
    'alpha': {'k': 0.5872, 'xc': -0.01893, 'dm': 3.648, 's': 2.831, 'a0': 3.228},
    'beta':  {'A': 90.48, 'gamma': 0.2371, 'dc': 0.1855, 'base': 2.896},
    'pnorm': {'p_base': 3.289, 'dm': 1.952, 'xc': 0.0708, 's': 3.221}
}


def compute_alpha(disp, scale=1):
    """计算给定 disparity 对应的 PSF alpha（宽度）参数"""
    disp_phys = disp * scale
    p_a = PSF_PARAMS['alpha']
    return (p_a['k']**2 * ((disp_phys - p_a['xc'])**2 - p_a['dm']**2)**2 /
            ((disp_phys - p_a['xc'])**2 + p_a['s']**2) + p_a['a0']**2)**0.5


def generate_psfs_batched(disparities, size=35, scale=9, device='cpu'):
    """
    根据一组 disparity 值批量生成 Lp-Norm PSF。

    Args:
        disparities: (bins,) tensor, 每个 bin 中心的 disparity 值
        size: PSF 的输出尺寸 (size x size)
        scale: 超采样倍率，>1 时先在高分辨率下生成再 avg pooling
        device: 计算设备
    Returns:
        psfs: (bins, 1, size, size) tensor, 归一化的 PSF 核
    """
    bins = disparities.shape[0]

    # 1. 物理映射：将低分辨率视差转换回高分辨率参数空间
    disp_phys = disparities * scale
    res_size = size * scale

    # 2. 坐标网格 (res_size x res_size)
    limit = (res_size - 1) / 2
    coords = torch.linspace(-limit, limit, res_size, device=device)
    X, Y = torch.meshgrid(coords, coords, indexing='ij')  # (res_size, res_size)

    # 3. 参数计算 (向量化，对所有 bins 同时计算)
    p_a = PSF_PARAMS['alpha']
    alpha = torch.sqrt(
        p_a['k']**2 * ((disp_phys - p_a['xc'])**2 - p_a['dm']**2)**2 /
        ((disp_phys - p_a['xc'])**2 + p_a['s']**2) + p_a['a0']**2
    )  # (bins,)

    p_b = PSF_PARAMS['beta']
    beta = p_b['A'] * (p_b['gamma']**2 / ((disp_phys - p_b['dc'])**2 + p_b['gamma']**2)) + p_b['base']

    p_p = PSF_PARAMS['pnorm']
    p_norm = p_p['p_base'] - (p_p['p_base'] - 2.0) * (
        p_p['s']**2 / ((torch.abs(disp_phys - p_p['xc']) - p_p['dm'])**2 + p_p['s']**2)
    )  # (bins,)

    # 4. Lp-Norm PSF 计算 (向量化)
    # alpha, beta, p_norm: (bins,) → (bins, 1, 1) for broadcast with (res_size, res_size)
    alpha_v = alpha.view(-1, 1, 1)
    beta_v = beta.view(-1, 1, 1)
    pnorm_v = p_norm.view(-1, 1, 1)

    term1 = (torch.abs(X / alpha_v)**pnorm_v + torch.abs(Y / alpha_v)**pnorm_v)**(beta_v / pnorm_v)
    psf_large = torch.exp(-term1)  # (bins, res_size, res_size)

    # 5. 降采样 (Avg Pooling)
    if scale > 1:
        psf = psf_large.reshape(bins, size, scale, size, scale).mean(dim=(2, 4))
    else:
        psf = psf_large  # (bins, size, size)

    # 6. 归一化并添加 channel 维度
    psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    psf = psf.unsqueeze(1)  # (bins, 1, size, size)

    return psf


def center2bin(center):
    interval = torch.diff(center)
    interval = torch.cat([interval[:1], interval], 0)
    bin = center - interval / 2
    bin = torch.cat([bin, center[-1:] + interval[-1:] / 2], 0)
    return bin


def continuous2bin(x: torch.Tensor, bin_edges: torch.Tensor):
    lb = x - bin_edges[:-1].reshape(torch.Size([-1]) + torch.Size(torch.ones(len(x.shape), dtype=torch.int)))
    ub = x - bin_edges[1:].reshape(torch.Size([-1]) + torch.Size(torch.ones(len(x.shape), dtype=torch.int)))
    volume_mask = ((lb >= 0) & (ub < 0))
    map = volume_mask.float().argmax(0).int()
    map[volume_mask.sum(0) == 0] = -1
    return map, volume_mask


def fftConv(x: torch.Tensor, kernel: torch.Tensor, padding='same'):
    input_fft = fft2(x, s=(kernel.shape[-2] + x.shape[-2] - 1, kernel.shape[-1] + x.shape[-1] - 1), dim=(-2, -1))
    kernel_fft = fft2(kernel, s=(kernel.shape[-2] + x.shape[-2] - 1, kernel.shape[-1] + x.shape[-1] - 1), dim=(-2, -1))
    conv_fft_real = input_fft.real * kernel_fft.real - input_fft.imag * kernel_fft.imag
    conv_fft_imag = input_fft.real * kernel_fft.imag + input_fft.imag * kernel_fft.real
    conv_fft = torch.complex(conv_fft_real, conv_fft_imag)
    if padding == 'same':
        conv_result = ifft2(conv_fft, dim=(-2, -1)).abs()[
            ..., floor(kernel.shape[-2] / 2):floor(kernel.shape[-2] / 2) + x.shape[-2],
            floor(kernel.shape[-1] / 2):floor(kernel.shape[-1] / 2) + x.shape[-1]
        ]
    elif padding == 'valid':
        conv_result = ifft2(conv_fft, dim=(-2, -1)).abs()[
            ..., kernel.shape[-2] - 1:kernel.shape[-2] - 1 + x.shape[-2] - kernel.shape[-2] + 1,
            kernel.shape[-1] - 1:kernel.shape[-1] - 1 + x.shape[-1] - kernel.shape[-1] + 1
        ]
    elif padding == 'full':
        conv_result = ifft2(conv_fft, dim=(-2, -1)).abs()
    else:
        raise ValueError('padding must be one of "same", "valid", or "full"')
    return conv_result


def optical_blur_batched(imgs, depths, psf, z_u, padding='same', mode='alpha'):
    """
    对 P 个视图同时施加光学模糊，共享同一组 PSF。

    imgs:   (P, C, H, W) float32
    depths: (P, 1, H, W) float32  (取负后的 disparity)
    psf:    (bins, 1, K, K) float32
    z_u:    (bins+1,) float32   bin edges

    返回:   (P, C, H, W) float32
    """
    P, C, H, W = imgs.shape

    # continuous2bin 天然支持 batched depth:
    # depths (P, 1, H, W) → volume_mask (bins, P, 1, H, W)
    _, volume_mask = continuous2bin(depths, z_u)

    # volume: (bins, P, C, H, W) — 通过 broadcast
    volume = (volume_mask * imgs.unsqueeze(0)).float()  # (bins, P, C, H, W)

    # Flatten P*C → 一次性做 FFT
    # (bins, P, C, H, W) → (bins, P*C, H, W)
    volume_flat = volume.reshape(volume.shape[0], P * C, H, W)

    if mode == 'naive':
        blurred_flat = fftConv(volume_flat, psf, padding)  # (bins, P*C, H, W)
        capimg_flat = torch.sum(blurred_flat, 0)  # (P*C, H, W)

    elif mode == 'alpha':
        # volume_mask: (bins, P, 1, H, W) → flatten: (bins, P*1, H, W) = (bins, P, H, W)
        vm_flat = volume_mask.reshape(volume_mask.shape[0], P, H, W)

        cumsum_alpha = torch.cumsum(vm_flat.flip(0), dim=0).flip(0).float()
        blurred_cumsum_alpha = fftConv(cumsum_alpha, psf, padding)  # (bins, P, H, W)

        # 把 blurred_cumsum_alpha 扩展到 P*C 维度，和 volume_flat 对齐
        # (bins, P, H, W) → (bins, P, 1, H, W) → (bins, P*C, H, W)
        bca_expanded = blurred_cumsum_alpha.unsqueeze(2).expand_as(volume).reshape(volume.shape[0], P * C, H, W)

        blurred_alpha_rgb = fftConv(vm_flat, psf, padding)  # (bins, P, H, W)
        bar_expanded = blurred_alpha_rgb.unsqueeze(2).expand_as(volume).reshape(volume.shape[0], P * C, H, W)

        blurred_alpha_rgb_norm = bar_expanded / (bca_expanded + 1e-3)
        blurred_volume = fftConv(volume_flat, psf, padding) / (bca_expanded + 1e-3)

        over_alpha = torch.zeros_like(blurred_alpha_rgb_norm)
        over_alpha[0] = 1.0
        over_alpha[1:] = 1 - blurred_alpha_rgb_norm[0:-1]
        over_alpha = torch.cumprod(over_alpha, dim=0)
        capimg_flat = torch.sum(over_alpha * blurred_volume, 0)  # (P*C, H, W)

    # Reshape back: (P*C, H, W) → (P, C, H, W)
    return capimg_flat.reshape(P, C, H, W)


class GPUPostAugment(nn.Module):
    """
    在 GPU 上执行数据增强（CoC Blur / 高斯模糊 / 光度增强 / 噪声注入）。
    不参与梯度计算。
    """

    def __init__(self, cfg):
        super().__init__()
        # CoC Blur 参数
        self.add_cocblur = cfg.get('add_cocblur', False)
        self.cocblur_ratio = cfg.get('cocblur_ratio', 0.3)  # 触发概率
        self.cocblur_binsnum = cfg.get('cocblur_binsnum', 11)
        self.cocblur_method = cfg.get('cocblur_method', 'alpha')
        self.cocblur_psf_size = cfg.get('cocblur_psf_size', 33)
        self.cocblur_psf_scale = cfg.get('cocblur_psf_scale', 1)
        # cocblur_num_parallel: 并行处理的 view 数量
        #   1  = 逐 view 处理，显存最友好 (~250MB)
        #   9  = 逐 sample 内所有 view 并行 (~2.2GB)
        #   36 = 全 batch 并行 (~8.5GB)
        self.cocblur_num_parallel = cfg.get('cocblur_num_parallel', 1)

        # 高斯模糊参数
        self.randomblur = cfg.get('randomblur', False)
        self.randomblur_ratio = cfg.get('randomblur_ratio', 0.3)  # 触发概率
        self.blur_sigma_min = cfg.get('blur_sigma_min', 0.1)
        self.blur_sigma_max = cfg.get('blur_sigma_max', 3.0)

        # 光度增强参数
        self.random_photometric_aug = cfg.get('random_photometric_aug', False)
        self.random_photometric_aug_ratio = cfg.get('random_photometric_aug_ratio', 0.3)
        self.brightness = cfg.get('brightness', 0.4)
        self.contrast = cfg.get('contrast', 0.4)
        self.gamma = cfg.get('gamma', 0.0)

        # 噪声参数
        self.randomnoise = cfg.get('randomnoise', False)
        self.noise_fwc = cfg.get('noise_fwc', 1000)
        self.noise_std = cfg.get('noise_std', 0.01)

        # 随机灰度参数（仅在 RGB 输入时生效）
        self.randomgray = cfg.get('randomgray', False)
        self.randomgray_ratio = cfg.get('randomgray_ratio', 0.3)

    # ------------------------------------------------------------------ #
    #  CoC Blur
    # ------------------------------------------------------------------ #
    def _get_cocblur_psfs(self, disp_min, disp_max, num_bins, device):
        """
        根据 disparity 范围生成 Lp-Norm PSF 核（可被多个 view 共享）。
        返回: psfs (num_bins, 1, K, K), centers (num_bins,)
        """
        center = torch.linspace(disp_max, disp_min, num_bins, device=device)
        psfs = generate_psfs_batched(
            center,
            size=self.cocblur_psf_size,
            scale=self.cocblur_psf_scale,
            device=device
        )
        return psfs, center

    def _compute_num_bins(self, disp_min, disp_max):
        """
        根据 disp range 内 alpha 的实际变化幅度，动态决定 bins 数量。
        在整个范围内采样 alpha 以捕获极值（alpha 曲线在 disp≈0.5 附近有极小值）。
        """
        # 在 disp range 内采样多个点，找到 alpha 的真实 min/max
        sample_disps = [disp_min + (disp_max - disp_min) * t / 20 for t in range(21)]
        alphas = [compute_alpha(d, self.cocblur_psf_scale) for d in sample_disps]
        a_min, a_max = min(alphas), max(alphas)
        alpha_ratio = a_max / (a_min + 1e-6)
        # alpha_ratio ~1 → 3 bins; ~4+ → 用满 max_bins
        num_bins = int(alpha_ratio * 2)
        return max(3, min(num_bins, self.cocblur_binsnum))

    def _apply_cocblur(self, viewimgs, views_disp):
        """
        对所有视图施加 CoC Blur，根据 cocblur_num_parallel 控制并行度。

        viewimgs:   (B, N, C, H, W)
        views_disp: (B, N, H, W)
        返回:       (B, N, C, H, W)
        """
        B, N, C, H, W = viewimgs.shape
        device = viewimgs.device
        P = self.cocblur_num_parallel

        # 展平为 (BN, C, H, W) 和 (BN, H, W)
        imgs_flat = viewimgs.reshape(B * N, C, H, W)
        disp_flat = views_disp.reshape(B * N, H, W)
        result = torch.empty_like(imgs_flat)
        BN = B * N

        # 按 chunk 大小分批处理
        for start in range(0, BN, P):
            end = min(start + P, BN)
            chunk_imgs = imgs_flat[start:end]    # (p, C, H, W)
            chunk_disp = disp_flat[start:end]    # (p, H, W)

            # 使用 chunk 内的全局 disp range 生成共享 PSF
            disp_min = chunk_disp.min().item()
            disp_max = chunk_disp.max().item()
            num_bins = self._compute_num_bins(disp_min, disp_max)
            psfs, centers = self._get_cocblur_psfs(disp_min, disp_max, num_bins, device)

            # depth 参数 = -disp, 升维为 (p, 1, H, W)
            chunk_depth = -chunk_disp.unsqueeze(1)

            blurred = optical_blur_batched(
                chunk_imgs, chunk_depth,
                psfs, -center2bin(centers),
                mode=self.cocblur_method
            )
            result[start:end] = blurred

        return result.reshape(B, N, C, H, W)

    # ------------------------------------------------------------------ #
    #  高斯模糊
    # ------------------------------------------------------------------ #
    def _apply_gaussian_blur(self, viewimgs):
        """
        对所有视图施加相同 sigma 的高斯模糊。
        viewimgs: (B, N, C, H, W)
        返回: (B, N, C, H, W)
        """
        B, N, C, H, W = viewimgs.shape
        sigma = torch.empty(1, device=viewimgs.device).uniform_(
            self.blur_sigma_min, self.blur_sigma_max).item()
        kernel_size = int((sigma * 6) + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        x = torch.arange(kernel_size, device=viewimgs.device, dtype=torch.float32) - kernel_size // 2
        y = x.view(-1, 1)
        x = x.view(1, -1)
        kernel = torch.exp(-0.5 * ((x / sigma)**2 + (y / sigma)**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # Reshape to (B*N*C, 1, H, W) for grouped conv
        imgs_flat = viewimgs.reshape(B * N * C, 1, H, W)
        blurred = conv2d(imgs_flat, kernel, padding=kernel_size // 2, groups=1)
        return blurred.reshape(B, N, C, H, W)

    # ------------------------------------------------------------------ #
    #  光度增强 (纯 Tensor)
    # ------------------------------------------------------------------ #
    def _apply_photometric_aug(self, viewimgs):
        """
        对所有视图施加相同的随机光度变换（亮度/对比度/gamma）。
        在 batch 内每张图的所有 view 共享相同的随机参数。
        viewimgs: (B, N, C, H, W)
        返回: (B, N, C, H, W)
        """
        device = viewimgs.device
        B = viewimgs.shape[0]

        for b in range(B):
            img = viewimgs[b]  # (N, C, H, W)

            ops = []
            if self.brightness > 0:
                bval = torch.empty(1, device=device).uniform_(
                    max(0, 1 - self.brightness), 1 + self.brightness).item()
                ops.append(lambda x, b=bval: x * b)

            if self.contrast > 0:
                cval = torch.empty(1, device=device).uniform_(
                    max(0, 1 - self.contrast), 1 + self.contrast).item()
                mean = img.mean(dim=(-2, -1), keepdim=True)
                ops.append(lambda x, c=cval, m=mean: (x - m) * c + m)

            # 随机打乱 brightness/contrast 的顺序
            perm = torch.randperm(len(ops), device=device)
            for idx in perm:
                img = ops[idx](img)

            if self.gamma > 0:
                gval = torch.empty(1, device=device).uniform_(
                    max(1e-6, 1 - self.gamma), 1 + self.gamma).item()
                img = torch.pow(img.clamp(0, 1), gval)

            viewimgs[b] = img.clamp(0, 1)

        return viewimgs

    # ------------------------------------------------------------------ #
    #  噪声注入 (纯 Tensor)
    # ------------------------------------------------------------------ #
    def _apply_noise(self, viewimgs):
        """
        对所有视图注入 shot noise + read noise + salt-pepper noise。
        viewimgs: (B, N, C, H, W)
        返回: (B, N, C, H, W)
        """
        device = viewimgs.device

        # 随机 shot noise FWC 和 read noise std
        log_fwc = torch.empty(1, device=device).uniform_(
            torch.tensor(self.noise_fwc, dtype=torch.float32).log().item(),
            torch.tensor(1e12, dtype=torch.float32).log().item()
        ).item()
        noise_fwc = torch.tensor(log_fwc).exp().item()
        noise_std = torch.empty(1, device=device).uniform_(0.0001, self.noise_std).item()

        # Shot noise + read noise
        var_map = viewimgs / noise_fwc + noise_std**2
        sigma_map = torch.sqrt(var_map)
        noise = torch.randn_like(viewimgs) * sigma_map
        viewimgs = viewimgs + noise

        # Salt-and-pepper noise
        noise_ratio = torch.empty(1, device=device).uniform_(0, 0.001).item()
        if noise_ratio > 0:
            mask = torch.rand_like(viewimgs) < noise_ratio
            num_noise = mask.sum().item()
            if num_noise > 0:
                salt_pepper = torch.randint(0, 2, (int(num_noise),),
                                            device=device, dtype=viewimgs.dtype)
                viewimgs[mask] = salt_pepper

        return viewimgs.clamp(0, 1)

    # ------------------------------------------------------------------ #
    #  随机灰度 (仅 RGB)
    # ------------------------------------------------------------------ #
    def _apply_randomgray(self, viewimgs):
        """
        按 batch item 维度随机将 RGB 转为灰度图，并保持 3 通道输出。
        viewimgs: (B, N, 3, H, W)
        返回: (B, N, 3, H, W)
        """
        if viewimgs.shape[2] != 3:
            return viewimgs

        B = viewimgs.shape[0]
        device = viewimgs.device
        apply_mask = (torch.rand(B, device=device) < self.randomgray_ratio).view(B, 1, 1, 1, 1)
        if not apply_mask.any():
            return viewimgs

        # ITU-R BT.601 luma weights
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=viewimgs.dtype, device=device).view(1, 1, 3, 1, 1)
        gray = (viewimgs * weights).sum(dim=2, keepdim=True)
        gray3 = gray.repeat(1, 1, 3, 1, 1)

        return torch.where(apply_mask, gray3, viewimgs)

    # ------------------------------------------------------------------ #
    #  主入口
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, batch: dict) -> dict:
        """
        在 GPU 上执行全部后处理增强。仅在 training 时调用。

        batch['viewimgs']:   (B, N, C, H, W)
        batch['views_disp']: (B, N, H, W)
        """
        viewimgs = batch['viewimgs']      # (B, N, C, H, W)
        views_disp = batch.get('views_disp', None)  # (B, N, H, W)

        # 1. CoC Blur（根据 cocblur_ratio 按概率触发）
        if self.add_cocblur and views_disp is not None and torch.rand(1).item() < self.cocblur_ratio:
            viewimgs = self._apply_cocblur(viewimgs, views_disp)

        # 2. 随机高斯模糊（根据 randomblur_ratio 按概率触发）
        if self.randomblur and torch.rand(1).item() < self.randomblur_ratio:
            viewimgs = self._apply_gaussian_blur(viewimgs)

        # 3. 随机光度增强
        if self.random_photometric_aug and torch.rand(1).item() < self.random_photometric_aug_ratio:
            viewimgs = self._apply_photometric_aug(viewimgs)

        # 4. 噪声注入
        if self.randomnoise:
            viewimgs = self._apply_noise(viewimgs)

        # 5. 随机灰度（放在末尾，确保最终输出满足灰度三通道）
        if self.randomgray and viewimgs.shape[2] == 3:
            viewimgs = self._apply_randomgray(viewimgs)

        batch['viewimgs'] = viewimgs
        return batch
