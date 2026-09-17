import cv2
from utils.rendering import render_viewsimg
import numpy as np
from torch.nn.functional import conv2d
from torch.fft import fft2, ifft2
from math import floor
import torch


class PhotometricAugmentation(object):
    def __init__(self, brightness, contrast, gamma):
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
    def __call__(self, img):
        '''
        img: (C, H, W) float32, range [0, 1]
        '''
        ops = []
        if self.brightness > 0:
            b = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            ops.append(lambda x: x*b)
        if self.contrast > 0:
            c = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            mean = img.mean(axis=(-2, -1), keepdims=True)
            ops.append(lambda x: (x - mean) * c + mean)
        final_op = lambda x: x
        if self.gamma > 0:
            g = np.random.uniform(max(1e-6, 1 - self.gamma), 1 + self.gamma)
            final_op = lambda x: np.power(np.clip(x, 0, 1), g)
        np.random.shuffle(ops)
        for op in ops:
            img = op(img)
        return final_op(img).clip(0, 1)



def _compute_region_masks(disparity, rgb_image, edge_threshold=5, safety_margin=10):
    """
    Compute two region masks for augmentation region selection.

    Returns:
        flat_mask: (H, W) bool - depth-flat regions (interior of smooth surface patches),
                   derived from normal-map variance. True = flat region.
        texture_mask: (H, W) bool - RGB-rich texture regions from Canny edges.
                      True = has texture.
    Both can be AND-combined for center-point sampling (flat & textured),
    or used independently for offset-application (flat only).
    """
    h, w = disparity.shape

    # --- Flat mask: normal-map based interior detection ---
    d_min, d_max = np.min(disparity), np.max(disparity)
    d_range = (d_max - d_min) if (d_max - d_min) > 1e-5 else 1.0
    norm_depth = (disparity - d_min) / d_range

    zx = cv2.Sobel(norm_depth, cv2.CV_64F, 1, 0, ksize=3)
    zy = cv2.Sobel(norm_depth, cv2.CV_64F, 0, 1, ksize=3)
    normal = np.dstack((-zx * 40, -zy * 40, np.ones_like(norm_depth)))
    norm_len = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= (norm_len + 1e-6)

    ng_x = cv2.Sobel(normal, cv2.CV_64F, 1, 0, ksize=3)
    ng_y = cv2.Sobel(normal, cv2.CV_64F, 0, 1, ksize=3)
    normal_diff = np.sum(np.abs(ng_x) + np.abs(ng_y), axis=2)

    normal_diff_norm = cv2.normalize(normal_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(normal_diff_norm, edge_threshold, 255, cv2.THRESH_BINARY)
    if safety_margin > 0:
        kernel = np.ones((safety_margin, safety_margin), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    flat_mask = (cv2.bitwise_not(edges) > 0)   # True = flat

    # --- Texture mask: Canny edge map on grayscale RGB ---
    if rgb_image is not None:
        if rgb_image.ndim == 2:
            gray = rgb_image                          # (H, W) already grayscale
        elif rgb_image.shape[2] == 1:
            gray = rgb_image[:, :, 0]                 # (H, W, 1) → squeeze
        else:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)  # (H, W, 3/4)
        if gray.dtype == np.uint8:
            gray = gray.astype(np.float32) / 255.0
        tex = extract_texture_edges(gray)
        texture_mask = tex > 0.5
    else:
        texture_mask = np.ones((h, w), dtype=bool)

    return flat_mask, texture_mask

def _generate_tilted_plane(y_points: int, x_points: int,
                           d_lo: float, d_hi: float) -> np.ndarray:
    """
    Generate a tilted 2D disparity plane that is strictly within [d_lo, d_hi].
    The center disparity is uniformly sampled in [d_lo, d_hi].
    The plane slopes are constrained in an L1-ball to prevent out-of-bounds.

    Args:
        y_points, x_points: Height and width of the bounding box/patch.
        d_lo, d_hi: Lower and upper bounds for the generated plane's values.

    Returns:
        plane_map: (y_points, x_points) float32 array representing the plane.
    """
    d_center = float(np.random.uniform(d_lo, d_hi))
    budget = min(d_hi - d_center, d_center - d_lo)

    slope_a = float(np.random.uniform(-budget, budget))
    remaining = budget - abs(slope_a)
    slope_b = float(np.random.uniform(-remaining, remaining))

    ys_n = (np.arange(y_points)[:, None] - y_points / 2.0) / (y_points / 2.0 + 1e-6)
    xs_n = (np.arange(x_points)[None, :] - x_points / 2.0) / (x_points / 2.0 + 1e-6)

    plane_map = (d_center + slope_a * xs_n + slope_b * ys_n).astype(np.float32)
    return plane_map


class SpecularAugmentor:
    def __init__(self,
                 min_disp=-10.0,
                 max_disp=10.0,
                 edge_threshold=5,
                 safety_margin=10,
                 num_shapes_range=(5, 20),
                 shape_size_range=(0.05, 0.25),
                 plane_prob=0.5):
        """
        Specular/Highlight Augmentor
        Simulates virtual images from mirror/specular surfaces.
        Virtual images are always behind the mirror surface (smaller disparity = farther away).

        Region selection: only applies to regions that are both depth-flat and have RGB texture.

        Two modes selected randomly with probability plane_prob:

        Mode A - offset: adds a uniform negative constant to the original disparity.
            The virtual image geometry mirrors the mirror surface geometry (depth-shifted copy).
            offset ~ Uniform(global_mindisp - min(patch_disp), 0)

        Mode B - plane: replaces the patch disparity with an independent tilted/flat plane,
            entirely decoupled from the mirror surface geometry.
            d_fake(x,y) = d_center + slope_a * x_norm + slope_b * y_norm
            where d_center ~ Uniform(min_disp, min(patch_disp)), slopes constrained
            so the plane stays strictly behind the mirror surface.

        NOTE: Only modifies dispforrender (used for view rendering), NOT the GT disparity.

        Args:
        - min_disp: global minimum disparity (camera physical lower bound)
        - max_disp: global maximum disparity (camera physical upper bound)
        - edge_threshold: normal gradient threshold, smaller = more sensitive
        - safety_margin: edge dilation radius (pixels), keeps patches away from depth boundaries
        - num_shapes_range: number of specular patches per image [min, max]
        - shape_size_range: patch size as fraction of short side [min, max]
        - plane_prob: probability of using plane mode (vs offset mode)
        """
        self.min_disp = float(min_disp)
        self.max_disp = float(max_disp)
        self.edge_threshold = edge_threshold
        self.safety_margin = safety_margin
        self.num_shapes_range = num_shapes_range
        self.shape_size_range = shape_size_range
        self.plane_prob = float(plane_prob)


    def __call__(self, disparity, rgb_image):
        """
        在 dispforrender（注意：不是 GT disparity）上叠加镜面虚像偏移。

        Args:
            disparity: (H, W) float32，用于渲染的视差图（将被原地修改副本）
            rgb_image: (H, W, 3) uint8，原始 RGB 图（用于纹理区域检测）
        Returns:
            augmented_disp: (H, W) float32，叠加了镜面偏移的视差图
        """
        h, w = disparity.shape
        aug_disp = disparity.copy()

        # 计算区域 mask（共用逻辑）
        flat_mask, texture_mask = _compute_region_masks(
            disparity, rgb_image, self.edge_threshold, self.safety_margin)

        # 中心选取候选区域 = 深度平坦 AND RGB有纹理
        avail_mask = flat_mask & texture_mask

        short_side = min(h, w)
        n_shapes = np.random.randint(self.num_shapes_range[0], self.num_shapes_range[1] + 1)
        offset_map = np.zeros((h, w), dtype=np.float32)

        for _ in range(n_shapes):
            ys, xs = np.where(avail_mask)
            if len(xs) == 0:
                break

            idx = np.random.randint(len(xs))
            cy, cx = int(ys[idx]), int(xs[idx])

            size_ratio = np.random.uniform(self.shape_size_range[0], self.shape_size_range[1])
            base_size = int(short_side * size_ratio)

            # 构建二值形状掩码
            shape_mask = np.zeros((h, w), dtype=np.uint8)
            if np.random.rand() < 0.5:  # 50% rect, 50% circle
                rw = int(base_size * np.random.uniform(0.7, 1.4))
                rh = int(base_size * np.random.uniform(0.7, 1.4))
                x1, y1 = cx, cy
                x2, y2 = min(w - 1, cx + rw), min(h - 1, cy + rh)
                cv2.rectangle(shape_mask, (x1, y1), (x2, y2), 1, -1)
            else:
                r = max(2, base_size // 2)
                cv2.circle(shape_mask, (cx, cy), r, 1, -1)

            shape_bool = shape_mask.astype(bool) & flat_mask  # 只在 flat 区域内应用

            if not shape_bool.any():
                continue

            patch_min_disp = float(disparity[shape_bool].min())

            use_plane = (np.random.rand() < self.plane_prob)

            if use_plane:
                # --- Mode B: plane ---
                # 在镜面背后放置一个独立的倾斜/平行平面，与原几何完全解耦。
                # d_fake(x,y) = d_center + slope_a * x_norm + slope_b * y_norm
                # 约束：d_fake 在整个 patch 上均满足 d_fake <= patch_min_disp - margin
                #       且 d_fake >= global_min_disp
                margin = 0.5   # 强制虚像与镜面拉开至少 0.5 视差
                d_hi = patch_min_disp - margin  # 平面中心的严格上界
                d_lo = self.min_disp
                if d_lo >= d_hi:
                    continue   # 深度空间不足，放弃

                # 在 shape 的 bounding box 内生成归一化坐标
                ys_idx, xs_idx = np.where(shape_bool)
                y_min_s, y_max_s = int(ys_idx.min()), int(ys_idx.max())
                x_min_s, x_max_s = int(xs_idx.min()), int(xs_idx.max())
                sh = y_max_s - y_min_s + 1
                sw = x_max_s - x_min_s + 1

                plane_patch = _generate_tilted_plane(sh, sw, d_lo, d_hi)

                # 放入全图对应的 bounding box 内，广播机制需要精准索引
                aug_disp[y_min_s:y_max_s+1, x_min_s:x_max_s+1][shape_bool[y_min_s:y_max_s+1, x_min_s:x_max_s+1]] = plane_patch[shape_bool[y_min_s:y_max_s+1, x_min_s:x_max_s+1]]

            else:
                # --- Mode A: offset ---
                # 在原有几何结构上加一个常数偏移，虚像几何完全复制镜面几何（深度平移）
                offset_lo = self.min_disp - patch_min_disp   # 确保 min(d_fake) >= global_mindisp
                offset_hi = 0.0                               # offset < 0 → 视差更小（更远）
                if offset_lo >= offset_hi:
                    continue  # patch 视差已在 global_mindisp 下方，无法再推远

                offset = float(np.random.uniform(offset_lo, offset_hi))
                offset_map[shape_bool] = offset

            # 从候选池移除该区域，避免下一个 patch 重叠选中同一中心
            avail_mask[shape_bool] = False
            flat_mask[shape_bool] = False

        aug_disp = aug_disp + offset_map  # plane 模式的 shape_bool 位置 offset_map 为 0，不受影响

        return aug_disp




def extract_texture_edges(rgb_image, low_thresh=50, high_thresh=150):
    """
    从RGB图中提取纹理边缘。

    Args:
        rgb_image: (H, W, 3) uint8 或 float32 图像
        low_thresh: Canny 的低阈值
        high_thresh: Canny 的高阈值
        blur_k: 高斯模糊核大小，用于过滤细微噪声
    """
    # 1. 转为灰度
    if rgb_image.ndim == 2:
        gray = rgb_image.copy()
    elif rgb_image.shape[2] == 1:
        gray = rgb_image[:, :, 0]
    else:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)


    # # 3. 增强对比度 (CLAHE) - 这一步能让阴影处的纹理也被提取出来
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, low_thresh, high_thresh)

    # 5. 可选：形态学膨胀，让边缘稍微粗一点，方便做 Mask
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=3)

    return edges_dilated.astype(np.float32) / 255.0


class TransparencyAugmentor:
    """
    透明体数据增强器。

    物理模型：模拟两种透明体场景：

    Mode A: front_glass (前景玻璃/反光/污溍)
        - 在原始场景（作为背景）前方贴一层带视差偏移的纹理层（玻璃表面）
        - fake_disp = max(patch_disparity) * (1 + offset)，确保纹理层永远在原场景最前方
        - 该模式下，patch 区域的 GT disparity 被更新为 d_fake（玻璃才是最近的表面）

    Mode B: back_texture (后景透视/复杂玻璃体)
        - 原始场景本身扮演复杂形状的透明物体，在其后方贴一层平面假背景
        - fake_disp = min(patch_disparity) * (1 - offset)，确保假背景永远在最后方
        - 该模式下GT disparity 不修改（原始场景的深度仍是最近的表面）

    两种模式均使用 patch 内的极值计算 d_fake（全 patch 统一常量），
    保证假纹理层是物理空间中的纯平面挟板，不受深度边界不连续的影响。
    """

    ALL_TEXTURE_TYPES = ['perlin', 'noise', 'stripe', 'gradient', 'caustic']

    def __init__(self,
                 alpha_range=(0.05, 0.25),
                 min_disp=-10.0,
                 max_disp=10.0,
                 num_patches=(1, 3),
                 patch_size_range=(0.15, 0.4),
                 front_glass_ratio=0.5,
                 edge_threshold=5,
                 safety_margin=10,
                 texture_prob='mixed'):
        """
        Args:
        - alpha_range: alpha blend strength range [min, max]
        - min_disp: global min disparity (camera physical lower bound)
        - max_disp: global max disparity (camera physical upper bound)
          front_glass: d_fake ~ Uniform(max(patch), max_disp)  -- tilted plane
          back_texture: d_fake ~ Uniform(min_disp, min(patch)) -- constant plane
        - num_patches: number of transparent patches per image [min, max]
        - patch_size_range: patch size as fraction of short side [min, max]
        - front_glass_ratio: probability of using front_glass mode
        - edge_threshold: normal gradient threshold for flat region detection
        - safety_margin: edge dilation radius (pixels) for flat mask
        - texture_prob: texture type probabilities, 'mixed' or dict like
          {'perlin': 0.3, 'caustic': 0.3, ...}
        """
        self.alpha_range = tuple(alpha_range)
        self.min_disp = float(min_disp)
        self.max_disp = float(max_disp)
        self.num_patches = tuple(num_patches)
        self.patch_size_range = tuple(patch_size_range)
        self.front_glass_ratio = front_glass_ratio
        self.edge_threshold = edge_threshold
        self.safety_margin = safety_margin
        self.texture_prob = texture_prob

    def _get_texture_type(self):
        """根据 texture_prob 配置采样纹理类型"""
        if isinstance(self.texture_prob, str) and self.texture_prob == 'mixed':
            return np.random.choice(self.ALL_TEXTURE_TYPES)
        elif hasattr(self.texture_prob, 'keys') and hasattr(self.texture_prob, 'values'):
            keys = list(self.texture_prob.keys())
            probs = [float(p) for p in self.texture_prob.values()]
            return np.random.choice(keys, p=probs)
        return 'perlin'

    def _generate_texture(self, h, w, channels, texture_type=None):
        """
        生成随机纹理图像 (H, W, C) uint8。

        Args:
            h, w: 图像尺寸
            channels: 通道数
            texture_type: 指定类型，None 则按 texture_prob 采样
        Returns:
            texture: (H, W, C) uint8
        """
        if texture_type is None:
            texture_type = self._get_texture_type()

        if texture_type == 'perlin':
            # 多尺度 Perlin-like noise：低分辨率随机矩阵 bicubic 上采样
            scale = np.random.randint(8, 32)
            base = np.random.randn(max(2, h // scale), max(2, w // scale)).astype(np.float32)
            texture = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
            texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6) * 255
            texture = np.stack([texture] * channels, axis=-1).astype(np.uint8)

        elif texture_type == 'noise':
            # 高斯噪声 + 轻微模糊，模拟毛玻璃
            texture = np.random.randint(0, 256, (h, w, channels), dtype=np.uint8)
            ksize = np.random.choice([3, 5, 7])
            texture = cv2.GaussianBlur(texture, (ksize, ksize), 0)
            # cv2.GaussianBlur 对 (H,W,1) 输入会 squeeze 掉通道维度，这里保证三维
            if texture.ndim == 2:
                texture = texture[:, :, np.newaxis]

        elif texture_type == 'stripe':
            # 空间调频条纹（Chirp Stripe）：沿随机方向的条纹频率随空间线性变化，
            # 破坏全局周期性，同时保留划痕/百叶窗的视觉质感
            angle = np.random.uniform(0, np.pi)
            phase = np.random.uniform(0, 2 * np.pi)
            freq_start = np.random.uniform(0.005, 0.04)   # 起始频率
            freq_end   = np.random.uniform(0.04,  0.12)   # 终止频率（更高）
            # 沿主方向的投影坐标 t ∈ [0, 1]
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            proj = xs * np.cos(angle) + ys * np.sin(angle)  # (H, W)
            proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)  # [0, 1]
            # 瞬时频率随 proj_norm 线性插值
            inst_freq  = freq_start + (freq_end - freq_start) * proj_norm
            # 积分累相 → chirp stripe
            phase_map  = 2 * np.pi * inst_freq * proj + phase
            stripe = np.sin(phase_map)
            # 可选：加入横向的轻微 perlin 扰动，让条纹边缘稍不规则
            noise_scale = np.random.randint(4, 12)
            perturb = np.random.randn(max(2, h // noise_scale),
                                      max(2, w // noise_scale)).astype(np.float32)
            perturb = cv2.resize(perturb, (w, h), interpolation=cv2.INTER_CUBIC)
            perturb = perturb / (np.abs(perturb).max() + 1e-6) * 0.25  # 小幅扰动
            stripe = np.clip(stripe + perturb, -1, 1)
            stripe = ((stripe + 1) / 2 * 255).astype(np.uint8)
            texture = np.stack([stripe] * channels, axis=-1)

        elif texture_type == 'gradient':
            # 随机方向线性渐变，模拟大面积变厚度玻璃
            angle = np.random.uniform(0, np.pi)
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            grad_val = xs * np.cos(angle) + ys * np.sin(angle)
            grad_val = (grad_val - grad_val.min()) / (grad_val.max() - grad_val.min() + 1e-6) * 255
            texture = np.stack([grad_val.astype(np.uint8)] * channels, axis=-1)

        elif texture_type == 'caustic':
            # 多组随机高斯核叠加 + 非线性映射，模拟水面/弧面玻璃焦散
            texture = np.zeros((h, w), dtype=np.float32)
            num_blobs = np.random.randint(10, 40)
            for _ in range(num_blobs):
                cy = np.random.randint(0, h)
                cx = np.random.randint(0, w)
                sigma = np.random.uniform(10, 60)
                amp = np.random.uniform(0.5, 1.0)
                ys, xs = np.ogrid[0:h, 0:w]
                blob = amp * np.exp(-((ys - cy)**2 + (xs - cx)**2) / (2 * sigma**2))
                texture += blob
            # 非线性映射增加焦散感
            texture = np.sin(texture * np.pi * 2)
            texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6) * 255
            texture = np.stack([texture.astype(np.uint8)] * channels, axis=-1)
        else:
            raise ValueError(f"Unknown texture type: {texture_type}")

        return texture


    def __call__(self, viewsimg, disparity, depth, viewspos,
                 pad='zero', interp_method='linear'):
        """
        在已渲染的光场视图上叠加透明层。

        front_glass: 全图随机采样，生成倾斜平面视差（模拟倾斜角度的玻璃前表面）。
        back_texture: 仅在"深度平坦 + RGB有纹理"区域应用（同 SpecularAugmentor 选区逻辑）。

        Returns:
            augmented_viewsimg: (N, H, W, C) uint8
            augmented_disparity: (H, W) float32  --- front_glass 下部分区域被更新
        """
        N, H, W, C = viewsimg.shape
        result = viewsimg.copy()
        out_disparity = disparity.copy()
        short_side = min(H, W)

        # --- 预计算区域 mask（共用 _compute_region_masks）---
        # flat_mask: 深度平坦区域; texture_mask: RGB纹理丰富区域
        flat_mask, texture_mask = _compute_region_masks(
            disparity, viewsimg[N // 2], self.edge_threshold, self.safety_margin)
        avail_mask = flat_mask & texture_mask   # 中心点候选区域（两个模式均使用）

        n_patches = np.random.randint(self.num_patches[0], self.num_patches[1] + 1)

        for _ in range(n_patches):
            use_front_glass = np.random.rand() < self.front_glass_ratio

            # --- 中心点选取：两种模式都从 flat+textured 区域采样 ---
            avail_ys, avail_xs = np.where(avail_mask)
            if len(avail_xs) == 0:
                break
            idx = np.random.randint(len(avail_xs))
            cy, cx = int(avail_ys[idx]), int(avail_xs[idx])

            # --- patch 边界 ---
            size_ratio = np.random.uniform(self.patch_size_range[0], self.patch_size_range[1])
            ph = int(short_side * size_ratio)
            pw = int(short_side * size_ratio * np.random.uniform(0.7, 1.4))
            y1, y2 = max(0, cy - ph // 2), min(H, cy + ph // 2)
            x1, x2 = max(0, cx - pw // 2), min(W, cx + pw // 2)
            if (y2 - y1) < 10 or (x2 - x1) < 10:
                continue

            patch_disp = disparity[y1:y2, x1:x2]

            # --- d_fake 计算 ---
            if use_front_glass:
                # Mode A: 前景倾斜玻璃
                # 保证 d_fake 在 [d_lo, max_disp] 内完全不被截断的条件：
                d_lo = float(np.max(patch_disp))
                d_hi = self.max_disp
                if d_lo >= d_hi:
                    continue
                ph_a, pw_a = y2 - y1, x2 - x1
                d_fake_patch = _generate_tilted_plane(ph_a, pw_a, d_lo, d_hi)

            else:
                # Mode B: 后景透视（常量 offset，与 SpecularAugmentor 一致）
                d_hi_bt = float(np.min(patch_disp))
                d_lo_bt = self.min_disp
                if d_lo_bt >= d_hi_bt:
                    continue
                d_fake_val = float(np.random.uniform(d_lo_bt, d_hi_bt))
                d_fake_patch = np.full((y2 - y1, x2 - x1), d_fake_val, dtype=np.float32)

            # --- 生成纹理（只生成 patch + bleed 区域，节省计算）---
            # bleed: 其他视角 warp 会把 patch 边缘向外偏移 d_fake * viewspos 个像素，
            # 纹理需要额外覆盖这个 bleed 区域才不会被截断。
            max_disp_fake = float(np.max(np.abs(d_fake_patch)))
            expand_x = int(np.ceil(max_disp_fake * float(np.max(np.abs(viewspos[:, 0])))))
            expand_y = int(np.ceil(max_disp_fake * float(np.max(np.abs(viewspos[:, 1])))))
            ty1, ty2 = max(0, y1 - expand_y), min(H, y2 + expand_y)
            tx1, tx2 = max(0, x1 - expand_x), min(W, x2 + expand_x)

            patch_tex = self._generate_texture(ty2 - ty1, tx2 - tx1, C)
            texture_img = np.zeros((H, W, C), dtype=np.uint8)
            patch_tex = patch_tex.astype(np.uint8)
            if patch_tex.ndim == 2:  # 小概率安全保障：确保总是 3D
                patch_tex = patch_tex[:, :, np.newaxis]
            texture_img[ty1:ty2, tx1:tx2] = patch_tex

            mask_full = np.zeros((H, W), dtype=np.float32)
            mask_full[y1:y2, x1:x2] = 1.0   # blend 范围仍限定在原始 patch

            # --- 构造虚假视差图和深度 ---
            fake_disp_map = np.zeros_like(disparity)
            fake_disp_map[y1:y2, x1:x2] = d_fake_patch
            fake_depth_map = np.full_like(depth, depth.max())
            fake_depth_map[y1:y2, x1:x2] = depth.max() + 1.0

            # --- 渲染纹理层 ---
            texture_views, _, _ = render_viewsimg(
                texture_img, fake_depth_map, fake_disp_map, viewspos,
                pad='zero', interp_method=interp_method)

            # --- Alpha Blend ---
            # front_glass: 全区域混合（玻璃可出现任意位置）
            # back_texture: 仅在深度平坦区域混合（非平坦处不叠加，物理上更合理）
            alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
            for i in range(N):
                rendered_nonzero = texture_views[i].sum(axis=-1) > 0
                if use_front_glass:
                    blend_mask = rendered_nonzero & mask_full.astype(bool)
                else:
                    blend_mask = rendered_nonzero & mask_full.astype(bool) & flat_mask
                if not np.any(blend_mask):
                    continue
                # 对齐通道数：render_viewsimg 总是输出 3 通道，
                # 但 result 可能是 1 通道（灰度 lambertian）
                tv = texture_views[i]
                if tv.shape[-1] != C:
                    if C == 1:
                        # 三通道 → 灰度（weighted average）
                        tv = (0.299 * tv[..., 0] + 0.587 * tv[..., 1] + 0.114 * tv[..., 2])[..., np.newaxis]
                    else:
                        # 单通道 → 复制到 C 通道
                        tv = np.repeat(tv[..., :1], C, axis=-1)
                tv = tv.astype(np.uint8)
                result[i][blend_mask] = (
                    (1 - alpha) * result[i][blend_mask].astype(np.float32) +
                    alpha * tv[blend_mask].astype(np.float32)
                ).clip(0, 255).astype(np.uint8)

            # --- front_glass: 更新 GT disparity（倾斜玻璃表面的视差） ---
            if use_front_glass:
                out_disparity[y1:y2, x1:x2] = d_fake_patch

        return result, out_disparity


def center2bin(center):
    interval = torch.diff(center)
    interval = torch.cat([interval[:1], interval], 0)
    bin = center - interval/2
    bin = torch.cat([bin, center[-1:] + interval[-1:] / 2], 0)
    return bin


def continuous2bin(x: torch.Tensor, bin_edges: torch.Tensor):
    # Replicate bin edges to match the dimensions of x
    lb = x - bin_edges[:-1].reshape(torch.Size([-1]) + torch.Size(torch.ones(len(x.shape), dtype=torch.int)))
    ub = x - bin_edges[1:].reshape(torch.Size([-1]) + torch.Size(torch.ones(len(x.shape), dtype=torch.int)))

    # Create the volume mask: True if x is in the interval [lb, ub)
    volume_mask = ((lb >= 0) & (ub < 0))

    # Find the bin index for each element in x
    map = volume_mask.float().argmax(0).int()  # +1 because NumPy is 0-indexed

    # Set elements not in any bin to NaN
    map[volume_mask.sum(0) == 0] = -1

    return map, volume_mask


def fftConv(x: torch.Tensor, kernel: torch.Tensor, padding='same'):
    # try:
    input_fft = fft2(x, s=(kernel.shape[-2] + x.shape[-2] - 1, kernel.shape[-1] + x.shape[-1] - 1), dim=(-2, -1))
    # except Exception as e:
    kernel_fft = fft2(kernel,s=(kernel.shape[-2] + x.shape[-2] - 1, kernel.shape[-1] + x.shape[-1] - 1), dim=(-2, -1))
    conv_fft_real = input_fft.real * kernel_fft.real - input_fft.imag * kernel_fft.imag
    conv_fft_imag = input_fft.real * kernel_fft.imag + input_fft.imag * kernel_fft.real
    conv_fft = torch.complex(conv_fft_real, conv_fft_imag)
    # 直接做乘法C=A*B，但C[i,j] 和 A[i,j]*B[i,j]有细微差别
    if padding =='same':
        conv_result = ifft2(conv_fft, dim=(-2, -1)).abs()[..., floor(kernel.shape[-2]/2):floor(kernel.shape[-2]/2) + x.shape[-2],
                                                                floor(kernel.shape[-1]/2):floor(kernel.shape[-1]/2) + x.shape[-1]]
    elif padding == 'valid':
        conv_result = ifft2(conv_fft, dim=(-2, -1)).abs()[..., kernel.shape[-2]-1:kernel.shape[-2]-1 + x.shape[-2] - kernel.shape[-2] + 1,
                                                                kernel.shape[-1]-1:kernel.shape[-1]-1 + x.shape[-1] - kernel.shape[-1] + 1]
    elif padding == 'full':
        conv_result = ifft2(conv_fft, dim=(-2, -1)).abs()
    else:
        raise ValueError('padding must be one of "same", "valid", or "full"')
    return conv_result


def optical_blur(img, depth, psf, z_u, padding='same', mode='alpha'):
    '''
    img: BxHxW
    depth: BxHxW
    psf: NxBxKxK
    z: Nx1
    '''
    _, volume_mask = continuous2bin(depth, z_u)
    volume = (volume_mask * img).float()
    if mode == 'naive':
        blurred_volume = fftConv(volume, psf, padding)
        capimg = torch.sum(blurred_volume, 0)
    elif mode == 'alpha':
        cumsum_alpha = torch.cumsum(volume_mask.flip(0), dim=0).flip(0).float()
        blurred_cumsum_alpha = fftConv(cumsum_alpha, psf, padding)
        blurred_alpha_rgb = fftConv(volume_mask, psf, padding) / (blurred_cumsum_alpha + 1e-3)
        blurred_volume = fftConv(volume, psf, padding) / (blurred_cumsum_alpha + 1e-3)
        over_alpha = torch.zeros_like(blurred_alpha_rgb)
        over_alpha[0] = 1.0
        over_alpha[1:] = 1 - blurred_alpha_rgb[0:-1]
        over_alpha = torch.cumprod(over_alpha, dim=0)
        capimg = torch.sum(over_alpha*blurred_volume, 0)

    return capimg


def gaussian_blur_same_sigma(imgs, sigma_min=0.1, sigma_max=3.0):
    """
    对 [N, 1, H, W] 图像批次进行高斯模糊，所有图像共享同一随机 sigma
    """
    N, C, H, W = imgs.shape
    assert C == 1, "当前版本仅支持单通道，可轻松扩展到RGB"

    # 随机一个 sigma（所有图像共用）
    sigma = torch.empty(1, device=imgs.device).uniform_(sigma_min, sigma_max).item()
    kernel_size = int((sigma * 6) + 1)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    # 生成二维高斯核
    x = torch.arange(kernel_size, device=imgs.device) - kernel_size // 2
    y = x.view(-1, 1)
    x = x.view(1, -1)
    kernel = torch.exp(-0.5 * ((x/sigma)**2 + (y/sigma)**2))
    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    blurred = conv2d(imgs, kernel, padding=kernel_size//2, groups=1)

    return blurred, sigma
