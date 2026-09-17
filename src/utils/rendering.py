import numpy as np
import torch
import cv2
from torch.nn.functional import grid_sample
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from utils.depth_ops import depth2normal, rescale
from numba import njit


def fill_holes_nearest(img, mask, max_iter=2):
    """
    img: (H, W, C) uint8
    max_iter: 扩散轮数，默认 2 可补大部分空洞
    """
    mask = mask.astype(np.uint8)
    H, W, C = img.shape
    for _ in range(max_iter):
        # dilate mask
        dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8))

        # find new pixels to fill (where dilated==1 but mask==0)
        new_pixels = (dilated == 1) & (mask == 0)

        for c in range(C):
            # get the current channel
            channel = img[:, :, c]
            dilated_channel = cv2.dilate(channel, np.ones((3, 3), img.dtype))
            channel[new_pixels] = dilated_channel[new_pixels]
            img[:, :, c] = channel
        mask = np.any(img != 0, axis=2).astype(np.uint8)

    return img, mask

def fill_holes_linear(img, mask):
    '''
    img: (H, W, C) float32
    mask: (H, W) bool
    max_iter: 扩散轮数，默认 1 可补大部分空洞
    '''
    # mask = mask.astype(np.uint8)
    H, W, C = img.shape
    num_mask_pixels = (~mask).sum()
    mask_pixels = np.stack(np.where(~mask), axis=1)
    values = np.zeros((num_mask_pixels, C, 4))
    weight = np.zeros((num_mask_pixels, 4))

    sub = mask_pixels[:, 0] - 1
    submask = sub >= 0
    values[submask, :, 0] = img[sub[submask], mask_pixels[submask, 1]]
    weight[submask, 0] = mask[sub[submask], mask_pixels[submask, 1]]

    sub = mask_pixels[:, 0] + 1
    submask = sub < H
    values[submask, :, 1] = img[sub[submask], mask_pixels[submask, 1]]
    weight[submask, 1] = mask[sub[submask], mask_pixels[submask, 1]]

    sub = mask_pixels[:, 1] - 1
    submask = sub >= 0
    values[submask, :, 2] = img[mask_pixels[submask, 0], sub[submask]]
    weight[submask, 2] = mask[mask_pixels[submask, 0], sub[submask]]

    sub = mask_pixels[:, 1] + 1
    submask = sub < W
    values[submask, :, 3] = img[mask_pixels[submask, 0], sub[submask]]
    weight[submask, 3] = mask[mask_pixels[submask, 0], sub[submask]]

    values_mask = np.sum(weight, axis=-1) > 0
    mask[mask_pixels[:, 0], mask_pixels[:, 1]] = values_mask
    img[mask_pixels[values_mask, 0], mask_pixels[values_mask, 1]] = np.sum(values[values_mask] * weight[values_mask][:, None], axis=-1) / np.sum(weight[values_mask], axis=-1, keepdims=True)

    return img, mask

def highspeed_zbuffer_render(u, v, z, colors, pad='noise', interp_method='linear'):
    u_round = np.floor(u + 0.5).astype(np.int32)
    v_round = np.floor(v + 0.5).astype(np.int32)
    H, W, C = colors.shape
    valid = (
        (u_round >= 0) & (u_round < H) &
        (v_round >= 0) & (v_round < W) &
        (z > 0)
    )

    u_round, v_round, z, colors = u_round[valid], v_round[valid], z[valid], colors[valid]

    idx_sort = np.lexsort((z, v_round, u_round))  # sort by u, v, z
    u_sorted, v_sorted, z_sorted = u_round[idx_sort], v_round[idx_sort], z[idx_sort]
    colors_sorted = colors[idx_sort]

    uv = np.stack((u_sorted, v_sorted), axis=1)
    _, unique_indices = np.unique(uv, axis=0, return_index=True)

    u_final = u_sorted[unique_indices]
    v_final = v_sorted[unique_indices]
    colors_final = colors_sorted[unique_indices]

    if pad == 'noise':
        pad_value = np.random.randn(H, W, C) * colors.std() + colors.mean()
        pad_value = pad_value.clip(0, np.max(colors))
    else:
        pad_value = np.zeros((H, W, C), dtype=colors.dtype)
    mask = np.zeros((H, W), dtype=np.bool_)
    mask[u_final, v_final] = 1
    grid_x, grid_y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    img = griddata(
            points=np.stack([u[valid][idx_sort][unique_indices], v[valid][idx_sort][unique_indices]], axis=1),
            values=colors_final,
            xi=(grid_x, grid_y),
            method=interp_method,
            fill_value=0)
    img[~mask] = pad_value[~mask]

    return img, mask

def pad_edge_extend(out_img, mask):
    """
    使用最近有效像素进行边缘延展填充
    out_img: (H, W, C)
    mask: (H, W) bool
    """
    H, W, C = out_img.shape
    holes = ~mask
    if not np.any(holes):
        return out_img

    # distance_transform_edt 返回：
    # - 距离
    # - 最近的 mask=True 像素的索引
    _, indices = distance_transform_edt(
        holes,
        return_indices=True
    )

    # indices shape: (2, H, W)
    nearest_r = indices[0]
    nearest_c = indices[1]

    # 对洞像素拷贝最近有效像素的颜色
    out_img[holes] = out_img[
        nearest_r[holes],
        nearest_c[holes]
    ]

    return out_img

@njit(fastmath=True, cache=True)
def get_mirror_idx(base_idx, step, valid_len):
    """
    计算镜像采样的索引 (Ping-Pong Pattern)。
    例如 valid_len=3 (Indices: 0,1,2), step 递增:
    step=0 -> 2 (边界)
    step=1 -> 1
    step=2 -> 0
    step=3 -> 0 (反弹)
    step=4 -> 1
    ...
    """
    if valid_len <= 0:
        return base_idx

    # 归一化步长到周期 2*valid_len
    cycle = step % (2 * valid_len)

    if cycle < valid_len:
        # 正向阶段 (从边界向内): valid_len-1, valid_len-2...
        offset = cycle
        return base_idx - 1 - offset
    else:
        # 反向阶段 (弹回来): 0, 1, 2...
        offset = cycle - valid_len
        return base_idx - valid_len + offset

@njit(fastmath=True, cache=True)
def fill_holes_advanced(img, depth, mask, z_threshold=0.05):
    """
    高级自适应填充：
    1. 优先使用深度较大(背景)的一侧。
    2. 使用镜像(Mirror)平铺代替重复，消除接缝。
    3. 若两侧深度相近(z_diff < threshold)，则进行距离加权融合(Lerp)。
    """
    H, W, C = img.shape

    # 用于标记是否已被Horizontal Pass填充，避免Vertical Pass覆盖好的纹理
    # 0: 未填充, 1: 已填充
    # 我们直接利用 img 是否为 0 来判断（假设 pad_val=0 且有效像素不纯黑）
    # 或者为了严谨，用一个局部布尔数组
    filled_map = np.zeros((H, W), dtype=np.bool_)

    # ===============================
    # Pass 1: 水平扫描 (Horizontal)
    # ===============================
    for r in range(H):
        c = 0
        while c < W:
            if mask[r, c]:
                c += 1
                continue

            start = c
            while c < W and not mask[r, c]:
                c += 1
            end = c
            length = end - start

            # 获取深度
            z_left = depth[r, start-1] if start > 0 else -1.0
            z_right = depth[r, end] if end < W else -1.0

            # 决策逻辑
            mode = 0 # 0:Skip, 1:Left, 2:Right, 3:Blend

            # 有效性检查
            valid_L = (z_left > 0)
            valid_R = (z_right > 0)

            if not valid_L and not valid_R:
                mode = 0
            elif valid_L and not valid_R:
                mode = 1
            elif not valid_L and valid_R:
                mode = 2
            else:
                # 两侧都有效，比较深度差异
                # 这里假设 depth 是差异越大越远 (Z-buffer)
                diff = abs(z_left - z_right)
                max_z = max(z_left, z_right)

                # 相对误差判断 (深度相差 5% 以内视为同一平面)
                if diff / (max_z + 1e-6) < z_threshold:
                    mode = 3 # Blend
                elif z_left > z_right:
                    mode = 1 # Left is background
                else:
                    mode = 2 # Right is background

            # 执行填充
            if mode != 0:
                len_L = 0
                if mode == 1 or mode == 3:
                    temp = start - 1
                    while temp >= 0 and mask[r, temp]:
                        temp -= 1
                    len_L = (start - 1) - temp

                # 计算右侧可用纹理长度
                len_R = 0
                if mode == 2 or mode == 3:
                    temp = end
                    while temp < W and mask[r, temp]:
                        temp += 1
                    len_R = temp - end

                # 填充循环
                for i in range(length):
                    curr_c = start + i

                    if mode == 1: # Left Only (Mirror)
                        src_idx = get_mirror_idx(start, i, len_L)
                        for k in range(C):
                            img[r, curr_c, k] = img[r, src_idx, k]

                    elif mode == 2: # Right Only (Mirror)
                        # 右侧镜像逻辑：从 end 开始往右走 i 步的镜像
                        # 相当于把坐标系反过来
                        # base=end, step=i, valid_len=len_R
                        # 但 get_mirror_idx 是往"左"减的，我们需要往"右"加
                        # 手动写右侧逻辑：
                        cycle = i % (2 * len_R)
                        if cycle < len_R:
                            offset = cycle
                            src_idx = end + offset # 0 -> end
                        else:
                            offset = cycle - len_R
                            src_idx = end + (len_R - 1) - offset

                        for k in range(C):
                            img[r, curr_c, k] = img[r, src_idx, k]

                    elif mode == 3: # Bi-directional Blend
                        # 左侧源
                        src_idx_L = get_mirror_idx(start, i, len_L)

                        # 右侧源 (同上)
                        cycle = (length - 1 - i) % (2 * len_R) # 注意：对于右侧，i越小离得越远
                        # 其实对于右侧，距离是 (length - 1 - i)
                        dist_R = length - 1 - i
                        cycle_R = dist_R % (2 * len_R)

                        if cycle_R < len_R:
                            src_idx_R = end + cycle_R
                        else:
                            src_idx_R = end + (len_R - 1) - (cycle_R - len_R)

                        # 计算权重 (距离反比)
                        # i=0 (近左) -> w_L=1
                        # i=length-1 (近右) -> w_L=0
                        w_R = float(i + 1) / float(length + 1)
                        w_L = 1.0 - w_R

                        for k in range(C):
                            val_L = img[r, src_idx_L, k]
                            val_R = img[r, src_idx_R, k]
                            img[r, curr_c, k] = w_L * val_L + w_R * val_R

                    filled_map[r, curr_c] = True

    # ===============================
    # Pass 2: 垂直扫描 (Vertical)
    # ===============================
    # 逻辑类似，但增加了对 filled_map 的检查
    # 只有当 mask 为 False 且 Pass 1 也没填的时候才处理 (或者你可以允许覆盖)

    for c in range(W):
        r = 0
        while r < H:
            # 跳过原始有效像素 OR 已经被水平填好的像素
            if mask[r, c] or filled_map[r, c]:
                r += 1
                continue

            start = r
            while r < H and (not mask[r, c] and not filled_map[r, c]):
                r += 1
            end = r
            length = end - start

            z_top = depth[start-1, c] if start > 0 else -1.0
            z_bottom = depth[end, c] if end < H else -1.0

            valid_T = (z_top > 0)
            valid_B = (z_bottom > 0)

            mode = 0
            if not valid_T and not valid_B: mode = 0
            elif valid_T and not valid_B: mode = 1
            elif not valid_T and valid_B: mode = 2
            else:
                diff = abs(z_top - z_bottom)
                max_z = max(z_top, z_bottom)
                if diff / (max_z + 1e-6) < z_threshold:
                    mode = 3
                elif z_top > z_bottom:
                    mode = 1
                else:
                    mode = 2

            if mode != 0:
                # 寻找可用长度 (简化: 向上/向下找50个像素 max)
                len_T = 0
                if mode == 1 or mode == 3:
                    temp = start - 1
                    while temp >= 0 and mask[temp, c]: # 注意这里只用了mask，没用filled_map
                        temp -= 1
                    len_T = (start - 1) - temp
                    if len_T < 1: len_T = 1 # 避免除0

                len_B = 0
                if mode == 2 or mode == 3:
                    temp = end
                    while temp < H and mask[temp, c]:
                        temp += 1
                    len_B = temp - end
                    if len_B < 1: len_B = 1

                for i in range(length):
                    curr_r = start + i

                    if mode == 1: # Top Mirror
                        src_r = get_mirror_idx(start, i, len_T)
                        for k in range(C):
                            img[curr_r, c, k] = img[src_r, c, k]

                    elif mode == 2: # Bottom Mirror
                        cycle = i % (2 * len_B)
                        if cycle < len_B:
                            src_r = end + cycle
                        else:
                            src_r = end + (len_B - 1) - (cycle - len_B)
                        for k in range(C):
                            img[curr_r, c, k] = img[src_r, c, k]

                    elif mode == 3: # Blend
                        src_r_T = get_mirror_idx(start, i, len_T)

                        dist_B = length - 1 - i
                        cycle_B = dist_B % (2 * len_B)
                        if cycle_B < len_B:
                            src_r_B = end + cycle_B
                        else:
                            src_r_B = end + (len_B - 1) - (cycle_B - len_B)

                        w_B = float(i + 1) / float(length + 1)
                        w_T = 1.0 - w_B

                        for k in range(C):
                            img[curr_r, c, k] = w_T * img[src_r_T, c, k] + w_B * img[src_r_B, c, k]

    return img

@njit(fastmath=True, cache=True)
def zbuffer_render_linear_fast(u, v, z, colors, H, W, only_u=False, only_v=False, pad_val=0.0):
    """
    Numba 加速的 Z-buffer 线性渲染核心函数。
    包含完整的 2D 插值以及 only_u/only_v 的 1D 插值逻辑。
    """
    # 1. 数据展平与预处理
    u_flat = u.ravel()
    v_flat = v.ravel()
    z_flat = z.ravel()
    # 确保颜色是 (N, C) 形状
    colors_flat = colors.reshape(-1, colors.shape[-1])
    num_pixels = u_flat.shape[0]
    C = colors.shape[-1]

    # 2. 深度缓冲区初始化 (Pass 1)
    # 用于记录每个整数网格点(y, x)对应的最近源像素的索引
    min_depth = np.full((H, W), np.inf, dtype=np.float32)
    closest_idx = np.full((H, W), -1, dtype=np.int32)

    # === Pass 1: Z-Buffering ===
    # 找到落在每个整数像素中心且深度最小的源像素
    for i in range(num_pixels):
        ui = u_flat[i]
        vi = v_flat[i]
        zi = z_flat[i]

        # 计算最近的整数坐标 (Round)
        u_r = int(np.floor(ui + 0.5))
        v_r = int(np.floor(vi + 0.5))

        # 边界检查与深度测试
        if 0 <= u_r < H and 0 <= v_r < W and zi > 0:
            if zi < min_depth[u_r, v_r]:
                min_depth[u_r, v_r] = zi
                closest_idx[u_r, v_r] = i

    # 3. 累积缓冲区初始化 (Pass 2)
    output_img = np.zeros((H, W, C), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    # === Pass 2: Splatting (累积混合) ===
    # 遍历所有被标记为"可见"的像素位置，将源像素颜色"泼洒"到周围邻域
    for r in range(H):
        for c in range(W):
            idx = closest_idx[r, c]
            if idx == -1:
                continue

            # 取出该位置对应的原始浮点坐标和颜色
            curr_u = u_flat[idx]
            curr_v = v_flat[idx]
            color = colors_flat[idx]

            # 计算 floor/ceil
            u_f = int(np.floor(curr_u))
            u_c = int(np.ceil(curr_u))
            v_f = int(np.floor(curr_v))
            v_c = int(np.ceil(curr_v))

            # 基础权重计算
            diff_u = curr_u - u_f
            diff_v = curr_v - v_f

            # === 分支 1: 仅 U 方向插值 ===
            if only_u:
                # 逻辑：V 方向锁定为 ceil (参考原代码 v_final = v_ceil)
                # 这是一个 1D 插值，只在 U 方向扩散
                target_v = v_c

                if 0 <= target_v < W:
                    w_uf = 1.0 - diff_u
                    w_uc = diff_u # 等同于 1 - (u_c - curr_u)

                    # 邻居 1: (ceil, target_v)
                    if 0 <= u_c < H:
                        for k in range(C):
                            output_img[u_c, target_v, k] += color[k] * w_uc
                        weight_sum[u_c, target_v] += w_uc

                    # 邻居 2: (floor, target_v)
                    if 0 <= u_f < H:
                        for k in range(C):
                            output_img[u_f, target_v, k] += color[k] * w_uf
                        weight_sum[u_f, target_v] += w_uf

            # === 分支 2: 仅 V 方向插值 ===
            elif only_v:
                # 逻辑：U 方向锁定为 ceil (参考原代码 u_final = u_ceil)
                # 这是一个 1D 插值，只在 V 方向扩散
                target_u = u_c

                if 0 <= target_u < H:
                    w_vf = 1.0 - diff_v
                    w_vc = diff_v

                    # 邻居 1: (target_u, ceil)
                    if 0 <= v_c < W:
                        for k in range(C):
                            output_img[target_u, v_c, k] += color[k] * w_vc
                        weight_sum[target_u, v_c] += w_vc

                    # 邻居 2: (target_u, floor)
                    if 0 <= v_f < W:
                        for k in range(C):
                            output_img[target_u, v_f, k] += color[k] * w_vf
                        weight_sum[target_u, v_f] += w_vf

            # === 分支 3: 完整的 2D 双线性插值 (默认) ===
            else:
                w_uf = 1.0 - diff_u
                w_uc = diff_u
                w_vf = 1.0 - diff_v
                w_vc = diff_v

                # 邻域 1: (ceil, ceil)
                if 0 <= u_c < H and 0 <= v_c < W:
                    w = w_uc * w_vc
                    for k in range(C):
                        output_img[u_c, v_c, k] += color[k] * w
                    weight_sum[u_c, v_c] += w

                # 邻域 2: (ceil, floor)
                if 0 <= u_c < H and 0 <= v_f < W:
                    w = w_uc * w_vf
                    for k in range(C):
                        output_img[u_c, v_f, k] += color[k] * w
                    weight_sum[u_c, v_f] += w

                # 邻域 3: (floor, ceil)
                if 0 <= u_f < H and 0 <= v_c < W:
                    w = w_uf * w_vc
                    for k in range(C):
                        output_img[u_f, v_c, k] += color[k] * w
                    weight_sum[u_f, v_c] += w

                # 邻域 4: (floor, floor)
                if 0 <= u_f < H and 0 <= v_f < W:
                    w = w_uf * w_vf
                    for k in range(C):
                        output_img[u_f, v_f, k] += color[k] * w
                    weight_sum[u_f, v_f] += w

    # 4. 归一化和初步填充
    mask = weight_sum > 0

    for r in range(H):
        for c in range(W):
            if mask[r, c]:
                for k in range(C):
                    output_img[r, c, k] /= (weight_sum[r, c] + 1e-6)
            else:
                for k in range(C):
                    output_img[r, c, k] = pad_val

    return output_img, mask, min_depth

def zbuffer_render_linear_wrapper(u, v, z, colors, pad='noise', only_u=False, only_v=False):
    """
    Python 包装器，用于处理 Numba 不方便处理的随机噪声填充逻辑。
    """
    H, W, C = colors.shape
    u = np.ascontiguousarray(u, dtype=u.dtype)
    v = np.ascontiguousarray(v, dtype=v.dtype)
    z = np.ascontiguousarray(z, dtype=z.dtype)
    colors = np.ascontiguousarray(colors, dtype=colors.dtype)
    out_img, mask, min_depth = zbuffer_render_linear_fast(u, v, z, colors, H, W, only_u, only_v)

    if pad == 'zero':
        out_img[~mask] = 0
    elif pad == 'noise':
        num_holes = np.sum(~mask)
        if num_holes > 0:
            std = np.random.rand() * (colors.std() - 1) + 1
            mean = np.random.rand() * (colors.mean() - 1) + 1
            noise = np.random.randn(num_holes, C) * std + mean
            out_img[~mask] = noise.clip(0, np.max(colors))
    elif pad == 'constant':
        out_img[~mask] = np.random.uniform(0, 255, (1, C))
    elif pad == 'edge':
        out_img = pad_edge_extend(out_img, mask)
    elif pad == 'edge_fill':
        out_img = fill_holes_advanced(out_img, min_depth, mask)
    else:
        raise ValueError(f'Invalid pad: {pad}')
    return out_img, mask


def render_viewsimg(img, depth, disparity, viewspos, pad='noise', interp_method='linear',
random_occ=False, random_occ_ratio=0.1, occ_num=3, occ_size=(50, 50), occ_color=False,
random_disp_jitter=False):
    '''
    img: (H, W, C)
    depth: (H, W) float32
    disparity: (H, W) float32
    viewspos: (N, 2) float32

    Returns:
        viewsimg: (N, H, W, C)
        masks: (N, H, W)
        views_disp: (N, H, W) — 每个视角的 warped disparity
    '''
    if pad == 'random': pad = np.random.choice(['zero', 'constant', 'edge_fill'])
    N = viewspos.shape[0]
    H, W, C = img.shape
    sub1, sub2 = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    u = viewspos[:, 0]
    v = viewspos[:, 1]

    # 将 disparity 拼接为额外通道，一起渲染以获得 per-view warped disparity
    img_for_render = np.concatenate([img.astype(np.float32), disparity[..., np.newaxis].astype(np.float32)], axis=-1)
    C_render = C + 1

    viewsimg_full = np.zeros((N, H, W, C_render), dtype=np.float32)
    masks = np.zeros((N, H, W), dtype=np.bool_)
    for i in range(N):
        if u[i] == 0 and v[i] == 0:
            viewsimg_full[i] = img_for_render
            masks[i] = np.ones((H, W), dtype=np.bool_)
            continue
        warped_sub1 = sub1 - disparity * u[i]
        warped_sub2 = sub2 - disparity * v[i]
        if random_disp_jitter:
            disp_jitter = np.random.randn(H, W) * 0.03
            warped_sub1 = warped_sub1 + disp_jitter * (u[i] == 0)
            warped_sub2 = warped_sub2 + disp_jitter * (v[i] == 0)
        if interp_method == 'linear':
            if not random_disp_jitter:
                rendered, mask = zbuffer_render_linear_wrapper(warped_sub1, warped_sub2, depth, img_for_render, pad, v[i]==0, u[i]==0)
            else:
                rendered, mask = zbuffer_render_linear_wrapper(warped_sub1, warped_sub2, depth, img_for_render, pad)
        elif interp_method == 'nearest':
            rendered, mask = highspeed_zbuffer_render(warped_sub1, warped_sub2, depth, img_for_render, pad, interp_method)
        elif interp_method == 'backwarp':
            warped_sub1 = (torch.tensor(sub1 + disparity * u[i]) / (H-1)) * 2 - 1
            warped_sub2 = (torch.tensor(sub2 + disparity * v[i]) / (W-1)) * 2 - 1
            rendered = grid_sample(torch.tensor(img_for_render).permute(2,0,1).unsqueeze(0).float(), torch.stack([warped_sub2, warped_sub1], dim=-1).unsqueeze(0).float(), align_corners=True).squeeze(0)
            mask = np.ones((H, W), dtype=np.bool_)
            rendered = rendered.clamp(0, 255).permute(1,2,0).numpy()
        else:
            raise ValueError(f'Invalid interp_method: {interp_method}')
        if random_occ and np.random.rand() < random_occ_ratio:
            for _ in range(occ_num):
                bh = np.random.randint(occ_size[0], occ_size[1])
                bw = np.random.randint(occ_size[0], occ_size[1])
                loc1 = np.random.randint(0, H-bh)
                loc2 = np.random.randint(0, W-bw)
                if occ_color:
                    rendered[loc1:loc1+bh, loc2:loc2+bw, :C] = np.random.rand(C)
                else:
                    rendered[loc1:loc1+bh, loc2:loc2+bw, :C] = rendered[:,:,:C].mean(axis=(0,1))
        viewsimg_full[i] = rendered
        masks[i] = mask.squeeze()

    viewsimg = viewsimg_full[:, :, :, :C]
    views_disp = viewsimg_full[:, :, :, C]
    views_disp[~masks] = np.median(disparity)  # 遮挡区域的 disparity 被 pad 噪声污染，清零
    return viewsimg, masks, views_disp


def render_viewsimg_non_lambertian(lambertian, residual, depth, disparity, viewspos, pad='noise', interp_method='linear'):
    '''
    lambertian: (H, W, C)
    residual: (H, W, C)
    depth: (H, W) float32
    disparity: (H, W) float32
    viewspos: (N, 2) float32
    '''
    N = viewspos.shape[0]
    H, W, C = lambertian.shape
    sub1, sub2 = np.meshgrid(np.arange(lambertian.shape[0]), np.arange(lambertian.shape[1]), indexing='ij')
    u = viewspos[:, 0]
    v = viewspos[:, 1]
    viewsimg = np.zeros((N, H, W, C), dtype=lambertian.dtype)
    masks = np.zeros((N, H, W), dtype=np.bool_)

    normal = depth2normal(torch.tensor(rescale(disparity, 0, 10))).numpy()
    maxloc1, maxloc2 = np.where(residual.squeeze() == residual.max())
    max_normal = normal[:,maxloc1[0],maxloc2[0]].squeeze()
    # 计算两个和max_normal正交的单位向量
    # max_normal 可能有shape (3,) 或 (N,3)，此处我们只取第一个max_normal
    n_vec = max_normal[0] if max_normal.ndim > 1 else max_normal
    n_vec = n_vec / np.linalg.norm(n_vec)
    # 找到与 n 不平行的一个向量
    if abs(n_vec[0]) < 0.9:
        v_vec = np.array([1, 0, 0], dtype=n_vec.dtype)
    else:
        v_vec = np.array([0, 1, 0], dtype=n_vec.dtype)
    # 使用叉乘得到正交向量
    ortho1 = np.cross(n_vec, v_vec)
    ortho1 = ortho1 / np.linalg.norm(ortho1)
    ortho2 = np.cross(n_vec, ortho1)
    ortho2 = ortho2 / np.linalg.norm(ortho2)
    x1 = normal[0]*ortho1[0] + normal[1]*ortho1[1] + normal[2]*ortho1[2]
    x2 = normal[0]*ortho2[0] + normal[1]*ortho2[1] + normal[2]*ortho2[2]
    sigma = np.random.uniform(2,6)
    factor = np.exp(-0.5*(x1**2+x2**2)/sigma**2).clip(0.2, 1.0)
    norm_residual = residual / factor[..., None]
    for i in range(N):
        if u[i] == 0 and v[i] == 0:
            viewsimg[i] = lambertian + residual
            masks[i] = np.ones((H, W), dtype=np.bool_)
            continue
        warped_sub1 = sub1 - disparity * u[i]
        warped_sub2 = sub2 - disparity * v[i]
        img = np.concatenate([lambertian, norm_residual], axis=-1)
        if interp_method == 'linear':
            img_rendered, mask = zbuffer_render_linear(warped_sub1, warped_sub2, depth, img, pad, v[i]==0, u[i]==0)
        elif interp_method == 'nearest':
            img_rendered, mask = highspeed_zbuffer_render(warped_sub1, warped_sub2, depth, img, pad, interp_method)
        else:
            raise ValueError(f'Invalid interp_method: {interp_method}')
        lambertian_rendered = img_rendered[..., :C]
        norm_residual_rendered = img_rendered[..., C:]
        factor = np.exp(-0.5*((x1-u[i])**2+(x2-v[i])**2)/sigma**2)
        residual_rendered = norm_residual_rendered * factor[..., None]
        img_rendered = (lambertian_rendered + residual_rendered)
        viewsimg[i] = img_rendered
        masks[i] = mask.squeeze()
    return viewsimg, masks
