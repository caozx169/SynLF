import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _z_buffer_pass_kernel(
    disp_ptr,
    disp_max_ptr,
    viewspos_ptr,
    z_min_ptr,
    H, W, V,
    stride_disp_b, stride_disp_h, stride_disp_w,
    stride_vpos_b, stride_vpos_v, stride_vpos_d,
    stride_zmin_b, stride_zmin_v, stride_zmin_m, stride_zmin_h, stride_zmin_w,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2) # B * V

    bi = pid_z // V
    vi = pid_z % V

    xs = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    ys = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)

    mask_x = xs < W
    mask_y = ys < H
    mask = mask_y[:, None] & mask_x[None, :]

    disp_offsets = bi * stride_disp_b + ys[:, None] * stride_disp_h + xs[None, :] * stride_disp_w
    disp_vals = tl.load(disp_ptr + disp_offsets, mask=mask, other=0.0)

    disp_max = tl.load(disp_max_ptr + bi)
    z_vals = 1.0 + (disp_max - disp_vals)

    du = tl.load(viewspos_ptr + bi * stride_vpos_b + vi * stride_vpos_v + 0 * stride_vpos_d)
    dv = tl.load(viewspos_ptr + bi * stride_vpos_b + vi * stride_vpos_v + 1 * stride_vpos_d)

    x_proj = xs[None, :] - disp_vals * dv
    y_proj = ys[:, None] - disp_vals * du

    for mode_x in range(2):
        for mode_y in range(2):
            if mode_x == 0:
                xi = tl.math.floor(x_proj)
            else:
                xi = tl.math.ceil(x_proj)

            if mode_y == 0:
                yi = tl.math.floor(y_proj)
            else:
                yi = tl.math.ceil(y_proj)

            xi_int = xi.to(tl.int32)
            yi_int = yi.to(tl.int32)

            valid = mask & (xi_int >= 0) & (xi_int < W) & (yi_int >= 0) & (yi_int < H)

            mode_idx = mode_x * 2 + mode_y
            zmin_offsets = (bi * stride_zmin_b +
                            vi * stride_zmin_v +
                            mode_idx * stride_zmin_m +
                            yi_int * stride_zmin_h +
                            xi_int * stride_zmin_w)

            # Using atomic min to compute Z-buffer
            tl.atomic_min(z_min_ptr + zmin_offsets, z_vals, mask=valid)

@triton.jit
def _visibility_pass_kernel(
    disp_ptr,
    disp_max_ptr,
    viewspos_ptr,
    z_min_ptr,
    vis_out_ptr,
    H, W, V,
    z_tol,
    stride_disp_b, stride_disp_h, stride_disp_w,
    stride_vpos_b, stride_vpos_v, stride_vpos_d,
    stride_zmin_b, stride_zmin_v, stride_zmin_m, stride_zmin_h, stride_zmin_w,
    stride_vis_b, stride_vis_v, stride_vis_h, stride_vis_w,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2) # B * V

    bi = pid_z // V
    vi = pid_z % V

    xs = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    ys = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)

    mask_x = xs < W
    mask_y = ys < H
    mask = mask_y[:, None] & mask_x[None, :]

    disp_offsets = bi * stride_disp_b + ys[:, None] * stride_disp_h + xs[None, :] * stride_disp_w
    disp_vals = tl.load(disp_ptr + disp_offsets, mask=mask, other=0.0)

    disp_max = tl.load(disp_max_ptr + bi)
    z_vals = 1.0 + (disp_max - disp_vals)

    du = tl.load(viewspos_ptr + bi * stride_vpos_b + vi * stride_vpos_v + 0 * stride_vpos_d)
    dv = tl.load(viewspos_ptr + bi * stride_vpos_b + vi * stride_vpos_v + 1 * stride_vpos_d)

    x_proj = xs[None, :] - disp_vals * dv
    y_proj = ys[:, None] - disp_vals * du

    is_visible = tl.zeros([BLOCK_Y, BLOCK_X], dtype=tl.int1)

    for mode_x in range(2):
        for mode_y in range(2):
            if mode_x == 0:
                xi = tl.math.floor(x_proj)
            else:
                xi = tl.math.ceil(x_proj)

            if mode_y == 0:
                yi = tl.math.floor(y_proj)
            else:
                yi = tl.math.ceil(y_proj)

            xi_int = xi.to(tl.int32)
            yi_int = yi.to(tl.int32)

            valid = mask & (xi_int >= 0) & (xi_int < W) & (yi_int >= 0) & (yi_int < H)

            mode_idx = mode_x * 2 + mode_y
            zmin_offsets = (bi * stride_zmin_b +
                            vi * stride_zmin_v +
                            mode_idx * stride_zmin_m +
                            yi_int * stride_zmin_h +
                            xi_int * stride_zmin_w)

            z_front = tl.load(z_min_ptr + zmin_offsets, mask=valid, other=float('inf'))
            keep = valid & (z_vals <= z_front + z_tol)

            is_visible = is_visible | keep

    vis_offsets = bi * stride_vis_b + vi * stride_vis_v + ys[:, None] * stride_vis_h + xs[None, :] * stride_vis_w
    tl.store(vis_out_ptr + vis_offsets, is_visible.to(tl.float32), mask=mask)

_kernel_cache = {}

def _get_kernels(device, dtype):
    key = (str(device), dtype)
    if key not in _kernel_cache:
        k3 = torch.ones((1, 1, 3, 3), device=device, dtype=dtype)
        k2 = torch.ones((1, 1, 2, 2), device=device, dtype=dtype)
        kernel_1d = torch.tensor([0.25, 0.5, 0.25], device=device, dtype=dtype)
        kernel_2d = (kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)).view(1, 1, 3, 3)
        _kernel_cache[key] = (k3, k2, kernel_2d)
    return _kernel_cache[key]

def soft_visibility_forward_splat_triton(disp, viewspos_src):
    b, _, h, w = disp.shape
    device = disp.device
    dtype = disp.dtype

    disp_s = disp[:, 0].contiguous()
    disp_max = disp_s.flatten(1).amax(dim=1).contiguous()
    viewspos_src = viewspos_src.contiguous()

    V = viewspos_src.shape[1]

    # Initialize Z-buffer with infinity
    # Shape: (B, V, 4, H, W)
    z_min = torch.full((b, V, 4, h, w), float('inf'), device=device, dtype=dtype)

    vis_out = torch.zeros((b, V, h, w), device=device, dtype=dtype)

    BLOCK_X = 16
    BLOCK_Y = 16
    grid = (triton.cdiv(w, BLOCK_X), triton.cdiv(h, BLOCK_Y), b * V)

    # Pass 1: Compute Z-buffer
    _z_buffer_pass_kernel[grid](
        disp_s,
        disp_max,
        viewspos_src,
        z_min,
        h, w, V,
        disp_s.stride(0), disp_s.stride(1), disp_s.stride(2),
        viewspos_src.stride(0), viewspos_src.stride(1), viewspos_src.stride(2),
        z_min.stride(0), z_min.stride(1), z_min.stride(2), z_min.stride(3), z_min.stride(4),
        BLOCK_X=BLOCK_X,
        BLOCK_Y=BLOCK_Y,
    )

    # Pass 2: Visibility Test
    z_tol = 1e-4
    _visibility_pass_kernel[grid](
        disp_s,
        disp_max,
        viewspos_src,
        z_min,
        vis_out,
        h, w, V,
        z_tol,
        disp_s.stride(0), disp_s.stride(1), disp_s.stride(2),
        viewspos_src.stride(0), viewspos_src.stride(1), viewspos_src.stride(2),
        z_min.stride(0), z_min.stride(1), z_min.stride(2), z_min.stride(3), z_min.stride(4),
        vis_out.stride(0), vis_out.stride(1), vis_out.stride(2), vis_out.stride(3),
        BLOCK_X=BLOCK_X,
        BLOCK_Y=BLOCK_Y,
    )

    # Shape of vis_out is (B, V, H, W). We need (B*V, 1, H, W) for morphology
    vis_flat = vis_out.view(b * V, 1, h, w)
    vis_bin = vis_flat > 0.5

    k3, k2, kernel_2d = _get_kernels(device, dtype)

    # fill 1px holes
    n3 = F.conv2d(vis_bin.to(dtype), k3, padding=1)
    fill_hole = (~vis_bin) & (n3 >= 8.0)
    vis_bin = vis_bin | fill_hole

    # 2x2 support
    full2 = F.conv2d(vis_bin.to(dtype), k2, padding=0) >= 4.0
    support = (
        F.pad(full2, (0, 1, 0, 1))
        | F.pad(full2, (1, 0, 0, 1))
        | F.pad(full2, (0, 1, 1, 0))
        | F.pad(full2, (1, 0, 1, 0))
    )
    vis_clean = (vis_bin & support).to(dtype)

    # gaussian blur
    vis_blur = F.conv2d(F.pad(vis_clean, (1, 1, 1, 1), mode="replicate"), kernel_2d, padding=0)
    blur_alpha = 0.15
    vis_soft = ((1.0 - blur_alpha) * vis_clean + blur_alpha * vis_blur).clamp_(0.0, 1.0)

    return vis_soft.view(b, V, 1, h, w)
