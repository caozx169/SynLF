import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# 0. Helper Functions
# -----------------------------------------------------------------------------

@triton.jit
def _bilinear_read(
    ptr_base, h_coords, w_coords, H, W, stride_h, stride_w, mask_in
):
    h0 = tl.floor(h_coords).to(tl.int32)
    h1 = h0 + 1
    w0 = tl.floor(w_coords).to(tl.int32)
    w1 = w0 + 1

    alpha_h = h_coords - h0
    alpha_w = w_coords - w0

    valid_h0 = (h0 >= 0) & (h0 < H)
    valid_h1 = (h1 >= 0) & (h1 < H)
    valid_w0 = (w0 >= 0) & (w0 < W)
    valid_w1 = (w1 >= 0) & (w1 < W)

    v00 = tl.load(ptr_base + h0 * stride_h + w0 * stride_w, mask=mask_in & valid_h0 & valid_w0, other=0.0)
    v01 = tl.load(ptr_base + h0 * stride_h + w1 * stride_w, mask=mask_in & valid_h0 & valid_w1, other=0.0)
    v10 = tl.load(ptr_base + h1 * stride_h + w0 * stride_w, mask=mask_in & valid_h1 & valid_w0, other=0.0)
    v11 = tl.load(ptr_base + h1 * stride_h + w1 * stride_w, mask=mask_in & valid_h1 & valid_w1, other=0.0)

    top = v00 * (1.0 - alpha_w) + v01 * alpha_w
    bot = v10 * (1.0 - alpha_w) + v11 * alpha_w
    return top * (1.0 - alpha_h) + bot * alpha_h

# -----------------------------------------------------------------------------
# 1. Forward Kernel (Fused Warp + GWC + AvgPool)
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 16}, num_warps=4, num_stages=3),
    ],
    key=[], # 这里的 key 为空意味着：整个程序生命周期只 Tune 一次 (假设你的 H, W 训练时固定)
)
@triton.jit
def lf_shift_gwc_forward_kernel(
    x_ptr, ref_ptr, y_ptr, views_ptr,
    B, N, C, H, W, G, C_per_G,
    shift, downfactor,
    stride_xb, stride_xn, stride_xc, stride_xh, stride_xw,
    stride_rb, stride_rc, stride_rh, stride_rw,
    stride_yb, stride_yn, stride_yg, stride_yh, stride_yw,
    stride_vb, stride_vn, stride_vc, # Added stride_vb for [B, N, 2]
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # PID Mapping corresponds to OUTPUT (Low-Res) dimensions
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)

    total_ng = N * G
    idx_b = pid_z // total_ng
    rem = pid_z % total_ng
    idx_n = rem // G
    idx_g = rem % G

    # Read per-batch view offsets.
    ptr_u = views_ptr + idx_b * stride_vb + idx_n * stride_vn + 0 * stride_vc
    ptr_v = views_ptr + idx_b * stride_vb + idx_n * stride_vn + 1 * stride_vc
    u = tl.load(ptr_u).to(tl.float32)
    v = tl.load(ptr_v).to(tl.float32)

    delta_h = u * shift
    delta_w = v * shift

    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    H_out = H // downfactor
    W_out = W // downfactor

    mask_h = out_h[:, None] < H_out
    mask_w = out_w[None, :] < W_out
    store_mask = mask_h & mask_w

    base_x_bn = x_ptr + (idx_b * stride_xb) + (idx_n * stride_xn)
    base_ref_b = ref_ptr + (idx_b * stride_rb)

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    base_h_in = out_h * downfactor
    base_w_in = out_w * downfactor

    for k in range(C_per_G):
        real_c = idx_g * C_per_G + k
        ptr_x_c = base_x_bn + real_c * stride_xc
        ptr_ref_c = base_ref_b + real_c * stride_rc

        for dy in range(downfactor):
            for dx in range(downfactor):
                curr_h = base_h_in[:, None] + dy
                curr_w = base_w_in[None, :] + dx

                src_h = curr_h - delta_h
                src_w = curr_w - delta_w

                val_x = _bilinear_read(ptr_x_c, src_h, src_w, H, W, stride_xh, stride_xw, store_mask)

                valid_ref = (curr_h < H) & (curr_w < W)
                ptr_ref = ptr_ref_c + curr_h * stride_rh + curr_w * stride_rw
                val_ref = tl.load(ptr_ref, mask=store_mask & valid_ref, other=0.0)

                acc += val_x * val_ref

    scale = 1.0 / (downfactor * downfactor)
    acc = acc * scale

    out_ptr = y_ptr + (idx_b * stride_yb) + (idx_n * stride_yn) + (idx_g * stride_yg) + \
              (out_h[:, None] * stride_yh) + (out_w[None, :] * stride_yw)
    tl.store(out_ptr, acc, mask=store_mask)

# -----------------------------------------------------------------------------
# 2. Backward Kernel 1: Grad X
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 16}, num_warps=4, num_stages=3),
    ],
    key=[], # 这里的 key 为空意味着：整个程序生命周期只 Tune 一次 (假设你的 H, W 训练时固定)
)
@triton.jit
def lf_shift_gwc_backward_dx_kernel(
    grad_y_ptr, ref_ptr, grad_x_ptr, views_ptr,
    B, N, C, H, W, G, C_per_G,
    shift, downfactor,
    stride_gy_b, stride_gy_n, stride_gy_g, stride_gy_h, stride_gy_w,
    stride_rb, stride_rc, stride_rh, stride_rw,
    stride_xb, stride_xn, stride_xc, stride_xh, stride_xw,
    stride_vb, stride_vn, stride_vc, # Added stride_vb
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)

    total_nc = N * C
    idx_b = pid_z // total_nc
    rem = pid_z % total_nc
    idx_n = rem // C
    idx_c = rem % C
    idx_g = idx_c // C_per_G

    # Read per-batch view offsets.
    ptr_u = views_ptr + idx_b * stride_vb + idx_n * stride_vn + 0 * stride_vc
    ptr_v = views_ptr + idx_b * stride_vb + idx_n * stride_vn + 1 * stride_vc
    u = tl.load(ptr_u).to(tl.float32)
    v = tl.load(ptr_v).to(tl.float32)

    delta_h = u * shift
    delta_w = v * shift

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_h = offs_h[:, None] < H
    mask_w = offs_w[None, :] < W
    store_mask = mask_h & mask_w

    target_h = offs_h[:, None] + delta_h
    target_w = offs_w[None, :] + delta_w

    h0 = tl.floor(target_h).to(tl.int32)
    h1 = h0 + 1
    w0 = tl.floor(target_w).to(tl.int32)
    w1 = w0 + 1

    alpha_h = target_h - h0
    alpha_w = target_w - w0

    valid_h0 = (h0 >= 0) & (h0 < H)
    valid_h1 = (h1 >= 0) & (h1 < H)
    valid_w0 = (w0 >= 0) & (w0 < W)
    valid_w1 = (w1 >= 0) & (w1 < W)

    base_ref = ref_ptr + idx_b * stride_rb + idx_c * stride_rc
    base_gy = grad_y_ptr + idx_b * stride_gy_b + idx_n * stride_gy_n + idx_g * stride_gy_g

    scale = 1.0 / (downfactor * downfactor)

    gh0 = h0 // downfactor
    gw0 = w0 // downfactor
    ptr_gy00 = base_gy + gh0 * stride_gy_h + gw0 * stride_gy_w
    gy00 = tl.load(ptr_gy00, mask=store_mask & valid_h0 & valid_w0, other=0.0)

    gh0 = h0 // downfactor
    gw1 = w1 // downfactor
    ptr_gy01 = base_gy + gh0 * stride_gy_h + gw1 * stride_gy_w
    gy01 = tl.load(ptr_gy01, mask=store_mask & valid_h0 & valid_w1, other=0.0)

    gh1 = h1 // downfactor
    gw0 = w0 // downfactor
    ptr_gy10 = base_gy + gh1 * stride_gy_h + gw0 * stride_gy_w
    gy10 = tl.load(ptr_gy10, mask=store_mask & valid_h1 & valid_w0, other=0.0)

    gh1 = h1 // downfactor
    gw1 = w1 // downfactor
    ptr_gy11 = base_gy + gh1 * stride_gy_h + gw1 * stride_gy_w
    gy11 = tl.load(ptr_gy11, mask=store_mask & valid_h1 & valid_w1, other=0.0)

    r00 = tl.load(base_ref + h0 * stride_rh + w0 * stride_rw, mask=store_mask & valid_h0 & valid_w0, other=0.0)
    r01 = tl.load(base_ref + h0 * stride_rh + w1 * stride_rw, mask=store_mask & valid_h0 & valid_w1, other=0.0)
    r10 = tl.load(base_ref + h1 * stride_rh + w0 * stride_rw, mask=store_mask & valid_h1 & valid_w0, other=0.0)
    r11 = tl.load(base_ref + h1 * stride_rh + w1 * stride_rw, mask=store_mask & valid_h1 & valid_w1, other=0.0)

    p00 = gy00 * r00
    p01 = gy01 * r01
    p10 = gy10 * r10
    p11 = gy11 * r11

    top = p00 * (1.0 - alpha_w) + p01 * alpha_w
    bot = p10 * (1.0 - alpha_w) + p11 * alpha_w
    grad_val = (top * (1.0 - alpha_h) + bot * alpha_h) * scale

    out_ptr = grad_x_ptr + idx_b * stride_xb + idx_n * stride_xn + idx_c * stride_xc + \
              offs_h[:, None] * stride_xh + offs_w[None, :] * stride_xw
    tl.store(out_ptr, grad_val, mask=store_mask)

# -----------------------------------------------------------------------------
# 3. Backward Kernel 2: Grad Ref
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 16}, num_warps=4, num_stages=3),
    ],
    key=[], # 这里的 key 为空意味着：整个程序生命周期只 Tune 一次 (假设你的 H, W 训练时固定)
)
@triton.jit
def lf_shift_gwc_backward_dref_kernel(
    grad_y_ptr, x_ptr, grad_ref_ptr, views_ptr,
    B, N, C, H, W, G, C_per_G,
    shift, downfactor,
    stride_gy_b, stride_gy_n, stride_gy_g, stride_gy_h, stride_gy_w,
    stride_xb, stride_xn, stride_xc, stride_xh, stride_xw,
    stride_rb, stride_rc, stride_rh, stride_rw,
    stride_vb, stride_vn, stride_vc, # Added stride_vb
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)

    idx_b = pid_z // C
    idx_c = pid_z % C
    idx_g = idx_c // C_per_G

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (offs_h[:, None] < H) & (offs_w[None, :] < W)
    store_mask = mask

    acc_grad = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    scale = 1.0 / (downfactor * downfactor)

    gy_h_idx = offs_h[:, None] // downfactor
    gy_w_idx = offs_w[None, :] // downfactor

    for n in range(N):
        # Read per-batch view offsets.
        ptr_u = views_ptr + idx_b * stride_vb + n * stride_vn + 0 * stride_vc
        ptr_v = views_ptr + idx_b * stride_vb + n * stride_vn + 1 * stride_vc
        u = tl.load(ptr_u).to(tl.float32)
        v = tl.load(ptr_v).to(tl.float32)
        delta_h = u * shift
        delta_w = v * shift

        src_h = offs_h[:, None] - delta_h
        src_w = offs_w[None, :] - delta_w

        base_x = x_ptr + idx_b * stride_xb + n * stride_xn + idx_c * stride_xc
        val_x = _bilinear_read(base_x, src_h, src_w, H, W, stride_xh, stride_xw, store_mask)

        ptr_gy = grad_y_ptr + idx_b * stride_gy_b + n * stride_gy_n + idx_g * stride_gy_g + \
                 gy_h_idx * stride_gy_h + gy_w_idx * stride_gy_w
        val_gy = tl.load(ptr_gy, mask=store_mask, other=0.0)

        acc_grad += val_x * val_gy

    acc_grad = acc_grad * scale

    out_ptr = grad_ref_ptr + idx_b * stride_rb + idx_c * stride_rc + \
              offs_h[:, None] * stride_rh + offs_w[None, :] * stride_rw
    tl.store(out_ptr, acc_grad, mask=store_mask)

# -----------------------------------------------------------------------------
# 4. Wrapper
# -----------------------------------------------------------------------------
class LFShiftGWCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ref, views, shift, groups, cost_downfactor):
        x = x.contiguous()
        ref = ref.contiguous()
        views = views.contiguous() # [B, N, 2]

        device = x.device

        B, N, C, H, W = x.shape
        ctx.shift_val = shift.item() if isinstance(shift, torch.Tensor) else shift
        ctx.groups = groups
        ctx.down = cost_downfactor

        H_out = H // cost_downfactor
        W_out = W // cost_downfactor

        # [REVERTED] Output is back to [B, N, Groups, H, W]
        y = torch.empty((B, N, groups, H_out, W_out), device=device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(W_out, meta['BLOCK_W']), triton.cdiv(H_out, meta['BLOCK_H']), B * N * groups)

        with torch.cuda.device(device):
            lf_shift_gwc_forward_kernel[grid](
                x, ref, y, views,
                B, N, C, H, W, groups, C // groups,
                ctx.shift_val, cost_downfactor,
                *x.stride(), *ref.stride(), *y.stride(), *views.stride()
            )

        ctx.save_for_backward(x, ref, views)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        x, ref, views = ctx.saved_tensors
        shift = ctx.shift_val
        groups = ctx.groups
        down = ctx.down

        device = x.device

        B, N, C, H, W = x.shape
        C_per_G = C // groups

        grad_x = torch.empty_like(x)
        grad_ref = torch.empty_like(ref)

        # Grids
        grid_dx = lambda meta: (triton.cdiv(W, meta['BLOCK_W']), triton.cdiv(H, meta['BLOCK_H']), B * N * C)
        grid_dref = lambda meta: (triton.cdiv(W, meta['BLOCK_W']), triton.cdiv(H, meta['BLOCK_H']), B * C)

        with torch.cuda.device(device):
            # Launch Grad X
            lf_shift_gwc_backward_dx_kernel[grid_dx](
                grad_output, ref, grad_x, views,
                B, N, C, H, W, groups, C_per_G,
                shift, down,
                *grad_output.stride(), *ref.stride(), *grad_x.stride(), *views.stride()
            )

            # Launch Grad Ref
            lf_shift_gwc_backward_dref_kernel[grid_dref](
                grad_output, x, grad_ref, views,
                B, N, C, H, W, groups, C_per_G,
                shift, down,
                *grad_output.stride(), *x.stride(), *grad_ref.stride(), *views.stride()
            )

        return grad_x, grad_ref, None, None, None, None


def lf_shift_gwc(x, ref, views, shift, groups, cost_downfactor=1):
    return LFShiftGWCFunction.apply(x, ref, views, shift, groups, cost_downfactor)
