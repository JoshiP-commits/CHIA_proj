import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    x_ptr,
    out_ptr,
    row_stride,
    n_cols,
    n_cols_f32,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x_ptrs = x_ptr + row_idx * row_stride + offsets
    out_ptrs = out_ptr + row_idx * row_stride + offsets

    x = tl.load(x_ptrs, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / n_cols_f32
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols_f32

    inv_std = tl.math.rsqrt(var + eps)
    out = x_centered * inv_std

    tl.store(out_ptrs, out, mask=mask)

def launch_kernel(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    eps = 1e-5

    grid = (n_rows,)
    graphsynth_kernel[grid](
        x,
        out,
        x.stride(0),
        n_cols,
        float(n_cols),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out