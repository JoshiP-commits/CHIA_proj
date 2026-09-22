import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    sm_scale,
    NUM_BLOCKS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    # Head index is available as a program id -- derive per-head constants from it.
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Base pointers with per-head offsets
    q_base = q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_base = k_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = v_ptr + pid_b * stride_vb + pid_h * stride_vh
    out_base = out_ptr + pid_b * stride_ob + pid_h * stride_oh

    # Q pointers
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs)

    # K pointers (swapped strides for implicit transpose: [HEAD_DIM, BLOCK_N])
    k_ptrs = k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks

    # V pointers ([BLOCK_N, HEAD_DIM])
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    # Init running max with tl.full([BLOCK_M], -1e9, tl.float32), not -inf.
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, NUM_BLOCKS_N):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        # Compute scores
        qk = tl.dot(q, k) * sm_scale

        # Compute the stride test arithmetically in the kernel
        n_idx = start_n * BLOCK_N + offs_n
        diff = offs_m[:, None] - n_idx[None, :]
        is_valid = diff >= 0
        safe_diff = tl.where(is_valid, diff, 0)
        cond = is_valid & ((safe_diff % 4) == 0)

        # Use tl.where(cond, a, b) for masking
        qk = tl.where(cond, qk, -1e9)

        # Running max and sum (FlashAttention-style tiling)
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(qk - m_i_new[:, None])
        
        # Zero out beta for masked elements to prevent accumulating exp(-1e9 - (-1e9)) = 1.0
        beta = tl.where(cond, beta, 0.0)

        l_i_new = alpha * l_i + tl.sum(beta, 1)

        p = beta.to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p, v)

        m_i = m_i_new
        l_i = l_i_new

        # Advance pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Normalize and store
    acc = acc / l_i[:, None]
    out_ptrs = out_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(q.dtype))

def launch_kernel(q, k, v):
    B, H, S, D = q.shape
    out = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    NUM_BLOCKS_N = S // BLOCK_N

    grid = (S // BLOCK_M, B, H)
    
    # Compute 1/sqrt(D) on the host, pass as a float
    sm_scale = float(1.0 / (D ** 0.5))

    graphsynth_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sm_scale,
        NUM_BLOCKS_N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=3
    )

    return out