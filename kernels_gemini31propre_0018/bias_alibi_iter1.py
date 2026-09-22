import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    Q, K, V, Out,
    stride_b, stride_h, stride_s, stride_d,
    scale: tl.float32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    offset = batch_idx * stride_b + head_idx * stride_h
    q_ptr = Q + offset
    k_ptr = K + offset
    v_ptr = V + offset
    o_ptr = Out + offset

    # Load Q: [BLOCK_M, BLOCK_D]
    q_ptrs = q_ptr + (offs_m[:, None] * stride_s + offs_d[None, :] * stride_d)
    q = tl.load(q_ptrs)

    # Load K transposed: [BLOCK_D, BLOCK_N]
    k_ptrs = k_ptr + (offs_d[:, None] * stride_d + offs_n[None, :] * stride_s)
    # Load V: [BLOCK_N, BLOCK_D]
    v_ptrs = v_ptr + (offs_n[:, None] * stride_s + offs_d[None, :] * stride_d)

    m_i = tl.full([BLOCK_M], -1e9, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    # Bound loop range based on query block (causality)
    n_end = (pid_m + 1) * BLOCK_M
    for start_n in range(0, n_end, BLOCK_N):
        n_idx = start_n + offs_n

        k_curr = tl.load(k_ptrs + start_n * stride_s)
        v_curr = tl.load(v_ptrs + start_n * stride_s)

        # Q @ K^T
        s = tl.dot(q, k_curr) * scale

        # ALiBi bias
        slope = head_idx + 1.0
        alibi = slope * (n_idx[None, :] - offs_m[:, None])
        s = s + alibi

        # Causal mask (j <= i)
        mask = n_idx[None, :] <= offs_m[:, None]
        s = tl.where(mask, s, -1e9)

        # FlashAttention updates
        m_ij = tl.max(s, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(s - m_i_new[:, None])

        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        # fp32 accumulation
        acc = acc * alpha[:, None] + tl.dot(p.to(v_curr.dtype), v_curr)

        m_i = m_i_new
        l_i = l_i_new

    # Normalize and store
    out = acc / l_i[:, None]
    out_ptrs = o_ptr + (offs_m[:, None] * stride_s + offs_d[None, :] * stride_d)
    tl.store(out_ptrs, out.to(q.dtype))


def launch_kernel(q, k, v):
    B, H, S, D = q.shape
    out = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    grid = (B, H, S // BLOCK_M)
    scale = float(1.0 / math.sqrt(D))

    stride_b = q.stride(0)
    stride_h = q.stride(1)
    stride_s = q.stride(2)
    stride_d = q.stride(3)

    graphsynth_kernel[grid](
        q, k, v, out,
        stride_b, stride_h, stride_s, stride_d,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D
    )

    return out