import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    Q, K, V, Out,
    scale: tl.float32,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_id = pid_bh // H
    head_id = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + batch_id * stride_qb + head_id * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    
    # Load K with swapped strides to mimic transpose (no trans_b in tl.dot)
    k_ptrs = K + batch_id * stride_kb + head_id * stride_kh + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks
    
    v_ptrs = V + batch_id * stride_vb + head_id * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    q = tl.load(q_ptrs)

    # Initialize running max and running sum
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Fixed loop without break/continue, masking out invalid tokens using tl.where
    for start_n in range(0, 2048, BLOCK_N):
        curr_n = start_n + offs_n

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        # q is [BLOCK_M, D], k is [D, BLOCK_N], dot yields [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * scale

        # Query i attends to key j if (j <= i) OR (j < 256)
        mask = (curr_n[None, :] <= offs_m[:, None]) | (curr_n[None, :] < 256)
        qk = tl.where(mask, qk, -1e9)

        # Running max
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        # Scaling factors
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        # Running sum
        l_i_new = l_i * alpha + tl.sum(p, axis=1)

        # Accumulator update
        p_bf16 = p.to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc += tl.dot(p_bf16, v)

        # State updates
        m_i = m_i_new
        l_i = l_i_new

        # Advance pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Final normalization
    acc = acc / l_i[:, None]
    out = acc.to(tl.bfloat16)

    out_ptrs = Out + batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, H, S, D = q.shape
    out = torch.empty_like(q)

    # Compute 1/sqrt(D) on the host and pass as float
    scale = float(1.0 / (D ** 0.5))

    # A100 SRAM is plenty for 64x64 blocks in bfloat16
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    graphsynth_kernel[grid](
        q, k, v, out,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H=H, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D
    )

    return out