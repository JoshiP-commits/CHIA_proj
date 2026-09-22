import triton
import triton.language as tl
import torch

@triton.jit
def graphsynth_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, D_HEAD,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    z = pid_bh // H
    h = pid_bh % H

    q_offset = z * stride_qz + h * stride_qh
    k_offset = z * stride_kz + h * stride_kh
    v_offset = z * stride_vz + h * stride_vh
    o_offset = z * stride_oz + h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)

    q_ptrs = q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)

    # Bound loop range: only KV blocks overlapping the bidirectional band abs(i - j) < 256
    lo = (pid_m * BLOCK_M - 255) // BLOCK_N * BLOCK_N
    start_n = tl.maximum(0, lo)
    hi = (pid_m * BLOCK_M + BLOCK_M + 255 + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    end_n = tl.minimum(N_CTX, hi)

    # Load K with swapped strides (transposed) to avoid tl.dot trans_b
    k_ptrs = k_ptr + k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_ptrs = v_ptr + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    i_idx = offs_m[:, None]

    for start_j in range(start_n, end_n, BLOCK_N):
        curr_k_ptrs = k_ptrs + start_j * stride_kn
        curr_v_ptrs = v_ptrs + start_j * stride_vn

        k = tl.load(curr_k_ptrs, mask=offs_n[None, :] + start_j < N_CTX, other=0.0)
        v = tl.load(curr_v_ptrs, mask=offs_n[:, None] + start_j < N_CTX, other=0.0)

        qk = tl.dot(q, k)
        qk = qk.to(tl.float32) * scale

        j_idx = start_j + offs_n[None, :]
        dist = i_idx - j_idx
        
        mask = (dist > -256) & (dist < 256) & (j_idx < N_CTX) & (i_idx < N_CTX)
        qk = tl.where(mask, qk, -1e9)

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        l_i_new = alpha * l_i + tl.sum(p, 1)

        p_bf16 = p.to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc += tl.dot(p_bf16, v).to(tl.float32)

        m_i = m_i_new
        l_i = l_i_new

    acc = acc / l_i[:, None]

    out_ptrs = out_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < N_CTX)


def launch_kernel(q, k, v):
    Z, H, N_CTX, D_HEAD = q.shape
    out = torch.empty_like(q)

    scale = float(1.0 / (D_HEAD ** 0.5))
    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    graphsynth_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX, D_HEAD,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=3
    )
    return out