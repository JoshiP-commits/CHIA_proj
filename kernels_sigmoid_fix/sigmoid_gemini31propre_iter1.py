import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    Q, K, V, Out,
    scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // 32
    head_idx = pid_bh % 32

    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_offset = batch_idx * stride_ob + head_idx * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs)

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # No break/continue; sequence length S bounds the loop uniformly for all threads.
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K transposed by swapping stride directions
        k_ptrs = K + k_offset + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks
        k = tl.load(k_ptrs)

        # 1. s = (q @ k^T) * 0.0883883476
        qk = tl.dot(q, k)
        qk = qk.to(tl.float32) * scale

        # 2. p = sigmoid(s)
        p = tl.sigmoid(qk)

        # 3. p = p * (j <= i)
        mask = offs_m[:, None] >= offs_n[None, :]
        p = tl.where(mask, p, 0.0)
        p = p.to(tl.bfloat16)

        # Load V
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs)

        # 4. out = p @ v
        acc += tl.dot(p, v)

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16))


def launch_kernel(q, k, v):
    B, H, S, D = q.shape
    out = torch.empty_like(q)

    # Pass 1/sqrt(D) as a host float 
    scale = 0.0883883476

    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    graphsynth_kernel[grid](
        q, k, v, out,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=3
    )

    return out