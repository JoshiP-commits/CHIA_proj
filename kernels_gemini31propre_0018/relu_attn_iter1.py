import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    batch = off_hz // H
    head = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # Q shape: [BLOCK_M, BLOCK_D]
    q_ptrs = Q + (batch * stride_qz + head * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    
    # K loaded transposed to avoid trans_b: shape [BLOCK_D, BLOCK_N]
    # stride_kd is along dim 0 (features), stride_kn is along dim 1 (sequence length)
    k_ptrs = K + (batch * stride_kz + head * stride_kh + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn)
    
    # V shape: [BLOCK_N, BLOCK_D]
    v_ptrs = V + (batch * stride_vz + head * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # Initialize running max, running sum, and fp32 accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -1e9, tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    q = tl.load(q_ptrs)

    # Bound loop range strictly on host-provided N_CTX and mask elements causally
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)
        qk = qk.to(tl.float32) * sm_scale
        
        # Causal mask j <= i (masked contribute 0)
        causal_mask = offs_n_curr[None, :] <= offs_m[:, None]
        qk = tl.where(causal_mask, qk, 0.0)
        
        # Elementwise ReLU (NO softmax)
        p = tl.where(qk > 0.0, qk, 0.0)
        
        # FlashAttention-style tiling requirements: update running max and running sum
        m_i = tl.maximum(m_i, tl.max(p, axis=1))
        l_i = l_i + tl.sum(p, axis=1)

        # Accumulate unnormalized out = p @ v
        v = tl.load(v_ptrs)
        acc += tl.dot(p.to(tl.bfloat16), v)
        
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    out_ptrs = Out + (batch * stride_oz + head * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(out_ptrs, acc.to(tl.bfloat16))

def launch_kernel(q, k, v):
    Z, H, N_CTX, D = q.shape
    out = torch.empty_like(q)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 128
    
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H, 1)
    
    # Compute scale on host and pass as float
    sm_scale = float(1.0 / (D ** 0.5))
    
    graphsynth_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N
    )
    return out