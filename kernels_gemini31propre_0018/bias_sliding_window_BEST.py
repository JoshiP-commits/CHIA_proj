import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid_m = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    # Derive per-head constants as constrained
    q_head_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_head_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_head_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_head_offset = batch_idx * stride_ob + head_idx * stride_oh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_init = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    q_ptrs = Q + q_head_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs)
    
    # K is loaded with swapped strides to simulate trans_b (Shape: [D, BLOCK_N])
    k_ptrs = K + k_head_offset + offs_d[:, None] * stride_kd + offs_n_init[None, :] * stride_ks
    
    # V is loaded normally (Shape: [BLOCK_N, D])
    v_ptrs = V + v_head_offset + offs_n_init[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Init running max with -1e9 as specified
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Bound the KV loop range based on sliding window to skip entirely masked blocks
    lo = tl.maximum(0, (pid_m * BLOCK_M - 256) // BLOCK_N * BLOCK_N)
    hi = (pid_m + 1) * BLOCK_M
    
    k_ptrs += lo * stride_ks
    v_ptrs += lo * stride_vs
    
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + offs_n_init
        
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        qk = tl.dot(q, k)
        qk = qk.to(tl.float32) * sm_scale
        
        # Masking using tl.where
        mask = (offs_n[None, :] <= offs_m[:, None]) & (offs_m[:, None] - offs_n[None, :] < 256)
        qk = tl.where(mask, qk, -1e9)
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])
        p = tl.where(mask, p, 0.0)
        
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        p = p.to(tl.bfloat16)
        acc += tl.dot(p, v)
        
        m_i = m_ij
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    acc = acc / l_i[:, None]
    acc = acc.to(tl.bfloat16)
    
    out_ptrs = Out + o_head_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc)

def launch_kernel(q, k, v):
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    
    # Compute 1/sqrt(D) on the host to avoid missing .to() on Python ints inside kernel
    sm_scale = float(1.0 / (D ** 0.5))
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    graphsynth_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D
    )
    return out