import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_km, stride_kd,
    stride_vb, stride_vh, stride_vm, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seqlen,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch = pid_bh // 32
    head = pid_bh % 32
    
    t_h = 0.5 + head / 32.0
    
    q_off = batch * stride_qb + head * stride_qh
    k_off = batch * stride_kb + head * stride_kh
    v_off = batch * stride_vb + head * stride_vh
    o_off = batch * stride_ob + head * stride_oh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)
    
    q_ptrs = q_ptr + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = k_ptr + k_off + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_km
    v_ptrs = v_ptr + v_off + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd
    o_ptrs = out_ptr + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    
    q_mask = offs_m[:, None] < seqlen
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    
    for start_n in range(0, seqlen, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        k_mask = (start_n + offs_n[None, :]) < seqlen
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        v_mask = (start_n + offs_n[:, None]) < seqlen
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        qk = tl.dot(q, k)
        qk = qk.to(tl.float32)
        
        qk = qk * sm_scale
        qk = qk * t_h
        
        causal_mask = (start_n + offs_n[None, :]) <= offs_m[:, None]
        qk = tl.where(causal_mask, qk, -1e9)
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        p = tl.where(causal_mask, p, 0.0)
        
        l_new = l_i * alpha + tl.sum(p, 1)
        
        acc = acc * alpha[:, None]
        
        p_bfloat = p.to(tl.bfloat16)
        acc += tl.dot(p_bfloat, v)
        
        m_i = m_new
        l_i = l_new
        
        k_ptrs += BLOCK_N * stride_km
        v_ptrs += BLOCK_N * stride_vm

    acc = acc / l_i[:, None]
    
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < seqlen)

def launch_kernel(q, k, v):
    B, H, M, D = q.shape
    out = torch.empty_like(q)
    BLOCK_M = 128
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), B * H, 1)
    sm_scale = float(1.0 / (D ** 0.5))
    
    graphsynth_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out