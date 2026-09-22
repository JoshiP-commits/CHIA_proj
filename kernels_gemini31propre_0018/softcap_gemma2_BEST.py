import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    sm_scale,
    B, H, N, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    start_m = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    # Derive per-head and batch constants
    batch_id = batch_head_id // H
    head_id = batch_head_id % H

    # Offset base pointers
    q_base = Q + batch_id * stride_qb + head_id * stride_qh
    k_base = K + batch_id * stride_kb + head_id * stride_kh
    v_base = V + batch_id * stride_vb + head_id * stride_vh
    o_base = Out + batch_id * stride_ob + head_id * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Pointers
    q_ptrs = q_base + (offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    # Load K transposed by swapping its strides (no trans_b needed in tl.dot)
    k_ptrs = k_base + (offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn)
    v_ptrs = v_base + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # Load queries for this block
    q = tl.load(q_ptrs)

    # Initialize running max and sum for online softmax
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # fp32 accumulator for the output
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Causal sequence masking: block-level bound
    n_steps = start_m + 1 
    
    for i in range(n_steps):
        start_n = i * BLOCK_N
        current_offs_n = start_n + offs_n
        
        # Advance pointers along sequence length N
        current_k_ptrs = k_ptrs + start_n * stride_kn
        current_v_ptrs = v_ptrs + start_n * stride_vn
        
        k = tl.load(current_k_ptrs)
        v = tl.load(current_v_ptrs)
        
        # Compute dot product
        qk = tl.dot(q, k)
        qk = qk * sm_scale
        
        # Gemma-2 logit soft-capping: tanh(x/30)*30, expanded using sigmoid
        qk = qk / 30.0
        qk = (2.0 * tl.sigmoid(2.0 * qk) - 1.0) * 30.0
        
        # Causal mask (j <= i)
        mask = current_offs_n[None, :] <= offs_m[:, None]
        qk = tl.where(mask, qk, -1e9)
        
        # Online softmax logic
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, 1)
        
        p_v = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(p_v, v)
        
        m_i = m_i_new
        
    acc = acc / l_i[:, None]
    
    # Store output in bfloat16
    out_ptrs = o_base + (offs_m[:, None] * stride_on + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(tl.bfloat16))


def launch_kernel(q, k, v):
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128
    
    # Compute scale on host and pass as float (no python int .to())
    sm_scale = float(1.0 / (D ** 0.5))
    
    grid = (N // BLOCK_M, B * H)
    
    graphsynth_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sm_scale,
        B, H, N, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
    )
    
    return out