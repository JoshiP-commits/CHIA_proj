import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    Q, K, V, Out,
    scale,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    start_m = pid_m * BLOCK_M
    # Document size is 512. Start of the document for this query block.
    doc_start = (start_m // 512) * 512
    # Causal constraint means we only need to visit keys up to start_m + BLOCK_M
    loop_end = start_m + BLOCK_M

    # Derive per-head constants from program id
    offset_bh = batch_idx * stride_b + head_idx * stride_h

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + offset_bh + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d
    q = tl.load(q_ptrs)

    m_i = tl.full([BLOCK_M], -1e9, tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop exactly over the causal portion of the diagonal document block
    for start_n in range(doc_start, loop_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K with swapped strides to avoid trans_b in tl.dot
        # Resulting k shape will be [BLOCK_DMODEL, BLOCK_N]
        k_ptrs = K + offset_bh + offs_d[:, None] * stride_d + offs_n[None, :] * stride_s
        k = tl.load(k_ptrs)

        # Load V normally [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = V + offset_bh + offs_n[:, None] * stride_s + offs_d[None, :] * stride_d
        v = tl.load(v_ptrs)

        # Compute Q @ K^T
        qk = tl.dot(q, k) * scale

        # Masking: Apply causal mask using tl.where
        mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(mask, qk, -1e9)

        # FlashAttention running max and running sum updates
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(beta, axis=1)

        # Update accumulators
        p = beta.to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc += tl.dot(p, v)

        m_i = m_i_new
        l_i = l_i_new

    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    out_ptrs = Out + offset_bh + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d
    tl.store(out_ptrs, acc.to(tl.bfloat16))

def launch_kernel(q, k, v):
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Grid handles seq_len block, head, and batch
    grid = (triton.cdiv(S, BLOCK_M), H, B)
    scale = float(1.0 / (D ** 0.5))
    
    stride_b = q.stride(0)
    stride_h = q.stride(1)
    stride_s = q.stride(2)
    stride_d = q.stride(3)
    
    graphsynth_kernel[grid](
        q, k, v, out,
        scale,
        stride_b, stride_h, stride_s, stride_d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D
    )
    
    return out