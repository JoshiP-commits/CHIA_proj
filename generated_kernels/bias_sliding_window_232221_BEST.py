import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    Q, K, V, Out,
    # Stride useful for indexing
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Other parameters
    Z, H, N_CTX,
    # Kernels arguments
    sm_scale: float,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton kernel for sliding-window causal attention.
    Each program instance computes a BLOCK_M x D_HEAD block of the output.
    """
    # Program IDs
    pid_bh = tl.program_id(0)  # Index for batch and head axis
    pid_m = tl.program_id(1)   # Index for sequence axis (query block)

    # De-mux batch and head indices
    pid_z = pid_bh // H
    pid_h = pid_bh % H

    # Offset pointers to the correct batch and head
    q_ptr = Q + pid_z * stride_qz + pid_h * stride_qh
    k_ptr = K + pid_z * stride_kz + pid_h * stride_kh
    v_ptr = V + pid_z * stride_vz + pid_h * stride_vh
    o_ptr = Out + pid_z * stride_oz + pid_h * stride_oh

    # Compute offsets for the current query block
    start_m = pid_m * BLOCK_M
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load query block
    q_ptrs = q_ptr + (start_m + offs_m[:, None]) * stride_qm + offs_d[None, :] * stride_qk
    # Mask to avoid loading out-of-bounds queries
    q_mask = (start_m + offs_m[:, None]) < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators for online softmax
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Initialize running max to a very small number
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)

    # Scale queries
    q = (q * sm_scale).to(tl.bfloat16)

    # --- Optimized loop over key/value blocks ---
    # Compute the range of KV blocks that this query block can attend to.
    # Lower bound: first key block that overlaps with the sliding window.
    # Upper bound: last key block that is causally before the end of the query block.
    start_k_idx = (start_m - WINDOW_SIZE + 1) // BLOCK_N
    # The range start must be non-negative.
    start_k_idx = max(start_k_idx, 0)
    end_k_idx = (start_m + BLOCK_M - 1) // BLOCK_N + 1
    
    # Loop over key/value blocks.
    for k_block_idx in range(start_k_idx, end_k_idx):
        start_n = k_block_idx * BLOCK_N
        offs_n = tl.arange(0, BLOCK_N)

        # --- Load K and V blocks ---
        # Load K transposed (D_HEAD x BLOCK_N)
        k_trans_ptrs = k_ptr + offs_d[:, None] * stride_kk + (start_n + offs_n[None, :]) * stride_kn
        # Load V (BLOCK_N x D_HEAD)
        v_ptrs = v_ptr + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        
        # Mask for loading K and V to handle sequence length
        kv_mask = (start_n + offs_n) < N_CTX
        k = tl.load(k_trans_ptrs, mask=kv_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        
        # --- Compute scores (S = Q @ K^T) ---
        # q: [BLOCK_M, D_HEAD], k: [D_HEAD, BLOCK_N] -> s: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k)

        # --- Apply causal and sliding window mask ---
        i_indices = start_m + offs_m
        j_indices = start_n + offs_n
        
        causal_mask = i_indices[:, None] >= j_indices[None, :]
        window_mask = (i_indices[:, None] - j_indices[None, :]) < WINDOW_SIZE
        # Combine masks. Also mask out padding tokens in keys.
        padding_mask = j_indices[None, :] < N_CTX
        mask = causal_mask & window_mask & padding_mask

        # Apply mask to scores before exponentiation
        s = tl.where(mask, s, -float('inf'))
        
        # --- Online softmax update ---
        # Get new max of scores for this block
        m_ij = tl.max(s, axis=1)
        # Compute new running max
        m_new = tl.maximum(m_i, m_ij)
        
        # Rescale current scores with new max
        s_exp = tl.exp(s - m_new[:, None])
        p = s_exp
        
        # Rescale old accumulator and running sum of exponents
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # Update accumulator with current block's contribution
        # p must be cast to bfloat16 for `dot` on SM80
        acc = acc + tl.dot(p.to(tl.bfloat16), v)
        
        # Update running sum of exponents
        l_i = l_i + tl.sum(p, axis=1)
        
        # Update running max
        m_i = m_new

    # --- Finalize output ---
    # Normalize accumulator by the sum of exponents
    # Add a small epsilon to avoid division by zero
    acc = acc / (l_i[:, None] + 1e-8)

    # Cast to bfloat16 and store to output tensor
    out_ptrs = o_ptr + (start_m + offs_m[:, None]) * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=q_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launch the Triton kernel for sliding-window causal attention.

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, S, D]
        k (torch.Tensor): Key tensor of shape [B, H, S, D]
        v (torch.Tensor): Value tensor of shape [B, H, S, D]

    Returns:
        torch.Tensor: Output tensor of shape [B, H, S, D]
    """
    # Ensure inputs are bfloat16 and on CUDA
    assert all(t.dtype == torch.bfloat16 for t in [q, k, v])
    assert all(t.is_cuda for t in [q, k, v])

    # Shape parameters
    Z, H, N_CTX, D_HEAD = q.shape
    
    # Output tensor
    out = torch.empty_like(q)

    # Kernel constants
    # Using powers of 2 for block sizes is generally good for performance
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = D_HEAD
    WINDOW_SIZE = 256
    
    # Grid dimensions
    # Each program computes one query block for one head in one batch
    grid = (Z * H, triton.cdiv(N_CTX, BLOCK_M))

    # Pre-calculate scaling factor
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    # Launch kernel
    graphsynth_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        sm_scale=sm_scale,
        WINDOW_SIZE=WINDOW_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        # Using num_warps=4 is a good default
        num_warps=4,
        # Using num_stages > 2 can improve performance by hiding memory latency
        num_stages=3,
    )
    
    return out