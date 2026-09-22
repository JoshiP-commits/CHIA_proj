import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to matrices
    Q, K, V, O,
    # Strides
    s_qb, s_qh, s_qm, s_qd,
    s_kb, s_kh, s_kn, s_kd,
    s_vb, s_vh, s_vn, s_vd,
    s_ob, s_oh, s_om, s_od,
    # Head and Context dimensions
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    # Operator scale
    scale: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    """
    Triton kernel for sliding window attention on SM80.
    Each program instance computes a BLOCK_M x D_HEAD block of the output.
    
    Grid: (cdiv(N_CTX, BLOCK_M), B * H)
    
    Masking Logic:
    - Causal: j <= i
    - Sliding Window: i - j < WINDOW_SIZE
    - Implemented by iterating over all K/V blocks and applying a combined
      mask to the score matrix block before the softmax update. This is the
      idiomatic Triton approach that avoids dynamic loops or divergent branches.
    """
    # 1. Get program IDs to identify the current block
    pid_m = tl.program_id(axis=0)  # Block index along the sequence length dimension
    pid_bh = tl.program_id(axis=1) # Combined batch and head index

    # Decode batch and head indices from the combined program ID
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # 2. Compute pointers to the start of the current block
    # `start_m` is the row index of the first query in this block
    start_m = pid_m * BLOCK_M
    
    # `offs_m` are the row indices (sequence positions) for the Q block
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # `offs_d` are the column indices (head dimension)
    offs_d = tl.arange(0, D_HEAD)

    # Pointers to the base of the current batch/head
    q_base = Q + pid_b * s_qb + pid_h * s_qh
    k_base = K + pid_b * s_kb + pid_h * s_kh
    v_base = V + pid_b * s_vb + pid_h * s_vh
    o_base = O + pid_b * s_ob + pid_h * s_oh

    # 3. Load the Q block from DRAM into SRAM
    # Pointers for the [BLOCK_M, D_HEAD] block of Q
    q_ptrs = q_base + (offs_m[:, None] * s_qm + offs_d[None, :] * s_qd)
    # Mask for padding; `offs_m` must not exceed sequence length
    q_mask = offs_m[:, None] < N_CTX
    # Load Q. Padded elements will be 0.
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 4. Initialize accumulators for the online softmax
    # `acc`: Stores the running sum of (p_ij * v_j) in fp32
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    # `m_i`: Stores the running max of s_ij
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    # `l_i`: Stores the running sum of exp(s_ij - m_i)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # 5. Loop over blocks of K and V
    # Loop over the entire sequence length. Masking inside the loop handles
    # out-of-window and causal blocks efficiently.
    for start_n in range(0, N_CTX, BLOCK_N):
        # `offs_n` are the row indices (sequence positions) for the K/V block
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 6. Load K and V blocks
        # Load K with swapped strides for a transposed dot product (q @ k^T)
        # Pointers for a [D_HEAD, BLOCK_N] block of K
        k_ptrs = k_base + (offs_d[:, None] * s_kd + offs_n[None, :] * s_kn)
        # Pointers for a [BLOCK_N, D_HEAD] block of V
        v_ptrs = v_base + (offs_n[:, None] * s_vn + offs_d[None, :] * s_vd)
        
        # Mask for padding; `offs_n` must not exceed sequence length
        kv_mask_n = offs_n[None, :] < N_CTX
        
        # Load K, ensuring transposed shape for dot product without trans_b
        k = tl.load(k_ptrs, mask=kv_mask_n, other=0.0)
        # Load V, mask needs to match V's shape for loading
        v = tl.load(v_ptrs, mask=kv_mask_n.T, other=0.0)

        # 7. Compute score block S = (Q @ K^T) * scale
        s = tl.dot(q, k)
        s *= scale

        # 8. Apply causal and sliding window mask
        # `mask` is True for elements we want to keep.
        # Condition 1: Causal (j <= i)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        # Condition 2: Sliding Window (i - j < WINDOW_SIZE)
        window_mask = (offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE
        # Condition 3: Sequence Padding (j < N_CTX)
        padding_mask = offs_n[None, :] < N_CTX
        
        mask = causal_mask & window_mask & padding_mask

        # Apply mask to scores, setting masked elements to -1e9 for softmax
        s = tl.where(mask, s, -1e9)
        
        # 9. Online softmax update (FlashAttention algorithm)
        # Find the max of the current score block
        m_ij = tl.max(s, axis=1)
        # Get the new running max
        m_new = tl.maximum(m_i, m_ij)
        
        # Rescale previous accumulator and sum of exps using the new max
        alpha = tl.math.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # Compute probabilities for the current block, p_ij = exp(s_ij - m_new)
        beta = tl.math.exp(s - m_new[:, None])
        
        # Update sum of exps
        l_i += tl.sum(beta, axis=1)
        
        # Update accumulator: acc += p_ij @ v_j
        # Cast beta to V's dtype for the dot product
        acc += tl.dot(beta.to(v.dtype), v)
        
        # Update running max for the next iteration
        m_i = m_new

    # 10. Finalize and store the output block
    # Rescale accumulator by the reciprocal of the total sum of exps
    l_i_reciprocal = 1.0 / l_i
    acc = acc * l_i_reciprocal[:, None]
    
    # Pointers to the output block
    o_ptrs = o_base + (offs_m[:, None] * s_om + offs_d[None, :] * s_od)
    # Store the final result, applying the padding mask and casting to bf16
    tl.store(o_ptrs, acc.to(o.dtype.element_ty), mask=q_mask)

def launch_kernel(q, k, v):
    """
    Launches the Triton kernel for a sliding-window attention operation.

    Args:
        q (torch.Tensor): Query tensor of shape [2, 32, 2048, 128], dtype=bfloat16.
        k (torch.Tensor): Key tensor of shape [2, 32, 2048, 128], dtype=bfloat16.
        v (torch.Tensor): Value tensor of shape [2, 32, 2048, 128], dtype=bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype as inputs.
    """
    B, H, N_CTX, D_HEAD = q.shape
    
    # Create the output tensor on the correct device
    o = torch.empty_like(q)

    # Define block sizes for the kernel. These are hardcoded for this specific
    # problem but could be tuned for performance.
    BLOCK_M = 128
    BLOCK_N = 64
    WINDOW_SIZE = 256 # As specified in the operator definition

    # Set up the grid for the kernel launch.
    # Each program instance computes one BLOCK_M-sized block of rows of the output.
    # The grid is 2D: (number of blocks in N_CTX) x (batch_size * num_heads)
    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)
    
    # Calculate scale factor on the host as a float
    scale = 1.0 / math.sqrt(D_HEAD)

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        # Strides are passed to enable memory-coalesced access
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Kernel parameters
        H=H,
        N_CTX=N_CTX,
        D_HEAD=D_HEAD,
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        WINDOW_SIZE=WINDOW_SIZE,
    )

    return o