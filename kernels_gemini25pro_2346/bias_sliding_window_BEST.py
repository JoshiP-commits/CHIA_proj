import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to matrices
    Q, K, V, O,
    # Strides for tensors
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Other parameters
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    """
    Triton kernel for sliding-window causal attention, inspired by FlashAttention.

    Grid: (num_m_blocks, num_heads, num_batches)
    - num_m_blocks: ceil(N_CTX / BLOCK_M)
    - num_heads: Number of attention heads
    - num_batches: Batch size

    Each program instance computes one BLOCK_M x D_HEAD output block for one
    head and one batch item.
    """
    # 1. Get program IDs to identify the current block/head/batch
    pid_m = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    # 2. Setup pointers for the Q block
    # Offsets for the M-dimension (sequence length for Q)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Offsets for the D-dimension (head dimension)
    offs_d = tl.arange(0, D_HEAD)
    
    # Pointers to the current Q block
    q_ptrs = Q + (pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)

    # 3. Initialize accumulators and running statistics for softmax
    # Running max for stable softmax, initialized to a very small number
    m_i = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)
    # Running sum for softmax denominator
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Output accumulator, initialized to zeros
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # 4. Load the Q block. This remains constant throughout the loop.
    q = tl.load(q_ptrs)

    # 5. Loop over blocks of K and V.
    # The loop upper bound is `(pid_m + 1) * BLOCK_M` to enforce causality.
    # Keys with index `j > i` are not attended to.
    for start_n in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        # -- a. Setup pointers for K and V blocks --
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Pointers to the K block, loaded in a transposed layout for `tl.dot`.
        # Shape of loaded K will be [D_HEAD, BLOCK_N]
        k_ptrs = K + (pid_b * stride_kb + pid_h * stride_kh + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn)
        
        # Pointers to the V block. Shape of loaded V will be [BLOCK_N, D_HEAD]
        v_ptrs = V + (pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

        # -- b. Load K and V with boundary checks --
        # Use a mask to avoid loading out-of-bounds data for the last block.
        k = tl.load(k_ptrs, mask=(offs_n[None, :] < N_CTX), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_CTX), other=0.0)

        # -- c. Compute scores S = (Q @ K^T) * scale --
        s_ij = tl.dot(q, k)
        s_ij *= SCALE

        # -- d. Apply causal and sliding window mask --
        # `offs_m` are the query indices, `offs_n` are the key indices.
        # Mask is True for elements where `j <= i` AND `i - j < WINDOW_SIZE`.
        mask = (offs_m[:, None] >= offs_n[None, :]) & ((offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE)
        
        # Apply mask by setting masked-out elements to a large negative number.
        s_ij = tl.where(mask, s_ij, -1e9)

        # -- e. Perform FlashAttention online softmax update --
        # Find the new running max for the row
        m_j = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_j)
        
        # Calculate probabilities for the current block, scaled by the new max
        p_ij = tl.exp(s_ij - m_new[:, None])
        
        # Rescale the old accumulator and denominator based on the change in max
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # Update the accumulator with the new values
        p_ij_casted = p_ij.to(Q.dtype.element_ty)
        acc = tl.dot(p_ij_casted, v, acc)
        
        # Update the denominator
        l_j = tl.sum(p_ij, axis=1)
        l_i += l_j
        
        # Update the running max for the next iteration
        m_i = m_new

    # 6. Post-loop normalization and write to output
    # Normalize the accumulator by the final denominator.
    l_i_inv = 1.0 / (l_i + 1e-6)
    acc = acc * l_i_inv[:, None]
    
    # Pointers to the output block
    o_ptrs = O + (pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od)
    
    # Write the final result.
    tl.store(o_ptrs, acc.to(O.dtype.element_ty))


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Lauches the Triton kernel for a custom sliding-window causal attention operation.
    
    Args:
        q: Query tensor of shape [2, 32, 2048, 128], dtype bfloat16.
        k: Key tensor of shape [2, 32, 2048, 128], dtype bfloat16.
        v: Value tensor of shape [2, 32, 2048, 128], dtype bfloat16.

    Returns:
        Output tensor of shape [2, 32, 2048, 128], dtype bfloat16.
    """
    # Input validation
    assert all(t.is_cuda and t.dtype == torch.bfloat16 for t in [q, k, v])
    assert q.shape == k.shape == v.shape and q.shape == (2, 32, 2048, 128)
    
    B, H, N_CTX, D_HEAD = q.shape
    
    # Create the output tensor
    o = torch.empty_like(q)
    
    # Define block sizes for the kernel
    BLOCK_M = 128
    BLOCK_N = 64
    WINDOW_SIZE = 256
    
    # Pre-calculate the scale factor as a float on the host
    scale = 1.0 / math.sqrt(D_HEAD)

    # Define the grid for the kernel launch
    # Each program computes one output block of size BLOCK_M x D_HEAD
    grid = (triton.cdiv(N_CTX, BLOCK_M), H, B)

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N_CTX=N_CTX,
        D_HEAD=D_HEAD,
        SCALE=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        WINDOW_SIZE=WINDOW_SIZE,
    )
    
    return o