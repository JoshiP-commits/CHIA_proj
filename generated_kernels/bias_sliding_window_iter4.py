import torch
import triton
import triton.language as tl

# This is the single Python code block required by the prompt.

@triton.jit
def graphsynth_kernel(
    # Pointers to matrices
    Q, K, V, O,
    # Stride variables for tensors
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Matrix dimensions
    Z, H, N_CTX,
    # Kernel constants
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    """
    Triton kernel for sliding-window causal attention.

    This kernel implements a FlashAttention-style algorithm to compute attention
    scores for a given query (Q), key (K), and value (V) tensor.

    Key Features:
    - Online Softmax: Avoids materializing the full (N_CTX, N_CTX) score matrix,
      saving memory and bandwidth. Accumulation is done in float32 for precision.
    - Tiling: Q is processed in blocks of size (BLOCK_M, D_HEAD), and the inner loop
      iterates over blocks of K and V of size (BLOCK_N, D_HEAD).
    - Sliding-Window Causal Masking:
        1. Causal: A query at index `i` can only attend to keys at index `j <= i`.
        2. Sliding Window: A query at `i` can only attend to `j` where `i - j < WINDOW_SIZE`.
    - Optimized Block Skipping: Instead of naively looping over all possible key/value blocks,
      this kernel calculates the valid range of KV blocks for each query block based on the
      sliding-window and causal constraints. It then predicates the entire loop body,
      ensuring that memory loads and computations are only performed for blocks that
      can contain unmasked elements. This is crucial for performance.
    """
    # 1. Get program IDs to identify the current work item
    # We parallelize over batch*heads (dim 0) and query sequence length (dim 1)
    pid_bh = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # Unpack batch and head indices
    pid_z = pid_bh // H
    pid_h = pid_bh % H

    # 2. Compute pointers to the start of the current head's Q, K, V matrices
    q_offset = pid_z * stride_qz + pid_h * stride_qh
    k_offset = pid_z * stride_kz + pid_h * stride_kh
    v_offset = pid_z * stride_vz + pid_h * stride_vh
    o_offset = pid_z * stride_oz + pid_h * stride_oh

    Q_ptr = Q + q_offset
    K_ptr = K + k_offset
    V_ptr = V + v_offset
    O_ptr = O + o_offset

    # 3. Initialize pointers, accumulators, and online softmax statistics
    # `start_m` is the row index of the first query in the current block
    start_m = pid_m * BLOCK_M
    
    # Per-query-block range for K/V. `offs_m` are the row indices for the current Q block.
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # Per-K/V-block range for K/V. `offs_d` are the column indices for the head dimension.
    offs_d = tl.arange(0, D_HEAD)

    # Online softmax statistics:
    # `m_i`: running max of scores
    # `l_i`: running sum of exponentiated scores
    # `acc`: accumulator for the output
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # 4. Load the current block of queries
    # This block of Q is held in SRAM and reused for all K/V blocks.
    q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    # Masking for the last block of queries if N_CTX is not a multiple of BLOCK_M
    mask_m = offs_m < N_CTX
    q = tl.load(q_ptrs, mask=mask_m[:, None])
    
    # Scale factor for dot products
    sm_scale = tl.math.rsqrt(D_HEAD.to(tl.float32))

    # 5. Main loop over key and value blocks (FlashAttention-style)
    # This loop must have static bounds for Triton to compile.
    # The optimization is done by predicating the entire loop body with an `if`.
    for j in range(0, tl.cdiv(N_CTX, BLOCK_N)):
        start_n = j * BLOCK_N

        # 6. OPTIMIZATION: Check if the current KV block is relevant
        # A KV block is relevant if it can contain keys that are visible to
        # at least one query in the current Q block.
        # Condition 1 (Causality): The KV block must start before the end of the Q block.
        # `start_n < start_m + BLOCK_M`
        # Condition 2 (Sliding Window): The KV block must not be too far in the past.
        # Its end must be after the start of the window for the first query in the Q block.
        # `(start_n + BLOCK_N) > (start_m - WINDOW_SIZE)`
        # The entire loop body is wrapped in this `if` to skip non-relevant blocks.
        # This avoids the unsupported `continue` statement and allows the compiler
        # to predicate the expensive operations.
        if (start_n < start_m + BLOCK_M) and ((start_n + BLOCK_N) > (start_m - WINDOW_SIZE)):
            # 7. Load K and V blocks
            offs_n = start_n + tl.arange(0, BLOCK_N)
            
            k_ptrs = K_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
            v_ptrs = V_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

            # Masking for the last block of K/V if N_CTX is not a multiple of BLOCK_N
            mask_n = offs_n < N_CTX
            k = tl.load(k_ptrs, mask=mask_n[:, None])
            v = tl.load(v_ptrs, mask=mask_n[:, None])

            # 8. Compute attention scores (Q @ K^T)
            s_ij = tl.dot(q, tl.trans(k)) * sm_scale

            # 9. Apply combined sliding-window causal mask
            # Create a matrix of query indices `i` and key indices `j`
            i = offs_m[:, None]
            j = offs_n[None, :]
            
            # The mask is true for elements where `j <= i` AND `i - j < WINDOW_SIZE`
            causal_mask = (i >= j)
            window_mask = ((i - j) < WINDOW_SIZE)
            full_mask = causal_mask & window_mask

            # Apply mask by setting masked-out elements to negative infinity
            s_ij = tl.where(full_mask, s_ij, -float('inf'))

            # 10. Online softmax update
            # `m_j`: current block's max score
            # `p_ij`: current block's exponentiated scores (unnormalized)
            # `l_j`: current block's sum of exponentiated scores
            m_j = tl.max(s_ij, axis=1)
            p_ij = tl.exp(s_ij - m_j[:, None])
            l_j = tl.sum(p_ij, axis=1)

            # Find the new running max
            m_new = tl.maximum(m_i, m_j)
            
            # Rescale previous accumulator and sum
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_j - m_new)
            
            l_new = alpha * l_i + beta * l_j
            
            # Rescale and update the output accumulator
            # This is the core of the online softmax algorithm
            acc = acc * (alpha / l_new)[:, None]
            
            # Compute `p_ij @ v` and add its contribution to the accumulator
            p_ij_v = tl.dot(p_ij.to(v.dtype), v) # Cast p_ij to bf16 to use TC
            acc += (beta / l_new)[:, None] * p_ij_v

            # Update running statistics
            l_i = l_new
            m_i = m_new

    # 11. Finalize and store the output
    # The accumulator `acc` now holds the correctly normalized output.
    # No final division by `l_i` is needed because it's part of the online update.
    
    # Cast to output dtype (bfloat16) before storing
    acc = acc.to(O.dtype.element_ty)
    
    # Pointers to the output block
    o_ptrs = O_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc, mask=mask_m[:, None])


def launch_kernel(q, k, v):
    """
    Launcher function for the sliding-window causal attention kernel.

    Args:
        q (torch.Tensor): Query tensor of shape [Z, H, N_CTX, D_HEAD].
        k (torch.Tensor): Key tensor of shape [Z, H, N_CTX, D_HEAD].
        v (torch.Tensor): Value tensor of shape [Z, H, N_CTX, D_HEAD].

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype as inputs.
    """
    # 1. Ensure inputs are on the correct device and have the right dtype
    assert all(t.is_cuda and t.dtype == torch.bfloat16 for t in [q, k, v])
    assert q.shape == k.shape == v.shape
    assert q.shape == (2, 32, 2048, 128)

    # 2. Get tensor dimensions and strides
    Z, H, N_CTX, D_HEAD = q.shape
    
    # 3. Create the output tensor
    o = torch.empty_like(q)

    # 4. Define kernel parameters. These are tuned for A100.
    # A block of 128 queries processes the sequence in chunks of 64 keys/values.
    BLOCK_M = 128
    BLOCK_N = 64
    WINDOW_SIZE = 256
    num_warps = 4
    num_stages = 3

    # 5. Define the grid for launching the kernel.
    # Each program instance computes one block of `BLOCK_M` queries for one head.
    grid = (Z * H, triton.cdiv(N_CTX, BLOCK_M))

    # 6. Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        WINDOW_SIZE=WINDOW_SIZE,
        num_warps=num_warps,
        num_stages=num_stages
    )

    return o