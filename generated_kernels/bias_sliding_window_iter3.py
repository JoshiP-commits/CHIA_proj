import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_stride_bh, q_stride_s, q_stride_d,
    k_stride_bh, k_stride_s, k_stride_d,
    v_stride_bh, v_stride_s, v_stride_d,
    o_stride_bh, o_stride_s, o_stride_d,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    """
    Triton kernel for sliding-window causal attention.
    Each program instance computes a BLOCK_M x head_dim block of the output.
    """
    # 1. Get program IDs to identify the current work item
    pid_m = tl.program_id(0)  # ID for the query block dimension
    pid_bh = tl.program_id(1) # ID for the batch and head dimension

    # 2. Calculate pointers to the base of the current batch/head
    q_base_ptr = q_ptr + pid_bh * q_stride_bh
    k_base_ptr = k_ptr + pid_bh * k_stride_bh
    v_base_ptr = v_ptr + pid_bh * v_stride_bh
    o_base_ptr = o_ptr + pid_bh * o_stride_bh

    # 3. Initialize pointers and offsets for the current query block
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # 4. Initialize accumulators for the online softmax in high precision (fp32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)

    # 5. Load the current query block. It's constant for the inner loop.
    q_block_ptr = q_base_ptr + (offs_m[:, None] * q_stride_s) + (offs_d[None, :] * q_stride_d)
    # Apply mask for the last block if seq_len is not a multiple of BLOCK_M
    q_load_mask = offs_m[:, None] < seq_len
    q = tl.load(q_block_ptr, mask=q_load_mask, other=0.0)

    # Pre-scale Q for efficiency. The result of Q@K.T will be correctly scaled.
    sm_scale = (head_dim ** -0.5)
    q = (q * sm_scale).to(tl.bfloat16)

    # 6. OPTIMIZATION: Calculate loop bounds to iterate only over relevant KV blocks.
    # A query `i` attends to key `j` if (j <= i) and (i - j < WINDOW_SIZE).
    # This is equivalent to `i - WINDOW_SIZE < j <= i`.
    # For a query block `offs_m`, we find the min and max `j` to consider.
    # Smallest `j` is for smallest `i` (`start_m`): `start_m - WINDOW_SIZE + 1`.
    # Largest `j` is for largest `i` (`start_m + BLOCK_M - 1`): `start_m + BLOCK_M - 1`.
    # The loop should start at the beginning of the block containing the smallest `j`.
    loop_start_n = tl.maximum(0, start_m - WINDOW_SIZE + 1)
    loop_start_n = (loop_start_n // BLOCK_N) * BLOCK_N
    # The loop must end at or after the current query block for causality.
    loop_end_n = tl.minimum(start_m + BLOCK_M, seq_len)

    # 7. Loop over KV blocks within the calculated sliding window
    for start_n in range(loop_start_n, loop_end_n, BLOCK_N):
        # Offsets for the current key/value block
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_load_mask = offs_n[None, :] < seq_len
        v_load_mask = offs_n[:, None] < seq_len

        # Load K and V blocks
        k_block_ptr = k_base_ptr + (offs_n[None, :] * k_stride_s) + (offs_d[:, None] * k_stride_d)
        v_block_ptr = v_base_ptr + (offs_n[:, None] * v_stride_s) + (offs_d[None, :] * v_stride_d)
        k = tl.load(k_block_ptr, mask=k_load_mask, other=0.0)
        v = tl.load(v_block_ptr, mask=v_load_mask, other=0.0)

        # Compute scores S = Q @ K.T, accumulate in float32
        s = tl.dot(q, k, out_dtype=tl.float32)

        # Create and apply the combined sliding-window causal mask
        mask = (offs_m[:, None] >= offs_n[None, :]) & ((offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE)
        s = tl.where(mask, s, -float('inf'))

        # --- Online softmax update ---
        # Find max of current scores
        m_j = tl.max(s, axis=1)
        # Get new running max, safe for first iteration where m_i is -inf
        m_i_new = tl.maximum(m_i, m_j)
        # Compute weights for current block, P_j = exp(S_j - m_i_new)
        p_j = tl.exp(s - m_i_new[:, None])
        # Compute sum of weights for current block, l_j = sum(P_j)
        l_j = tl.sum(p_j, axis=1)
        
        # Rescale old accumulator and running sum to match the new max
        scale = tl.exp(m_i - m_i_new)
        acc = acc * scale[:, None]
        l_i = l_i * scale

        # Update accumulator with new values: acc += P_j @ V_j
        p_j = p_j.to(tl.bfloat16) # Cast probabilities to match V's dtype for dot product
        acc += tl.dot(p_j, v)
        
        # Update running sum and max
        l_i += l_j
        m_i = m_i_new
        # --- End of online softmax update ---

    # 8. Final normalization and store to output
    # Invert the final sum of weights to multiply instead of divide
    l_i_inv = 1.0 / l_i
    # Handle cases where l_i is zero (e.g., all-masked rows) to avoid NaN
    l_i_inv = tl.where(l_i == 0, 0.0, l_i_inv)
    acc = acc * l_i_inv[:, None]

    # Write the final normalized block to the output tensor
    o_block_ptr = o_base_ptr + (offs_m[:, None] * o_stride_s) + (offs_d[None, :] * o_stride_d)
    tl.store(o_block_ptr, acc.to(tl.bfloat16), mask=q_load_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launch function for the sliding-window causal attention kernel.

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, S, D].
        k (torch.Tensor): Key tensor of shape [B, H, S, D].
        v (torch.Tensor): Value tensor of shape [B, H, S, D].

    Returns:
        torch.Tensor: Output tensor of the same shape as inputs.
    """
    B, H, S, D = q.shape
    assert q.shape == k.shape == v.shape, "Input tensors must have the same shape"
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "Inputs must be bfloat16"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on GPU"
    assert D == 128 and S == 2048, "Shape must be [2, 32, 2048, 128]"

    # Create an output tensor of the same shape and dtype
    o = torch.empty_like(q)

    # Triton kernels generally require contiguous tensors
    q, k, v, o = [x.contiguous() for x in (q, k, v, o)]
    
    # Reshape inputs to merge Batch and Head dimensions for the kernel grid
    q_r, k_r, v_r, o_r = [x.view(B * H, S, D) for x in (q, k, v, o)]

    # Kernel configuration constants
    BLOCK_M = 128
    BLOCK_N = 64
    WINDOW_SIZE = 256
    num_warps = 4
    num_stages = 3

    # Define the grid for launching the kernel
    # Each program in the grid computes one BLOCK_M slice of the output
    grid = (triton.cdiv(S, BLOCK_M), B * H)

    # Launch the kernel
    graphsynth_kernel[grid](
        q_r, k_r, v_r, o_r,
        q_r.stride(0), q_r.stride(1), q_r.stride(2),
        k_r.stride(0), k_r.stride(1), k_r.stride(2),
        v_r.stride(0), v_r.stride(1), v_r.stride(2),
        o_r.stride(0), o_r.stride(1), o_r.stride(2),
        seq_len=S,
        head_dim=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        WINDOW_SIZE=WINDOW_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o