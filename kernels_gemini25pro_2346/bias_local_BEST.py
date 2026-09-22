import torch
import triton
import triton.language as tl
import math


@triton.jit
def graphsynth_kernel(
    # Pointers to tensors
    Q, K, V, O,
    start_n_indices, end_n_indices,
    # Stride variables for tensors
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    # Other parameters
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    scale: tl.constexpr,
    # Block constants
    BANDWIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Computes banded attention using a FlashAttention-style tiling algorithm.

    Each program operates on a single query block for a specific batch and head.
    The kernel loops over key/value blocks that overlap with the attention band,
    with loop bounds pre-calculated on the host to optimize performance.
    """
    # 1. Get program IDs and calculate offsets for the current work item.
    pid_m = tl.program_id(0)  # ID for the query block in the sequence.
    pid_bh = tl.program_id(1) # ID for the batch and head, flattened.

    # Unpack batch and head indices from the flattened ID.
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    start_m = pid_m * BLOCK_M

    # 2. Load loop bounds pre-calculated on the host.
    # This determines which KV blocks to visit for the current query block.
    start_n_block_idx = tl.load(start_n_indices + pid_m)
    end_n_block_idx = tl.load(end_n_indices + pid_m)

    # 3. Initialize pointers and FlashAttention accumulators.
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh

    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # 4. Load the Q block (it's constant throughout the inner loop).
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 5. Main loop over relevant K and V blocks.
    # A `while` loop is used for dynamic loop bounds loaded from the host.
    n_block_idx = start_n_block_idx
    while n_block_idx < end_n_block_idx:
        start_n = n_block_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block, transposing it on the fly for the dot product.
        # This is required because tl.dot does not have a `trans_b` argument.
        k_ptrs = k_ptr + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
        k_mask = offs_n[None, :] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute scores: S = (Q @ K^T) * scale
        s_ij = tl.dot(q, k)
        s_ij *= scale

        # Apply the bidirectional band attention mask via tl.where.
        # A query i can attend to a key j only if abs(i-j) < BANDWIDTH.
        dist_mask = tl.abs(offs_m[:, None] - offs_n[None, :]) < BANDWIDTH

        # Also apply sequence boundary mask to prevent out-of-bounds attention.
        full_mask = dist_mask & (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        s_ij = tl.where(full_mask, s_ij, -1e9)

        # --- FlashAttention online softmax update ---
        # Find the new running maximum of scores.
        m_j = tl.max(s_ij, 1)
        m_new = tl.maximum(m_i, m_j)

        # Rescale the previous accumulator and sum of exponents based on the new max.
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # Compute probabilities for the current block and update the sum of exponents.
        p_ij = tl.exp(s_ij - m_new[:, None])
        l_i += tl.sum(p_ij, 1)

        # Load the V block.
        v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = offs_n[:, None] < N_CTX
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Update the accumulator with the weighted values from the current block.
        acc += tl.dot(p_ij.to(V.dtype.element_ty), v)

        # Update the running maximum for the next iteration.
        m_i = m_new
        n_block_idx += 1

    # 6. Finalize and store the output block.
    # Normalize the accumulator by the final sum of exponents.
    # A safe reciprocal is used to prevent division by zero.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    l_i_inv = 1.0 / l_i_safe
    out = acc * l_i_inv[:, None]

    # Write the final output block to global memory.
    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=q_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the GraphSynth attention kernel.

    Args:
        q: Query tensor of shape [2, 32, 2048, 128], dtype=bfloat16.
        k: Key tensor of shape [2, 32, 2048, 128], dtype=bfloat16.
        v: Value tensor of shape [2, 32, 2048, 128], dtype=bfloat16.

    Returns:
        Output tensor of the same shape and dtype.
    """
    # 1. Input validation and shape extraction.
    assert q.shape == (2, 32, 2048, 128) and k.shape == (2, 32, 2048, 128) and v.shape == (2, 32, 2048, 128)
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert q.is_cuda and k.is_cuda and v.is_cuda

    B, H, N_CTX, D_HEAD = q.shape
    o = torch.empty_like(q)

    # 2. Kernel configuration and constants.
    BLOCK_M = 64
    BLOCK_N = 64
    BANDWIDTH = 256
    scale = 1.0 / math.sqrt(D_HEAD)

    # 3. Grid definition. The grid is 2D, with one program per query block per head/batch.
    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)

    # 4. Pre-calculate loop bounds on the host for each query block.
    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    num_n_blocks = triton.cdiv(N_CTX, BLOCK_N)
    
    start_n_indices = torch.empty(num_m_blocks, dtype=torch.int32, device=q.device)
    end_n_indices = torch.empty(num_m_blocks, dtype=torch.int32, device=q.device)

    for m_block_idx in range(num_m_blocks):
        q_start_pos = m_block_idx * BLOCK_M
        
        # Calculate the min and max key indices this query block can possibly attend to.
        # The attention condition is abs(i-j) < BANDWIDTH.
        # For a block of queries, we take the union of their attention windows.
        k_min_pos = q_start_pos - BANDWIDTH + 1
        k_max_pos = (q_start_pos + BLOCK_M - 1) + BANDWIDTH - 1
        
        # Convert absolute key positions to key block indices.
        start_n_block = k_min_pos // BLOCK_N
        # The end block index is exclusive, so we find the block containing k_max_pos and add 1.
        end_n_block = (k_max_pos // BLOCK_N) + 1
        
        # Clamp indices to the valid range [0, num_n_blocks].
        start_n_indices[m_block_idx] = max(0, start_n_block)
        end_n_indices[m_block_idx] = min(num_n_blocks, end_n_block)

    # 5. Launch the kernel.
    graphsynth_kernel[grid](
        q, k, v, o,
        start_n_indices, end_n_indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H=H,
        N_CTX=N_CTX,
        D_HEAD=D_HEAD,
        scale=scale,
        BANDWIDTH=BANDWIDTH,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,   # Recommended for good performance on A100.
        num_stages=3,  # Helps hide memory latency by pre-fetching data.
    )

    return o