import torch
import triton
import triton.language as tl


@triton.jit
def graphsynth_kernel(
    # Pointers to matrices
    Q, K, V, Out,
    # Stride variables for tensors
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Other metadata
    Z, H, N_CTX,
    # Heuristics and compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    """
    Computes sliding-window causal attention using a FlashAttention-style algorithm.
    Each program instance processes one BLOCK_M x BLOCK_DMODEL block of queries.
    """
    # 1. Get program IDs to identify the current batch, head, and query block.
    # The grid is 2D: (Z * H, N_CTX / BLOCK_M)
    start_m = tl.program_id(1) * BLOCK_M
    off_zh = tl.program_id(0)
    off_z = off_zh // H
    off_h = off_zh % H

    # 2. Compute pointers to the start of the Q, K, V matrices for the current batch/head.
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    Q_ptr = Q + q_offset
    K_ptr = K + k_offset
    V_ptr = V + v_offset

    # 3. Initialize accumulators for the online softmax.
    # `acc` stores the running sum of values, weighted by softmax probabilities.
    # We use float32 for accumulation to maintain precision.
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # `l_i` stores the running sum of the softmax denominator.
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # `m_i` stores the running maximum of the scores, for numerical stability.
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)

    # 4. Load the block of queries. This is constant across the inner loop.
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Mask to prevent out-of-bounds access for the last query block.
    q_mask = offs_m[:, None] < N_CTX
    # Load queries, masking out-of-bounds elements to 0.0.
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 5. Initialize scaling factor for the dot product.
    sm_scale = 1.0 / (BLOCK_DMODEL**0.5)

    # 6. Loop over blocks of keys and values. This is the core of FlashAttention.
    # The loop iterates causally, only up to the current query block.
    for start_n in range(0, start_m + BLOCK_M, BLOCK_N):
        # 7. OPTIMIZATION: Early exit for KV blocks outside the sliding window.
        # A KV block is entirely outside the window if for all queries `i` in the
        # current query block and all keys `j` in the current key block, `i - j >= WINDOW_SIZE`.
        # The minimum value of `(i - j)` in a tile is `min(i) - max(j)`.
        # `min(i)` is `start_m`. `max(j)` is `start_n + BLOCK_N - 1`.
        # If `start_m - (start_n + BLOCK_N - 1) >= WINDOW_SIZE`, we can safely skip the block.
        if (start_m - start_n - BLOCK_N + 1) >= WINDOW_SIZE:
            continue

        # 8. Load the current block of keys (K_j).
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k_mask = offs_n[None, :] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # 9. Compute scores: S_ij = (Q_i @ K_j^T) / sqrt(d_k).
        s_ij = tl.dot(q, k) * sm_scale

        # 10. Apply the combined causal and sliding-window mask.
        # A score is valid if `j <= i` (causal) AND `i - j < WINDOW_SIZE` (sliding window).
        mask = (offs_m[:, None] >= offs_n[None, :]) & (offs_m[:, None] - offs_n[None, :] < WINDOW_SIZE)
        s_ij = tl.where(mask, s_ij, -float("inf"))

        # 11. Perform the online softmax update.
        # Find the max of the current scores tile.
        m_ij = tl.max(s_ij, axis=1)
        # Find the new overall running max.
        m_i_new = tl.maximum(m_i, m_ij)
        # Rescale the previous accumulator and denominator based on the new max.
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        # Compute the softmax probabilities for the current tile, p_ij.
        p_ij = tl.exp(s_ij - m_i_new[:, None])
        # Update the denominator sum, l_i.
        l_ij = tl.sum(p_ij, axis=1)
        l_i += l_ij

        # 12. Load the corresponding block of values (V_j).
        v_ptrs = V_ptr + offs_n[None, :] * stride_vn + offs_d[:, None] * stride_vk
        # k_mask has the same dimensions needed for V, so we can reuse it.
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # 13. Update the accumulator with the weighted values.
        # The probabilities p_ij must be cast to the same dtype as V for the dot product.
        acc += tl.dot(p_ij.to(V.dtype.element_ty), v)

        # 14. Update the running max for the next iteration.
        m_i = m_i_new

    # 15. After the loop, normalize the accumulator.
    # Handle the case where l_i is 0 (e.g., for padding tokens) to avoid division by zero.
    l_i_reciprocal = 1.0 / l_i
    l_i_reciprocal = tl.where(l_i > 0, l_i_reciprocal, 0.0)
    acc = acc * l_i_reciprocal[:, None]

    # 16. Write the final output block to global memory.
    out_offset = off_z * stride_oz + off_h * stride_oh
    Out_ptr = Out + out_offset
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    # Cast the fp32 accumulator to the output dtype (bfloat16) before storing.
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launches the Triton kernel for sliding-window causal attention.

    Args:
        q: Query tensor of shape [Z, H, N_CTX, D_HEAD] and dtype bfloat16.
        k: Key tensor of shape [Z, H, N_CTX, D_HEAD] and dtype bfloat16.
        v: Value tensor of shape [Z, H, N_CTX, D_HEAD] and dtype bfloat16.

    Returns:
        Output tensor of the same shape and dtype as the inputs.
    """
    # Ensure inputs are in the expected format
    assert q.shape == k.shape == v.shape, "Input tensors must have the same shape"
    assert q.dtype == torch.bfloat16, "Input tensors must be bfloat16"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Input tensors must be on a CUDA device"
    assert q.shape == (2, 32, 2048, 128), "Input shape must be [2, 32, 2048, 128]"

    # Tensor dimensions
    Z, H, N_CTX, D_HEAD = q.shape

    # Create an empty output tensor
    out = torch.empty_like(q)

    # Define Triton kernel parameters, tuned for A100 (sm80)
    BLOCK_M = 128
    BLOCK_N = 64
    WINDOW_SIZE = 256
    num_warps = 8
    num_stages = 3

    # The grid defines how many instances of the kernel will run.
    # Each instance computes one `BLOCK_M`-sized chunk of the output queries.
    # Grid is 2D: (Z * H, N_CTX / BLOCK_M)
    grid = (Z * H, triton.cdiv(N_CTX, BLOCK_M))

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, out,
        # Strides for Q, K, V, and Out tensors
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # Metadata
        Z, H, N_CTX,
        # Compile-time constants
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=D_HEAD,
        BLOCK_N=BLOCK_N,
        WINDOW_SIZE=WINDOW_SIZE,
        # Performance tuning hints
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out