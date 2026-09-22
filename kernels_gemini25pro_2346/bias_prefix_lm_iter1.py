import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DHEAD: tl.constexpr,
    GRAPH_CUTOFF: tl.constexpr,
):
    """
    Triton kernel for GraphSynth attention.

    Computes attention with a custom mask:
    s_ij is unmasked if (j <= i) OR (j < GRAPH_CUTOFF).
    - i: query index
    - j: key index
    - GRAPH_CUTOFF: First N tokens are globally visible (e.g., 256).

    This kernel implements FlashAttention-style tiling to avoid materializing
    the large [N_CTX, N_CTX] score matrix. It processes Q in blocks of `BLOCK_M`
    and iterates over K/V blocks of size `BLOCK_N`.
    """
    # Program IDs identify the current work item.
    # Grid is 2D: (Batch * Heads, Num_M_Blocks)
    pid_bh = tl.program_id(axis=0)  # Index for batch and head
    pid_m = tl.program_id(axis=1)   # Index for the M-block of the sequence

    # De-serialize batch and head indices
    # We have H heads per item in the batch.
    pid_z = pid_bh // H
    pid_h = pid_bh % H

    # Pointers to the start of the current head's data
    q_ptr = Q + pid_z * stride_qz + pid_h * stride_qh
    k_ptr = K + pid_z * stride_kz + pid_h * stride_kh
    v_ptr = V + pid_z * stride_vz + pid_h * stride_vh
    o_ptr = O + pid_z * stride_oz + pid_h * stride_oh

    # --- Tiling setup ---
    # `start_m` is the starting row index of the Q block we are processing.
    start_m = pid_m * BLOCK_M
    # `offs_m` are the row indices for the current Q block [start_m, start_m + BLOCK_M).
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # `offs_d` are the column indices for the head dimension [0, BLOCK_DHEAD).
    offs_d = tl.arange(0, BLOCK_DHEAD)

    # --- Load Q tile ---
    # Pointers to the Q tile of shape [BLOCK_M, BLOCK_DHEAD].
    q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    # Mask for padding tokens in Q.
    q_mask = (offs_m < N_CTX)[:, None]
    # Load the Q tile, masking out padding rows.
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # --- FlashAttention accumulators ---
    # `acc` stores the running sum of (p_ij * v_j) values, in fp32.
    acc = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)
    # `m_i` stores the running maximum of s_ij values for each row in Q.
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    # `l_i` stores the running sum of exp(s_ij - m_i) values (the softmax denominator).
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # --- Main loop over KV blocks ---
    # We iterate over K and V in blocks of size BLOCK_N.
    for start_n in range(0, N_CTX, BLOCK_N):
        # --- Load K and V tiles ---
        # `offs_n` are the row indices for the current K/V block.
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Pointers for transposed K load [BLOCK_DHEAD, BLOCK_N].
        k_ptrs = k_ptr + (offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn)
        # Pointers for V load [BLOCK_N, BLOCK_DHEAD].
        v_ptrs = v_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

        # Mask for padding tokens in K and V.
        kv_mask_n = (offs_n < N_CTX)
        # Load K (transposed) and V, masking padding tokens.
        k = tl.load(k_ptrs, mask=kv_mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask_n[:, None], other=0.0)

        # --- Compute scores S = Q @ K^T ---
        # `q` is [BLOCK_M, BLOCK_DHEAD], `k` is [BLOCK_DHEAD, BLOCK_N].
        # `s_ij` is the score block [BLOCK_M, BLOCK_N].
        s_ij = tl.dot(q, k) * sm_scale

        # --- Apply the custom causal/graph mask ---
        # Condition: j <= i OR j < GRAPH_CUTOFF
        # `offs_m` are query indices `i`, `offs_n` are key indices `j`.
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        graph_mask = offs_n[None, :] < GRAPH_CUTOFF
        attn_mask = causal_mask | graph_mask
        
        # Mask out scores for padding tokens.
        # This is redundant with masked loads but safer.
        padding_mask = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        final_mask = attn_mask & padding_mask

        # Set masked-out scores to a large negative number.
        s_ij = tl.where(final_mask, s_ij, -1e9)

        # --- Online Softmax Update ---
        # 1. Find the new block-wise maximum `m_ij` and update the overall running max `m_i`.
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # 2. Rescale the current accumulators `acc` and `l_i` based on the change in max.
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # 3. Calculate the numerator of the softmax for the current block, `p_ij`.
        p_ij = tl.exp(s_ij - m_new[:, None])
        
        # 4. Update the denominator `l_i` with the sum of the new numerators.
        l_ij = tl.sum(p_ij, axis=1)
        l_i += l_ij
        
        # 5. Update the output accumulator `acc` by adding the weighted values from `v`.
        # `p_ij` is fp32, `v` is bf16. Cast `p_ij` to bf16 before dot product.
        acc += tl.dot(p_ij.to(V.dtype.element_ty), v)

        # 6. Update the running max for the next iteration.
        m_i = m_new

    # --- Finalization ---
    # Normalize the final accumulator to get the output.
    # Add a small epsilon to `l_i` to avoid division by zero.
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_i_safe[:, None]

    # --- Write output tile to global memory ---
    o_ptrs = o_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    # Use the q_mask to only write valid rows.
    tl.store(o_ptrs, o.to(O.dtype.element_ty), mask=q_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the GraphSynth Triton kernel.

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, N, D] and dtype bfloat16.
        k (torch.Tensor): Key tensor of shape [B, H, N, D] and dtype bfloat16.
        v (torch.Tensor): Value tensor of shape [B, H, N, D] and dtype bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype.
    """
    # Tensor shape validation
    assert q.shape == k.shape == v.shape, "All tensors must have the same shape"
    assert q.dim() == 4, "Input tensors must be 4D"
    assert q.dtype == torch.bfloat16, "Input tensors must be bfloat16"

    # Shape parameters
    Z, H, N_CTX, D_HEAD = q.shape

    # Output tensor
    o = torch.empty_like(q)

    # Triton block sizes
    BLOCK_M = 64
    BLOCK_N = 64

    # The custom mask depends on a cutoff. Let's hardcode it as required.
    GRAPH_CUTOFF = 256

    # Pre-compute scaling factor on the host
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    # Grid definition
    # Each program instance handles a `BLOCK_M`-sized chunk of the sequence
    # for a specific batch and head.
    grid = (Z * H, triton.cdiv(N_CTX, BLOCK_M))

    # Launch the kernel
    graphsynth_kernel[grid](
        # Tensors
        q, k, v, o,
        # Strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Shape constants
        Z, H, N_CTX,
        # Kernel constants
        sm_scale=sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DHEAD=D_HEAD,
        GRAPH_CUTOFF=GRAPH_CUTOFF,
    )

    return o