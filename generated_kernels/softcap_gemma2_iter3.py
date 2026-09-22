import torch
import triton
import triton.language as tl
import math

# This is a self-contained Python code block with a single Triton kernel and its launcher.

@triton.jit
def graphsynth_kernel(
    # Pointers to matrices
    Q, K, V, Out,
    # Stride variables for tensors
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # Matrix dimensions
    Z, H, SEQ_LEN, D_HEAD,
    # Kernel constants
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DHEAD: tl.constexpr,
):
    """
    Triton kernel for causal attention with Gemma-2 style logit soft-capping.
    This kernel implements a FlashAttention-style algorithm to avoid materializing
    the large S = Q @ K^T matrix.

    Grid: (B * H, ceil_div(SEQ_LEN, BLOCK_M))
    - Each program instance computes one BLOCK_M x D_HEAD block of the output.
    - pid_bh: Batch and Head index.
    - pid_m: Block index along the sequence length dimension for Q.
    """
    # --------------------------------------------------------------------------
    # Program and Grid Setup
    # --------------------------------------------------------------------------
    # This program instance is responsible for a single block of queries.
    pid_m = tl.program_id(1)
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Pointers to the current query block. This block is fixed for this program.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DHEAD)
    q_ptrs = Q + (pid_b * stride_qb + pid_h * stride_qh +
                  offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

    # Pointers to the base of K and V for the current batch and head.
    # The offset along the sequence length dimension will be updated in the loop.
    k_base_ptr = K + (pid_b * stride_kb + pid_h * stride_kh)
    v_base_ptr = V + (pid_b * stride_vb + pid_h * stride_vh)

    # Pointer to the output block.
    o_ptrs = Out + (pid_b * stride_ob + pid_h * stride_oh +
                    offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------
    # Accumulator for the output, initialized to zeros. Use fp32 for precision.
    acc = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)
    # Running max and sum for the online softmax.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Mask for queries that are beyond the actual sequence length (padding).
    q_mask = offs_m < SEQ_LEN
    # Load the query block, zeroing out padding queries.
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Constants for the attention calculation.
    sm_scale = 1.0 / (D_HEAD ** 0.5)

    # --------------------------------------------------------------------------
    # Main Loop over Key/Value Blocks
    # --------------------------------------------------------------------------
    # The loop iterates over key/value blocks. Causal masking is enforced by
    # only iterating up to the current query block's end position.
    end_m_idx = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_m_idx, BLOCK_N):
        # --- Early exit for padding KV blocks ---
        # If the start of the KV block is already past the sequence length, we can skip.
        # This is equivalent to `if start_n >= SEQ_LEN: continue` but avoids the
        # unsupported `continue` statement in older Triton versions.
        if start_n < SEQ_LEN:
            # -- Load K and V for the current block --
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # Pointers to the current K block (transposed) and V block.
            k_ptrs = k_base_ptr + (offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn)
            v_ptrs = v_base_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

            # Mask for keys/values that are padding.
            kv_mask = offs_n < SEQ_LEN
            k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

            # -- Compute scores S = Q @ K^T --
            s = tl.dot(q, k, out_dtype=tl.float32)
            s *= sm_scale

            # -- Apply Gemma-2 Logit Soft-Capping --
            # s_capped = tanh(s / 30.0) * 30.0
            # using the identity tanh(x) = 2*sigmoid(2x) - 1
            s_div_30 = s / 30.0
            s = (2.0 * tl.sigmoid(2.0 * s_div_30) - 1.0) * 30.0

            # -- Apply Causal Mask --
            # A query at row `m` cannot attend to a key at column `n` if `n > m`.
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, -float('inf'))

            # -- Online Softmax Update --
            # 1. Find the new row-wise maximum.
            m_new = tl.maximum(m_i, tl.max(s, axis=1))

            # 2. Calculate probabilities with the new maximum.
            p = tl.exp(s - m_new[:, None])

            # 3. Correct the running sum `l_i` and accumulator `acc` for the change in max.
            alpha = tl.exp(m_i - m_new)
            acc *= alpha[:, None]
            l_i *= alpha

            # 4. Update the running sum and accumulator.
            # p @ v is done in bfloat16 for performance on Tensor Cores.
            acc += tl.dot(p.to(V.dtype.element_ty), v)
            l_i += tl.sum(p, axis=1)

            # 5. Update the running maximum.
            m_i = m_new

    # --------------------------------------------------------------------------
    # Finalization and Store
    # --------------------------------------------------------------------------
    # Rescale the accumulator by the inverse of the softmax denominator `l_i`.
    # Guard against division by zero for padding rows where l_i could be 0.
    l_i_safe = tl.where(l_i == 0, 1, l_i)
    out = acc / l_i_safe[:, None]

    # Store the final result, casting to bfloat16 and applying the padding mask.
    tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=q_mask[:, None])


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the GraphSynth causal attention kernel.

    Args:
        q: Query tensor of shape [B, H, S, D] and dtype bfloat16.
        k: Key tensor of shape [B, H, S, D] and dtype bfloat16.
        v: Value tensor of shape [B, H, S, D] and dtype bfloat16.

    Returns:
        Output tensor of the same shape and dtype as the inputs.
    """
    # Ensure inputs are valid
    assert q.shape == k.shape == v.shape, "Input tensors must have the same shape."
    assert q.dtype == torch.bfloat16, "Inputs must be bfloat16."
    assert q.is_cuda, "Inputs must be on a CUDA device."
    assert q.shape[-1] == 128, "D_HEAD must be 128."

    # Tensor dimensions
    Z, H, SEQ_LEN, D_HEAD = q.shape

    # Output tensor
    out = torch.empty_like(q)

    # Triton kernel config
    # These block sizes are chosen for A100 (sm_80) to balance parallelism,
    # register usage, and shared memory capacity.
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Grid definition
    grid = (Z * H, triton.cdiv(SEQ_LEN, BLOCK_M))
    
    # Number of warps and other tuning parameters
    num_warps = 8 if D_HEAD >= 128 else 4

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, SEQ_LEN, D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DHEAD=D_HEAD,
        num_warps=num_warps,
    )

    return out