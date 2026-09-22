import torch
import triton
import triton.language as tl


@triton.jit
def graphsynth_kernel(
    # Pointers to Matrices
    Q, K, V, O,
    # Stride Cfg
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Matrix Dimensions
    Z, H, N_CTX,
    # Kernel-specific
    sm_scale,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """
    Triton kernel for causal attention with Gemma-2 style logit soft-capping.
    This kernel implements a FlashAttention-style algorithm to avoid materializing
    the full score matrix, using an online softmax update.
    """
    # -----------------------------------------------------------
    # Program and Offsets
    # This kernel processes one query block at a time.
    # The grid is 2D: (batch*heads, sequence_length / BLOCK_M)
    start_m = tl.program_id(1)
    off_zh = tl.program_id(0)

    # Offsets for the current query block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    offs_n = tl.arange(0, BLOCK_N)

    # -----------------------------------------------------------
    # Pointer Setup
    # Pointers to the input and output tensors for the current batch and head
    q_ptr = Q + off_zh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # K is loaded transposed: D_HEAD x BLOCK_N, by swapping n and k strides in pointer math
    k_ptr = K + off_zh * stride_kh + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_ptr = V + off_zh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    o_ptr = O + off_zh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    
    # -----------------------------------------------------------
    # Accumulators and Initial State
    # Accumulator for the output, in fp32 for precision
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    # Running sum of exponents for softmax normalization
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Running max score for stable softmax calculation
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)

    # -----------------------------------------------------------
    # Load Query Block
    # This block of Q is used for all KV blocks in the inner loop.
    # Masking is applied for rows where m >= N_CTX (padding).
    q = tl.load(q_ptr, mask=offs_m[:, None] < N_CTX, other=0.0)

    # -----------------------------------------------------------
    # Loop over Key-Value Blocks
    # The loop bound `(start_m + 1) * BLOCK_M` ensures causality at the block level.
    # We only compute attention for key blocks that are before or at the current query block.
    end_n = (start_m + 1) * BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        # -- Load K and V for the current block --
        k = tl.load(k_ptr + start_n * stride_kn, mask=(start_n + offs_n)[None, :] < N_CTX, other=0.0)
        v = tl.load(v_ptr + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # -- Compute Scores (s = q @ k^T) --
        # q: [BLOCK_M, D_HEAD], k: [D_HEAD, BLOCK_N] -> s: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k)
        s *= sm_scale

        # -- Apply Gemma-2 Logit Soft-Capping --
        # tanh(s / 30.0) * 30.0, using the required sigmoid-based formula: 2*sigmoid(2x)-1
        s_gated = s / 30.0
        s = (2.0 * tl.sigmoid(2.0 * s_gated) - 1.0) * 30.0

        # -- Apply Causal Mask --
        # This mask handles the fine-grained causality within the diagonal block.
        causal_mask = (offs_m[:, None] >= (start_n + offs_n[None, :]))
        s = tl.where(causal_mask, s, -1e9)

        # -----------------------------------------------------------
        # Online Softmax Update (FlashAttention algorithm)
        # 1. Get the max of the current scores
        m_ij = tl.max(s, axis=1)
        
        # 2. Update the running max
        m_prev = m_i
        m_new = tl.maximum(m_prev, m_ij)
        
        # 3. Calculate scale factor and update probabilities for the current block
        scale = tl.exp(m_prev - m_new)
        p_ij = tl.exp(s - m_new[:, None])
        
        # 4. Rescale previous accumulator and sum of exponents
        acc = acc * scale[:, None]
        l_i = l_i * scale
        
        # 5. Update accumulator and sum of exponents with new values
        # Cast p_ij to bfloat16 for efficient dot product on Tensor Cores
        p_ij = p_ij.to(v.dtype)
        acc += tl.dot(p_ij, v)
        l_i += tl.sum(p_ij, axis=1)

        # 6. Update the running max for the next iteration
        m_i = m_new
    
    # -----------------------------------------------------------
    # Final Normalization and Store
    # Divide the accumulator by the final sum of exponents.
    # Handle the case where the sum is zero (due to masking) to avoid NaN.
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc_normalized = acc / l_i_safe[:, None]

    # Cast to bfloat16 and store the final output block
    tl.store(o_ptr, acc_normalized.to(tl.bfloat16), mask=offs_m[:, None] < N_CTX)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launches the Triton kernel for causal attention with Gemma-2 style logit soft-capping.

    Args:
        q: Query tensor of shape [B, H, S, D] and dtype bfloat16.
        k: Key tensor of shape [B, H, S, D] and dtype bfloat16.
        v: Value tensor of shape [B, H, S, D] and dtype bfloat16.

    Returns:
        Output tensor of the same shape and dtype as inputs.
    """
    # Shape checks and constants
    B, H, S, D_HEAD = q.shape
    
    # Output tensor
    o = torch.empty_like(q)

    # Scaling factor for the dot product
    sm_scale = 1.0 / (D_HEAD**0.5)

    # Triton block size configuration for sm80
    BLOCK_M = 128
    BLOCK_N = 64

    # Grid definition for the kernel launch
    # Each program on the grid handles one query block for one head
    grid = (B * H, S // BLOCK_M)

    # Launch the kernel
    graphsynth_kernel[grid](
        # Tensors
        q, k, v, o,
        # Strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Dimensions
        B, H, S,
        # Kernel parameters
        sm_scale,
        # Compile-time constants
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D_HEAD=D_HEAD,
        # Performance tuning for A100
        num_warps=4,
        num_stages=2,
    )

    return o