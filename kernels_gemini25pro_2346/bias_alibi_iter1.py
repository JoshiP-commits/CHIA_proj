import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    Q, K, V, Out,
    # Stride variables for tensors
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # Other parameters
    Z, H, N_CTX,
    # Kernel constants
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # Scale factor
    sm_scale: tl.constexpr,
):
    """
    Triton kernel for GraphSynth attention.

    This kernel implements a FlashAttention-style algorithm for a custom attention
    mechanism with an ALiBi bias and causal masking.

    Grid: (B * H, N_CTX / BLOCK_M)
    - pid_bh: Identifies the batch and head.
    - pid_m: Identifies the block of queries to process.
    """
    # -----------------------------------------------------------
    # Map program ids to batch, head, and query block
    pid_bh = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # Decompose batch-head pid to get batch and head indices
    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    # Define offsets for the current block of queries
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)

    # -----------------------------------------------------------
    # Initialize pointers to Q, K, V
    q_ptrs = Q + (batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_base_ptr = K + (batch_idx * stride_kb + head_idx * stride_kh)
    v_base_ptr = V + (batch_idx * stride_vb + head_idx * stride_vh)

    # -----------------------------------------------------------
    # Initialize FlashAttention accumulators and running statistics in SRAM
    # Accumulator for the output, in fp32
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    # Running sum of exponentials for softmax normalization, in fp32
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Running maximum of scores, in fp32
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)

    # Load the current block of queries from HBM to SRAM
    # Mask to avoid out-of-bounds access for sequences not perfectly divisible by BLOCK_M
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Compute ALiBi slope for the current head (slope_h = h + 1)
    slope = (head_idx + 1.0)

    # -----------------------------------------------------------
    # Main loop over key/value blocks (FlashAttention-style)
    # Causal attention: loop up to the current query block's end position
    end_n = start_m + BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        # -- Load K block (transposed) --
        # To compute Q@K.T without trans_b, we load K as [D_HEAD, BLOCK_N]
        offs_n_block = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base_ptr + (offs_d[:, None] * stride_kk + offs_n_block[None, :] * stride_kn)
        k = tl.load(k_ptrs, mask=offs_n_block[None, :] < N_CTX, other=0.0)

        # -- Compute scores S = Q @ K.T --
        scores = tl.dot(q, k)
        scores *= sm_scale

        # -- Add ALiBi bias: slope_h * (j - i) --
        # j = key indices, i = query indices
        alibi_bias = (offs_n_block[None, :] - offs_m[:, None]) * slope
        scores += alibi_bias

        # -- Apply causal mask (j <= i) --
        causal_mask = offs_m[:, None] >= offs_n_block[None, :]
        scores = tl.where(causal_mask, scores, -1e9)

        # -----------------------------------------------------------
        # Online softmax update (FlashAttention algorithm)
        # -- Compute local max, rescaled probabilities, and local sum --
        m_ij = tl.max(scores, axis=1)
        p_ij = tl.exp(scores - m_ij[:, None])
        l_ij = tl.sum(p_ij, axis=1)

        # -- Compute new global max and rescaling factors --
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)

        # -- Update running sum and rescale old accumulator --
        l_i_new = alpha * l_i + beta * l_ij
        acc = acc * alpha[:, None]

        # -- Load V and update accumulator --
        v_ptrs = v_base_ptr + (offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        v = tl.load(v_ptrs, mask=offs_n_block[:, None] < N_CTX, other=0.0)

        # Cast probabilities to bf16 before dot product for performance
        p_ij = p_ij.to(v.dtype)
        acc += tl.dot(p_ij, v)

        # -- Update running statistics for the next iteration --
        l_i = l_i_new
        m_i = m_i_new

    # -----------------------------------------------------------
    # Finalize and store the output
    # Normalize the accumulator by the final sum
    l_i_reciprocal = 1.0 / (l_i + 1e-6) # Add epsilon for stability
    out = acc * l_i_reciprocal[:, None]

    # Write the output block back to HBM
    out_ptrs = Out + (batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(out_ptrs, out.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher function for the GraphSynth Triton kernel.

    Args:
        q: Query tensor of shape [B, H, N, D], dtype bfloat16.
        k: Key tensor of shape [B, H, N, D], dtype bfloat16.
        v: Value tensor of shape [B, H, N, D], dtype bfloat16.

    Returns:
        Output tensor of shape [B, H, N, D], dtype bfloat16.
    """
    # Input tensor shapes and dimensions
    B, H, N, D_HEAD = q.shape

    # Allocate the output tensor on the same device as inputs
    o = torch.empty_like(q, dtype=torch.bfloat16)

    # Choose block sizes for tiling. These are tunable parameters.
    # For A100, BLOCK_M=128 and BLOCK_N=64 are common choices.
    BLOCK_M = 128
    BLOCK_N = 64

    # The grid is 2D. First dimension is batch*heads, second is sequence length tiled by BLOCK_M.
    grid = (B * H, triton.cdiv(N, BLOCK_M))

    # Pre-compute the scale factor on the host
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    # Launch the Triton kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        # Strides for each tensor
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Other parameters
        B, H, N,
        # Constexpr parameters
        D_HEAD=D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        sm_scale=sm_scale,
    )

    return o