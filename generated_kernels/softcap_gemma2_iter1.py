import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    # Pointers to matrices
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Stride variables for tensors
    stride_q_bh, stride_q_s, stride_q_d,
    stride_k_bh, stride_k_s, stride_k_d,
    stride_v_bh, stride_v_s, stride_v_d,
    stride_o_bh, stride_o_s, stride_o_d,
    # Metadata
    SEQ_LEN: tl.constexpr,
    # Tiling constants
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Fused Causal Attention Kernel with Gemma-2 Logit Soft-capping.
    This kernel computes attention for a single query block of size BLOCK_M.
    """
    # --------------------------------------------------------------------------
    # Program and Tile identification
    # --------------------------------------------------------------------------
    # This program operates on a single head for a batch item.
    pid_bh = tl.program_id(axis=0)
    # This program operates on a single block of queries of size BLOCK_M.
    pid_m = tl.program_id(axis=1)

    # Offsets for the current query block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # --------------------------------------------------------------------------
    # Pointer Setup and Initialization
    # --------------------------------------------------------------------------
    # Pointers to the current query block
    q_ptrs = Q_ptr + pid_bh * stride_q_bh + (offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d)

    # Base pointers for K and V for the current head
    k_base_ptr = K_ptr + pid_bh * stride_k_bh
    v_base_ptr = V_ptr + pid_bh * stride_v_bh

    # Accumulator, running max, and running sum for online softmax
    # All must be float32 for precision
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load the query block once
    # Masking ensures we don't read out of bounds if SEQ_LEN is not a multiple of BLOCK_M
    q_mask = offs_m[:, None] < SEQ_LEN
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Scale for dot product
    sm_scale = 1.0 / (BLOCK_D**0.5)

    # --------------------------------------------------------------------------
    # Main Loop over Key/Value Blocks
    # --------------------------------------------------------------------------
    # The loop iterates over key/value blocks. Causal masking is enforced by
    # only iterating up to the current query block's end position.
    end_n = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        # -- Early exit for padding KV blocks --
        # If the start of the KV block is already past the sequence length, we can skip.
        if start_n >= SEQ_LEN:
            continue
            
        # -- Step 1: Load K and V blocks --
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Pointers to the current K and V blocks
        k_ptrs = k_base_ptr + (offs_n[None, :] * stride_k_s + offs_d[:, None] * stride_k_d)
        v_ptrs = v_base_ptr + (offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d)

        # Masking for K and V loads
        kv_mask = offs_n[None, :] < SEQ_LEN
        
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        # Transpose K is not needed, tl.dot handles it. k shape: [BLOCK_D, BLOCK_N]
        # v shape: [BLOCK_N, BLOCK_D]
        v = tl.load(v_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)

        # -- Step 2: Compute scores s = (q @ k^T) / sqrt(d) --
        s = tl.dot(q, k, out_dtype=tl.float32) * sm_scale

        # -- Step 3: Apply Gemma-2 logit soft-capping and causal mask --
        # 3.1: Gemma-2 logit soft-capping: s_capped = tanh(s / 30.0) * 30.0
        # using tanh(x) = 2*sigmoid(2x) - 1
        s_scaled = s / 30.0
        s_capped = (2.0 * tl.sigmoid(2.0 * s_scaled) - 1.0) * 30.0

        # 3.2: Causal mask: query i attends to key j only if j <= i
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        
        # Apply causal mask *after* capping to ensure masked elements are -inf.
        # tanh(-inf) = -1, which would corrupt the softmax.
        s_masked = tl.where(causal_mask, s_capped, -float('inf'))

        # -- Step 4: Online softmax update --
        m_ij = tl.max(s_masked, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Calculate P_ij = exp(s_ij - m_new)
        p = tl.exp(s_masked - m_new[:, None])

        # Rescale l_i and accumulator
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # Update l_i and accumulator with current block
        l_i += tl.sum(p, axis=1)
        
        # p is fp32, cast to bfloat16 for efficient dot product on A100
        p = p.to(q.dtype)
        acc += tl.dot(p, v)

        # Update running max
        m_i = m_new

    # --------------------------------------------------------------------------
    # Finalization and Store
    # --------------------------------------------------------------------------
    # Normalize the accumulator
    # Add a small epsilon to l_i to avoid division by zero
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    o = acc / l_i_safe[:, None]

    # Pointers to the output block
    out_ptrs = Out_ptr + pid_bh * stride_o_bh + (offs_m[:, None] * stride_o_s + offs_d[None, :] * stride_o_d)
    
    # Store the final result, casting from fp32 to bfloat16
    tl.store(out_ptrs, o, mask=q_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the GraphSynth causal attention kernel.

    Args:
        q: Query tensor of shape [B, H, S, D], dtype=bfloat16.
        k: Key tensor of shape [B, H, S, D], dtype=bfloat16.
        v: Value tensor of shape [B, H, S, D], dtype=bfloat16.

    Returns:
        Output tensor of shape [B, H, S, D], dtype=bfloat16.
    """
    # Extract dimensions
    B, H, S, D = q.shape
    assert D == 128, "Head dimension must be 128"

    # Create an empty output tensor
    out = torch.empty_like(q)

    # Reshape tensors to [B*H, S, D] for simpler kernel logic
    q_r = q.view(B * H, S, D)
    k_r = k.view(B * H, S, D)
    v_r = v.view(B * H, S, D)
    o_r = out.view(B * H, S, D)
    
    # Kernel tiling parameters
    BLOCK_M = 128
    BLOCK_N = 64

    # Define the Triton grid
    # Each program instance computes one block of queries for one head
    grid = (B * H, triton.cdiv(S, BLOCK_M))

    # Launch the kernel
    graphsynth_kernel[grid](
        q_r, k_r, v_r, o_r,
        # Strides for each tensor (reshaped)
        q_r.stride(0), q_r.stride(1), q_r.stride(2),
        k_r.stride(0), k_r.stride(1), k_r.stride(2),
        v_r.stride(0), v_r.stride(1), v_r.stride(2),
        o_r.stride(0), o_r.stride(1), o_r.stride(2),
        # Metadata and constants
        SEQ_LEN=S,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        # Performance tuning hints
        num_warps=4,
        num_stages=2,
    )

    return out