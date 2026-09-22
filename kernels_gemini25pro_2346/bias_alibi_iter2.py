import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    Q, K, V, O,
    # Stride variables
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    # Other parameters
    seq_len,
    scale,
    # Bounded constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """
    Computes scaled dot-product attention with a causal mask and an ALiBi bias.
    This kernel is styled after FlashAttention to avoid materializing the
    large score matrix in DRAM.
    """
    # 1. Program and Block Identification
    pid_m = tl.program_id(0)  # Block index along the sequence length dimension
    pid_h = tl.program_id(1)  # Head index
    pid_z = tl.program_id(2)  # Batch index

    # 2. Pointer Setup
    # Advance pointers to the correct batch and head
    q_ptr = Q + pid_z * stride_qz + pid_h * stride_qh
    k_ptr = K + pid_z * stride_kz + pid_h * stride_kh
    v_ptr = V + pid_z * stride_vz + pid_h * stride_vh
    o_ptr = O + pid_z * stride_oz + pid_h * stride_oh

    # 3. Initialize Accumulators and Running Statistics
    # Running max and sum for online softmax, and the output accumulator
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # 4. Load Query Block
    # Offsets for the current block of queries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q_mask = offs_m < seq_len
    # Load and cast to float32 for high-precision score computation
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # 5. Main Loop over Key/Value Blocks (Causal)
    # The loop upper bound ensures we only attend to keys up to the current query block
    end_n = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        # -- Load K block (transposed) --
        offs_n_block = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n_block < seq_len
        # Load K transposed by swapping stride order in pointer arithmetic
        # This avoids using `trans_b=True` in tl.dot, as required.
        k_ptrs = k_ptr + (offs_d[:, None] * stride_kd + offs_n_block[None, :] * stride_kn)
        k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

        # -- Compute scores s_ij --
        s_ij = tl.dot(q, k)
        s_ij *= scale

        # -- Add ALiBi Bias and Causal Mask --
        # ALiBi: slope_h * (j - i)
        slope_h = (pid_h + 1.0) # head index is 0-based
        alibi_bias = (offs_n_block[None, :] - offs_m[:, None]) * slope_h
        s_ij += alibi_bias
        # Causal mask (j <= i) for elements within the current block
        causal_mask = offs_m[:, None] >= offs_n_block[None, :]
        s_ij = tl.where(causal_mask, s_ij, -1e9)

        # -- Online Softmax Update --
        # 1. New running max
        m_i_new = tl.maximum(m_i, tl.max(s_ij, 1))
        # 2. Rescale previous accumulator and sum
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        # 3. Compute new probabilities and update sum
        p_ij = tl.exp(s_ij - m_i_new[:, None])
        l_i += tl.sum(p_ij, 1)
        # 4. Load V and update accumulator
        v_ptrs = v_ptr + (offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        # Cast p_ij to bfloat16 for efficient dot product on A100
        acc += tl.dot(p_ij.to(v.dtype), v)
        # 5. Update running max
        m_i = m_i_new

    # 6. Finalization and Write Output
    # Rescale accumulator by the inverse of the final sum
    l_i_inv = 1.0 / l_i
    # Handle rows that were all-masked (l_i=0) to avoid division by zero
    acc = acc * tl.where(l_i == 0.0, 0.0, l_i_inv)[:, None]

    # Write output block, casting back to the original bfloat16 dtype
    o_ptrs = o_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=q_mask[:, None])


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the GraphSynth Triton kernel.

    Args:
        q: Query tensor of shape [B, H, N, D] and dtype bfloat16.
        k: Key tensor of shape [B, H, N, D] and dtype bfloat16.
        v: Value tensor of shape [B, H, N, D] and dtype bfloat16.

    Returns:
        Output tensor of the same shape and dtype as the inputs.
    """
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = q.shape
    o = torch.empty_like(q)

    # Kernel configuration for NVIDIA A100 (sm80)
    BLOCK_M = 128
    BLOCK_N = 64
    num_warps = 4
    num_stages = 2 # Use 2 for more aggressive data prefetching

    # Grid dimensions: (num_blocks_in_seq, num_heads, num_batches)
    grid = (triton.cdiv(SEQ_LEN, BLOCK_M), N_HEADS, BATCH)

    # Pre-compute scale factor on the host as a float
    scale = 1.0 / math.sqrt(D_HEAD)

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        SEQ_LEN,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D_HEAD=D_HEAD,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o