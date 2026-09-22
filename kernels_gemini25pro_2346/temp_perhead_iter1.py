import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    Q, K, V, O,
    # Strides
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Head and Context dimensions
    N_HEADS: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    # Kernel Constants
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for custom attention on an NVIDIA A100 (sm80).
    - FlashAttention-style implementation to avoid materializing the score matrix.
    - Applies a per-head temperature scaling to the scores.
    - Handles causal masking.
    - Grid: (num_blocks_seq_len, batch_size * num_heads)
    """
    # 1. Get program IDs to identify the current block
    pid_m = tl.program_id(0)  # Block index along the sequence dimension (M)
    pid_z = tl.program_id(1)  # Combined batch and head index

    # 2. Derive batch and head indices from the combined Z-dimension ID
    batch_idx = pid_z // N_HEADS
    head_idx = pid_z % N_HEADS

    # 3. Calculate per-head temperature scalar t_h = 0.5 + h/32
    # The division must be a float operation.
    t_h = 0.5 + head_idx / 32.0

    # 4. Initialize pointers and offsets for the current block
    # Start of the current Q block (rows)
    start_m = pid_m * BLOCK_M
    # Row offsets for the Q block and output block
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # Column offsets for Q, K, V (head dimension)
    offs_d = tl.arange(0, D_HEAD)

    # Base pointers for Q, K, V for the current batch and head
    q_base_ptr = Q + batch_idx * stride_qz + head_idx * stride_qh
    k_base_ptr = K + batch_idx * stride_kz + head_idx * stride_kh
    v_base_ptr = V + batch_idx * stride_vz + head_idx * stride_vh

    # Pointers to the current Q block
    q_ptrs = q_base_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk

    # 5. Initialize accumulators for online softmax using fp32
    # Running max of scores, for numerical stability
    m_i = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)
    # Running sum of exp(scores), the denominator of softmax
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Output accumulator
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # 6. Load the Q block
    # Masking is applied to prevent out-of-bounds access for the last block
    q_mask = offs_m[:, None] < N_CTX
    # Load Q, filling with 0.0 for masked-out elements
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 7. Main loop over K and V blocks (causal)
    # The loop upper bound ensures we only process K blocks up to the current Q block
    end_m = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_m, BLOCK_N):
        # -- Load K (transposed) --
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # Pointers for a transposed K block [D_HEAD, BLOCK_N]
        # Strides are swapped (stride_kk, stride_kn) to achieve transpose
        k_ptrs = k_base_ptr + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
        k_mask = offs_n[None, :] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # -- Compute scores S = Q @ K^T --
        # Result is a [BLOCK_M, BLOCK_N] matrix of scores
        s = tl.dot(q, k)

        # -- Apply scaling and per-head temperature --
        s = s * scale * t_h

        # -- Apply causal mask --
        # Prevents attention to "future" tokens within the current block
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        # Set scores for masked-out positions to a large negative number
        s = tl.where(causal_mask, s, -1e9)

        # -- Online softmax update --
        # 1. Find the new row-wise max
        m_ij = tl.max(s, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        # 2. Calculate rescaling factor and update accumulators
        p_scale = tl.exp(m_i - m_i_new)
        acc = acc * p_scale[:, None]
        l_i = l_i * p_scale

        # 3. Compute probabilities for the current block and update l_i
        p_ij = tl.exp(s - m_i_new[:, None])
        l_ij = tl.sum(p_ij, 1)
        l_i = l_i + l_ij

        # 4. Update the running max
        m_i = m_i_new

        # -- Update output accumulator O = P @ V --
        # 1. Load V block
        v_ptrs = v_base_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = offs_n[:, None] < N_CTX
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # 2. Accumulate weighted V
        # tl.dot can handle mixed precision (fp32 @ bf16 -> fp32)
        acc = acc + tl.dot(p_ij, v)

    # 8. Finalize and store the output block
    # Normalize the output accumulator by the final sum of exponents
    # Use a safe reciprocal to avoid division by zero for fully masked rows
    l_i_safe_rcp = 1.0 / tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc * l_i_safe_rcp[:, None]

    # Pointers to the output tensor O
    o_base_ptr = O + batch_idx * stride_oz + head_idx * stride_oh
    o_ptrs = o_base_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = offs_m[:, None] < N_CTX

    # Store the final result, casting back to the input dtype
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=o_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launch function for the custom GraphSynth attention kernel.

    Args:
        q (torch.Tensor): Query tensor of shape [2, 32, 2048, 128] and dtype bfloat16.
        k (torch.Tensor): Key tensor of shape [2, 32, 2048, 128] and dtype bfloat16.
        v (torch.Tensor): Value tensor of shape [2, 32, 2048, 128] and dtype bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype as inputs.
    """
    # Check tensor properties against hard requirements
    assert all(t.is_cuda and t.dtype == torch.bfloat16 for t in [q, k, v])
    shape = (2, 32, 2048, 128)
    assert q.shape == shape and k.shape == shape and v.shape == shape

    BATCH, N_HEADS, N_CTX, D_HEAD = q.shape

    # Create the output tensor
    o = torch.empty_like(q)

    # Kernel configuration
    # These block sizes are chosen as a reasonable default for A100.
    BLOCK_M = 64
    BLOCK_N = 64

    # Define the grid for launching the kernel.
    # Each program in the grid processes one BLOCK_M chunk of the output for one head.
    grid = (triton.cdiv(N_CTX, BLOCK_M), BATCH * N_HEADS)

    # Pre-calculate the scale factor on the host as a Python float
    scale = 1.0 / math.sqrt(D_HEAD)

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        # Strides for 4D tensors
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Tensor and head dimensions
        N_HEADS,
        N_CTX,
        D_HEAD,
        # Kernel constants
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        # Tuning hints for NVIDIA A100 (sm_80)
        num_warps=4,
        num_stages=3,
    )

    return o