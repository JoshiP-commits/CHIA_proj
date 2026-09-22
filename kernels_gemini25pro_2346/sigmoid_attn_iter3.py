import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to matrices
    Q, K, V, Out,
    # Stride variables
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Other metadata
    Z, H, N_CTX,
    # Kernel constants
    sm_scale: float,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DHEAD: tl.constexpr,
):
    """
    Triton kernel for a custom attention-like operation on A100 (sm80).
    Operator: out = sigmoid((q @ k.T) * scale) @ v, with causal masking.
    This kernel implements FlashAttention-style tiling to avoid materializing the
    large score matrix in DRAM. It uses fp32 accumulators for precision.
    """
    # -----------------------------------------------------------
    # Map program ids to batch, head, and sequence block
    # This kernel is launched in a 2D grid.
    # axis=0 corresponds to the batch * head dimension.
    # axis=1 corresponds to the block index along the sequence length (M dimension).
    pid_m_block = tl.program_id(axis=1)
    pid_bh = tl.program_id(axis=0)

    # Decompose the combined batch/head index to get specific z and h indices.
    pid_z = pid_bh // H
    pid_h = pid_bh % H

    # -----------------------------------------------------------
    # Compute pointers for the current block
    # Create offsets for the M and D_HEAD dimensions.
    offs_m = pid_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DHEAD)

    # Q pointers: pointing to a [BLOCK_M, BLOCK_DHEAD] block
    q_offset_bh = pid_z * stride_qz + pid_h * stride_qh
    q_ptrs = Q + q_offset_bh + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

    # K and V base pointers for the current batch and head
    k_offset_bh = pid_z * stride_kz + pid_h * stride_kh
    K_ptrs_base = K + k_offset_bh
    v_offset_bh = pid_z * stride_vz + pid_h * stride_vh
    V_ptrs_base = V + v_offset_bh

    # Output pointers, similar to Q
    out_offset_bh = pid_z * stride_oz + pid_h * stride_oh
    out_ptrs = Out + out_offset_bh + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)

    # -----------------------------------------------------------
    # Load the Q block and initialize the accumulator
    # Masking is applied to prevent out-of-bounds access for sequences
    # not perfectly divisible by BLOCK_M.
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize the accumulator for the output block in high precision (float32).
    acc = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)

    # -----------------------------------------------------------
    # Main loop over K and V blocks (FlashAttention-style)
    # The upper bound of the loop is determined by the current Q block's position,
    # ensuring causality (a query can only attend to keys that came before it).
    causal_end_n = (pid_m_block + 1) * BLOCK_M
    for lo in range(0, causal_end_n, BLOCK_N):
        # --- Create pointers and load K (transposed) and V for the current block ---
        offs_n_loop = lo + tl.arange(0, BLOCK_N)

        # K pointers for loading a [D_HEAD, BLOCK_N] block. This is K transposed.
        # This is required because tl.dot does not support trans_b.
        k_ptrs = K_ptrs_base + (offs_d[:, None] * stride_kk + offs_n_loop[None, :] * stride_kn)

        # V pointers for loading a [BLOCK_N, D_HEAD] block.
        v_ptrs = V_ptrs_base + (offs_n_loop[:, None] * stride_vn + offs_d[None, :] * stride_vk)

        # Load K and V with masking to handle sequence boundaries.
        kv_mask = offs_n_loop[None, :] < N_CTX
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n_loop[:, None] < N_CTX, other=0.0)

        # --- Compute scores S = (Q @ K^T) * scale ---
        # tl.dot accumulates in fp32 by default for bfloat16 inputs.
        s = tl.dot(q, k, out_dtype=tl.float32)
        s *= sm_scale

        # --- Apply the causal mask (j <= i) ---
        # Create a mask where True indicates an element should be kept.
        causal_mask = offs_m[:, None] >= offs_n_loop[None, :]
        # Apply the mask. Where the mask is False, substitute a large negative number.
        # This ensures that after the sigmoid, these elements become ~0 and do not
        # contribute to the output, as per the operator requirements.
        s = tl.where(causal_mask, s, -1e9)

        # --- Compute P = sigmoid(S) ---
        p = tl.sigmoid(s)

        # Cast P from fp32 to bfloat16 to match V's dtype for the second dot product.
        p = p.to(tl.bfloat16)

        # --- Accumulate output: acc += P @ V ---
        acc += tl.dot(p, v)

    # -----------------------------------------------------------
    # Write the final accumulated block to the output tensor
    # Cast the fp32 accumulator back to bfloat16 for storage.
    acc = acc.to(tl.bfloat16)
    tl.store(out_ptrs, acc, mask=q_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launch the Triton kernel for the custom attention-like operation.

    Args:
        q: Query tensor of shape [Z, H, N_CTX, D_HEAD] (bfloat16, on CUDA).
        k: Key tensor of shape [Z, H, N_CTX, D_HEAD] (bfloat16, on CUDA).
        v: Value tensor of shape [Z, H, N_CTX, D_HEAD] (bfloat16, on CUDA).

    Returns:
        An output tensor of the same shape and dtype as the inputs.
    """
    # Input validation
    assert q.shape == k.shape == v.shape, "Input tensors must have the same shape"
    assert q.dtype == torch.bfloat16, "Input tensors must be of dtype bfloat16"
    assert q.is_cuda, "Input tensors must be on a CUDA device"
    assert q.shape[-1] == 128, "D_HEAD must be 128 as per requirements"
    assert q.dim() == 4, "Input tensors must be 4-dimensional"

    # Get tensor dimensions from the query tensor
    Z, H, N_CTX, D_HEAD = q.shape

    # Create the output tensor with the same properties as the input
    out = torch.empty_like(q)

    # Heuristics for block sizes on A100
    BLOCK_M = 128
    BLOCK_N = 64

    # Define the grid for launching the kernel
    # Each program in the grid handles one block of M for one batch/head combination.
    grid = (Z * H, triton.cdiv(N_CTX, BLOCK_M))

    # Compute the scaling factor on the host as a float
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    # Launch the kernel
    graphsynth_kernel[grid](
        # Tensors
        q, k, v, out,
        # Strides for each tensor
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # Other metadata
        Z, H, N_CTX,
        # Kernel constants passed as arguments
        sm_scale=sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DHEAD=D_HEAD,
        # Tuning parameters for sm80 (A100)
        num_warps=4,
        num_stages=2,
    )

    return out