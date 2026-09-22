import torch
import triton
import triton.language as tl
import math


@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    Q, K, V, Out,
    # Stride variables for each tensor
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Other parameters
    Z, H, N_CTX,
    scale: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DHEAD: tl.constexpr,
):
    """
    Triton kernel for a custom attention-like operation on sm80.
    Computes Out = relu((Q @ K^T) * scale) @ V with causal masking.
    This implementation uses a tiled approach similar to FlashAttention
    to avoid materializing the large score matrix S in DRAM.
    """
    # 1. Get program IDs to identify the current work item
    pid_m = tl.program_id(0)  # Block index along the sequence dimension (M)
    pid_z = tl.program_id(1)  # Batch index
    pid_h = tl.program_id(2)  # Head index

    # 2. Compute pointers to the start of the current head's data
    q_ptr = Q + pid_z * stride_qz + pid_h * stride_qh
    k_ptr = K + pid_z * stride_kz + pid_h * stride_kh
    v_ptr = V + pid_z * stride_vz + pid_h * stride_vh
    out_ptr = Out + pid_z * stride_oz + pid_h * stride_oh

    # 3. Initialize accumulator for the output block
    # The accumulator is in fp32 for higher precision during summation.
    accumulator = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)

    # 4. Define offsets for the current block of Q queries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DHEAD)
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk

    # 5. Load the block of Q, masking for rows outside the actual sequence length
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 6. Loop over blocks of K and V along the sequence dimension (N)
    # The loop bound is determined by causality: a query at position `m` can only
    # attend to keys at positions `n <= m`.
    end_n_loop_bound = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_n_loop_bound, BLOCK_N):
        # -- Load K block (transposed) --
        # To compute Q @ K^T without `trans_b` in `tl.dot`, we load K
        # into a [BLOCK_DHEAD, BLOCK_N] layout.
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_ptr + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
        k_mask = offs_n[None, :] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # -- Compute scores S = (Q @ K^T) * scale --
        s = tl.dot(q, k)
        s *= scale

        # -- Apply causal mask --
        # This ensures that for a query i, we only use keys j where j <= i.
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        # Set scores for masked-out elements to a large negative number.
        # This guarantees that relu(s) will be 0 for these elements,
        # fulfilling the "masked contribute 0" requirement.
        s = tl.where(causal_mask, s, -1e9)

        # -- Compute P = relu(S) --
        # Triton does not have tl.relu, so we implement it with tl.where.
        p = tl.where(s > 0, s, 0.0)
        # Convert P to the same dtype as V for the dot product.
        p = p.to(tl.bfloat16)

        # -- Load V block --
        v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = offs_n[:, None] < N_CTX
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # -- Accumulate output: accumulator += P @ V --
        accumulator += tl.dot(p, v)

    # 7. Store the final accumulated block to the output tensor
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    out_mask = offs_m[:, None] < N_CTX
    tl.store(out_ptrs, accumulator, mask=out_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launchpad for the `graphsynth_kernel`.

    Args:
        q (torch.Tensor): Query tensor of shape [2, 32, 2048, 128] and dtype bfloat16.
        k (torch.Tensor): Key tensor of shape [2, 32, 2048, 128] and dtype bfloat16.
        v (torch.Tensor): Value tensor of shape [2, 32, 2048, 128] and dtype bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype as the inputs.
    """
    # Extract tensor dimensions and validate inputs
    Z, H, N_CTX, D_HEAD = q.shape

    # Create the output tensor on the same device as the inputs
    o = torch.empty_like(q)

    # Define tiling sizes. These are chosen to fit into A100's 64KB SRAM.
    # Q (128x128, bf16) = 32KB
    # K (64x128, bf16)  = 16KB
    # V (64x128, bf16)  = 16KB
    # Total = 64KB, which is a good fit.
    BLOCK_M = 128
    BLOCK_N = 64

    # Define the grid for launching the kernel. Each kernel instance processes
    # one BLOCK_M-sized block of rows from the output matrix.
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z, H)

    # Pre-compute the scaling factor on the host as a float.
    scale = 1.0 / math.sqrt(D_HEAD)

    # Launch the Triton kernel.
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DHEAD=D_HEAD,
        num_warps=8,
        num_stages=3
    )

    return o