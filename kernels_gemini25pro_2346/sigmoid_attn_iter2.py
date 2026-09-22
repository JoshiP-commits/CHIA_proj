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
    stride_vz, stride_vh, stride_vm, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Matrix dimensions
    Z, H, N_CTX,
    # Kernel parameters
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # Scale factor
    scale: tl.constexpr,
):
    """
    Computes attention with element-wise sigmoid activation instead of softmax.
    Operator: out = (sigmoid((q @ k.T) * scale)) @ v
    Causal masking is applied.
    """
    # -----------------------------------------------------------
    # Map program ids to batch, head, and query block
    # -----------------------------------------------------------
    # This program instance computes a single block of output rows for one head
    # The grid is (num_m_blocks, B * H)
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    # Decompose batch/head id
    pid_z = pid_bh // H
    pid_h = pid_bh % H

    # Offset pointers to the start of the batch/head
    Q += pid_z * stride_qz + pid_h * stride_qh
    K += pid_z * stride_kz + pid_h * stride_kh
    V += pid_z * stride_vz + pid_h * stride_vh
    Out += pid_z * stride_oz + pid_h * stride_oh

    # -----------------------------------------------------------
    # Initialize pointers and accumulator
    # -----------------------------------------------------------
    # Offsets for the current block of Q rows
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # Offsets for the head dimension
    offs_d = tl.arange(0, D_HEAD)

    # Initialize accumulator with zeros
    # This will store the dot(p, v) result for the current Q block
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # -----------------------------------------------------------
    # Load Q block
    # -----------------------------------------------------------
    # Pointers to the Q block
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Mask for padding tokens in Q
    mask_m = offs_m < N_CTX
    # Load Q, zeroing out padding tokens
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # -----------------------------------------------------------
    # Loop over K and V blocks
    # -----------------------------------------------------------
    # The loop bound is `start_m + BLOCK_M` to enforce causality. We only
    # need to compute scores against keys up to the current query's position.
    end_n = start_m + BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        # -- Load K block (transposed) --
        # Offsets for the current block of K columns (and V rows)
        offs_n_block = start_n + tl.arange(0, BLOCK_N)
        # Pointers to the K block, loading it transposed [D_HEAD, BLOCK_N]
        k_ptrs = K + offs_d[:, None] * stride_kk + offs_n_block[None, :] * stride_kn
        # Mask for padding tokens in K
        mask_n = offs_n_block < N_CTX
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # -- Compute scores S = (Q @ K^T) * scale --
        # q is [BLOCK_M, D_HEAD], k is [D_HEAD, BLOCK_N]
        # s is [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k)
        s *= scale

        # -- Apply causal mask --
        # A score s_ij is only computed if query_i's index >= key_j's index
        causal_mask = offs_m[:, None] >= offs_n_block[None, :]
        # Set scores for masked-out elements to a large negative number.
        # sigmoid(-1e9) is effectively 0, so these elements won't contribute.
        s = tl.where(causal_mask, s, -1e9)

        # -- Compute P = sigmoid(S) --
        p = tl.sigmoid(s)
        # Cast p to the same dtype as V for the dot product
        p = p.to(Q.dtype.element_ty)

        # -- Load V block --
        # Pointers to the V block [BLOCK_N, D_HEAD]
        v_ptrs = V + offs_n_block[:, None] * stride_vm + offs_d[None, :] * stride_vk
        # No need to re-mask with mask_n, as p already zeroes out contributions
        # from padding tokens in K. However, it's safer to mask the load.
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # -- Accumulate output --
        # acc += P @ V
        acc += tl.dot(p, v)

    # -----------------------------------------------------------
    # Write output block to DRAM
    # -----------------------------------------------------------
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the GraphSynth Triton kernel.

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, N_CTX, D_HEAD] and dtype bfloat16.
        k (torch.Tensor): Key tensor of shape [B, H, N_CTX, D_HEAD] and dtype bfloat16.
        v (torch.Tensor): Value tensor of shape [B, H, N_CTX, D_HEAD] and dtype bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype as the inputs.
    """
    # Ensure inputs have the expected shape and dtype
    assert q.shape == (2, 32, 2048, 128) and k.shape == (2, 32, 2048, 128) and v.shape == (2, 32, 2048, 128)
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert q.is_cuda and k.is_cuda and v.is_cuda

    Z, H, N_CTX, D_HEAD = q.shape

    # Create an empty tensor for the output
    o = torch.empty_like(q)

    # Kernel configuration
    # These block sizes are chosen as a reasonable default for A100.
    BLOCK_M = 64
    BLOCK_N = 64

    # The grid defines how many instances of the kernel will be launched.
    # Each instance computes one block of M rows for one head and batch.
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    # Pre-compute scale factor on the host
    scale = 1.0 / (D_HEAD**0.5)

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        scale=scale,
    )

    return o