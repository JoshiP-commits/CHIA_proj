import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    Q, K, V, Out,
    # Strides
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # Other parameters
    batch_size, num_heads, seq_len,
    scale: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    """
    Triton kernel for a custom attention-like mechanism.

    Operator:
    s = (q @ k.T) * scale
    s is causally masked (j <= i)
    p = sigmoid(s)
    out = p @ v

    This kernel implements the operation using a tiled approach inspired by
    FlashAttention to avoid materializing the large score matrix in DRAM.
    """
    # 1. Get Program IDs to identify the work item
    pid_m = tl.program_id(axis=0)  # Block index along the sequence length dimension of Q
    pid_bh = tl.program_id(axis=1) # Combined batch and head index

    # Unpack batch and head indices
    pid_batch = pid_bh // num_heads
    pid_head = pid_bh % num_heads

    # 2. Compute pointer offsets for the current work item
    # Offsets for the M-dimension (rows of Q, rows of Out)
    rm_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Offsets for the D_HEAD dimension (columns of Q, K, V, Out)
    rk_offsets = tl.arange(0, BLOCK_DMODEL)

    # 3. Initialize pointers to the start of the current batch/head block
    Q_block_ptr = Q + pid_batch * stride_qb + pid_head * stride_qh
    K_block_ptr = K + pid_batch * stride_kb + pid_head * stride_kh
    V_block_ptr = V + pid_batch * stride_vb + pid_head * stride_vh
    Out_block_ptr = Out + pid_batch * stride_ob + pid_head * stride_oh

    # 4. Initialize accumulator for the output block with zeros
    # This accumulator will be in fp32 for precision.
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 5. Load the block of Q for the current work item
    # Pointers to the [BLOCK_M, BLOCK_DMODEL] tile of Q
    q_ptrs = Q_block_ptr + (rm_offsets[:, None] * stride_qm + rk_offsets[None, :] * stride_qk)
    # Boundary check mask for Q
    q_mask = rm_offsets[:, None] < seq_len
    # Load Q, apply scale factor, and cast to the appropriate data type
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q = (q * scale).to(q.dtype)

    # 6. Loop over blocks of K and V along the sequence length dimension
    # The loop is bounded to handle causality: a query at position `i`
    # can only attend to keys at positions `j <= i`.
    end_n = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        # -- a. Load K tile (transposed) --
        # Offsets for the N-dimension (sequence length of K/V)
        rn_offsets = start_n + tl.arange(0, BLOCK_N)
        # Pointers to the [D_HEAD, BLOCK_N] tile of K. Note the swapped strides
        # to perform the transpose during load, as tl.dot does not have trans_b.
        k_ptrs = K_block_ptr + (rk_offsets[:, None] * stride_kk + rn_offsets[None, :] * stride_km)
        # Boundary check mask for K
        k_mask = rn_offsets[None, :] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # -- b. Compute scores S = Q @ K^T --
        # q: [BLOCK_M, D_HEAD], k: [D_HEAD, BLOCK_N] -> s: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k)

        # -- c. Apply causal mask --
        # This handles the fine-grained mask within the current block.
        # Block-level causality is handled by the `end_n` loop boundary.
        causal_mask = rm_offsets[:, None] >= rn_offsets[None, :]
        # Set scores for masked-out elements to a large negative number
        # so that sigmoid(s) evaluates to approximately 0.
        s = tl.where(causal_mask, s, -1e9)

        # -- d. Compute P = sigmoid(S) --
        # Promote to float32 for sigmoid calculation to maintain precision
        p = tl.sigmoid(s.to(tl.float32))
        # Cast back to bfloat16 for the dot product with V
        p = p.to(q.dtype)

        # -- e. Load V tile --
        # Pointers to the [BLOCK_N, D_HEAD] tile of V
        v_ptrs = V_block_ptr + (rn_offsets[:, None] * stride_vm + rk_offsets[None, :] * stride_vk)
        # Boundary check mask for V
        v_mask = rn_offsets[:, None] < seq_len
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # -- f. Accumulate output --
        # p: [BLOCK_M, BLOCK_N], v: [BLOCK_N, D_HEAD] -> [BLOCK_M, D_HEAD]
        acc += tl.dot(p, v)

    # 7. Store the final accumulated output block to DRAM
    # Pointers to the [BLOCK_M, D_HEAD] tile of the output tensor
    out_ptrs = Out_block_ptr + (rm_offsets[:, None] * stride_om + rk_offsets[None, :] * stride_ok)
    # Boundary check mask for storing the output
    out_mask = rm_offsets[:, None] < seq_len
    # Cast accumulator to bfloat16 and store
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the graphsynth_kernel.

    Args:
        q: Query tensor of shape [B, H, N_CTX, D_HEAD], dtype=bfloat16.
        k: Key tensor of shape [B, H, N_CTX, D_HEAD], dtype=bfloat16.
        v: Value tensor of shape [B, H, N_CTX, D_HEAD], dtype=bfloat16.

    Returns:
        Output tensor of shape [B, H, N_CTX, D_HEAD], dtype=bfloat16.
    """
    # Ensure inputs are bfloat16 as per requirement
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    
    # Get tensor dimensions
    BATCH, N_HEAD, N_CTX, D_HEAD = q.shape

    # Create the output tensor, same shape and device as inputs
    o = torch.empty_like(q)

    # Set meta-parameters for the kernel
    # These are tuned for A100 performance
    BLOCK_M = 128
    BLOCK_N = 64
    num_warps = 4
    num_stages = 2

    # Define the grid for kernel launch
    # One program instance per block of Q rows per batch/head
    grid = (triton.cdiv(N_CTX, BLOCK_M), BATCH * N_HEAD)

    # Pre-compute the scale factor on the host
    scale = 1.0 / (D_HEAD**0.5)

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BATCH, N_HEAD, N_CTX,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D_HEAD,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o