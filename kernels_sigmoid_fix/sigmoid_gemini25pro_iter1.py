import torch
import triton
import triton.language as tl


@triton.jit
def graphsynth_kernel(
    # Tensors
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Strides for 3D tensors [B*H, N_CTX, D_HEAD]
    stride_qz, stride_qn,
    stride_kz, stride_kn,
    stride_vz, stride_vn,
    stride_oz, stride_on,
    # Dimensions
    N_CTX,
    D_HEAD,
    # Attention scale factor
    scale: float,
    # Compile-time constants for block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for causal sigmoid attention on sm80.
    Computes Out = (sigmoid((Q @ K^T) * scale) * causal_mask) @ V
    """
    # 1. Get program IDs to identify the current block
    # The grid is (B*H, N_CTX / BLOCK_M), so pid_z is the batch/head index
    # and pid_m is the index for the query block along the sequence length.
    pid_z = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # 2. Compute offsets for loading data
    # Offsets for the query block (rows)
    offs_m = (pid_m * BLOCK_M) + tl.arange(0, BLOCK_M)
    # Offsets for the head dimension (columns)
    offs_d = tl.arange(0, D_HEAD)

    # 3. Setup pointers and load query block
    # Base pointers for the current batch/head
    q_base_ptr = Q_ptr + pid_z * stride_qz
    k_base_ptr = K_ptr + pid_z * stride_kz
    v_base_ptr = V_ptr + pid_z * stride_vz
    out_base_ptr = Out_ptr + pid_z * stride_oz

    # Pointers to the query block of shape (BLOCK_M, D_HEAD)
    q_ptrs = q_base_ptr + (offs_m[:, None] * stride_qn) + (offs_d[None, :] * 1)

    # Mask to avoid loading from memory locations outside the sequence length
    q_mask = offs_m[:, None] < N_CTX
    # Load the query block
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 4. Initialize accumulator
    # Use float32 for the accumulator to maintain precision during summation
    acc = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)

    # 5. Loop over key/value blocks
    # We iterate over the sequence length of K and V in blocks of BLOCK_N.
    # Causality is enforced by ensuring the loop only goes up to the current query block.
    loop_end = (pid_m + 1) * BLOCK_M
    for start_n in range(0, loop_end, BLOCK_N):
        # -- a. Load K and V for the current block --
        # Offsets for the key/value block (columns in the score matrix)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Pointers for K block, loaded as if transposed: shape (D_HEAD, BLOCK_N)
        # This is to satisfy the tl.dot() constraint of no `trans_b`.
        k_ptrs = k_base_ptr + (offs_d[:, None] * 1) + (offs_n[None, :] * stride_kn)
        
        # Pointers for V block, loaded normally: shape (BLOCK_N, D_HEAD)
        v_ptrs = v_base_ptr + (offs_n[:, None] * stride_vn) + (offs_d[None, :] * 1)

        # Mask for keys/values to avoid loading from outside the sequence length
        kv_mask = offs_n[None, :] < N_CTX
        
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # -- b. Compute scores: s = (q @ k^T) * scale --
        # q: (BLOCK_M, D_HEAD), k: (D_HEAD, BLOCK_N) -> s: (BLOCK_M, BLOCK_N)
        s = tl.dot(q, k)
        s *= scale

        # -- c. Apply sigmoid element-wise --
        p = tl.sigmoid(s)

        # -- d. Apply causal mask AFTER sigmoid --
        # The mask is True for elements where query_index >= key_index (j <= i).
        # We use tl.where to set scores for masked-out positions to 0.0.
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        p = tl.where(causal_mask, p, 0.0)
        
        # -- e. Update accumulator: acc += p @ v --
        # Cast p to the value tensor's dtype before the dot product.
        p = p.to(V_ptr.dtype.element_ty)
        # p: (BLOCK_M, BLOCK_N), v: (BLOCK_N, D_HEAD) -> delta_out: (BLOCK_M, D_HEAD)
        delta_out = tl.dot(p, v)
        acc += delta_out

    # 6. Store final result
    # Pointers to the output block
    out_ptrs = out_base_ptr + (offs_m[:, None] * stride_on) + (offs_d[None, :] * 1)
    # Use the query mask to avoid writing outside the valid sequence length
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=q_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launches the Triton kernel for causal sigmoid attention.

    This function reshapes the 4D input tensors (Batch, Heads, SeqLen, HeadDim)
    into 3D tensors, sets up the Triton grid, and calls the `graphsynth_kernel`.

    Args:
        q (torch.Tensor): Query tensor of shape [2, 32, 2048, 128] and dtype bfloat16.
        k (torch.Tensor): Key tensor of shape [2, 32, 2048, 128] and dtype bfloat16.
        v (torch.Tensor): Value tensor of shape [2, 32, 2048, 128] and dtype bfloat16.

    Returns:
        torch.Tensor: The output tensor of the same shape and dtype as the inputs.
    """
    # Ensure inputs are contiguous for correct stride calculations
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    # Get dimensions from the input tensor
    BATCH, N_HEADS, N_CTX, D_HEAD = q.shape

    # Allocate the output tensor
    out = torch.empty_like(q)

    # Reshape tensors to 3D for the kernel: [B*H, N, D]
    # The kernel will operate on one sequence from the batch/head group at a time.
    q_3d = q.view(BATCH * N_HEADS, N_CTX, D_HEAD)
    k_3d = k.view(BATCH * N_HEADS, N_CTX, D_HEAD)
    v_3d = v.view(BATCH * N_HEADS, N_CTX, D_HEAD)
    out_3d = out.view(BATCH * N_HEADS, N_CTX, D_HEAD)

    # Tiling configuration for the kernel
    BLOCK_M = 128
    BLOCK_N = 64

    # Grid definition for the kernel launch
    # The grid is 2D:
    # - Dimension 0 covers all sequences in the batch (BATCH * N_HEADS).
    # - Dimension 1 tiles the sequence length into blocks of size BLOCK_M.
    grid = (BATCH * N_HEADS, triton.cdiv(N_CTX, BLOCK_M))

    # The scaling factor is 1/sqrt(D_HEAD), passed as a float.
    scale = D_HEAD**-0.5

    # Launch the kernel
    graphsynth_kernel[grid](
        # Tensors
        q_3d, k_3d, v_3d, out_3d,
        # Strides for the 3D tensors [Z, N, D]
        q_3d.stride(0), q_3d.stride(1),
        k_3d.stride(0), k_3d.stride(1),
        v_3d.stride(0), v_3d.stride(1),
        out_3d.stride(0), out_3d.stride(1),
        # Dimensions and scale factor
        N_CTX, D_HEAD, scale,
        # Compile-time constants for block sizes
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        # Performance hints for the Triton compiler
        num_warps=8,
        num_stages=2,
    )

    # The result is written in-place to `out`, so we can return it directly.
    return out