import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    q_ptr, k_ptr, v_ptr, o_ptr,
    # Strides
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    # Other parameters
    num_heads,
    n_ctx,
    scale,
    # Metaparameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton kernel for causal Sigmoid Attention.
    Computes Out = (sigmoid((Q @ K^T) * scale) * causal_mask) @ V
    """
    # 1. Get program IDs to identify the current work item
    pid_m = tl.program_id(axis=1)    # Block index along the sequence dimension (M)
    pid_bh = tl.program_id(axis=0)   # Combined batch and head index

    # Decompose batch/head index
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    # 2. Compute base pointers for the current batch and head
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh

    q_ptr += q_offset
    k_ptr += k_offset
    v_ptr += v_offset
    o_ptr += o_offset

    # 3. Initialize pointers and accumulator
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Accumulator for the output block, initialized to zeros in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # 4. Load the Q block. This is done once before the inner loop.
    q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :])
    q_mask = (offs_m[:, None] < n_ctx)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 5. Inner loop over K and V blocks
    # The loop iterates up to the current query block's end to maintain causality.
    loop_end = start_m + BLOCK_M
    for start_n in range(0, loop_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # -- Load K block with swapped strides --
        # To satisfy the `tl.dot()` constraint (no `trans_b`), we must load K
        # into a [BLOCK_DMODEL, BLOCK_N] tensor.
        k_ptrs = k_ptr + (offs_d[:, None] * 1 + offs_n[None, :] * stride_kn)
        
        # Mask for loading K and V (sequence length dimension)
        kv_mask = (offs_n[None, :] < n_ctx)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)

        # -- Step 1: s = (q @ k^T) * scale --
        # q is [M, D], k is loaded as [D, N], so dot(q, k) gives [M, N]
        s = tl.dot(q, k)
        s = s * scale

        # -- Step 2: p = sigmoid(s) --
        # The causal mask is applied *after* this step, as required.
        p = 1.0 / (1.0 + tl.exp(-s))

        # -- Step 3: p = p * (j <= i) --
        # Create the causal mask and apply it by zeroing out masked elements.
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        p = tl.where(causal_mask, p, 0.0)

        # Cast p to the value tensor's dtype for the second dot product
        p = p.to(v_ptr.dtype.element_ty)

        # -- Load V block --
        v_ptrs = v_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :])
        # The mask needs to match the pointer shape [N, D]. kv_mask is [1, N],
        # so kv_mask.T is [N, 1] which broadcasts correctly.
        v = tl.load(v_ptrs, mask=kv_mask.T, other=0.0)

        # -- Step 4: out = p @ v --
        # Accumulate the result in fp32. p is [M, N], v is [N, D].
        acc += tl.dot(p, v)

    # 6. Store the final accumulated result
    o_ptrs = o_ptr + (offs_m[:, None] * stride_om + offs_d[None, :])
    # Cast accumulator to output dtype (bfloat16) before storing
    acc = acc.to(o_ptr.dtype.element_ty)
    tl.store(o_ptrs, acc, mask=q_mask)

def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the Sigmoid Attention kernel.

    Args:
        q (torch.Tensor): Query tensor of shape [2, 32, 2048, 128], dtype=bfloat16.
        k (torch.Tensor): Key tensor of shape [2, 32, 2048, 128], dtype=bfloat16.
        v (torch.Tensor): Value tensor of shape [2, 32, 2048, 128], dtype=bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype.
    """
    # Hard requirement checks from prompt
    shape_req = (2, 32, 2048, 128)
    dtype_req = torch.bfloat16
    assert q.shape == shape_req, f"Input q shape must be {shape_req}"
    assert k.shape == shape_req, f"Input k shape must be {shape_req}"
    assert v.shape == shape_req, f"Input v shape must be {shape_req}"
    assert all(t.dtype == dtype_req for t in (q, k, v)), f"All inputs must be {dtype_req}"
    assert all(t.is_cuda for t in (q, k, v)), "All inputs must be on a CUDA device"

    # Tensor dimensions
    BATCH, N_HEADS, N_CTX, D_HEAD = q.shape

    # Create output tensor
    o = torch.empty_like(q)

    # Kernel metaparameters
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid for kernel launch
    grid = (BATCH * N_HEADS, triton.cdiv(N_CTX, BLOCK_M))

    # Scaling factor passed from host as float, as required
    # 1.0 / math.sqrt(128) = 0.0883883476...
    scale = 1.0 / math.sqrt(D_HEAD)

    # Launch kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        N_HEADS,
        N_CTX,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D_HEAD,
    )

    return o