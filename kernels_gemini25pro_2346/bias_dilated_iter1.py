import torch
import triton
import triton.language as tl


@triton.jit
def graphsynth_kernel(
    # Pointers to input/output tensors
    Q, K, V, O,
    # Strides for tensors
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # Other parameters
    SCALE: tl.constexpr,
    N_CTX,
    N_HEAD,
    # Block-level constants
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for custom graph-based attention.

    Computes attention with a custom mask: query `i` attends to key `j`
    only if `j <= i` AND `(i - j) % 4 == 0`.

    This kernel uses a FlashAttention-style algorithm to avoid materializing
    the large score matrix and to maintain numerical stability with online softmax.
    """
    # 1. Get program IDs to identify the current work item
    pid_m = tl.program_id(1)       # ID for the M-dimension block (query block)
    pid_bh = tl.program_id(0)      # ID for the batch and head

    # De-multiplex batch and head IDs
    pid_batch = pid_bh // N_HEAD
    pid_head = pid_bh % N_HEAD

    # 2. Compute pointers and offsets for the Q block
    # This program instance processes one [BLOCK_M, D_HEAD] block of queries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)

    # Base pointers for the current batch and head
    q_base_ptr = Q + pid_batch * stride_qb + pid_head * stride_qh
    k_base_ptr = K + pid_batch * stride_kb + pid_head * stride_kh
    v_base_ptr = V + pid_batch * stride_vb + pid_head * stride_vh
    o_base_ptr = O + pid_batch * stride_ob + pid_head * stride_oh

    # Pointers for the Q block
    q_ptrs = q_base_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk

    # 3. Initialize accumulators and running statistics for online softmax
    # Accumulator for the output values, in fp32 for precision
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    # Running max for stable softmax, initialized to a large negative number
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    # Running sum of exponentials (the denominator of softmax)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # 4. Load the Q block, masking for sequences shorter than N_CTX
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 5. Main loop over KV blocks (FlashAttention-style)
    # The loop iterates up to the current query block's end to enforce causality
    end_n = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        # -- a. Load K and V blocks --
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Pointers for K (transposed load, as required by tl.dot)
        # Resulting shape in SRAM is [D_HEAD, BLOCK_N]
        k_ptrs = k_base_ptr + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
        k_mask = offs_n[None, :] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Pointers for V (standard load)
        # Resulting shape in SRAM is [BLOCK_N, D_HEAD]
        v_ptrs = v_base_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = offs_n[:, None] < N_CTX
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # -- b. Compute scores S = (Q @ K^T) * scale --
        s_ij = tl.dot(q, k)
        s_ij *= SCALE

        # -- c. Apply custom causal and stride mask --
        # Create matrices of absolute query 'i' and key 'j' indices
        i_indices = offs_m[:, None]
        j_indices = offs_n[None, :]
        
        # Causal mask: j <= i
        causal_mask = i_indices >= j_indices
        # Stride mask: (i - j) % 4 == 0
        stride_mask = ((i_indices - j_indices) % 4) == 0
        
        # Combine masks
        combined_mask = causal_mask & stride_mask
        
        # Apply mask by setting masked-out elements to a large negative number
        s_ij = tl.where(combined_mask, s_ij, -1e9)

        # -- d. Perform online softmax update --
        # 1. Get max of current scores
        m_ij_curr = tl.max(s_ij, axis=1)
        # 2. Compute new running max
        m_new = tl.maximum(m_i, m_ij_curr)
        # 3. Compute weights for rescaling previous accumulator and current probabilities
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])
        # 4. Update accumulator
        acc *= alpha[:, None]
        acc += tl.dot(p_ij, v)
        # 5. Update running sum (denominator)
        l_i = l_i * alpha + tl.sum(p_ij, axis=1)
        # 6. Update running max
        m_i = m_new

    # 6. Post-loop normalization and storing the result
    # Normalize the accumulator with the final sum
    # Use tl.where to avoid division by zero for fully masked rows
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc /= l_i_safe[:, None]

    # Pointers to the output block
    o_ptrs = o_base_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    # Mask for storing, same as the Q loading mask
    o_mask = offs_m[:, None] < N_CTX
    # Store the final result, casting back to the input dtype
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=o_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the GraphSynth Triton kernel.

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, N, D], dtype bfloat16.
        k (torch.Tensor): Key tensor of shape [B, H, N, D], dtype bfloat16.
        v (torch.Tensor): Value tensor of shape [B, H, N, D], dtype bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype.
    """
    # 1. Get tensor dimensions
    BATCH, N_HEAD, N_CTX, D_HEAD = q.shape

    # Hard requirement check
    assert D_HEAD == 128, "D_HEAD must be 128"
    assert q.shape == k.shape == v.shape, "Input tensor shapes must be identical"
    assert q.dtype == torch.bfloat16, "Input tensors must be bfloat16"

    # 2. Allocate output tensor
    o = torch.empty_like(q)

    # 3. Define tiling constants for the kernel
    # These could be tuned for performance
    BLOCK_M = 128
    BLOCK_N = 64

    # 4. Set up the launch grid
    # Each program on the grid computes one [BLOCK_M, D_HEAD] block of the output
    grid = (BATCH * N_HEAD, triton.cdiv(N_CTX, BLOCK_M))
    
    # 5. Compute the scale factor on the host
    scale = 1.0 / (D_HEAD**0.5)

    # 6. Launch the kernel
    graphsynth_kernel[grid](
        # Tensors
        q, k, v, o,
        # Strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Meta-parameters
        SCALE=scale,
        N_CTX=N_CTX,
        N_HEAD=N_HEAD,
        # Kernel constants
        D_HEAD=D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return o