import math
import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    # Pointers to input/output tensors
    Q, K, V, O,
    # Stride variables for tensors
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_vz, stride_vh, stride_vm, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Other parameters
    sm_scale,
    N_CTX,
    NUM_HEADS: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for GraphSynth attention.
    Computes attention with a causal and strided mask: (j <= i) AND ((i - j) % 4 == 0).
    This kernel implements a FlashAttention-style algorithm to avoid materializing the
    large score matrix in DRAM.
    """
    # 1. Program and Tile Identification
    # Each program instance handles one query block from one head in one batch
    start_m = tl.program_id(0) * BLOCK_M
    pid_bh = tl.program_id(1)

    # Decompose the combined batch-head dimension
    batch = pid_bh // NUM_HEADS
    head = pid_bh % NUM_HEADS

    # 2. Pointer Setup
    # Advance pointers to the current batch and head
    q_base = Q + batch * stride_qz + head * stride_qh
    k_base = K + batch * stride_kz + head * stride_kh
    v_base = V + batch * stride_vz + head * stride_vh
    o_base = O + batch * stride_oz + head * stride_oh

    # 3. Initialize Accumulators and Offsets
    # Offsets for the current query block (M-dimension) and head dimension (D-dimension)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Initialize accumulators for the online softmax algorithm
    # `acc` stores the running weighted sum of values
    # `m_i` stores the running maximum of scores
    # `l_i` stores the running sum of exponentiated scores (the normalizer)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # 4. Load Query Block
    # Load the query block for this program instance once, as it's reused across all KV blocks
    q_ptrs = q_base + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    # Mask to prevent loading from padding tokens
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 5. Loop over Key-Value Blocks
    # The loop iterates over the sequence length in blocks of BLOCK_N.
    # The upper bound is set to handle the causal nature of the attention.
    loop_end = start_m + BLOCK_M
    for start_n in range(0, loop_end, BLOCK_N):
        # -- a. Load K and V for the current block --
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Pointers for K (transposed load) and V (standard load)
        # For K, we swap n and d to load a D x N tile for the dot product
        k_ptrs = k_base + (offs_d[:, None] * stride_kk + offs_n[None, :] * stride_km)
        v_ptrs = v_base + (offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vk)

        # Boundary mask for the N-dimension (sequence length)
        kv_mask = offs_n[None, :] < N_CTX
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        # -- b. Compute Attention Scores --
        # Q(M x D) @ K(D x N) -> S(M x N)
        s = tl.dot(q, k)
        s *= sm_scale

        # -- c. Apply Attention Mask --
        # The mask combines causal, strided, and boundary conditions
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        stride_mask = ((offs_m[:, None] - offs_n[None, :]) % 4) == 0
        boundary_mask = offs_n[None, :] < N_CTX
        
        full_mask = causal_mask & stride_mask & boundary_mask
        s = tl.where(full_mask, s, -1e9)

        # -- d. Perform Online Softmax Update --
        # 1. Get max of current scores
        m_ij = tl.max(s, axis=1)
        # 2. Find new running max
        m_i_new = tl.maximum(m_i, m_ij)
        # 3. Calculate rescaling factor for old accumulator and normalizer
        alpha = tl.exp(m_i - m_i_new)
        # 4. Rescale previous accumulator and normalizer
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        # 5. Calculate probabilities for the current block
        p_ij = tl.exp(s - m_i_new[:, None])
        # 6. Update running normalizer
        l_ij = tl.sum(p_ij, axis=1)
        l_i = l_i + l_ij
        # 7. Update accumulator with new values
        # Cast probabilities to V's dtype for the dot product
        p_ij = p_ij.to(v.dtype)
        acc = acc + tl.dot(p_ij, v)
        # 8. Update running max for the next iteration
        m_i = m_i_new

    # 6. Final Normalization and Store Output
    # Normalize the accumulator by the final normalizer sum
    # Use a safe guard against division by zero
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    # Pointers to the output block
    o_ptrs = o_base + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    # Mask to prevent writing to padding tokens
    o_mask = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=o_mask)

def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launch function for the GraphSynth Triton kernel.

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, N, D], dtype bfloat16.
        k (torch.Tensor): Key tensor of shape [B, H, N, D], dtype bfloat16.
        v (torch.Tensor): Value tensor of shape [B, H, N, D], dtype bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype as the inputs.
    """
    assert q.shape == k.shape and q.shape == v.shape, "Input tensors must have the same shape"
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16, "Inputs must be bfloat16"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on a CUDA device"
    
    B, H, N_CTX, D_HEAD = q.shape
    
    # Create an empty tensor for the output
    o = torch.empty_like(q)

    # Define kernel constants and tile sizes
    # These values are chosen for good performance on A100.
    BLOCK_M = 128
    BLOCK_N = 64

    # Calculate the softmax scaling factor
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    # Define the grid for launching the kernel. Each program instance
    # computes a BLOCK_M-sized chunk of the output for one head.
    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)

    # Call the Triton kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale=sm_scale,
        N_CTX=N_CTX,
        NUM_HEADS=H,
        BLOCK_M=BLOCK_M,
        BLOCK_D=D_HEAD,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return o