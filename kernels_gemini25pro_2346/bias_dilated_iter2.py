import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    Q, K, V, O,
    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    # Other parameters
    n_heads,
    seq_len,
    head_dim,
    scale,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Computes attention with a causal and a strided mask.
    The mask condition is (j <= i) AND ((i - j) % 4 == 0).
    This kernel is styled after FlashAttention.
    """
    # 1. Program and thread IDs
    pid_m = tl.program_id(axis=0)    # Block index along the sequence dimension (M)
    pid_bh = tl.program_id(axis=1)   # Combined batch and head index

    # Decompose the combined batch-head index to get individual batch and head IDs
    pid_batch = pid_bh // n_heads
    pid_head = pid_bh % n_heads

    # 2. Pointer setup
    # Base pointers for the current batch and head
    q_base_ptr = Q + pid_batch * stride_qb + pid_head * stride_qh
    k_base_ptr = K + pid_batch * stride_kb + pid_head * stride_kh
    v_base_ptr = V + pid_batch * stride_vb + pid_head * stride_vh
    o_base_ptr = O + pid_batch * stride_ob + pid_head * stride_oh

    # Offsets for the current query block (fixed for this program instance)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # Pointers to the Q block that this program will process
    q_ptrs = q_base_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd

    # 3. Initialise accumulators and statistics
    # Accumulator for the output
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    # Running sum of exponentials for softmax normalization
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Running max of scores for numerical stability
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)

    # 4. Load Q block (once per program)
    # Create a mask to avoid loading from out-of-bounds memory
    q_load_mask = offs_m[:, None] < seq_len
    # Load Q and promote to float32 for computation
    q = tl.load(q_ptrs, mask=q_load_mask, other=0.0).to(tl.float32)

    # 5. Main loop over KV blocks
    # The loop iterates up to the current query block's end to enforce causality
    end_n_causal = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_n_causal, BLOCK_N):
        # a. -- Load K and V blocks --
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Pointers to K (transposed layout for dot product)
        k_ptrs = k_base_ptr + (offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks)
        # Pointers to V (standard layout)
        v_ptrs = v_base_ptr + (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)

        # Load K and V with masking for sequences not perfectly divisible by BLOCK_N
        k_load_mask = offs_n[None, :] < seq_len
        v_load_mask = offs_n[:, None] < seq_len
        k = tl.load(k_ptrs, mask=k_load_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=v_load_mask, other=0.0).to(tl.float32)

        # b. -- Compute attention scores --
        s = tl.dot(q, k)
        s *= scale

        # c. -- Apply causal and strided masks --
        # Global indices for queries (i) and keys (j)
        i = offs_m[:, None]
        j = offs_n[None, :]
        
        # Create masks based on the operator definition
        causal_mask = (j <= i)
        stride_mask = ((i - j) % 4) == 0
        
        # Apply masks by setting masked-out scores to a large negative number
        s = tl.where(causal_mask & stride_mask, s, -1e9)

        # d. -- Perform online softmax update --
        # 1. Get max of current scores
        m_ij = tl.max(s, axis=1)
        # 2. Correct old statistics and accumulator for the new max
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        # 3. Calculate probabilities for the current block, scaled by new max
        p_ij = tl.math.exp(s - m_new[:, None])
        # 4. Update accumulator and statistics with current block's values
        l_i += tl.sum(p_ij, axis=1)
        acc += tl.dot(p_ij, v)
        # 5. Update the running max
        m_i = m_new

    # 6. Final normalization and write to output
    # Invert the final sum of exponentials, handling cases where it's zero
    l_i_inv = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    # Normalize the accumulator
    acc = acc * l_i_inv[:, None]

    # Pointers to the output block
    o_ptrs = o_base_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    # Create a mask to avoid writing to out-of-bounds memory
    o_write_mask = offs_m[:, None] < seq_len
    # Store the result, converting back to the input dtype
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=o_write_mask)

def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launch function for the GraphSynth Triton kernel.

    Args:
        q: Query tensor of shape [B, H, N_CTX, D_HEAD] and dtype bfloat16.
        k: Key tensor of shape [B, H, N_CTX, D_HEAD] and dtype bfloat16.
        v: Value tensor of shape [B, H, N_CTX, D_HEAD] and dtype bfloat16.

    Returns:
        Output tensor of the same shape and dtype as the inputs.
    """
    # Shape and dimension checks
    assert q.shape == k.shape == v.shape, "Input tensors must have the same shape"
    assert q.dim() == 4, "Input tensors must be 4-dimensional"
    assert q.dtype == torch.bfloat16, "Input tensors must be bfloat16"
    B, H, N_CTX, D_HEAD = q.shape

    # Allocate output tensor
    o = torch.empty_like(q)

    # Pre-compute scaling factor
    scale = 1.0 / math.sqrt(D_HEAD)

    # Kernel meta-parameters
    # These values are tuned for A100 performance, balancing SRAM usage and parallelism.
    BLOCK_M = 128
    BLOCK_N = 64 # Changed to 64 for better occupancy and compatibility with D_HEAD=128

    # Set up the launch grid. Each program instance computes one BLOCK_M chunk
    # of the output for a specific batch and head.
    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H,
        N_CTX,
        D_HEAD,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return o