import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    # Pointers to Matrices
    Q, K, V, O,
    # Stride information for each matrix
    s_qb, s_qh, s_qn, s_qd,
    s_kb, s_kh, s_kn, s_kd,
    s_vb, s_vh, s_vn, s_vd,
    s_ob, s_oh, s_on, s_od,
    # Other metadata
    B, H, N_CTX, D_HEAD,
    scale,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Computes causal self-attention with a per-head temperature scaling.
    This kernel is executed in a 2D grid:
    - grid_x (pid_bh): Batch and Head dimensions combined (B * H)
    - grid_y (pid_m): Sequence dimension blocks (N_CTX / BLOCK_M)
    """
    # 1. Get Program IDs and derive batch, head, and query block indices
    pid_m = tl.program_id(1)
    pid_bh = tl.program_id(0)

    # Decompose the combined batch-head program ID
    h = pid_bh % H
    b = pid_bh // H

    # Compute per-head temperature: t_h = 0.5 + h/32
    # HARD REQUIREMENT: Derive per-head constants from head index `h`
    t_h = 0.5 + h / 32.0

    # Pointers to the start of the current Q, K, V, O tensors for this head
    q_base_ptr = Q + b * s_qb + h * s_qh
    k_base_ptr = K + b * s_kb + h * s_kh
    v_base_ptr = V + b * s_vb + h * s_vh
    o_base_ptr = O + b * s_ob + h * s_oh

    # Offsets for the current query block this program is responsible for
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # 2. Initialize accumulators for the FlashAttention algorithm
    # `acc` stores the unnormalized output vector
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # `m_i` stores the running maximum of scores for the online softmax
    # HARD REQUIREMENT: Init running max with -1e9
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    # `l_i` stores the running sum of exp(scores - max)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # 3. Load the current query block from DRAM
    # q_ptrs shape: [BLOCK_M, BLOCK_DMODEL]
    q_ptrs = q_base_ptr + (offs_m[:, None] * s_qn + offs_d[None, :] * s_qd)
    q_mask = offs_m[:, None] < N_CTX
    # Load `q` block, converting to float32 for computation
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # 4. Main loop over key/value blocks (causal)
    # The loop bound `(pid_m + 1) * BLOCK_M` ensures block-level causality.
    # HARD REQUIREMENT: Do NOT use `continue` or `break` in the loop.
    for start_n in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        # 4a. Load K (transposed) and V blocks
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Load K^T, shape [BLOCK_DMODEL, BLOCK_N]
        # HARD REQUIREMENT: tl.dot() has NO `trans_b`. Load K with swapped strides.
        k_ptrs = k_base_ptr + (offs_d[:, None] * s_kd + offs_n[None, :] * s_kn)
        k_mask = offs_n[None, :] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Load V, shape [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = v_base_ptr + (offs_n[:, None] * s_vn + offs_d[None, :] * s_vd)
        v_mask = offs_n[:, None] < N_CTX
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # 4b. Compute scores S = Q @ K^T
        # q is [BLOCK_M, D_HEAD], k is [D_HEAD, BLOCK_N] -> s is [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k, allow_tf32=True)
        
        # Apply scaling and per-head temperature
        s = s * scale * t_h

        # 4c. Apply causal mask within the block
        # HARD REQUIREMENT: Use tl.where for masking; no boolean indexing.
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal_mask, s, -1e9)
        
        # 4d. Update softmax statistics (running max and sum)
        m_ij = tl.max(s, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)

        # 4e. Update the output accumulator
        acc = acc * alpha[:, None]
        p = p.to(v.dtype)
        acc = acc + tl.dot(p, v, allow_tf32=True)

        # 4f. Update state for the next iteration
        l_i = l_i_new
        m_i = m_i_new

    # 5. Normalize the final output accumulator
    l_i_reciprocal = 1.0 / l_i
    acc = acc * l_i_reciprocal[:, None]
    
    # 6. Write the final output block to DRAM
    o_ptrs = o_base_ptr + (offs_m[:, None] * s_on + offs_d[None, :] * s_od)
    o_mask = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=o_mask)

def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the GraphSynth Triton kernel.

    Args:
        q: Query tensor of shape [B, H, N, D] and dtype bfloat16.
        k: Key tensor of shape [B, H, N, D] and dtype bfloat16.
        v: Value tensor of shape [B, H, N, D] and dtype bfloat16.

    Returns:
        Output tensor of shape [B, H, N, D] and dtype bfloat16.
    """
    # HARD REQUIREMENT: Input/output tensors are bfloat16 [2, 32, 2048, 128]
    assert q.shape == (2, 32, 2048, 128) and q.dtype == torch.bfloat16
    assert k.shape == q.shape and k.dtype == q.dtype
    assert v.shape == q.shape and v.dtype == q.dtype
    
    B, H, N_CTX, D_HEAD = q.shape
    
    # Allocate the output tensor
    o = torch.empty_like(q)
    
    # Define block sizes for the kernel. These are reasonable defaults for A100.
    BLOCK_M = 128
    BLOCK_N = 64

    # HARD REQUIREMENT: Compute 1/sqrt(D) on the host and pass as a float.
    scale = D_HEAD**-0.5
    
    # Define the grid for the kernel launch.
    # Each program instance will handle one block of M queries for one head.
    grid = (B * H, triton.cdiv(N_CTX, BLOCK_M))

    # Get strides for each tensor.
    s_qb, s_qh, s_qn, s_qd = q.stride()
    s_kb, s_kh, s_kn, s_kd = k.stride()
    s_vb, s_vh, s_vn, s_vd = v.stride()
    s_ob, s_oh, s_on, s_od = o.stride()

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        s_qb, s_qh, s_qn, s_qd,
        s_kb, s_kh, s_kn, s_kd,
        s_vb, s_vh, s_vn, s_vd,
        s_ob, s_oh, s_on, s_od,
        B, H, N_CTX, D_HEAD,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D_HEAD,
        num_warps=4,
        num_stages=4
    )
    
    return o