import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    Q, K, V, Out,
    # Stride variables for tensors
    s_qb, s_qh, s_qn, s_qd,
    s_kb, s_kh, s_kn, s_kd,
    s_vb, s_vh, s_vn, s_vd,
    s_ob, s_oh, s_on, s_od,
    # Metadata
    B, H, N_CTX, D_HEAD,
    # Scale factor
    scale: float,
    # Triton-specific constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton kernel for a custom attention mechanism.
    Grid: (B * H, N_CTX // BLOCK_M)
    """
    # 1. Get program IDs
    pid_bh = tl.program_id(0)  # Program ID for the batch and head
    pid_m = tl.program_id(1)   # Program ID for the M-dimension block (query sequence)

    # 2. Derive batch and head indices from the 1D program ID
    b = pid_bh // H
    h = pid_bh % H

    # 3. Calculate per-head temperature as required
    # t_h = 0.5 + h/32
    t_h = 0.5 + h / 32.0

    # 4. Setup base pointers for the current batch and head
    q_base_ptr = Q + b * s_qb + h * s_qh
    k_base_ptr = K + b * s_kb + h * s_kh
    v_base_ptr = V + b * s_vb + h * s_vh
    o_base_ptr = Out + b * s_ob + h * s_oh

    # 5. Initialize offsets for the current Q block
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # 6. Initialize running accumulators for the online softmax
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 7. Load Q block for this program instance
    q_ptrs = q_base_ptr + (offs_m[:, None] * s_qn + offs_d[None, :] * s_qd)
    q_mask = (offs_m[:, None] < N_CTX)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # 8. Loop over K and V blocks in a FlashAttention-style manner
    # The loop bound is causal: we only need to iterate up to the current query block.
    loop_hi = (pid_m + 1) * BLOCK_M
    for start_n in range(0, loop_hi, BLOCK_N):
        # -- Load K block (transposed) --
        offs_n_loop = start_n + tl.arange(0, BLOCK_N)
        # To satisfy `tl.dot` without `trans_b`, we load K transposed.
        # Pointer shape is [BLOCK_DMODEL, BLOCK_N]
        k_ptrs = k_base_ptr + (offs_d[:, None] * s_kd + offs_n_loop[None, :] * s_kn)
        k_mask = (offs_n_loop[None, :] < N_CTX)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # -- Compute scores S = (Q @ K^T) * scale * temp --
        s_ij = tl.dot(q, k)
        s_ij *= scale
        s_ij *= t_h

        # -- Apply causal mask (j <= i) --
        causal_mask = (offs_m[:, None] >= offs_n_loop[None, :])
        s_ij = tl.where(causal_mask, s_ij, -1e9)

        # -- Online softmax update logic --
        # 1. Find new running max
        m_ij = tl.max(s_ij, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        # 2. Rescale previous accumulator and sum based on new max
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # 3. Compute current block's unnormalized probabilities
        p_ij_exp = tl.exp(s_ij - m_i_new[:, None])
        
        # 4. Update running sum (denominator)
        l_i += tl.sum(p_ij_exp, axis=1)

        # 5. Load V block and update accumulator
        # Pointer shape is [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = v_base_ptr + (offs_n_loop[:, None] * s_vn + offs_d[None, :] * s_vd)
        v_mask = (offs_n_loop[:, None] < N_CTX)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # p_ij must be converted to V's dtype for tl.dot
        p_ij = p_ij_exp.to(V.dtype.element_ty)
        acc += tl.dot(p_ij, v)

        # 6. Update running max for the next iteration
        m_i = m_i_new

    # 9. Finalize: normalize the accumulator
    # Safely handle cases where l_i is zero (e.g., fully masked rows) to avoid NaN
    l_i_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_i_safe[:, None]

    # 10. Store the final output block to global memory
    o_ptrs = o_base_ptr + (offs_m[:, None] * s_on + offs_d[None, :] * s_od)
    o_mask = (offs_m[:, None] < N_CTX)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launchpad function for the GraphSynth Triton kernel.

    Args:
        q: Query tensor of shape [2, 32, 2048, 128], dtype bfloat16.
        k: Key tensor of shape [2, 32, 2048, 128], dtype bfloat16.
        v: Value tensor of shape [2, 32, 2048, 128], dtype bfloat16.

    Returns:
        Output tensor of shape [2, 32, 2048, 128], dtype bfloat16.
    """
    B, H, N, D = q.shape
    
    # Create the output tensor
    o = torch.empty_like(q)

    # Triton kernel constants
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Grid definition: one program per (batch, head) and per query block
    grid = (B * H, triton.cdiv(N, BLOCK_M))
    
    # Pre-compute scale factor on the host as a float
    scale = 1.0 / math.sqrt(D)

    # Launch the kernel
    graphsynth_kernel[grid](
        # Tensors
        q, k, v, o,
        # Strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Metadata
        B, H, N, D,
        # Scale factor
        scale,
        # Triton constants
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D,
        # Performance tuning for sm80 (A100)
        num_warps=4,
        num_stages=3,
    )

    return o