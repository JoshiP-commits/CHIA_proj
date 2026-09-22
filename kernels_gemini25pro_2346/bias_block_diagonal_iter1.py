import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    Q, K, V, O,
    # Stride Info
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Other metadata
    B, H, N, D,
    # Constexprs
    scale: tl.constexpr,
    DOC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton kernel for attention with document-based causal masking.
    Each program computes a BLOCK_M x D block of the output O.
    """
    # 1. Get program IDs to identify the current M-block and head
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Decompose batch/head ID
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # 2. Compute offsets for the current query block
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # 3. Initialize pointers to Q, K, V, O
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    q_ptrs = q_base + (offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)

    # 4. Initialize FlashAttention accumulators
    # Running max and sum for stable softmax
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Output accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 5. Load Q block once. This block is reused across all K,V blocks.
    # Mask is needed for the last block of M if N is not a multiple of BLOCK_M
    q_load_mask = offs_m[:, None] < N
    q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)

    # 6. Loop over key/value blocks (N dimension)
    for start_n in range(0, N, BLOCK_N):
        # -- Compute offsets for the current K/V block --
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # -- Load K and V blocks --
        # K is loaded transposed to avoid using `trans_b` in tl.dot
        k_ptrs = k_base + (offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn)
        v_ptrs = v_base + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

        # Masks for loading K and V, needed for the last block of N
        k_load_mask = offs_n[None, :] < N
        v_load_mask = offs_n[:, None] < N

        k = tl.load(k_ptrs, mask=k_load_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_load_mask, other=0.0)

        # -- Compute scores S = (Q @ K^T) * scale --
        s = tl.dot(q, k, out_dtype=tl.float32)
        s *= scale

        # -- Apply attention mask --
        # Causal mask: j <= i
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        # Document mask: doc(i) == doc(j)
        doc_i = offs_m[:, None] // DOC_SIZE
        doc_j = offs_n[None, :] // DOC_SIZE
        doc_mask = (doc_i == doc_j)

        # Combine masks and apply to scores
        combined_mask = causal_mask & doc_mask
        s = tl.where(combined_mask, s, -1e9)

        # -- Update softmax statistics (running max and sum) --
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        p_ij = tl.exp(s - m_new[:, None])
        l_ij = tl.sum(p_ij, axis=1)

        # -- Update output accumulator --
        # Correct previous accumulator and l_i for the new max value
        alpha = tl.exp(m_i - m_new)
        acc *= alpha[:, None]

        # Add current block's contribution
        p_ij_casted = p_ij.to(v.dtype)
        acc += tl.dot(p_ij_casted, v)

        # Update running sum of normalizers
        l_i = l_i * alpha + l_ij

        # Update running max
        m_i = m_new

    # 7. Normalize and store output block
    # Guard against division by zero for rows that attend to nothing
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_i_safe[:, None]

    # Pointer to the output block
    o_ptrs = o_base + (offs_m[:, None] * stride_on + offs_d[None, :] * stride_od)
    # Mask for storing output, same as q_load_mask
    o_store_mask = offs_m[:, None] < N
    tl.store(o_ptrs, o.to(O.dtype.element_ty), mask=o_store_mask)


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launches the Triton kernel for document-aware causal attention.

    Args:
        q: Query tensor of shape [B, H, N, D], dtype=bfloat16.
        k: Key tensor of shape [B, H, N, D], dtype=bfloat16.
        v: Value tensor of shape [B, H, N, D], dtype=bfloat16.

    Returns:
        Output tensor of shape [B, H, N, D], dtype=bfloat16.
    """
    B, H, N, D = 2, 32, 2048, 128
    # Hard requirement checks
    assert q.shape == (B, H, N, D), f"Expected q shape ({B},{H},{N},{D}), got {q.shape}"
    assert k.shape == q.shape and v.shape == q.shape, "Input tensors must have the same shape"
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16, "Inputs must be bfloat16"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA device"

    # Allocate output tensor
    o = torch.empty_like(q)

    # Kernel constants
    DOC_SIZE = 512
    scale = 1.0 / math.sqrt(D)

    # Block sizes for tiling
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = D

    # Grid definition for kernel launch
    # One program per M-block per head
    grid = (triton.cdiv(N, BLOCK_M), B * H)

    graphsynth_kernel[grid](
        q, k, v, o,
        # Strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Other parameters
        B, H, N, D,
        # Constexprs
        scale=scale,
        DOC_SIZE=DOC_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        # Tuning for A100 (sm_80)
        num_warps=4,
        num_stages=2,
    )

    return o