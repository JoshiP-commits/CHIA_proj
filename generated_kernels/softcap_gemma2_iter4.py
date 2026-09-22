import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, SEQ_LEN, D_HEAD,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DHEAD: tl.constexpr,
    SOFTCAP_SCALE: tl.constexpr,
):
    """
    Triton kernel for causal attention with Gemma-2 logit soft-capping.
    """
    # -----------------------------------------------------------
    # Map program ids to batch/head/query-block
    # -----------------------------------------------------------
    pid_m = tl.program_id(1)  # Program ID along the sequence dimension (query blocks)
    pid_bh = tl.program_id(0) # Program ID for batch*head
    
    # Calculate batch and head indices
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # -----------------------------------------------------------
    # Offset pointers to the current batch and head
    # -----------------------------------------------------------
    Q += pid_b * stride_qb + pid_h * stride_qh
    K += pid_b * stride_kb + pid_h * stride_kh
    V += pid_b * stride_vb + pid_h * stride_vh
    O += pid_b * stride_ob + pid_h * stride_oh

    # -----------------------------------------------------------
    # Initialize pointers and offsets for the query block
    # -----------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DHEAD)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

    # -----------------------------------------------------------
    # Initialize accumulators and running stats for online softmax
    # -----------------------------------------------------------
    acc = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # -----------------------------------------------------------
    # Load the query block, applying a mask for padding
    # -----------------------------------------------------------
    q_mask = offs_m < SEQ_LEN
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Scale factor for dot product
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    # -----------------------------------------------------------
    # Main loop over key/value blocks
    # Causal attention is enforced by the loop's upper bound
    # -----------------------------------------------------------
    end_m = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_m, BLOCK_N):
        # -- Skip KV blocks that are entirely padding (optimization) --
        # Using `break` is supported and more efficient than a masked load/compute.
        if start_n >= SEQ_LEN:
            break

        # -- Pointers and offsets for the current key/value block --
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

        # -- Load K and V blocks with padding masks --
        kv_mask = offs_n < SEQ_LEN
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # -- Compute attention scores (S = Q @ K^T) --
        s = tl.dot(q, tl.trans(k))
        s *= sm_scale

        # -- Apply causal mask --
        # This mask ensures query i only attends to key j if j <= i.
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal_mask, s, -float('inf'))

        # -- Apply Gemma-2 logit soft-capping --
        # s_capped = tanh(s / 30.0) * 30.0
        s_capped = tl.math.tanh(s / SOFTCAP_SCALE) * SOFTCAP_SCALE

        # -- Online softmax update --
        # 1. Compute max and statistics for the current block
        m_j = tl.max(s_capped, axis=1)
        p_j = tl.exp(s_capped - m_j[:, None])
        l_j = tl.sum(p_j, axis=1)

        # 2. Find the new overall max
        m_new = tl.maximum(m_i, m_j)

        # 3. Rescale old accumulator and statistics
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # 4. Compute and rescale current block's output
        # Cast p_j to bf16 to use Tensor Cores on A100
        p_j_casted = p_j.to(v.dtype)
        acc_j = tl.dot(p_j_casted, v)
        beta = tl.exp(m_j - m_new)

        # 5. Update accumulator and statistics
        acc += acc_j * beta[:, None]
        l_i += l_j * beta
        m_i = m_new

    # -----------------------------------------------------------
    # Final normalization and output write
    # -----------------------------------------------------------
    # Normalize the accumulator by the sum of exp(score)
    # Protect against division by zero for padded rows where l_i is 0
    l_i_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_i_safe[:, None]

    # -- Pointers to output block --
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    
    # -- Store the final output, casting to bfloat16 --
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask[:, None])

def launch_kernel(q, k, v):
    """
    Lauches the Triton kernel for causal attention with Gemma-2 soft-capping.

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, SEQ_LEN, D_HEAD]
        k (torch.Tensor): Key tensor of shape [B, H, SEQ_LEN, D_HEAD]
        v (torch.Tensor): Value tensor of shape [B, H, SEQ_LEN, D_HEAD]

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype.
    """
    # Input validation
    assert q.shape == k.shape == v.shape, "Input tensors must have the same shape"
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "Inputs must be bfloat16"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on a CUDA device"
    assert q.dim() == 4, "Inputs must be 4D tensors"

    # Tensor dimensions
    B, H, SEQ_LEN, D_HEAD = q.shape

    # Create output tensor
    o = torch.empty_like(q)

    # Kernel configuration
    BLOCK_M = 128
    BLOCK_N = 64
    SOFTCAP_SCALE = 30.0

    # Grid setup: one program per head and per query block
    grid = (B * H, triton.cdiv(SEQ_LEN, BLOCK_M))

    # Get tensor strides
    stride_qb, stride_qh, stride_qm, stride_qk = q.stride()
    stride_kb, stride_kh, stride_kn, stride_kk = k.stride()
    stride_vb, stride_vh, stride_vn, stride_vk = v.stride()
    stride_ob, stride_oh, stride_om, stride_ok = o.stride()

    # Launch the kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        B, H, SEQ_LEN, D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DHEAD=D_HEAD,
        SOFTCAP_SCALE=SOFTCAP_SCALE,
    )

    return o