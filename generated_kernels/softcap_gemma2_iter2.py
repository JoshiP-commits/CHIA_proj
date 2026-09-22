import torch
import triton
import triton.language as tl
import math

@triton.jit
def graphsynth_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, SEQ_LEN, D_HEAD,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DHEAD: tl.constexpr,
):
    """
    Triton kernel for causal attention with Gemma-2 style logit soft-capping.

    This kernel implements a fused attention mechanism that avoids materializing the
    full score matrix, using an online softmax algorithm similar to FlashAttention.

    Grid: (triton.cdiv(SEQ_LEN, BLOCK_M), Z, H)
    B, H, SEQ_LEN, D_HEAD = [2, 32, 2048, 128]

    OPERATOR:
      1. s = (q @ k^T) / sqrt(D_HEAD)
      2. s = tanh(s / 30.0) * 30.0
      3. mask: query i attends to key j only if j <= i
      4. p = softmax(s, dim=-1)
      5. out = p @ v
    """
    # --------------------------------------------------------------------------
    # Grid and Program ID Setup
    # --------------------------------------------------------------------------
    # Each program instance computes one BLOCK_M x D_HEAD block of the output.
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Offsets for batch and head dimensions.
    q_offset = pid_z * stride_qz + pid_h * stride_qh
    k_offset = pid_z * stride_kz + pid_h * stride_kh
    v_offset = pid_z * stride_vz + pid_h * stride_vh
    o_offset = pid_z * stride_oz + pid_h * stride_oh

    # Pointers to the start of the Q, K, V, Out tensors for the current head.
    Q_ptrs = Q + q_offset
    K_ptrs = K + k_offset
    V_ptrs = V + v_offset
    Out_ptrs = Out + o_offset

    # --------------------------------------------------------------------------
    # Initialize Accumulators and Pointers for the Q-block
    # --------------------------------------------------------------------------
    # Running accumulators for the online softmax algorithm.
    # All accumulators are in float32 for precision.
    # acc: holds the numerator (p @ v), initialized to zeros.
    # l_i: holds the denominator (sum of exp(scores)), initialized to zeros.
    # m_i: holds the running max of scores, initialized to -infinity.
    acc = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)

    # Pointers to the current query block. This block is fixed for this program.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DHEAD)
    q_ptrs = Q_ptrs + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

    # Mask for queries that are beyond the actual sequence length (padding).
    q_mask = offs_m < SEQ_LEN
    # Load the query block, zeroing out padding queries.
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Constants for the attention calculation.
    sm_scale = 1.0 / math.sqrt(D_HEAD)
    softcap_const = 30.0

    # --------------------------------------------------------------------------
    # Main Loop over Key/Value Blocks
    # --------------------------------------------------------------------------
    # The loop iterates over key/value blocks. Causal masking is enforced at the
    # block level by setting the loop's end point.
    end_n = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        # -- Load Key and Value Blocks --
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptrs + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
        v_ptrs = V_ptrs + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

        # Mask for keys/values that are beyond sequence length (padding).
        kv_mask = offs_n < SEQ_LEN
        # There's no need for an `if start_n >= SEQ_LEN: continue` because
        # the mask on tl.load handles out-of-bounds memory access gracefully.
        # The loop must have a static trip count.
        k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # -- Compute Scores (s = q @ k^T) --
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        s *= sm_scale

        # -- Apply Gemma-2 Logit Soft-Capping --
        # s_capped = tanh(s / 30.0) * 30.0
        # This is applied BEFORE the softmax calculation (including masking).
        s = tl.math.tanh(s / softcap_const) * softcap_const

        # -- Apply Causal Mask --
        # We mask out scores where key_pos > query_pos.
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        # Apply the mask by setting masked-out elements to -infinity.
        s = tl.where(causal_mask, s, -float('inf'))

        # -- Online Softmax Update --
        # 1. Find the new maximum score for the current block.
        m_i_new = tl.max(s, axis=1)
        # 2. Get the new running max.
        m_i_prime = tl.maximum(m_i, m_i_new)
        # 3. Calculate probabilities for the current block, scaled by the new max.
        p = tl.exp(s - m_i_prime[:, None])
        # 4. Rescale the current accumulator and running denominator using the
        #    difference between the old and new max values.
        alpha = tl.exp(m_i - m_i_prime)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        # 5. Update the running denominator.
        l_i += tl.sum(p, axis=1)
        # 6. Update the accumulator with the new values (p @ v).
        #    p must be cast to the input dtype for the dot product.
        p = p.to(V.dtype.element_ty)
        acc += tl.dot(p, v)
        # 7. Update the running max for the next iteration.
        m_i = m_i_prime

    # --------------------------------------------------------------------------
    # Finalization and Store
    # --------------------------------------------------------------------------
    # Normalize the accumulator with the final denominator.
    # Use a safe division to avoid NaN where l_i is zero (e.g., padding rows).
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    # Pointers to the output block.
    out_ptrs = Out_ptrs + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    
    # Cast the final result to the output dtype (bfloat16) and store.
    out = acc.to(Out.dtype.element_ty)
    tl.store(out_ptrs, out, mask=q_mask[:, None])


def launch_kernel(q, k, v):
    """
    Launches the Triton kernel for causal attention with Gemma-2 soft-capping.

    Args:
        q (torch.Tensor): Query tensor of shape [2, 32, 2048, 128], dtype bfloat16.
        k (torch.Tensor): Key tensor of shape [2, 32, 2048, 128], dtype bfloat16.
        v (torch.Tensor): Value tensor of shape [2, 32, 2048, 128], dtype bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype as inputs.
    """
    if not (q.shape == k.shape == v.shape and q.shape == (2, 32, 2048, 128)):
        raise ValueError("Inputs must be bfloat16 tensors of shape [2, 32, 2048, 128]")
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
         raise ValueError("Input tensors must have dtype torch.bfloat16")

    B, H, SEQ_LEN, D_HEAD = q.shape
    o = torch.empty_like(q, dtype=torch.bfloat16)

    # Kernel configuration for A100.
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DHEAD = D_HEAD # Must be a power of 2 and match D_HEAD for tl.dot.
    num_warps = 4
    num_stages = 2
    
    # Grid dimensions. Each program instance processes one query block.
    grid = (triton.cdiv(SEQ_LEN, BLOCK_M), B, H)

    # Launch the kernel.
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, SEQ_LEN, D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DHEAD=BLOCK_DHEAD,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o