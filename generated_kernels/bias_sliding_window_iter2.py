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
    Z, H, N_CTX,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for sliding-window causal attention.
    Each program instance computes a BLOCK_M x DMODEL block of the output.
    """
    # 1. Get program IDs to identify the current batch, head, and query block.
    start_m = tl.program_id(0) * BLOCK_M
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)

    # 2. Create pointers to Q and Out tensors for the current head and batch.
    q_offset = pid_z * stride_qz + pid_h * stride_qh
    k_offset = pid_z * stride_kz + pid_h * stride_kh
    v_offset = pid_z * stride_vz + pid_h * stride_vh
    o_offset = pid_z * stride_oz + pid_h * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    Out_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    # 3. Initialize accumulators for the online softmax.
    # `acc` stores the unnormalized output in fp32.
    # `m_i` stores the running maximum of scores.
    # `l_i` stores the running sum of exponentiated scores.
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)

    # 4. Load the query block once.
    q = tl.load(Q_block_ptr, boundary_check=(0,))

    # 5. Set up the loop over key/value blocks.
    # This is the core optimization for sliding window attention.
    # Instead of iterating from 0, we calculate the earliest KV block that can be
    # inside the sliding window and start there, using a while loop.
    sm_scale = tl.math.rsqrt(BLOCK_DMODEL)
    causal_end_n = start_m + BLOCK_M
    
    # First query in block `start_m` attends to keys `j > start_m - WINDOW_SIZE`.
    # Find the block containing this key index and start our loop there, aligned to BLOCK_N.
    kv_loop_start = tl.maximum(0, (start_m - WINDOW_SIZE) // BLOCK_N * BLOCK_N)

    # Initialize K and V pointers to the start of our restricted loop.
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, kv_loop_start),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(kv_loop_start, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    start_n = kv_loop_start
    while start_n < causal_end_n:
        # 6. Load K and V blocks for the current iteration.
        k = tl.load(K_block_ptr, boundary_check=(1,))
        v = tl.load(V_block_ptr, boundary_check=(0,))

        # 7. Compute attention scores (S = Q @ K^T).
        s_ij = tl.dot(q, k) * sm_scale

        # 8. Apply the combined causal and sliding window mask.
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        window_mask = (offs_m[:, None] - offs_n[None, :]) < WINDOW_SIZE

        # Combine masks and apply to scores, setting masked elements to -inf.
        mask = causal_mask & window_mask
        s_ij = tl.where(mask, s_ij, -float('inf'))

        # 9. Perform the online softmax update for numerical stability.
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Rescale the old accumulator and `l_i` with the new max.
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # Calculate new probabilities `p` and update `l_i` and `acc`.
        p_ij = tl.exp(s_ij - m_new[:, None])
        l_i += tl.sum(p_ij, axis=1)
        
        # Cast `p` to bfloat16 before the dot product with `v`.
        p_ij = p_ij.to(Q.dtype.element_ty)
        acc += tl.dot(p_ij, v)

        # Update the running max.
        m_i = m_new

        # 10. Advance to the next KV block.
        start_n += BLOCK_N
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # 11. Normalize the output and store it.
    # A safe division handles rows that were fully masked (where l_i=0).
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    # Store the final fp32 accumulator, casting to bfloat16 on write.
    tl.store(Out_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0,))


def launch_kernel(q, k, v):
    """
    Launch function for the sliding-window causal attention kernel.

    Args:
        q (torch.Tensor): Query tensor of shape [2, 32, 2048, 128], bfloat16.
        k (torch.Tensor): Key tensor of shape [2, 32, 2048, 128], bfloat16.
        v (torch.Tensor): Value tensor of shape [2, 32, 2048, 128], bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype as inputs.
    """
    # Hard requirements check
    shape_req = (2, 32, 2048, 128)
    dtype_req = torch.bfloat16
    assert q.shape == k.shape == v.shape == shape_req, f"Input tensors must have shape {shape_req}"
    assert q.dtype == k.dtype == v.dtype == dtype_req, f"Input tensors must be {dtype_req}"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Input tensors must be on a CUDA device"

    # Shape and constants
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = q.shape
    
    # Create output tensor
    o = torch.empty_like(q)

    # Triton config - tuned for A100 performance
    BLOCK_M = 128
    BLOCK_N = 64
    WINDOW_SIZE = 256
    num_warps = 4
    num_stages = 3

    # Grid dimensions
    grid = (triton.cdiv(SEQ_LEN, BLOCK_M), N_HEADS, BATCH)

    # Launch kernel
    graphsynth_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z=BATCH, H=N_HEADS, N_CTX=SEQ_LEN,
        WINDOW_SIZE=WINDOW_SIZE,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=D_HEAD, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o