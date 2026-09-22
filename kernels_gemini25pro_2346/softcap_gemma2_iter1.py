import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    # Pointers to Tensors
    Q, K, V, O,
    # Stride variables for tensor access
    stride_qz, stride_qh, stride_qn, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_on, stride_od,
    # Meta-parameters
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # Scale for the dot product
    scale: tl.constexpr,
):
    """
    Triton kernel for a custom attention-like operation.
    Computes O = softmax(tanh( (Q @ K^T * scale) / 30) * 30) @ V with causal masking.
    This implementation uses FlashAttention-style tiling and online softmax.
    """
    # Get program IDs to identify the current work item
    pid_m = tl.program_id(0)  # Block index along the sequence dimension
    pid_h = tl.program_id(1)  # Head index
    pid_z = tl.program_id(2)  # Batch index

    #
    # 1. Setup Pointers and Offsets
    #
    # Base offsets for the current batch and head
    q_offset = pid_z * stride_qz + pid_h * stride_qh
    k_offset = pid_z * stride_kz + pid_h * stride_kh
    v_offset = pid_z * stride_vz + pid_h * stride_vh
    o_offset = pid_z * stride_oz + pid_h * stride_oh

    # Block pointers for Q, K, V, and O
    # Q is processed in blocks of size [BLOCK_M, D_HEAD]
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_qn, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    # K is loaded transposed for the dot product: [D_HEAD, BLOCK_N]
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(D_HEAD, N_CTX),
        strides=(stride_kd, stride_kn), # Swapped strides for transpose
        offsets=(0, 0),
        block_shape=(D_HEAD, BLOCK_N),
        order=(0, 1) # Swapped order for transpose
    )
    # V is processed in blocks of size [BLOCK_N, D_HEAD]
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, D_HEAD),
        order=(1, 0)
    )
    # O is written to in blocks of size [BLOCK_M, D_HEAD]
    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_on, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )

    #
    # 2. Initialize Accumulators and Load Q
    #
    # Load the Q block for the current program instance. This is done once.
    q = tl.load(Q_block_ptr, boundary_check=(0,))

    # Initialize accumulators for the online softmax algorithm
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)  # Output accumulator
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)      # Running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)           # Running sum of exp(s - m)

    #
    # 3. Main Loop over K and V Blocks
    #
    # The loop bound `(pid_m + 1) * BLOCK_M` implements causal attention.
    # A query block `m` only attends to key blocks `n` where `n <= m`.
    end_n = (pid_m + 1) * BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        # -- Load K and V for the current block --
        # Advance pointers to the current block in the inner loop
        K_block_ptr_loop = tl.advance(K_block_ptr, (0, start_n))
        V_block_ptr_loop = tl.advance(V_block_ptr, (start_n, 0))
        k = tl.load(K_block_ptr_loop, boundary_check=(1,))
        v = tl.load(V_block_ptr_loop, boundary_check=(0,))

        # -- Compute scores (S) --
        s_ij = tl.dot(q, k)
        s_ij *= scale

        # -- Apply Causal and Padding Mask --
        # Row and column indices for the current score block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # Causal mask: ensures a query at position `i` only attends to keys at positions `j <= i`
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        # Padding mask: handles sequences shorter than N_CTX
        padding_mask = offs_n[None, :] < N_CTX
        # Combine masks
        full_mask = causal_mask & padding_mask
        # Apply mask by setting masked-out elements to a large negative number
        s_ij = tl.where(full_mask, s_ij, -1e9)

        # -- Gemma-2 Logit Soft-Capping --
        # s_cap = tanh(s / 30.0) * 30.0
        # Implemented as 2.0*sigmoid(2.0*x) - 1.0 for tanh
        s_ij_scaled = s_ij / 30.0
        s_ij_sigmoid = tl.sigmoid(2.0 * s_ij_scaled)
        s_ij = (2.0 * s_ij_sigmoid - 1.0) * 30.0

        # -- Online Softmax Update --
        # 1. Find the new running max
        m_ij = tl.max(s_ij, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        # 2. Rescale previous accumulators based on the new max
        alpha = tl.exp(m_i - m_i_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # 3. Compute probabilities for the current block, p_ij
        p_ij = tl.exp(s_ij - m_i_new[:, None])
        
        # 4. Update the running sum of exponentials, l_i
        l_ij = tl.sum(p_ij, axis=1)
        l_i += l_ij

        # 5. Update the output accumulator, acc
        # Cast p_ij to the same dtype as v to use fast bfloat16 dot product
        p_ij = p_ij.to(v.dtype)
        acc += tl.dot(p_ij, v)

        # 6. Update the running max for the next iteration
        m_i = m_i_new

    #
    # 4. Final Normalization and Store Output
    #
    # Normalize the output accumulator by the final sum of exponents
    # Add a small epsilon to avoid division by zero for rows with no attention
    l_i = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i[:, None]

    # Cast to output dtype and store to global memory
    tl.store(O_block_ptr, acc.to(O.dtype.element_ty), boundary_check=(0,))


def launch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Launcher for the graphsynth_kernel.

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, N, D], dtype bfloat16.
        k (torch.Tensor): Key tensor of shape [B, H, N, D], dtype bfloat16.
        v (torch.Tensor): Value tensor of shape [B, H, N, D], dtype bfloat16.

    Returns:
        torch.Tensor: Output tensor of the same shape and dtype as inputs.
    """
    # Ensure inputs have the correct shape and data type
    assert q.shape == k.shape == v.shape, "Input tensors must have the same shape"
    assert len(q.shape) == 4, "Input tensors must be 4-dimensional"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Input tensors must be on a CUDA device"
    assert q.dtype == torch.bfloat16, "Input tensors must have bfloat16 dtype"

    # Extract dimensions
    B, H, N_CTX, D_HEAD = q.shape

    # Create the output tensor on the same device as the inputs
    o = torch.empty_like(q)

    # Define tuning parameters for the kernel, optimized for A100 (sm80)
    # BLOCK_M: Tile size for the query sequence length (rows of the score matrix)
    # BLOCK_N: Tile size for the key sequence length (columns of the score matrix)
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Define the grid for launching the kernel. Each program instance handles
    # one query block for one head in one batch.
    # grid = (num_m_blocks, num_heads, num_batches)
    grid = (triton.cdiv(N_CTX, BLOCK_M), H, B)

    # Calculate the scale factor (1 / sqrt(D_HEAD)) on the host
    scale = D_HEAD**-0.5

    # Launch the kernel
    graphsynth_kernel[grid](
        # Tensors
        q, k, v, o,
        # Strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Meta-parameters (passed as compile-time constants)
        N_CTX=N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        scale=scale,
    )

    return o