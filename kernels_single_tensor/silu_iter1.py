import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data from DRAM
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU (Swish) activation: x * sigmoid(x)
    out = x * tl.sigmoid(x)
    
    # Store the result back to DRAM
    tl.store(out_ptr + offsets, out, mask=mask)

def launch_kernel(x: torch.Tensor) -> torch.Tensor:
    # Pre-allocate output tensor to ensure we only write to DRAM exactly once
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Choose a block size that gives optimal occupancy and coalesced memory accesses for A100
    BLOCK_SIZE = 1024
    
    # 1D grid since this is a pure element-wise memory-bound operation
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch Triton kernel
    graphsynth_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out