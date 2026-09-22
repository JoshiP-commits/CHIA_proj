import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Element-wise operation, so we can treat the 2D tensor as a flat 1D array
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with bounds checking and padding
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Scale x for the erf function: x / sqrt(2)
    x_scaled = x * 0.7071067811865476
    
    # Constants for Abramowitz and Stegun polynomial approximation of erf
    # Accurate to 1.5e-7
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    
    # Compute approximation
    sign = tl.where(x_scaled >= 0.0, 1.0, -1.0)
    abs_x = tl.abs(x_scaled)
    t = 1.0 / (1.0 + p * abs_x)
    
    # Horner's method to evaluate the polynomial
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    erf_val = sign * (1.0 - poly * tl.exp(-x_scaled * x_scaled))
    
    # Exact GELU formula: 0.5 * x * (1 + erf(x / sqrt(2)))
    out = 0.5 * x * (1.0 + erf_val)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def launch_kernel(x: torch.Tensor) -> torch.Tensor:
    # Ensure input is continuous to achieve peak DRAM bandwidth
    if not x.is_contiguous():
        x = x.contiguous()
        
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # 1024 is a very safe block size for element-wise ops that ensures good occupancy
    # and coalesced memory access on an A100.
    BLOCK_SIZE = 1024
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    graphsynth_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out