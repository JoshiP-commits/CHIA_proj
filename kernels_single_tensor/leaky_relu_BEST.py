import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    in_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask)
    out = tl.where(x > 0.0, x, x * 0.01)
    tl.store(out_ptr + offsets, out, mask=mask)

def launch_kernel(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    graphsynth_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out