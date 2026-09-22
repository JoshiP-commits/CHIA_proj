import torch
import triton
import triton.language as tl

@triton.jit
def graphsynth_kernel(
    in_ptr, out_ptr, 
    n_rows, n_cols,
    in_stride, out_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Map each program to a single row
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    # Calculate starting pointers for this specific row
    row_start_ptr = in_ptr + row_idx * in_stride
    out_row_start_ptr = out_ptr + row_idx * out_stride
    
    # Create column offsets and mask for valid elements
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Read the entire row from DRAM into SRAM exactly once
    # For out-of-bounds elements, use -inf so they don't affect the max
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute row max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Compute exponentials (exp(x - max))
    numerator = tl.exp(row - row_max)
    
    # Ensure out-of-bounds values contribute exactly 0.0 to the sum
    numerator = tl.where(mask, numerator, 0.0)
    
    # Sum across the row to find the denominator
    denominator = tl.sum(numerator, axis=0)
    
    # Compute the final softmax probabilities
    softmax_out = numerator / denominator
    
    # Write the entire row's output back to DRAM exactly once
    tl.store(out_row_start_ptr + col_offsets, softmax_out, mask=mask)

def launch_kernel(x: torch.Tensor) -> torch.Tensor:
    # Validate properties
    n_rows, n_cols = x.shape
    
    # Output tensor allocation
    out = torch.empty_like(x)
    
    # For a row length of 8192, the next power of two is exactly 8192.
    # This block size is large but fully supported and fits neatly into A100's SRAM.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Using 16 warps (512 threads) provides optimal occupancy and register utilization 
    # for a block size of 8192 elements, translating to 16 elements per thread.
    num_warps = 16
    
    # We launch exactly one program per row
    grid = (n_rows,)
    
    graphsynth_kernel[grid](
        x, out,
        n_rows, n_cols,
        x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return out