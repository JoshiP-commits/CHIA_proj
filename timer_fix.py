import torch
PEAK_BW = 1555.0e9

def gpu_ms(fn, warmup=10, iters=50):
    """CUDA events around a LOOP. No profiler, no per-call averages,
    launch overhead amortised over `iters`."""
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters): fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters      # ms per call

# SANITY CHECK: a pure copy can never exceed 100% of peak bandwidth.
N = 8192
x = torch.randn(N, N, device='cuda', dtype=torch.float32)
y = torch.empty_like(x)
t = gpu_ms(lambda: y.copy_(x))
bytes_moved = 2 * N * N * 4
bw = bytes_moved / (t * 1e-3)
print(f"copy  {t*1000:8.1f} us   {bw/1e9:7.1f} GB/s   {bw/PEAK_BW*100:5.1f}% of peak")
print("-> if this is under 100%, the timer is trustworthy\n")
