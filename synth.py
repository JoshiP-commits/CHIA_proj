import os, re, math, json, time, tempfile, importlib.util
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from google import genai

PROJECT  = os.environ.get("GCP_PROJECT", "a3-chia-hack26ath-7724")
LOCATION = os.environ.get("GCP_LOCATION", "global")
MODEL    = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
MAX_ITER = int(os.environ.get("MAX_ITER", 3))
HOME     = os.path.expanduser("~")
import datetime as _dt
RUN = _dt.datetime.now().strftime("%H%M%S")
OUTDIR   = os.path.join(HOME, "generated_kernels"); os.makedirs(OUTDIR, exist_ok=True)

DT, dev = torch.bfloat16, "cuda"
B, H, S, D = 2, 32, 2048, 128
SOFTCAP, WINDOW = 30.0, 256
REL_TOL = 3e-2

def dtime(e):
    for a in ("device_time_total","cuda_time_total","device_time","cuda_time"):
        v = getattr(e, a, None)
        if v: return float(v)
    return 0.0

def gpu_us(fn):
    fn(); torch.cuda.synchronize()
    e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); fn(); e1.record(); torch.cuda.synchronize()
    nm = 20 if e0.elapsed_time(e1) < 5 else 5
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(nm): fn()
        torch.cuda.synchronize()
    evs = [e for e in pr.key_averages() if dtime(e) > 0]
    tot = sum(dtime(e) for e in evs)/nm
    if tot <= 0: tot = e0.elapsed_time(e1)*1000.0
    return tot, sum(e.count for e in evs)/nm

def ref_softcap(q, k, v):
    s = (q @ k.transpose(-2,-1)) / math.sqrt(q.shape[-1])
    s = torch.tanh(s / SOFTCAP) * SOFTCAP
    i = torch.arange(s.shape[-2], device=dev).view(-1,1)
    j = torch.arange(s.shape[-1], device=dev).view(1,-1)
    s = s.masked_fill(j > i, float("-inf"))
    return torch.softmax(s.float(), -1).to(DT) @ v

def ref_window(q, k, v):
    s = (q @ k.transpose(-2,-1)) / math.sqrt(q.shape[-1])
    i = torch.arange(s.shape[-2], device=dev).view(-1,1)
    j = torch.arange(s.shape[-1], device=dev).view(1,-1)
    ok = (j <= i) & ((i - j) < WINDOW)
    s = s.masked_fill(~ok, float("-inf"))
    return torch.softmax(s.float(), -1).to(DT) @ v

COMMON = """
You are an expert GPU kernel engineer writing Triton for an NVIDIA A100 (sm80).

HARD REQUIREMENTS
- Return ONE ```python code block, nothing else.
- Define a @triton.jit kernel named exactly `graphsynth_kernel`.
- Define `def launch_kernel(q, k, v):` taking exactly three bfloat16 tensors of
  shape [2, 32, 2048, 128] and returning ONE bfloat16 tensor of the same shape.
- Use FlashAttention-style tiling: loop over KV blocks, keep a running max and
  running sum for the online softmax, accumulate output in fp32 registers.
  NEVER materialise the full [S, S] score matrix in DRAM.
- Accumulate in float32; cast to bfloat16 only on the final store.
- Do NOT use `continue` or `break` inside the loop. Instead compute the valid
  KV block range on the HOST and pass it in, or bound `tl.range` so masked
  blocks are never visited.

TRITON API CONSTRAINTS FOR THIS EXACT VERSION -- violating any of these fails:
- `tl.math.tanh` does NOT exist. Use: 2.0 * tl.sigmoid(2.0 * x) - 1.0
- `tl.dot()` has NO `trans_b` argument. Transpose K by loading it with swapped
  strides instead.
- Python ints have no `.to()` method. Compute 1/sqrt(D) on the host in Python
  and pass it in as a float kernel argument.
- Use `tl.where(cond, a, b)` for masking. No boolean indexing.
- Initialise running max with `tl.full([BLOCK_M], -1e9, tl.float32)`, not -inf.
- Self-contained: import torch, triton, triton.language as tl.
"""

P_SOFTCAP = COMMON + f"""
OPERATOR: causal attention with Gemma-2 logit soft-capping.
  1. s = (q @ k^T) / sqrt({D})
  2. s = tanh(s / {SOFTCAP}) * {SOFTCAP}
  3. mask: query i attends to key j only if j <= i
  4. p = softmax(s, dim=-1)
  5. out = p @ v
Apply the soft-cap INSIDE the kernel before the running-max update.
If tl.math.tanh is unavailable use tanh(x) = 2*sigmoid(2x) - 1.
PyTorch eager: 10680 us. Fused causal flash on same tensors: 552 us.
Target: under 1100 us.
"""

P_WINDOW = COMMON + f"""
OPERATOR: sliding-window causal attention, window = {WINDOW}.
  1. s = (q @ k^T) / sqrt({D})
  2. query i attends to key j only if (j <= i) AND (i - j < {WINDOW})
  3. p = softmax(s, dim=-1)
  4. out = p @ v
Only ~12% of the score matrix is unmasked. For each query block compute the
range of KV blocks that can contain unmasked elements and iterate ONLY over
those. A kernel that visits every KV block will not beat the baseline.
cuDNN: 1802 us. Plain causal flash on same tensors: 552 us.
Target: under 400 us.
"""

TASKS = {
  "softcap_gemma2":      dict(prompt=P_SOFTCAP, ref=ref_softcap, target_us=1100.0),
  "bias_sliding_window": dict(prompt=P_WINDOW,  ref=ref_window,  target_us=400.0),
}

client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

def ask(prompt, retries=4):
    for a in range(retries):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt).text
        except Exception as e:
            if any(x in str(e) for x in ("503","UNAVAILABLE","429","RESOURCE_EXHAUSTED")):
                time.sleep(2**a); continue
            raise
    raise RuntimeError("Gemini unavailable")

def extract(txt):
    m = re.search(r"```python\s*(.*?)```", txt, re.S) or re.search(r"```\s*(.*?)```", txt, re.S)
    return (m.group(1) if m else txt).strip()

def load(code):
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, prefix="gs_")
    f.write(code); f.close()
    try:
        sp = importlib.util.spec_from_file_location(f"gs_{abs(hash(code))}", f.name)
        mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
        if not hasattr(mod, "launch_kernel"): return None, "no launch_kernel defined"
        return mod.launch_kernel, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def verify(fn, ref, q, k, v):
    worst, msgs = 0.0, []
    for nm, sc in (("standard",1.0), ("small",0.01)):
        qq,kk,vv = q*sc, k*sc, v*sc
        try:
            out, r = fn(qq,kk,vv), ref(qq,kk,vv)
        except Exception as e:
            return False, 9e9, f"{nm}: raised {type(e).__name__}: {e}"
        if out.shape != r.shape:   return False, 9e9, f"{nm}: shape {tuple(out.shape)} != {tuple(r.shape)}"
        if torch.isnan(out).any(): return False, 9e9, f"{nm}: NaN"
        if torch.isinf(out).any(): return False, 9e9, f"{nm}: Inf"
        rel = ((r-out).abs().max()/(r.abs().max()+1e-6)).item()
        worst = max(worst, rel); msgs.append(f"{nm} rel={rel:.2e}")
    return worst < REL_TOL, worst, "; ".join(msgs)

q = torch.randn(B,H,S,D, dtype=DT, device=dev)
k = torch.randn(B,H,S,D, dtype=DT, device=dev)
v = torch.randn(B,H,S,D, dtype=DT, device=dev)
flash_us, _ = gpu_us(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True))
print(f"flash ceiling (same tensors): {flash_us:.1f} us\n")

results = []
for name, T in TASKS.items():
    print("="*78); print(f"OP: {name}   target {T['target_us']:.0f} us"); print("="*78)
    base_us, base_nk = gpu_us(lambda: T["ref"](q,k,v))
    print(f"  eager baseline: {base_us:.1f} us  ({base_nk:.0f} kernels)")

    hist, best = [], None
    for it in range(1, MAX_ITER+1):
        print(f"\n  -- iteration {it}/{MAX_ITER} --")
        p = T["prompt"] + ("\n\nPREVIOUS ATTEMPTS (fix these):\n" + "\n".join(hist) if hist else "")
        try:
            code = extract(ask(p))
        except Exception as e:
            print(f"    gemini failed: {e}"); hist.append(f"{it}: gemini error"); continue
        open(f"{OUTDIR}/{name}_{RUN}_iter{it}.py","w").write(code)

        fn, err = load(code)
        if fn is None:
            print(f"    COMPILE FAIL: {err}")
            hist.append(f"{it}: compile failed -> {err}"); continue
        print("    compiled OK")

        ok, rel, detail = verify(fn, T["ref"], q, k, v)
        print(f"    verify: {'PASS' if ok else 'FAIL'}  ({detail})")
        if not ok:
            hist.append(f"{it}: numerically wrong, worst rel err {rel:.2e} ({detail}). "
                        f"Must be < {REL_TOL}. Check masking, scaling, softmax accumulation.")
            continue

        us, nk = gpu_us(lambda: fn(q,k,v))
        sp = base_us/us
        print(f"    latency {us:.1f} us   kernels {nk:.0f}   speedup {sp:.2f}x   "
              f"({us/flash_us:.2f}x flash ceiling)")
        if best is None or us < best[0]: best = (us, sp, rel, code, it)
        print("    -> ACCEPTED (verified correct)"); break
        hist.append(f"{it}: correct but slow: {us:.0f} us, need <= {T['target_us']:.0f} us. "
                    f"{nk:.0f} kernel launches. Fuse harder / skip more masked blocks / "
                    f"try BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3.")

    if best:
        us, sp, rel, code, it = best
        open(f"{OUTDIR}/{name}_{RUN}_BEST.py","w").write(code)
        results.append(dict(op=name, status="ACCEPTED" if us<=T["target_us"] else "BEST-EFFORT",
                            baseline_us=round(base_us,1), kernel_us=round(us,1),
                            speedup=round(sp,2), vs_flash=round(us/flash_us,2),
                            rel_err=f"{rel:.1e}", iters=it))
    else:
        results.append(dict(op=name, status="FAILED", baseline_us=round(base_us,1),
                            kernel_us=None, speedup=None, vs_flash=None, rel_err=None,
                            iters=MAX_ITER))

print("\n"+"="*94)
print(f"{'op':<22}{'status':<14}{'baseline':>10}{'kernel':>10}{'speedup':>10}{'x flash':>9}{'rel_err':>10}{'it':>4}")
print("-"*94)
for r in results:
    g = lambda x, s: (s % x) if x is not None else "--"
    print(f"{r['op']:<22}{r['status']:<14}{r['baseline_us']:>9.0f}u{g(r['kernel_us'],'%9.0fu')}"
          f"{g(r['speedup'],'%9.2fx')}{g(r['vs_flash'],'%8.2fx')}{str(r['rel_err'] or '--'):>10}{r['iters']:>4}")
print("="*94)
json.dump(results, open(os.path.join(HOME,"synth_results.json"),"w"), indent=2)
print(f"\nkernels -> {OUTDIR}/    results -> ~/synth_results.json")
