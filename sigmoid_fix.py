import os, re, math, json, time, tempfile, importlib.util, datetime
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from google import genai

PROJECT = "a3-chia-hack26ath-7724"
HOME    = os.path.expanduser("~")
DT, dev = torch.bfloat16, "cuda"
B,H,S,D = 2,32,2048,128
SC      = 1.0/math.sqrt(D)
REL_TOL = 3e-2
MAX_ITER= 3

def dtime(e):
    for a in ("device_time_total","cuda_time_total","device_time","cuda_time"):
        v = getattr(e,a,None)
        if v: return float(v)
    return 0.0

def gpu_us(fn):
    fn(); torch.cuda.synchronize()
    e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); fn(); e1.record(); torch.cuda.synchronize()
    ev = e0.elapsed_time(e1); nm = 20 if ev < 5 else 5
    for _ in range(5): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(nm): fn()
        torch.cuda.synchronize()
    evs=[e for e in pr.key_averages() if dtime(e)>0]
    t = sum(dtime(e) for e in evs)/nm
    return t if t>0 else ev*1000

def ref_sigmoid(q,k,v):
    """FIXED: sigmoid first, THEN zero the masked positions."""
    s = (q@k.transpose(-2,-1))*SC
    p = torch.sigmoid(s.float())
    i = torch.arange(S,device=dev).view(-1,1); j = torch.arange(S,device=dev).view(1,-1)
    p = p * (j<=i).float()                       # masked -> exactly 0
    return p.to(DT) @ v

PROMPT = f"""
You are an expert GPU kernel engineer writing Triton for an NVIDIA A100 (sm80).

HARD REQUIREMENTS
- Return ONE ```python code block, nothing else.
- @triton.jit kernel named exactly `graphsynth_kernel`.
- `def launch_kernel(q, k, v):` taking three bfloat16 tensors [2,32,2048,128],
  returning ONE bfloat16 tensor of the same shape.
- Tile over KV blocks, fp32 accumulator, never materialise the [S,S] matrix.
- No `continue`/`break` in the loop; bound the loop range on the host.

TRITON API CONSTRAINTS -- violating any fails:
- `tl.math.tanh` does not exist. Use 2.0*tl.sigmoid(2.0*x)-1.0
- `tl.dot()` has no `trans_b`. Load K with swapped strides.
- Python ints have no `.to()`. Pass 1/sqrt(D) from the host as a float.
- Use `tl.where(cond,a,b)` for masking.

OPERATOR: SIGMOID ATTENTION (causal). Exact order, this matters:
  1. s = (q @ k^T) * {SC:.10f}
  2. p = sigmoid(s)                  <-- elementwise sigmoid on ALL scores
  3. p = p * (j <= i)                <-- THEN zero the masked positions
  4. out = p @ v

CRITICAL: apply the causal mask AFTER the sigmoid, by multiplying by 0.
Do NOT set masked scores to 0 before the sigmoid -- sigmoid(0)=0.5, which would
give masked positions half weight. Do NOT use -inf either. There is NO softmax
here: no running max, no normalisation, no division by a row sum.
"""

client = genai.Client(vertexai=True, project=PROJECT, location="global")
def ask(p, model, r=4):
    for a in range(r):
        try: return client.models.generate_content(model=model, contents=p).text
        except Exception as e:
            if any(x in str(e) for x in ("503","UNAVAILABLE","429","RESOURCE_EXHAUSTED")):
                time.sleep(2**a); continue
            raise
    raise RuntimeError("gemini unavailable")

def extract(t):
    m = re.search(r"```python\s*(.*?)```", t, re.S) or re.search(r"```\s*(.*?)```", t, re.S)
    return (m.group(1) if m else t).strip()

def load(code):
    f = tempfile.NamedTemporaryFile("w",suffix=".py",delete=False,prefix="sg_")
    f.write(code); f.close()
    try:
        sp = importlib.util.spec_from_file_location(f"m{abs(hash(code))}", f.name)
        m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
        if not hasattr(m,"launch_kernel"): return None,"no launch_kernel"
        return m.launch_kernel, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

q = torch.randn(B,H,S,D,dtype=DT,device=dev)
k = torch.randn(B,H,S,D,dtype=DT,device=dev)
v = torch.randn(B,H,S,D,dtype=DT,device=dev)
flash = gpu_us(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True))
base    = gpu_us(lambda: ref_sigmoid(q,k,v))
print(f"flash {flash:.0f}us   sigmoid_attn eager baseline {base:.0f}us\n")

OUT = os.path.join(HOME,"kernels_sigmoid_fix"); os.makedirs(OUT, exist_ok=True)
res = []
for model in ("gemini-2.5-pro","gemini-3.1-pro-preview"):
    print("="*70); print(f"MODEL {model}"); print("="*70)
    hist, done = [], None
    for it in range(1, MAX_ITER+1):
        print(f"  -- iter {it}/{MAX_ITER} --")
        p = PROMPT + ("\n\nPREVIOUS ATTEMPTS (fix these):\n"+"\n".join(hist) if hist else "")
        try: code = extract(ask(p, model))
        except Exception as e:
            print(f"     gemini: {e}"); hist.append(f"{it}: gemini error"); continue
        tag = model.replace(".","").replace("-","")[:14]
        open(f"{OUT}/sigmoid_{tag}_iter{it}.py","w").write(code)
        fn, err = load(code)
        if fn is None:
            print(f"     COMPILE FAIL: {err[:90]}")
            hist.append(f"{it}: compile failed -> {err[:250]}"); continue
        worst, msgs, bad = 0.0, [], None
        for nm, sc in (("standard",1.0), ("small",0.01)):
            try:
                out, r = fn(q*sc,k*sc,v*sc), ref_sigmoid(q*sc,k*sc,v*sc)
            except Exception as e:
                bad = f"{nm}: {type(e).__name__}: {e}"; break
            if torch.isnan(out).any() or torch.isinf(out).any(): bad=f"{nm}: NaN/Inf"; break
            rel = ((r-out).abs().max()/(r.abs().max()+1e-6)).item()
            worst = max(worst,rel); msgs.append(f"{nm} rel={rel:.2e}")
        if bad:
            print(f"     verify FAIL ({bad[:70]})"); hist.append(f"{it}: {bad[:200]}"); continue
        ok = worst < REL_TOL
        print(f"     verify {'PASS' if ok else 'FAIL'} ({'; '.join(msgs)})")
        if not ok:
            hist.append(f"{it}: numerically wrong, rel={worst:.2e} ({'; '.join(msgs)}). "
                        f"Need < {REL_TOL}. Remember: sigmoid FIRST, then multiply by the "
                        f"causal mask. No softmax, no normalisation.")
            continue
        us = gpu_us(lambda: fn(q,k,v))
        print(f"     {us:.0f} us   speedup {base/us:.2f}x   ({us/flash:.2f}x flash)")
        open(f"{OUT}/sigmoid_{tag}_BEST.py","w").write(code)
        done = dict(model=model, status="ACCEPTED", baseline_us=round(base,1),
                    kernel_us=round(us,1), speedup=round(base/us,2),
                    vs_flash=round(us/flash,2), rel_err=f"{worst:.1e}", iters=it)
        break
    res.append(done or dict(model=model, status="FAILED", baseline_us=round(base,1),
                            kernel_us=None, speedup=None, vs_flash=None,
                            rel_err=None, iters=MAX_ITER))

print("\n"+"="*78)
print(f"{'model':<26}{'status':<11}{'baseline':>10}{'kernel':>9}{'speedup':>9}{'xflash':>8}{'it':>4}")
print("-"*78)
for r in res:
    g = lambda x,s: (s % x) if x is not None else "--"
    print(f"{r['model']:<26}{r['status']:<11}{r['baseline_us']:>9.0f}u{g(r['kernel_us'],'%8.0fu')}"
          f"{g(r['speedup'],'%8.2fx')}{g(r['vs_flash'],'%7.2fx')}{r['iters']:>4}")
print("="*78)
json.dump(res, open(f"{HOME}/results_sigmoid_fix.json","w"), indent=2)
print(f"\n-> {OUT}/   and ~/results_sigmoid_fix.json")
