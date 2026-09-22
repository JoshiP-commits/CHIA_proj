import os, re, math, json, time, tempfile, importlib.util, csv, statistics as st
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from google import genai

PROJECT="a3-chia-hack26ath-7724"; HOME=os.path.expanduser("~")
MODEL=os.environ.get("GEMINI_MODEL","gemini-3.1-pro-preview")
MAX_ITER=int(os.environ.get("MAX_ITER",3))
DT, dev = torch.float32, "cuda"          # fp32, matches the paper's dtype
N = 8192                                  # 8192x8192 = 268 MB/tensor, solidly memory-bound
REL_TOL, ROUNDS = 1e-4, 3
PEAK_BW = 1555.0e9
OUT=os.path.join(HOME,"kernels_single_tensor"); os.makedirs(OUT,exist_ok=True)
torch._dynamo.config.cache_size_limit=64

def dtime(e):
    for a in ("device_time_total","cuda_time_total","device_time","cuda_time"):
        v=getattr(e,a,None)
        if v: return float(v)
    return 0.0

def gpu_us(fn):
    fn(); torch.cuda.synchronize()
    e0,e1=torch.cuda.Event(True),torch.cuda.Event(True)
    e0.record(); fn(); e1.record(); torch.cuda.synchronize()
    ev=e0.elapsed_time(e1); nm=20 if ev<5 else 5
    for _ in range(5): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(nm): fn()
        torch.cuda.synchronize()
    evs=[e for e in pr.key_averages() if dtime(e)>0]
    t=sum(dtime(e) for e in evs)/nm
    return t if t>0 else ev*1000

OPS = {
 "relu":       (lambda x: F.relu(x),            "out = max(x, 0) elementwise"),
 "leaky_relu": (lambda x: F.leaky_relu(x,0.01), "out = x if x>0 else 0.01*x"),
 "sigmoid":    (lambda x: torch.sigmoid(x),     "out = 1/(1+exp(-x))"),
 "silu":       (lambda x: F.silu(x),            "out = x * sigmoid(x)"),
 "gelu":       (lambda x: F.gelu(x),            "out = 0.5*x*(1+erf(x/sqrt(2)))  (exact erf, not tanh approx)"),
 "softmax":    (lambda x: torch.softmax(x,-1),  "row-wise softmax along the LAST dim, numerically stable (subtract row max)"),
 "layernorm":  (lambda x: F.layer_norm(x,[N]),  "row-wise LayerNorm along the LAST dim, eps=1e-5, no affine weight/bias"),
}

COMMON=f"""
You are an expert GPU kernel engineer writing Triton for an NVIDIA A100 (sm80).

HARD REQUIREMENTS
- Return ONE ```python code block, nothing else.
- @triton.jit kernel named exactly `graphsynth_kernel`.
- `def launch_kernel(x):` taking ONE float32 tensor of shape [{N}, {N}] and
  returning ONE float32 tensor of the same shape.
- Read x from DRAM exactly once and write the output exactly once. This op is
  memory-bound: the only thing that matters is achieving peak DRAM bandwidth.
- Use fp32 accumulators. Choose BLOCK sizes that give good coalescing and
  occupancy; for row-wise ops use one program per row with
  BLOCK = triton.next_power_of_2(row_length) when it fits, otherwise tile.
- No `continue`/`break` inside loops.

TRITON API CONSTRAINTS -- violating any fails:
- `tl.math.tanh` does not exist. Use 2.0*tl.sigmoid(2.0*x)-1.0
- `tl.dot()` has no `trans_b`.
- Python ints have no `.to()`. Pass scalars from the host as floats.
- Use `tl.where(cond,a,b)` for masking; `other=` on tl.load for bounds.
- `tl.math.erf` may be unavailable; if so implement erf via a standard
  polynomial approximation accurate to better than 1e-6.

OPERATOR: """

client=genai.Client(vertexai=True,project=PROJECT,location="global")
def ask(p,r=4):
    for a in range(r):
        try: return client.models.generate_content(model=MODEL,contents=p).text
        except Exception as e:
            if any(x in str(e) for x in ("503","UNAVAILABLE","429","RESOURCE_EXHAUSTED")):
                time.sleep(2**a); continue
            raise
    raise RuntimeError("gemini unavailable")

def extract(t):
    m=re.search(r"```python\s*(.*?)```",t,re.S) or re.search(r"```\s*(.*?)```",t,re.S)
    return (m.group(1) if m else t).strip()

def load(code):
    f=tempfile.NamedTemporaryFile("w",suffix=".py",delete=False,prefix="st_")
    f.write(code); f.close()
    try:
        sp=importlib.util.spec_from_file_location(f"m{abs(hash(code))}",f.name)
        m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
        if not hasattr(m,"launch_kernel"): return None,"no launch_kernel"
        return m.launch_kernel,None
    except Exception as e:
        return None,f"{type(e).__name__}: {e}"

x = torch.randn(N,N,dtype=DT,device=dev)
MIN_BYTES = 2*N*N*4                      # one read + one write
print(f"MODEL={MODEL}   shape=[{N},{N}] fp32   floor={MIN_BYTES/PEAK_BW*1e6:.0f} us\n")

rows=[]
hdr=(f"{'op':<12}{'eager':>9}{'BW%':>7}{'compiled':>10}{'ind x':>8}"
     f"{'graphsyn':>10}{'vs eager':>10}{'vs comp':>9}{'BW%':>7}{'it':>4}")
print(hdr); print("-"*len(hdr))

for op,(ref,sem) in OPS.items():
    eager = st.median([gpu_us(lambda: ref(x)) for _ in range(ROUNDS)])
    comp_us,cnote=None,""
    try:
        cfn=torch.compile(ref); o=cfn(x); torch.cuda.synchronize()
        if (ref(x)-o).abs().max().item() < 1e-3:
            comp_us=st.median([gpu_us(lambda: cfn(x)) for _ in range(ROUNDS)])
        else: cnote="numdiff"
    except Exception as e: cnote=type(e).__name__[:8]

    prompt=COMMON+f"{op}\n{sem}\n\nPyTorch eager takes {eager:.0f} us for this op at this shape."
    hist,gs_us,gs_it=[],None,None
    for it in range(1,MAX_ITER+1):
        p=prompt+("\n\nPREVIOUS ATTEMPTS (fix these):\n"+"\n".join(hist) if hist else "")
        try: code=extract(ask(p))
        except Exception as e: hist.append(f"{it}: gemini error"); continue
        open(f"{OUT}/{op}_iter{it}.py","w").write(code)
        fn,err=load(code)
        if fn is None: hist.append(f"{it}: compile failed -> {err[:250]}"); continue
        try: out=fn(x)
        except Exception as e: hist.append(f"{it}: runtime {type(e).__name__}: {e}"); continue
        if out.shape!=x.shape: hist.append(f"{it}: shape {tuple(out.shape)}"); continue
        if torch.isnan(out).any() or torch.isinf(out).any(): hist.append(f"{it}: NaN/Inf"); continue
        r=ref(x); rel=((r-out).abs().max()/(r.abs().max()+1e-9)).item()
        if rel>=REL_TOL:
            hist.append(f"{it}: wrong, rel={rel:.2e}, need < {REL_TOL}"); continue
        gs_us=st.median([gpu_us(lambda: fn(x)) for _ in range(ROUNDS)]); gs_it=it
        open(f"{OUT}/{op}_BEST.py","w").write(code); break

    g=lambda v,s: (s%v) if v is not None else "--"
    print(f"{op:<12}{eager:>8.0f}u{MIN_BYTES/(eager*1e-6)/PEAK_BW*100:>6.1f}%"
          + (f"{comp_us:>9.0f}u{eager/comp_us:>7.2f}x" if comp_us else f"{cnote or 'fail':>10}{'--':>8}")
          + (f"{gs_us:>9.0f}u{eager/gs_us:>9.2f}x" if gs_us else f"{'--':>10}{'--':>10}")
          + (f"{comp_us/gs_us:>8.2f}x" if (comp_us and gs_us) else f"{'--':>9}")
          + (f"{MIN_BYTES/(gs_us*1e-6)/PEAK_BW*100:>6.1f}%" if gs_us else f"{'--':>7}")
          + f"{gs_it or MAX_ITER:>4}")
    rows.append(dict(op=op, eager_us=round(eager,1),
        eager_bw_pct=round(MIN_BYTES/(eager*1e-6)/PEAK_BW*100,1),
        compiled_us=round(comp_us,1) if comp_us else None, compile_note=cnote,
        inductor_speedup=round(eager/comp_us,2) if comp_us else None,
        graphsynth_us=round(gs_us,1) if gs_us else None,
        gs_vs_eager=round(eager/gs_us,2) if gs_us else None,
        gs_vs_compiled=round(comp_us/gs_us,2) if (comp_us and gs_us) else None,
        gs_bw_pct=round(MIN_BYTES/(gs_us*1e-6)/PEAK_BW*100,1) if gs_us else None,
        iters=gs_it, status="ACCEPTED" if gs_us else "FAILED", model=MODEL))

with open(f"{HOME}/single_tensor_results.csv","w",newline="") as fh:
    ks=sorted({k for r in rows for k in r})
    w=csv.DictWriter(fh,fieldnames=ks); w.writeheader(); w.writerows(rows)
n=sum(r["status"]=="ACCEPTED" for r in rows)
print(f"\n{n}/{len(rows)} accepted")
print("NOTE: eager BW% near 90-100% means PyTorch is already at the DRAM floor")
print(f"\n-> {OUT}/  and ~/single_tensor_results.csv")
