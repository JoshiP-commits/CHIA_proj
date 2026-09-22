import os, re, math, json, time, tempfile, importlib.util, datetime
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from google import genai

PROJECT  = os.environ.get("GCP_PROJECT", "a3-chia-hack26ath-7724")
MODEL    = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
MAX_ITER = int(os.environ.get("MAX_ITER", 3))
HOME     = os.path.expanduser("~")
TAG      = MODEL.replace(".","").replace("-","")[:14] + "_" + datetime.datetime.now().strftime("%H%M")
OUTDIR   = os.path.join(HOME, "kernels_" + TAG); os.makedirs(OUTDIR, exist_ok=True)

DT, dev = torch.bfloat16, "cuda"
B, H, S, D = 2, 32, 2048, 128
REL_TOL = 3e-2

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
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(nm): fn()
        torch.cuda.synchronize()
    evs = [e for e in pr.key_averages() if dtime(e) > 0]
    t = sum(dtime(e) for e in evs)/nm
    return (t if t > 0 else ev*1000), sum(e.count for e in evs)/nm

IDX = lambda: (torch.arange(S,device=dev).view(-1,1), torch.arange(S,device=dev).view(1,-1))

def base_scores(q,k):
    return (q @ k.transpose(-2,-1)) / math.sqrt(D)

def mk_ref(mod=None, maskfn=None, norm="softmax"):
    def ref(q,k,v):
        s = base_scores(q,k)
        if mod is not None: s = mod(s)
        i,j = IDX()
        ok = maskfn(i,j) if maskfn else (j <= i)
        s = s.masked_fill(~ok, float("-inf") if norm=="softmax" else 0.0)
        if   norm=="softmax": p = torch.softmax(s.float(),-1).to(DT)
        elif norm=="sigmoid": p = torch.sigmoid(s.float()).to(DT)
        else:                 p = F.relu(s.float()).to(DT)
        return p @ v
    return ref

_TEMP = (0.5 + torch.arange(H,device=dev,dtype=DT)/H).view(1,H,1,1)
_SLOPE = torch.arange(1,H+1,device=dev,dtype=torch.float32).view(1,H,1,1)

def ref_alibi(q,k,v):
    s = base_scores(q,k); i,j = IDX()
    s = s + (_SLOPE*(j-i).float()).to(DT)
    return torch.softmax(s.masked_fill(j>i,float("-inf")).float(),-1).to(DT) @ v

COMMON = """
You are an expert GPU kernel engineer writing Triton for an NVIDIA A100 (sm80).

HARD REQUIREMENTS
- Return ONE ```python code block, nothing else.
- @triton.jit kernel named exactly `graphsynth_kernel`.
- `def launch_kernel(q, k, v):` taking three bfloat16 tensors [2,32,2048,128],
  returning ONE bfloat16 tensor of the same shape.
- FlashAttention-style tiling: loop over KV blocks, running max + running sum,
  fp32 accumulators. NEVER materialise the [S,S] score matrix in DRAM.
- Do NOT use `continue` or `break` in the loop. Bound the loop range on the host
  or use tl.where.

TRITON API CONSTRAINTS -- violating any of these fails:
- `tl.math.tanh` does NOT exist. Use 2.0*tl.sigmoid(2.0*x) - 1.0
- `tl.dot()` has NO `trans_b`. Load K with swapped strides instead.
- Python ints have no `.to()`. Compute 1/sqrt(D) on the host, pass as a float.
- Use `tl.where(cond,a,b)` for masking; no boolean indexing.
- Init running max with `tl.full([BLOCK_M], -1e9, tl.float32)`, not -inf.
- Head index is available as a program id -- derive per-head constants from it.

OPERATOR (scale = 1/sqrt(128)):
"""

T = {}
def add(name, sem, ref):
    T[name] = dict(prompt=COMMON + sem, ref=ref)

add("sigmoid_attn",
 "s=(q@k^T)*scale; causal mask j<=i (masked entries contribute 0);\n"
 "p=sigmoid(s) elementwise (NO softmax, no running max needed -- much simpler);\n"
 "out=p@v",
 mk_ref(norm="sigmoid"))

add("relu_attn",
 "s=(q@k^T)*scale; causal mask j<=i (masked contribute 0);\n"
 "p=relu(s) elementwise (NO softmax); out=p@v",
 mk_ref(norm="relu"))

add("temp_perhead",
 "s=(q@k^T)*scale; multiply s by a PER-HEAD temperature t_h = 0.5 + h/32 where\n"
 "h is the head index (derive it from the program id, do not pass a tensor);\n"
 "causal mask j<=i; p=softmax(s); out=p@v",
 mk_ref(mod=lambda s: s*_TEMP))

add("bias_alibi",
 "s=(q@k^T)*scale; add ALiBi bias slope_h*(j-i) where slope_h = h+1 and h is the\n"
 "head index (compute it from the program id -- do NOT load a bias tensor, that\n"
 "would be 1.07 GB of avoidable DRAM traffic); causal mask j<=i;\n"
 "p=softmax(s); out=p@v",
 ref_alibi)

add("bias_dilated",
 "s=(q@k^T)*scale; query i attends to key j only if (j<=i) AND ((i-j) %% 4 == 0);\n"
 "p=softmax(s); out=p@v. Compute the stride test arithmetically in the kernel.",
 mk_ref(maskfn=lambda i,j: (j<=i) & (((i-j) % 4)==0)))

add("bias_local",
 "s=(q@k^T)*scale; query i attends to key j only if abs(i-j) < 256 (BIDIRECTIONAL\n"
 "band, not causal); p=softmax(s); out=p@v. Only KV blocks overlapping the band\n"
 "need visiting -- bound the loop range on the host accordingly.",
 mk_ref(maskfn=lambda i,j: (i-j).abs() < 256))

add("bias_block_diagonal",
 "s=(q@k^T)*scale; tokens are packed documents of length 512: doc(x)=x//512.\n"
 "query i attends to key j only if (doc(i)==doc(j)) AND (j<=i);\n"
 "p=softmax(s); out=p@v. Only the diagonal document block needs visiting.",
 mk_ref(maskfn=lambda i,j: ((i//512)==(j//512)) & (j<=i)))

add("bias_prefix_lm",
 "s=(q@k^T)*scale; prefix-LM mask: query i attends to key j if (j<=i) OR (j<256)\n"
 "-- the first 256 tokens are visible to everyone; p=softmax(s); out=p@v",
 mk_ref(maskfn=lambda i,j: (j<=i) | (j<256)))

add("softcap_gemma2",
 "s=(q@k^T)*scale; s=tanh(s/30)*30 (Gemma-2 logit soft-capping, applied BEFORE\n"
 "the running-max update); causal mask j<=i; p=softmax(s); out=p@v",
 mk_ref(mod=lambda s: torch.tanh(s/30)*30))

add("bias_sliding_window",
 "s=(q@k^T)*scale; query i attends to key j only if (j<=i) AND (i-j < 256);\n"
 "p=softmax(s); out=p@v. Only ~12%% of the matrix is unmasked -- bound the KV\n"
 "loop range on the host so masked blocks are never visited.",
 mk_ref(maskfn=lambda i,j: (j<=i) & ((i-j)<256)))

client = genai.Client(vertexai=True, project=PROJECT, location="global")
def ask(p, r=4):
    for a in range(r):
        try: return client.models.generate_content(model=MODEL, contents=p).text
        except Exception as e:
            if any(x in str(e) for x in ("503","UNAVAILABLE","429","RESOURCE_EXHAUSTED")):
                time.sleep(2**a); continue
            raise
    raise RuntimeError("gemini unavailable")

def extract(t):
    m = re.search(r"```python\s*(.*?)```", t, re.S) or re.search(r"```\s*(.*?)```", t, re.S)
    return (m.group(1) if m else t).strip()

def load(code):
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, prefix="gs_")
    f.write(code); f.close()
    try:
        sp = importlib.util.spec_from_file_location(f"m{abs(hash(code))}", f.name)
        m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
        if not hasattr(m,"launch_kernel"): return None,"no launch_kernel"
        return m.launch_kernel, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def verify(fn, ref, q,k,v):
    worst, msgs = 0.0, []
    for nm, sc in (("standard",1.0), ("small",0.01)):
        qq,kk,vv = q*sc,k*sc,v*sc
        try: out, r = fn(qq,kk,vv), ref(qq,kk,vv)
        except Exception as e: return False, 9e9, f"{nm}: {type(e).__name__}: {e}"
        if out.shape != r.shape:   return False,9e9,f"{nm}: shape {tuple(out.shape)}"
        if torch.isnan(out).any(): return False,9e9,f"{nm}: NaN"
        if torch.isinf(out).any(): return False,9e9,f"{nm}: Inf"
        rel = ((r-out).abs().max()/(r.abs().max()+1e-6)).item()
        worst = max(worst,rel); msgs.append(f"{nm} rel={rel:.2e}")
    return worst < REL_TOL, worst, "; ".join(msgs)

q = torch.randn(B,H,S,D,dtype=DT,device=dev)
k = torch.randn(B,H,S,D,dtype=DT,device=dev)
v = torch.randn(B,H,S,D,dtype=DT,device=dev)
flash,_ = gpu_us(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True))
print(f"MODEL={MODEL}   flash ceiling {flash:.0f} us   -> {OUTDIR}\n")

res = []
for name, task in T.items():
    print("="*74); print(f"OP: {name}"); print("="*74)
    try:
        bus, bnk = gpu_us(lambda: task["ref"](q,k,v))
    except Exception as e:
        print(f"  baseline failed: {e}"); continue
    print(f"  eager baseline {bus:.0f} us ({bnk:.0f} kernels)")
    hist, best = [], None
    for it in range(1, MAX_ITER+1):
        print(f"  -- iter {it}/{MAX_ITER} --")
        p = task["prompt"] + ("\n\nPREVIOUS ATTEMPTS (fix these):\n"+"\n".join(hist) if hist else "")
        try: code = extract(ask(p))
        except Exception as e:
            print(f"     gemini: {e}"); hist.append(f"{it}: gemini error"); continue
        open(f"{OUTDIR}/{name}_iter{it}.py","w").write(code)
        fn, err = load(code)
        if fn is None:
            print(f"     COMPILE FAIL: {err[:90]}")
            hist.append(f"{it}: compile failed -> {err[:250]}"); continue
        ok, rel, det = verify(fn, task["ref"], q,k,v)
        print(f"     verify {'PASS' if ok else 'FAIL'} ({det[:80]})")
        if not ok:
            hist.append(f"{it}: wrong, rel={rel:.2e} ({det[:120]}); need < {REL_TOL}")
            continue
        us, nk = gpu_us(lambda: fn(q,k,v))
        print(f"     {us:.0f} us   speedup {bus/us:.2f}x   ({us/flash:.2f}x flash)")
        best = (us, bus/us, rel, code, it)
        open(f"{OUTDIR}/{name}_BEST.py","w").write(code)
        break
    if best:
        us, sp, rel, code, it = best
        res.append(dict(op=name, model=MODEL, status="ACCEPTED", baseline_us=round(bus,1),
                        kernel_us=round(us,1), speedup=round(sp,2),
                        vs_flash=round(us/flash,2), rel_err=f"{rel:.1e}", iters=it))
    else:
        res.append(dict(op=name, model=MODEL, status="FAILED", baseline_us=round(bus,1),
                        kernel_us=None, speedup=None, vs_flash=None, rel_err=None,
                        iters=MAX_ITER))
    json.dump(res, open(f"{HOME}/results_{TAG}.json","w"), indent=2)

print("\n"+"="*92)
print(f"MODEL: {MODEL}")
print(f"{'op':<24}{'status':<11}{'baseline':>10}{'kernel':>9}{'speedup':>9}{'xflash':>8}{'relerr':>9}{'it':>4}")
print("-"*92)
n_ok = 0
for r in res:
    g = lambda x,s: (s % x) if x is not None else "--"
    n_ok += r["status"]=="ACCEPTED"
    print(f"{r['op']:<24}{r['status']:<11}{r['baseline_us']:>9.0f}u{g(r['kernel_us'],'%8.0fu')}"
          f"{g(r['speedup'],'%8.2fx')}{g(r['vs_flash'],'%7.2fx')}{str(r['rel_err'] or '--'):>9}{r['iters']:>4}")
print("-"*92)
print(f"{n_ok}/{len(res)} accepted")
print("="*92)
