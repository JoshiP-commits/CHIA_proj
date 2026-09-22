import glob, os, math, csv, importlib.util, statistics as st
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

DT, dev = torch.bfloat16, "cuda"
B, H, S, D = 2, 32, 2048, 128
HOME, ROUNDS, REL_TOL = os.path.expanduser("~"), 5, 3e-2

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
    evs = [e for e in pr.key_averages() if dtime(e) > 0]
    t = sum(dtime(e) for e in evs)/nm
    return t if t > 0 else ev*1000

IDX = lambda: (torch.arange(S,device=dev).view(-1,1), torch.arange(S,device=dev).view(1,-1))
SC  = 1.0/math.sqrt(D)
_TEMP  = (0.5 + torch.arange(H,device=dev,dtype=DT)/H).view(1,H,1,1)
_SLOPE = torch.arange(1,H+1,device=dev,dtype=torch.float32).view(1,H,1,1)

def mk(mod=None, maskfn=None, norm="softmax"):
    def ref(q,k,v):
        s = (q@k.transpose(-2,-1))*SC
        if mod is not None: s = mod(s)
        i,j = IDX(); ok = maskfn(i,j) if maskfn else (j<=i)
        s = s.masked_fill(~ok, float("-inf") if norm=="softmax" else 0.0)
        p = (torch.softmax(s.float(),-1) if norm=="softmax"
             else torch.sigmoid(s.float()) if norm=="sigmoid" else F.relu(s.float())).to(DT)
        return p@v
    return ref

def ref_alibi(q,k,v):
    s = (q@k.transpose(-2,-1))*SC; i,j = IDX()
    s = s + (_SLOPE*(j-i).float()).to(DT)
    return torch.softmax(s.masked_fill(j>i,float("-inf")).float(),-1).to(DT)@v

REFS = {
 "sigmoid_attn":        mk(norm="sigmoid"),
 "relu_attn":           mk(norm="relu"),
 "temp_perhead":        mk(mod=lambda s: s*_TEMP),
 "bias_alibi":          ref_alibi,
 "bias_dilated":        mk(maskfn=lambda i,j: (j<=i)&(((i-j)%4)==0)),
 "bias_local":          mk(maskfn=lambda i,j: (i-j).abs()<256),
 "bias_block_diagonal": mk(maskfn=lambda i,j: ((i//512)==(j//512))&(j<=i)),
 "bias_prefix_lm":      mk(maskfn=lambda i,j: (j<=i)|(j<256)),
 "softcap_gemma2":      mk(mod=lambda s: torch.tanh(s/30)*30),
 "bias_sliding_window": mk(maskfn=lambda i,j: (j<=i)&((i-j)<256)),
}

def load(path):
    sp = importlib.util.spec_from_file_location("k"+str(abs(hash(path))), path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m.launch_kernel

MODELS = {}
for d in sorted(glob.glob(os.path.join(HOME,"kernels_*"))):
    tag = os.path.basename(d).replace("kernels_","")
    name = "2.5-pro" if "25pro" in tag else "3.1-pro" if "31pro" in tag else tag
    MODELS.setdefault(name, d)
print("models found:", ", ".join(f"{k} -> {os.path.basename(v)}" for k,v in MODELS.items()), "\n")

q = torch.randn(B,H,S,D,dtype=DT,device=dev)
k = torch.randn(B,H,S,D,dtype=DT,device=dev)
v = torch.randn(B,H,S,D,dtype=DT,device=dev)

# load kernels + verify once
K, VERR = {}, {}
for op, ref in REFS.items():
    for mname, d in MODELS.items():
        p = os.path.join(d, f"{op}_BEST.py")
        if not os.path.exists(p): continue
        try:
            fn = load(p); out = fn(q,k,v); r = ref(q,k,v)
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"  drop {op}/{mname}: NaN/Inf"); continue
            rel = ((r-out).abs().max()/(r.abs().max()+1e-6)).item()
            if rel >= REL_TOL:
                print(f"  drop {op}/{mname}: rel={rel:.2e}"); continue
            K[(op,mname)] = fn; VERR[(op,mname)] = rel
        except Exception as e:
            print(f"  drop {op}/{mname}: {type(e).__name__}")

flash_r, base_r, kern_r = [], {op:[] for op in REFS}, {key:[] for key in K}
for rd in range(ROUNDS):
    flash_r.append(gpu_us(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True)))
    for op, ref in REFS.items():
        base_r[op].append(gpu_us(lambda: ref(q,k,v)))          # baseline
        for (o,mn), fn in K.items():                            # kernels, adjacent
            if o == op: kern_r[(o,mn)].append(gpu_us(lambda: fn(q,k,v)))
    print(f"  round {rd+1}/{ROUNDS} done")

flash = st.median(flash_r)
print(f"\nflash ceiling {flash:.0f} us  (spread {min(flash_r):.0f}-{max(flash_r):.0f})\n")

rows = []
names = list(MODELS.keys())
hdr = f"{'op':<22}{'baseline':>12}" + "".join(f"{n+' us':>12}{n+' x':>10}" for n in names)
print(hdr); print("-"*len(hdr))
for op in REFS:
    b = st.median(base_r[op]); bspread = max(base_r[op])/min(base_r[op])
    line = f"{op:<22}{b:>11.0f}u"
    row = dict(op=op, baseline_us=round(b,1), baseline_spread=round(bspread,2),
               flash_us=round(flash,1))
    for n in names:
        key = (op,n)
        if key in kern_r and kern_r[key]:
            kt = st.median(kern_r[key]); sp = b/kt
            line += f"{kt:>11.0f}u{sp:>9.2f}x"
            row[f"{n}_us"] = round(kt,1); row[f"{n}_speedup"] = round(sp,2)
            row[f"{n}_xflash"] = round(kt/flash,2); row[f"{n}_relerr"] = f"{VERR[key]:.1e}"
        else:
            line += f"{'--':>12}{'--':>10}"
    print(line); rows.append(row)

print("\nbaseline stability (max/min across rounds):")
for op in REFS:
    s = max(base_r[op])/min(base_r[op])
    flag = "  <-- UNSTABLE" if s > 1.15 else ""
    print(f"  {op:<22}{s:5.2f}x{flag}")

with open(f"{HOME}/final_benchmark.csv","w",newline="") as f:
    ks = sorted({k for r in rows for k in r})
    w = csv.DictWriter(f, fieldnames=ks); w.writeheader(); w.writerows(rows)

print("\nSPEEDUPS > 1.0 ONLY (what you can report):")
for r in rows:
    for n in names:
        sp = r.get(f"{n}_speedup")
        if sp and sp > 1.0:
            print(f"  {r['op']:<22}{n:<9}{sp:6.2f}x  ({r[f'{n}_xflash']}x flash, rel {r[f'{n}_relerr']})")
print("\n-> ~/final_benchmark.csv")
