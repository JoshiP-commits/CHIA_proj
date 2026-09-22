import glob, os, math, csv, time, importlib.util, statistics as st
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

DT, dev = torch.bfloat16, "cuda"
B,H,S,D = 2,32,2048,128
SC = 1.0/math.sqrt(D)
HOME, ROUNDS, REL_TOL = os.path.expanduser("~"), 3, 3e-2
torch._dynamo.config.cache_size_limit = 64

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

IDX = lambda: (torch.arange(S,device=dev).view(-1,1), torch.arange(S,device=dev).view(1,-1))
_TEMP  = (0.5 + torch.arange(H,device=dev,dtype=DT)/H).view(1,H,1,1)
_SLOPE = torch.arange(1,H+1,device=dev,dtype=torch.float32).view(1,H,1,1)

def mk(mod=None, maskfn=None, norm="softmax"):
    def ref(q,k,v):
        s = (q@k.transpose(-2,-1))*SC
        if mod is not None: s = mod(s)
        i,j = IDX(); ok = maskfn(i,j) if maskfn else (j<=i)
        if norm=="softmax":
            p = torch.softmax(s.masked_fill(~ok, float("-inf")).float(),-1)
        elif norm=="sigmoid":
            p = torch.sigmoid(s.float()) * ok.float()
        else:
            p = F.relu(s.float()) * ok.float()
        return p.to(DT) @ v
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

def load(p):
    sp = importlib.util.spec_from_file_location("k"+str(abs(hash(p))), p)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m.launch_kernel

GS = {}
for d in glob.glob(os.path.join(HOME,"kernels_*")):
    for p in glob.glob(os.path.join(d,"*_BEST.py")):
        bn = os.path.basename(p).replace("_BEST.py","")
        op = "sigmoid_attn" if bn.startswith("sigmoid_") else bn.split("_gemini")[0]
        if op in REFS: GS.setdefault(op, []).append(p)

q = torch.randn(B,H,S,D,dtype=DT,device=dev)
k = torch.randn(B,H,S,D,dtype=DT,device=dev)
v = torch.randn(B,H,S,D,dtype=DT,device=dev)
flash = gpu_us(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True))
print(f"flash ceiling {flash:.0f} us\n")

rows=[]
hdr=(f"{'op':<22}{'eager':>9}{'compiled':>10}{'induct x':>10}"
     f"{'graphsyn':>10}{'vs eager':>10}{'vs comp':>9}{'comp s':>8}")
print(hdr); print("-"*len(hdr))

for op, ref in REFS.items():
    eager = st.median([gpu_us(lambda: ref(q,k,v)) for _ in range(ROUNDS)])
    comp_us, ctime, cnote = None, None, ""
    try:
        cfn = torch.compile(ref)
        t0=time.time(); out=cfn(q,k,v); torch.cuda.synchronize(); ctime=time.time()-t0
        rel = ((ref(q,k,v)-out).abs().max()/(out.abs().max()+1e-6)).item()
        if rel < REL_TOL: comp_us = st.median([gpu_us(lambda: cfn(q,k,v)) for _ in range(ROUNDS)])
        else: cnote=f"rel{rel:.0e}"
    except Exception as e: cnote=type(e).__name__[:9]
    gs_us=None
    for p in GS.get(op,[]):
        try:
            fn=load(p); o=fn(q,k,v)
            if torch.isnan(o).any() or torch.isinf(o).any(): continue
            r=ref(q,k,v)
            if ((r-o).abs().max()/(r.abs().max()+1e-6)).item()>=REL_TOL: continue
            t=st.median([gpu_us(lambda: fn(q,k,v)) for _ in range(ROUNDS)])
            if gs_us is None or t<gs_us: gs_us=t
        except Exception: pass
    line=(f"{op:<22}{eager:>8.0f}u"
          + (f"{comp_us:>9.0f}u{eager/comp_us:>9.2f}x" if comp_us else f"{cnote or 'fail':>10}{'--':>10}")
          + (f"{gs_us:>9.0f}u{eager/gs_us:>9.2f}x" if gs_us else f"{'--':>10}{'--':>10}")
          + (f"{comp_us/gs_us:>8.2f}x" if (comp_us and gs_us) else f"{'--':>9}")
          + f"{ctime or 0:>7.1f}s")
    print(line)
    rows.append(dict(op=op, eager_us=round(eager,1),
        compiled_us=round(comp_us,1) if comp_us else None, compile_note=cnote,
        inductor_speedup=round(eager/comp_us,2) if comp_us else None,
        graphsynth_us=round(gs_us,1) if gs_us else None,
        gs_vs_eager=round(eager/gs_us,2) if gs_us else None,
        gs_vs_compiled=round(comp_us/gs_us,2) if (comp_us and gs_us) else None,
        compile_time_s=round(ctime,1) if ctime else None, flash_us=round(flash,1)))

with open(f"{HOME}/compile_comparison.csv","w",newline="") as fh:
    ks=sorted({k for r in rows for k in r})
    w=csv.DictWriter(fh,fieldnames=ks); w.writeheader(); w.writerows(rows)

both=[r for r in rows if r["gs_vs_compiled"]]
wins=[r for r in both if r["gs_vs_compiled"]>1.0]
print(f"\nGraphSynth beats torch.compile on {len(wins)}/{len(both)} ops")
for r in both:
    tag="WIN " if r["gs_vs_compiled"]>1.0 else "LOSS"
    print(f"  {tag} {r['op']:<22}{r['gs_vs_compiled']:6.2f}x")
print("\n-> ~/compile_comparison.csv")
