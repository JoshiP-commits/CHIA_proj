import glob, os, math, csv, time, importlib.util, statistics as st
import torch, torch.nn.functional as F

dev = "cuda"; HOME = os.path.expanduser("~")
PEAK_BW = 1555.0e9
ROUNDS = 3
torch._dynamo.config.cache_size_limit = 64

# ---------- validated timer: CUDA events around a loop ----------
def gpu_ms(fn, warmup=10, target_ms=300.0):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); fn(); e1.record(); torch.cuda.synchronize()
    one = max(e0.elapsed_time(e1), 1e-3)
    iters = max(5, min(200, int(target_ms/one)))
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/iters

def T(fn):  # median of ROUNDS, in microseconds
    return st.median([gpu_ms(fn) for _ in range(ROUNDS)])*1000.0

def bwpct(nbytes, t_us):
    return nbytes/(t_us*1e-6)/PEAK_BW*100.0

def load(p):
    sp = importlib.util.spec_from_file_location("k"+str(abs(hash(p))), p)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m.launch_kernel

# ================= sanity check =================
Nc=8192
xc=torch.randn(Nc,Nc,device=dev,dtype=torch.float32); yc=torch.empty_like(xc)
tc=T(lambda: yc.copy_(xc)); bc=bwpct(2*Nc*Nc*4, tc)
print(f"SANITY  copy {tc:.1f} us  {bc:.1f}% of peak  "
      f"{'OK' if bc<100 else 'TIMER BROKEN'}\n")
assert bc < 100, "timer invalid"
del xc,yc; torch.cuda.empty_cache()

# ================= PART 1: single-tensor ops (fp32) =================
N=8192; DT1=torch.float32
x=torch.randn(N,N,dtype=DT1,device=dev)
B1 = 2*N*N*4
OPS1 = {
 "relu":       lambda t: F.relu(t),
 "leaky_relu": lambda t: F.leaky_relu(t,0.01),
 "sigmoid":    lambda t: torch.sigmoid(t),
 "silu":       lambda t: F.silu(t),
 "gelu":       lambda t: F.gelu(t),
 "softmax":    lambda t: torch.softmax(t,-1),
 "layernorm":  lambda t: F.layer_norm(t,[N]),
}
KD1 = os.path.join(HOME,"kernels_single_tensor")

print("PART 1 -- single-tensor ops, fp32, [8192,8192]")
print(f"  DRAM floor = {B1/PEAK_BW*1e6:.0f} us\n")
h1=(f"{'op':<12}{'eager':>9}{'BW%':>7}{'compiled':>10}{'BW%':>7}{'ind x':>8}"
    f"{'graphsyn':>10}{'BW%':>7}{'vs eager':>10}{'vs comp':>9}")
print(h1); print("-"*len(h1))
rows1=[]
for op,ref in OPS1.items():
    te=T(lambda: ref(x))
    tc_,cn=None,""
    try:
        cf=torch.compile(ref); o=cf(x); torch.cuda.synchronize()
        if (ref(x)-o).abs().max().item()<1e-3: tc_=T(lambda: cf(x))
        else: cn="numdiff"
    except Exception as e: cn=type(e).__name__[:8]
    tg=None
    p=os.path.join(KD1,f"{op}_BEST.py")
    if os.path.exists(p):
        try:
            fn=load(p); o=fn(x); r=ref(x)
            if not (torch.isnan(o).any() or torch.isinf(o).any()) and \
               ((r-o).abs().max()/(r.abs().max()+1e-9)).item()<1e-4:
                tg=T(lambda: fn(x))
        except Exception: pass
    g=lambda v,s:(s%v) if v is not None else "--"
    print(f"{op:<12}{te:>8.0f}u{bwpct(B1,te):>6.1f}%"
          +(f"{tc_:>9.0f}u{bwpct(B1,tc_):>6.1f}%{te/tc_:>7.2f}x" if tc_ else f"{cn or 'fail':>10}{'--':>7}{'--':>8}")
          +(f"{tg:>9.0f}u{bwpct(B1,tg):>6.1f}%{te/tg:>9.2f}x" if tg else f"{'--':>10}{'--':>7}{'--':>10}")
          +(f"{tc_/tg:>8.2f}x" if (tc_ and tg) else f"{'--':>9}"))
    rows1.append(dict(op=op, dtype="fp32", shape=f"{N}x{N}",
        eager_us=round(te,1), eager_bw=round(bwpct(B1,te),1),
        compiled_us=round(tc_,1) if tc_ else None,
        compiled_bw=round(bwpct(B1,tc_),1) if tc_ else None,
        inductor_x=round(te/tc_,2) if tc_ else None, compile_note=cn,
        graphsynth_us=round(tg,1) if tg else None,
        graphsynth_bw=round(bwpct(B1,tg),1) if tg else None,
        gs_vs_eager=round(te/tg,2) if tg else None,
        gs_vs_compiled=round(tc_/tg,2) if (tc_ and tg) else None))
del x; torch.cuda.empty_cache()

# ================= PART 2: attention variants (bf16) =================
DT2=torch.bfloat16; Bn,H,S,D=2,32,2048,128
SC=1.0/math.sqrt(D)
q=torch.randn(Bn,H,S,D,dtype=DT2,device=dev)
k=torch.randn(Bn,H,S,D,dtype=DT2,device=dev)
v=torch.randn(Bn,H,S,D,dtype=DT2,device=dev)
B2 = 2*2*Bn*D*(H*S + H*S)          # bf16: read QKV + write O
IDX=lambda:(torch.arange(S,device=dev).view(-1,1), torch.arange(S,device=dev).view(1,-1))
_TEMP=(0.5+torch.arange(H,device=dev,dtype=DT2)/H).view(1,H,1,1)
_SLOPE=torch.arange(1,H+1,device=dev,dtype=torch.float32).view(1,H,1,1)

def mk(mod=None,maskfn=None,norm="softmax"):
    def ref(q,k,v):
        s=(q@k.transpose(-2,-1))*SC
        if mod is not None: s=mod(s)
        i,j=IDX(); ok=maskfn(i,j) if maskfn else (j<=i)
        if norm=="softmax": p=torch.softmax(s.masked_fill(~ok,float("-inf")).float(),-1)
        elif norm=="sigmoid": p=torch.sigmoid(s.float())*ok.float()
        else: p=F.relu(s.float())*ok.float()
        return p.to(DT2)@v
    return ref
def ref_alibi(q,k,v):
    s=(q@k.transpose(-2,-1))*SC; i,j=IDX()
    s=s+(_SLOPE*(j-i).float()).to(DT2)
    return torch.softmax(s.masked_fill(j>i,float("-inf")).float(),-1).to(DT2)@v

OPS2={
 "sigmoid_attn":        mk(norm="sigmoid"),
 "relu_attn":           mk(norm="relu"),
 "temp_perhead":        mk(mod=lambda s: s*_TEMP),
 "bias_alibi":          ref_alibi,
 "bias_dilated":        mk(maskfn=lambda i,j:(j<=i)&(((i-j)%4)==0)),
 "bias_local":          mk(maskfn=lambda i,j:(i-j).abs()<256),
 "bias_block_diagonal": mk(maskfn=lambda i,j:((i//512)==(j//512))&(j<=i)),
 "bias_prefix_lm":      mk(maskfn=lambda i,j:(j<=i)|(j<256)),
 "softcap_gemma2":      mk(mod=lambda s: torch.tanh(s/30)*30),
 "bias_sliding_window": mk(maskfn=lambda i,j:(j<=i)&((i-j)<256)),
}
GS={}
for d in glob.glob(os.path.join(HOME,"kernels_gemini*"))+glob.glob(os.path.join(HOME,"kernels_sigmoid_fix")):
    tag="2.5-pro" if "25pro" in d else "3.1-pro" if "31pro" in d else "fix"
    for p in glob.glob(os.path.join(d,"*_BEST.py")):
        bn=os.path.basename(p).replace("_BEST.py","")
        op="sigmoid_attn" if bn.startswith("sigmoid_") else bn.split("_gemini")[0]
        mt="2.5-pro" if "25pro" in bn else "3.1-pro" if "31pro" in bn else tag
        if op in OPS2: GS.setdefault(op,[]).append((mt,p))

tflash=T(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True))
print(f"\n\nPART 2 -- attention variants, bf16, [2,32,2048,128]")
print(f"  FlashAttention (causal) reference = {tflash:.0f} us\n")
h2=(f"{'op':<22}{'eager':>9}{'compiled':>10}{'ind x':>8}"
    f"{'graphsyn':>10}{'model':>9}{'vs eager':>10}{'vs comp':>9}{'x flash':>9}")
print(h2); print("-"*len(h2))
rows2=[]
for op,ref in OPS2.items():
    te=T(lambda: ref(q,k,v))
    tc_,cn=None,""
    try:
        cf=torch.compile(ref); o=cf(q,k,v); torch.cuda.synchronize()
        r=ref(q,k,v)
        if ((r-o).abs().max()/(r.abs().max()+1e-6)).item()<3e-2: tc_=T(lambda: cf(q,k,v))
        else: cn="numdiff"
    except Exception as e: cn=type(e).__name__[:8]
    best=None
    for mt,p in GS.get(op,[]):
        try:
            fn=load(p); o=fn(q,k,v)
            if torch.isnan(o).any() or torch.isinf(o).any(): continue
            r=ref(q,k,v)
            if ((r-o).abs().max()/(r.abs().max()+1e-6)).item()>=3e-2: continue
            t=T(lambda: fn(q,k,v))
            if best is None or t<best[0]: best=(t,mt)
        except Exception: pass
    tg,mt=(best if best else (None,""))
    g=lambda v,s:(s%v) if v is not None else "--"
    print(f"{op:<22}{te:>8.0f}u"
          +(f"{tc_:>9.0f}u{te/tc_:>7.2f}x" if tc_ else f"{cn or 'fail':>10}{'--':>8}")
          +(f"{tg:>9.0f}u{mt:>9}{te/tg:>9.2f}x" if tg else f"{'--':>10}{'--':>9}{'--':>10}")
          +(f"{tc_/tg:>8.2f}x" if (tc_ and tg) else f"{'--':>9}")
          +(f"{tg/tflash:>8.2f}x" if tg else f"{'--':>9}"))
    rows2.append(dict(op=op, dtype="bf16", shape="2x32x2048x128",
        eager_us=round(te,1), compiled_us=round(tc_,1) if tc_ else None,
        inductor_x=round(te/tc_,2) if tc_ else None, compile_note=cn,
        graphsynth_us=round(tg,1) if tg else None, best_model=mt,
        gs_vs_eager=round(te/tg,2) if tg else None,
        gs_vs_compiled=round(tc_/tg,2) if (tc_ and tg) else None,
        gs_vs_flash=round(tg/tflash,2) if tg else None,
        flash_us=round(tflash,1)))

for nm,rows in (("single_tensor",rows1),("attention",rows2)):
    with open(f"{HOME}/FINAL_{nm}.csv","w",newline="") as fh:
        ks=sorted({k for r in rows for k in r})
        w=csv.DictWriter(fh,fieldnames=ks); w.writeheader(); w.writerows(rows)

def geo(vals): return math.exp(sum(math.log(v) for v in vals)/len(vals))
for nm,rows in (("single-tensor",rows1),("attention",rows2)):
    e=[r["gs_vs_eager"] for r in rows if r["gs_vs_eager"]]
    c=[r["gs_vs_compiled"] for r in rows if r["gs_vs_compiled"]]
    print(f"\n{nm}: n={len(e)}  geomean vs eager {geo(e):.2f}x" + (f"  vs compiled {geo(c):.2f}x" if c else ""))
    w=[r['op'] for r in rows if r.get("gs_vs_compiled") and r["gs_vs_compiled"]>1.0]
    print(f"  beats torch.compile on {len(w)}/{len(c)}")

bad=[r for r in rows1 if (r.get("eager_bw") or 0)>100 or (r.get("graphsynth_bw") or 0)>100]
print(f"\nimpossible bandwidth rows: {len(bad)}  {'(clean)' if not bad else [r['op'] for r in bad]}")
print("\n-> ~/FINAL_single_tensor.csv  ~/FINAL_attention.csv")
