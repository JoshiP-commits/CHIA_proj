import glob, os, math, csv, time, importlib.util, statistics as st
import torch, torch.nn.functional as F

dev="cuda"; HOME=os.path.expanduser("~"); PEAK_BW=1555.0e9; ROUNDS=3
DT=torch.bfloat16; Bn,H,D=2,32,128
torch._dynamo.config.cache_size_limit=256

def gpu_ms(fn, warmup=10, target_ms=300.0):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    e0,e1=torch.cuda.Event(True),torch.cuda.Event(True)
    e0.record(); fn(); e1.record(); torch.cuda.synchronize()
    one=max(e0.elapsed_time(e1),1e-3)
    iters=max(5,min(200,int(target_ms/one)))
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/iters

def T(fn): return st.median([gpu_ms(fn) for _ in range(ROUNDS)])*1000.0

def load(p):
    sp=importlib.util.spec_from_file_location("k"+str(abs(hash(p))),p)
    m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m.launch_kernel

def relerr(r,o): return ((r-o).abs().max()/(r.abs().max()+1e-6)).item()

# ---------- references parameterised by S ----------
def build_refs(S):
    SC=1.0/math.sqrt(D)
    I=lambda:(torch.arange(S,device=dev).view(-1,1), torch.arange(S,device=dev).view(1,-1))
    TEMP=(0.5+torch.arange(H,device=dev,dtype=DT)/H).view(1,H,1,1)
    SLOPE=torch.arange(1,H+1,device=dev,dtype=torch.float32).view(1,H,1,1)
    def mk(mod=None,maskfn=None,norm="softmax"):
        def ref(q,k,v):
            s=(q@k.transpose(-2,-1))*SC
            if mod is not None: s=mod(s)
            i,j=I(); ok=maskfn(i,j) if maskfn else (j<=i)
            if norm=="softmax": p=torch.softmax(s.masked_fill(~ok,float("-inf")).float(),-1)
            elif norm=="sigmoid": p=torch.sigmoid(s.float())*ok.float()
            else: p=F.relu(s.float())*ok.float()
            return p.to(DT)@v
        return ref
    def alibi(q,k,v):
        s=(q@k.transpose(-2,-1))*SC; i,j=I()
        s=s+(SLOPE*(j-i).float()).to(DT)
        return torch.softmax(s.masked_fill(j>i,float("-inf")).float(),-1).to(DT)@v
    return {
     "sigmoid_attn":        mk(norm="sigmoid"),
     "relu_attn":           mk(norm="relu"),
     "temp_perhead":        mk(mod=lambda s: s*TEMP),
     "bias_alibi":          alibi,
     "bias_dilated":        mk(maskfn=lambda i,j:(j<=i)&(((i-j)%4)==0)),
     "bias_local":          mk(maskfn=lambda i,j:(i-j).abs()<256),
     "bias_block_diagonal": mk(maskfn=lambda i,j:((i//512)==(j//512))&(j<=i)),
     "bias_prefix_lm":      mk(maskfn=lambda i,j:(j<=i)|(j<256)),
     "softcap_gemma2":      mk(mod=lambda s: torch.tanh(s/30)*30),
     "bias_sliding_window": mk(maskfn=lambda i,j:(j<=i)&((i-j)<256)),
    }

# ---------- locate synthesized kernels ----------
GS={}
for d in glob.glob(os.path.join(HOME,"kernels_gemini*"))+glob.glob(os.path.join(HOME,"kernels_sigmoid_fix")):
    for p in glob.glob(os.path.join(d,"*_BEST.py")):
        bn=os.path.basename(p).replace("_BEST.py","")
        op="sigmoid_attn" if bn.startswith("sigmoid_") else bn.split("_gemini")[0]
        GS.setdefault(op,[]).append(p)

# =========================================================
# PART A -- FlexAttention on all 10 ops
# =========================================================
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

S=2048
refs=build_refs(S)
q=torch.randn(Bn,H,S,D,dtype=DT,device=dev)
k=torch.randn(Bn,H,S,D,dtype=DT,device=dev)
v=torch.randn(Bn,H,S,D,dtype=DT,device=dev)
slopes=torch.arange(1,H+1,device=dev,dtype=torch.float32)
temps =(0.5+torch.arange(H,device=dev,dtype=torch.float32)/H)

causal   = lambda b,h,qi,kj: kj<=qi
FLEX = {
 "temp_perhead":        (lambda sc,b,h,qi,kj: sc*temps[h],                       causal),
 "bias_alibi":          (lambda sc,b,h,qi,kj: sc+slopes[h]*(kj-qi),              causal),
 "softcap_gemma2":      (lambda sc,b,h,qi,kj: torch.tanh(sc/30.0)*30.0,          causal),
 "bias_dilated":        (None, lambda b,h,qi,kj: (kj<=qi)&(((qi-kj)%4)==0)),
 "bias_local":          (None, lambda b,h,qi,kj: (qi-kj).abs()<256),
 "bias_block_diagonal": (None, lambda b,h,qi,kj: ((qi//512)==(kj//512))&(kj<=qi)),
 "bias_prefix_lm":      (None, lambda b,h,qi,kj: (kj<=qi)|(kj<256)),
 "bias_sliding_window": (None, lambda b,h,qi,kj: (kj<=qi)&((qi-kj)<256)),
}
NOT_EXPRESSIBLE = {
 "sigmoid_attn":"flex_attention normalises with softmax; sigmoid normalisation cannot be expressed via score_mod",
 "relu_attn":   "flex_attention normalises with softmax; relu normalisation cannot be expressed via score_mod",
}

print("="*104)
print("PART A -- FlexAttention comparison, bf16, [2,32,2048,128]")
print("="*104)
fa=torch.compile(flex_attention)
h=(f"{'op':<22}{'eager':>9}{'compiled':>10}{'flex':>9}{'flex_cc':>9}"
   f"{'graphsyn':>10}{'GS/flex':>9}{'relerr':>9}  note")
print(h); print("-"*len(h))
rowsA=[]
for op,ref in refs.items():
    te=T(lambda: ref(q,k,v))
    tc=None
    try:
        cf=torch.compile(ref); o=cf(q,k,v); torch.cuda.synchronize()
        if relerr(ref(q,k,v),o)<3e-2: tc=T(lambda: cf(q,k,v))
    except Exception: pass
    tg=None
    for p in GS.get(op,[]):
        try:
            fn=load(p); o=fn(q,k,v)
            if torch.isnan(o).any() or torch.isinf(o).any(): continue
            if relerr(ref(q,k,v),o)>=3e-2: continue
            t=T(lambda: fn(q,k,v))
            if tg is None or t<tg: tg=t
        except Exception: pass

    tf,cc,fe,note=None,None,None,""
    if op in NOT_EXPRESSIBLE:
        note="NOT EXPRESSIBLE"
    else:
        smod,mmod=FLEX[op]
        try:
            bm=create_block_mask(mmod,B=None,H=None,Q_LEN=S,KV_LEN=S,device=dev)
            call=(lambda: fa(q,k,v,score_mod=smod,block_mask=bm)) if smod else \
                 (lambda: fa(q,k,v,block_mask=bm))
            t0=time.time(); o=call(); torch.cuda.synchronize(); cc=time.time()-t0
            fe=relerr(ref(q,k,v),o)
            if fe<3e-2: tf=T(call)
            else: note=f"numerical mismatch rel={fe:.1e}"
        except Exception as e:
            note=f"{type(e).__name__}: {str(e)[:40]}"

    g=lambda x,s:(s%x) if x is not None else "--"
    print(f"{op:<22}{te:>8.0f}u"+g(tc,'%9.0fu')+g(tf,'%8.0fu')+g(cc,'%8.1fs')
          +g(tg,'%9.0fu')+(f"{tf/tg:>8.2f}x" if (tf and tg) else f"{'--':>9}")
          +(f"{fe:>9.1e}" if fe is not None else f"{'--':>9}")+f"  {note}")
    rowsA.append(dict(op=op, eager_us=round(te,1),
        compiled_us=round(tc,1) if tc else None,
        flex_us=round(tf,1) if tf else None,
        flex_compile_s=round(cc,1) if cc else None,
        flex_relerr=f"{fe:.1e}" if fe is not None else None,
        graphsynth_us=round(tg,1) if tg else None,
        gs_vs_flex=round(tf/tg,2) if (tf and tg) else None, note=note))
del q,k,v; torch.cuda.empty_cache()

# =========================================================
# PART B -- sequence-length sweep
# =========================================================
print("\n"+"="*104)
print("PART B -- sequence-length sweep. Kernels were synthesized at S=2048 only;")
print("          S=1024 and S=4096 test whether they generalise to unseen shapes.")
print("="*104)
h=(f"{'op':<22}{'S':>6}{'eager':>9}{'compiled':>10}{'graphsyn':>10}"
   f"{'vs eager':>10}{'vs comp':>9}{'relerr':>9}  status")
print(h); print("-"*len(h))
rowsB=[]
for S in (1024,2048,4096):
    refs=build_refs(S)
    q=torch.randn(Bn,H,S,D,dtype=DT,device=dev)
    k=torch.randn(Bn,H,S,D,dtype=DT,device=dev)
    v=torch.randn(Bn,H,S,D,dtype=DT,device=dev)
    for op,ref in refs.items():
        te=T(lambda: ref(q,k,v))
        tc=None
        try:
            cf=torch.compile(ref); o=cf(q,k,v); torch.cuda.synchronize()
            if relerr(ref(q,k,v),o)<3e-2: tc=T(lambda: cf(q,k,v))
        except Exception: pass
        tg,re_,status=None,None,"no kernel"
        for p in GS.get(op,[]):
            try:
                fn=load(p); o=fn(q,k,v)
                if o.shape!=q.shape: status="wrong shape"; continue
                if torch.isnan(o).any() or torch.isinf(o).any(): status="NaN/Inf"; continue
                e=relerr(ref(q,k,v),o)
                if e>=3e-2: status=f"incorrect"; re_=e; continue
                t=T(lambda: fn(q,k,v))
                if tg is None or t<tg: tg,re_,status=t,e,"generalises"
            except Exception as ex:
                status=type(ex).__name__[:14]
        g=lambda x,s:(s%x) if x is not None else "--"
        print(f"{op:<22}{S:>6}{te:>8.0f}u"+g(tc,'%9.0fu')+g(tg,'%9.0fu')
              +(f"{te/tg:>9.2f}x" if tg else f"{'--':>10}")
              +(f"{tc/tg:>8.2f}x" if (tc and tg) else f"{'--':>9}")
              +(f"{re_:>9.1e}" if re_ is not None else f"{'--':>9}")+f"  {status}")
        rowsB.append(dict(op=op, S=S, eager_us=round(te,1),
            compiled_us=round(tc,1) if tc else None,
            graphsynth_us=round(tg,1) if tg else None,
            gs_vs_eager=round(te/tg,2) if tg else None,
            gs_vs_compiled=round(tc/tg,2) if (tc and tg) else None,
            relerr=f"{re_:.1e}" if re_ is not None else None, status=status))
    del q,k,v; torch.cuda.empty_cache()

for nm,rows in (("flexattention",rowsA),("seqsweep",rowsB)):
    with open(f"{HOME}/FINAL_{nm}.csv","w",newline="") as fh:
        ks=sorted({k for r in rows for k in r})
        w=csv.DictWriter(fh,fieldnames=ks); w.writeheader(); w.writerows(rows)

geo=lambda vs: math.exp(sum(math.log(x) for x in vs)/len(vs))
fx=[r["gs_vs_flex"] for r in rowsA if r["gs_vs_flex"]]
print(f"\nSUMMARY A: FlexAttention expressible on {sum(1 for r in rowsA if r['flex_us'])}/10 ops")
if fx:
    w=sum(1 for x in fx if x>1.0)
    print(f"  GraphSynth vs FlexAttention: geomean {geo(fx):.2f}x, faster on {w}/{len(fx)}")
print(f"  not expressible: {[r['op'] for r in rowsA if r['note']=='NOT EXPRESSIBLE']}")

print("\nSUMMARY B: shape generalisation")
for S in (1024,2048,4096):
    sub=[r for r in rowsB if r["S"]==S]
    ok=[r for r in sub if r["status"]=="generalises"]
    e=[r["gs_vs_eager"] for r in ok if r["gs_vs_eager"]]
    c=[r["gs_vs_compiled"] for r in ok if r["gs_vs_compiled"]]
    print(f"  S={S:<5} {len(ok)}/10 correct   "
          + (f"geomean vs eager {geo(e):.2f}x  vs compiled {geo(c):.2f}x" if e else "--"))
print("\n-> ~/FINAL_flexattention.csv  ~/FINAL_seqsweep.csv")
