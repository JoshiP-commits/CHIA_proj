import glob, os, math, csv, importlib.util, statistics as st
import torch, torch.nn.functional as F

dev="cuda"; HOME=os.path.expanduser("~"); ROUNDS=3
DT=torch.bfloat16; Bn,H,D=2,32,128
torch._dynamo.config.cache_size_limit=256

def gpu_ms(fn, warmup=10, target_ms=250.0):
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
def relerr(r,o): return ((r-o).abs().max()/(r.abs().max()+1e-6)).item()

def load(p):
    sp=importlib.util.spec_from_file_location("k"+str(abs(hash(p))),p)
    m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m.launch_kernel

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

def model_of(path):
    s = os.path.basename(path) + "|" + os.path.basename(os.path.dirname(path))
    if "31propre" in s or "31pro" in s: return "3.1-pro"
    if "25pro"    in s:                 return "2.5-pro"
    return "unknown"

OPS = list(build_refs(2048).keys())
KERNELS=[]   # (op, model, path)
for d in glob.glob(os.path.join(HOME,"kernels_gemini*"))+glob.glob(os.path.join(HOME,"kernels_sigmoid_fix")):
    for p in glob.glob(os.path.join(d,"*_BEST.py")):
        bn=os.path.basename(p).replace("_BEST.py","")
        op="sigmoid_attn" if bn.startswith("sigmoid_") else bn.split("_gemini")[0]
        if op in OPS: KERNELS.append((op, model_of(p), p))
KERNELS.sort()
print(f"{len(KERNELS)} kernels found\n")

rows=[]
for S in (1024,2048,4096):
    refs=build_refs(S)
    q=torch.randn(Bn,H,S,D,dtype=DT,device=dev)
    k=torch.randn(Bn,H,S,D,dtype=DT,device=dev)
    v=torch.randn(Bn,H,S,D,dtype=DT,device=dev)
    base={op: T(lambda: refs[op](q,k,v)) for op in OPS}
    comp={}
    for op in OPS:
        try:
            cf=torch.compile(refs[op]); o=cf(q,k,v); torch.cuda.synchronize()
            comp[op]=T(lambda: cf(q,k,v)) if relerr(refs[op](q,k,v),o)<3e-2 else None
        except Exception: comp[op]=None

    print("="*96)
    print(f"S = {S}   (kernels were synthesized at S=2048 only)")
    print("="*96)
    h=f"{'op':<22}{'model':<9}{'status':<14}{'kernel':>9}{'vs eager':>10}{'vs comp':>9}{'relerr':>9}"
    print(h); print("-"*len(h))
    for op,mdl,p in KERNELS:
        ref=refs[op]; status,t,re_="",None,None
        try:
            fn=load(p); o=fn(q,k,v)
            if o.shape!=q.shape:                 status="wrong shape"
            elif torch.isnan(o).any():           status="NaN"
            elif torch.isinf(o).any():           status="Inf"
            else:
                re_=relerr(ref(q,k,v),o)
                if re_>=3e-2:                    status="incorrect"
                else:
                    status="OK"; t=T(lambda: fn(q,k,v))
        except AssertionError as e:               status="AssertionError"
        except Exception as e:                    status=type(e).__name__[:13]
        b,c=base[op],comp.get(op)
        print(f"{op:<22}{mdl:<9}{status:<14}"
              +(f"{t:>8.0f}u{b/t:>9.2f}x" if t else f"{'--':>9}{'--':>10}")
              +(f"{c/t:>8.2f}x" if (t and c) else f"{'--':>9}")
              +(f"{re_:>9.1e}" if re_ is not None else f"{'--':>9}"))
        rows.append(dict(op=op, model=mdl, S=S, status=status,
            kernel_us=round(t,1) if t else None,
            eager_us=round(b,1), compiled_us=round(c,1) if c else None,
            vs_eager=round(b/t,2) if t else None,
            vs_compiled=round(c/t,2) if (t and c) else None,
            relerr=f"{re_:.1e}" if re_ is not None else None,
            synthesized_at_S=2048))
    del q,k,v; torch.cuda.empty_cache()
    print()

with open(f"{HOME}/FINAL_shape_generalisation.csv","w",newline="") as fh:
    ks=sorted({k for r in rows for k in r})
    w=csv.DictWriter(fh,fieldnames=ks); w.writeheader(); w.writerows(rows)

geo=lambda vs: math.exp(sum(math.log(x) for x in vs)/len(vs))
print("="*96); print("SUMMARY"); print("="*96)
for mdl in sorted({m for _,m,_ in KERNELS}):
    print(f"\n{mdl}:")
    for S in (1024,2048,4096):
        sub=[r for r in rows if r["model"]==mdl and r["S"]==S]
        ok=[r for r in sub if r["status"]=="OK"]
        e=[r["vs_eager"] for r in ok if r["vs_eager"]]
        c=[r["vs_compiled"] for r in ok if r["vs_compiled"]]
        print(f"  S={S:<5} {len(ok)}/{len(sub)} correct"
              + (f"   geomean vs eager {geo(e):5.2f}x  vs compiled {geo(c):5.2f}x" if e else ""))
    fails=sorted({(r["op"],r["status"]) for r in rows
                  if r["model"]==mdl and r["S"]!=2048 and r["status"]!="OK"})
    if fails:
        print(f"  shape-specialised (fail at S!=2048): {[f'{o} [{s}]' for o,s in fails]}")
    else:
        print("  all kernels generalise to unseen S")

print("\nPER-OP: does ANY model's kernel generalise to all three S?")
for op in OPS:
    per={}
    for mdl in sorted({m for _,m,_ in KERNELS}):
        st_=[r["status"] for r in rows if r["op"]==op and r["model"]==mdl]
        per[mdl]= "all S" if st_ and all(s=="OK" for s in st_) else \
                  ("partial" if any(s=="OK" for s in st_) else "none")
    ok = any(v=="all S" for v in per.values())
    print(f"  {op:<22}{'YES' if ok else 'NO ':<5} " + "  ".join(f"{m}:{v}" for m,v in per.items()))
print("\n-> ~/FINAL_shape_generalisation.csv")
