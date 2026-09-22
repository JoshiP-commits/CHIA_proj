import glob, math, importlib.util, torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
DT, dev = torch.bfloat16, "cuda"
B,H,S,D = 2,32,2048,128
SOFTCAP, WINDOW = 30.0, 256

def dtime(e):
    for a in ("device_time_total","cuda_time_total","device_time","cuda_time"):
        v = getattr(e,a,None)
        if v: return float(v)
    return 0.0
def gpu_us(fn):
    fn(); torch.cuda.synchronize()
    e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); fn(); e1.record(); torch.cuda.synchronize()
    ev = e0.elapsed_time(e1)
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(10): fn()
        torch.cuda.synchronize()
    evs=[e for e in pr.key_averages() if dtime(e)>0]
    t=sum(dtime(e) for e in evs)/10
    return t if t>0 else ev*1000

def ref_softcap(q,k,v):
    s=(q@k.transpose(-2,-1))/math.sqrt(D); s=torch.tanh(s/SOFTCAP)*SOFTCAP
    i=torch.arange(S,device=dev).view(-1,1); j=torch.arange(S,device=dev).view(1,-1)
    return torch.softmax(s.masked_fill(j>i,float('-inf')).float(),-1).to(DT)@v
def ref_window(q,k,v):
    s=(q@k.transpose(-2,-1))/math.sqrt(D)
    i=torch.arange(S,device=dev).view(-1,1); j=torch.arange(S,device=dev).view(1,-1)
    ok=(j<=i)&((i-j)<WINDOW)
    return torch.softmax(s.masked_fill(~ok,float('-inf')).float(),-1).to(DT)@v

q=torch.randn(B,H,S,D,dtype=DT,device=dev)
k=torch.randn(B,H,S,D,dtype=DT,device=dev)
v=torch.randn(B,H,S,D,dtype=DT,device=dev)
flash=gpu_us(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True))
base={"softcap_gemma2":gpu_us(lambda: ref_softcap(q,k,v)),
      "bias_sliding_window":gpu_us(lambda: ref_window(q,k,v))}
refs={"softcap_gemma2":ref_softcap,"bias_sliding_window":ref_window}
print(f"flash {flash:.0f}us  softcap base {base['softcap_gemma2']:.0f}us  window base {base['bias_sliding_window']:.0f}us\n")

for f in sorted(glob.glob("/home/devstar7724/generated_kernels/*.py")):
    op = "softcap_gemma2" if "softcap" in f else "bias_sliding_window"
    try:
        sp=importlib.util.spec_from_file_location("m"+str(abs(hash(f))),f)
        m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
        out=m.launch_kernel(q,k,v); r=refs[op](q,k,v)
        if torch.isnan(out).any(): print(f"{f.split('/')[-1]:<34} NaN"); continue
        rel=((r-out).abs().max()/(r.abs().max()+1e-6)).item()
        if rel<3e-2:
            us=gpu_us(lambda: m.launch_kernel(q,k,v))
            print(f"{f.split('/')[-1]:<34} PASS rel={rel:.2e}  {us:7.0f}us  "
                  f"speedup {base[op]/us:5.2f}x  ({us/flash:.2f}x flash)")
        else:
            print(f"{f.split('/')[-1]:<34} fail rel={rel:.2e}")
    except Exception as e:
        print(f"{f.split('/')[-1]:<34} err {type(e).__name__}")
