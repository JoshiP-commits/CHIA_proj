#!/usr/bin/env python3
"""Recompute every number in the paper from the released CSVs.

Run:  python verify_paper_numbers.py
Needs no GPU. Exits non-zero if any value disagrees with the paper.
"""
import csv, math, sys, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
rd  = lambda f: list(csv.DictReader(open(os.path.join(D, f))))

fails = []
def chk(label, got, want):
    ok = abs(got - want) < 5e-3
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:<42} {got:8.2f}   paper: {want}")
    if not ok:
        fails.append(label)

# ---------------------------------------------------------------- Table 1
S = rd("FINAL_single_tensor.csv")
print("\nTable 1 -- single-tensor control (FINAL_single_tensor.csv)")
chk("GraphSynth geo mean vs eager",
    geo([float(r["eager_us"]) / float(r["graphsynth_us"]) for r in S]), 1.08)
chk("GraphSynth geo mean vs torch.compile",
    geo([float(r["compiled_us"]) / float(r["graphsynth_us"]) for r in S]), 1.00)
chk("torch.compile geo mean vs eager",
    geo([float(r["eager_us"]) / float(r["compiled_us"]) for r in S]), 1.08)
bw = [float(r[k]) for r in S for k in ("eager_bw", "compiled_bw", "graphsynth_bw")]
print(f"  info  bandwidth spread {min(bw)}% .. {max(bw)}%   "
      f"(paper: 86.8-88.2 excluding eager softmax 75.4 / layernorm 58.4)")

# ---------------------------------------------------------------- Table 2
# Every Table 2 column comes from this one file, so each row is self-consistent.
A = rd("FINAL_flexattention.csv")
se = [float(r["eager_us"])    / float(r["graphsynth_us"]) for r in A]
sc = [float(r["compiled_us"]) / float(r["graphsynth_us"]) for r in A]
sf = [float(r["flex_us"])     / float(r["graphsynth_us"]) for r in A if r["flex_us"]]
ind= [float(r["eager_us"])    / float(r["compiled_us"])   for r in A]
print("\nTable 2 -- attention variants (FINAL_flexattention.csv)")
chk("GraphSynth geo mean vs eager",          geo(se), 10.83)
chk("GraphSynth geo mean vs torch.compile",  geo(sc),  3.04)
chk("GraphSynth geo mean vs flex_attention", geo(sf),  0.55)
chk("flex_attention faster by",            1/geo(sf),  1.81)
chk("torch.compile geo mean vs eager",      geo(ind),  3.56)
chk("torch.compile min vs eager",           min(ind),  3.18)
chk("torch.compile max vs eager",           max(ind),  4.29)
FLASH = 456.8
for op, want in (("bias_block_diagonal", 0.63), ("bias_sliding_window", 0.95)):
    g = float(next(r for r in A if r["op"] == op)["graphsynth_us"])
    chk(f"{op} vs FlashAttention", g / FLASH, want)
print(f"  info  flex_attention cannot express: "
      f"{[r['op'] for r in A if not r['flex_us']]}")

# ---------------------------------------------------------------- Table 3
R = rd("FINAL_shape_generalisation.csv")
print("\nTable 3 -- shape generalisation (FINAL_shape_generalisation.csv)")
want = {("2.5-pro","1024"):(5,9,5.71), ("2.5-pro","2048"):(9,9,6.77),
        ("2.5-pro","4096"):(5,9,7.42), ("3.1-pro","1024"):(10,10,4.79),
        ("3.1-pro","2048"):(10,10,6.60),("3.1-pro","4096"):(9,10,8.85)}
for (m, s), (n_ok, n_tot, g) in want.items():
    sub = [r for r in R if r["model"] == m and r["S"] == s]
    ok  = [r for r in sub if r["status"] == "OK"]
    assert (len(ok), len(sub)) == (n_ok, n_tot), f"{m} S={s}: {len(ok)}/{len(sub)}"
    chk(f"{m} S={s} geo mean vs eager ({n_ok}/{n_tot})",
        geo([float(r["vs_eager"]) for r in ok]), g)
b = max((r for r in R if r["status"] == "OK"), key=lambda r: float(r["vs_eager"]))
chk("best case: block-diagonal at S=4096", float(b["vs_eager"]), 60.88)
print(f"  info  that row: eager {b['eager_us']} us -> kernel {b['kernel_us']} us")

# the silent-correctness hazard reported in Section 4.5
bad = [r for r in R if r["status"] == "incorrect"]
print(f"\n  info  silently incorrect kernels: "
      f"{[(r['model'], r['op'], r['S'], r['relerr']) for r in bad]}")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
