# GraphSynth

Closed-loop synthesis of Triton kernels for PyTorch operators that Inductor's
fused-kernel catalog does not cover. An LLM writes a candidate kernel, a numerical
gate rejects it if it disagrees with the PyTorch reference, and hardware
measurement decides whether to accept it or feed the failure back into the prompt.

Artifact for *"GraphSynth: Autonomous Optimization of PyTorch Execution Graphs via
Search-Guided Compiler Pass & Triton Kernel Synthesis"*, A3 Workshop @ MICRO 2026.

Every script here is byte-identical to the one that produced the published numbers.
Nothing was cleaned up or rewritten for release.

## Verify the paper without a GPU

```bash
python verify_paper_numbers.py
```

Recomputes all 19 reported statistics from the released CSVs and fails loudly on
any disagreement. Takes about a second and needs only the standard library.

## Headline results

| Experiment | Result |
|---|---|
| 10 attention variants outside the fused catalog | verified kernel for 10/10; geo mean **10.83x** over eager, **3.04x** over `torch.compile` |
| vs hand-tuned FlashAttention | 2 of 10 kernels faster (0.63x and 0.95x of 457 us) |
| vs `flex_attention` | slower where it applies (geo mean 0.55x) but covers 10/10 against its 8/10 |
| 7 single-tensor operators (control) | **1.00x** vs `torch.compile`; every optimized path at 87-88% of peak DRAM bandwidth |
| Shape generalisation | speedups hold or improve at S = 1024 / 2048 / 4096; best case 60.88x |

The control experiment is a negative result and is meant to be. These operators sit
inside Inductor's coverage and both systems hit the same memory wall, so parity is
the correct answer; it is reported to show the measurement method does not
manufacture speedups.

## Requirements

- NVIDIA A100-40GB (sm80). Other GPUs will run but produce different numbers:
  the roofline constants in the scripts are hardcoded to this device
  (`PEAK_BW = 1555 GB/s`).
- PyTorch 2.9.1 + CUDA 12.9. Triton ships with PyTorch; do not install it separately.
  Install from the CUDA wheel index, since the default PyPI wheel may not match:
  ```bash
  pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu129
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```
  Install from the CUDA wheel index, or pip may give you a CPU-only build and every
  script will fail at `.cuda()`:
  ```bash
  pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu129
  ```
- Stage 2 (synthesis) additionally needs Vertex AI access:
  ```bash
  export GCP_PROJECT=<your-gcp-project>
  gcloud auth application-default login
  ```
  Re-benchmarking the included kernels needs **no** API access.

## Reproducing

You need your own machine with an NVIDIA A100-40GB. Nothing here depends on our
infrastructure: the VM these results were produced on no longer exists, and the scripts
talk only to the local GPU (Stage 2 additionally calls Vertex AI, and is optional).

The benchmark scripts locate kernels with `os.path.expanduser("~")`, which resolves to
**your own home directory on the machine you run them on** (`/home/<you>`), not to
anything from our setup. They also write their output CSVs there. So copy the kernel
directories into your home directory once:

```bash
cp -r kernels_* ~/
ls -d ~/kernels_*        # expect four directories
```

`generated_kernels/` is the two-operator prototype run and is not read by any benchmark,
so it does not need copying.

After that you can run the scripts from anywhere inside the clone. Their CSVs will
appear in `~/`, and you can diff them against the published copies in `results/`.

**1. Validate the timer first.** Everything downstream depends on it.

```bash
python timer_fix.py
```

Times a pure device-to-device copy of an 8192x8192 fp32 tensor, which moves exactly
536.9 MB and does no arithmetic. Expect ~389.6 us, i.e. 1376 GB/s or 88.6% of peak.
Anything above 100% means the timer is wrong, and an earlier version of this work
was wrong in exactly that way (see *Known issues*).

**2. Re-benchmark the released kernels** (no LLM calls, ~10 min):

```bash
python final_bench.py    # -> FINAL_single_tensor.csv   (Table 1)
                         #    FINAL_attention.csv       (earlier attention run, not quoted)
python extra_bench.py    # -> FINAL_flexattention.csv   (Table 2, every column)
                         #    FINAL_seqsweep.csv        (sequence-length sweep)
python shape_gen.py      # -> FINAL_shape_generalisation.csv   (Table 3)
```

Compare Table 2 against `FINAL_flexattention.csv` from `extra_bench.py`, not against
`FINAL_attention.csv`. Both measure the same ten operators and agree to within 0.3%, but
the paper quotes the former so that each row's latencies and speedups come from one run.

**3. Re-run synthesis from scratch** (needs Vertex AI; ~30 min; results will differ,
LLM sampling is stochastic):

```bash
GEMINI_MODEL=gemini-2.5-pro        MAX_ITER=3 python synth10.py
GEMINI_MODEL=gemini-3.1-pro-preview MAX_ITER=3 python synth10.py
python single_tensor.py     # 7-operator control
python sigmoid_fix.py       # sigmoid attention with the corrected reference
```

Each run writes to a timestamped `kernels_<model>_<HHMM>/` directory so a second run
cannot overwrite the first.

## Independent reproduction check

The packaged repository was unpacked into a clean directory on the same A100 and
re-run end to end. Two runs of the same measurement are not expected to be bit-identical,
and they are not:

| | Released CSVs | Fresh re-run | difference |
|---|---|---|---|
| single-tensor, geo mean vs eager | 1.08x | 1.08x | - |
| single-tensor, geo mean vs `torch.compile` | 1.00x | 1.00x | - |
| beats `torch.compile` on | 1/7 | 1/7 | - |
| attention, geo mean vs eager | 10.83x | 10.82x | 0.09% |
| attention, geo mean vs `torch.compile` | 3.04x | 3.04x | - |
| beats `torch.compile` on | 10/10 | 10/10 | - |
| copy-bandwidth calibration | 88.6% of peak | 88.5% | 0.1% |
| rows exceeding 100% of peak bandwidth | 0 | 0 | - |

Largest per-operator deviation was `temp_perhead`, 5.11x against 5.08x, i.e. 0.6%.
Treat roughly 0.5% as the run-to-run noise floor for these measurements. The paper
quotes the released CSVs, which `verify_paper_numbers.py` checks.

## Which file backs which table

| Paper | File |
|---|---|
| Table 1 (single-tensor control) | `results/FINAL_single_tensor.csv` |
| Table 2 (attention variants) | `results/FINAL_flexattention.csv` |
| Table 3 (shape generalisation) | `results/FINAL_shape_generalisation.csv` |
| Sec. 4.4 synthesis convergence | `results/logs/log_25pro.txt`, `log_31pro.txt`, `results/results_*.json` |

Every column of Table 2 is taken from `FINAL_flexattention.csv` so that each row is
internally consistent. `FINAL_attention.csv` is an earlier run of the same
measurement, kept for transparency; it differs by under 0.3%.

## Layout

**Pipeline**

| File | Role |
|---|---|
| `synth.py`, `synth10.py` | Stages 1-3: profile, prompt, synthesize, verify |
| `single_tensor.py` | end-to-end run for the 7-operator control |
| `sigmoid_fix.py` | sigmoid attention with the corrected reference |
| `timer_fix.py` | the validated timer and its falsification test |
| `recover.py` | re-imports kernels from a crashed run |

**Benchmarks**

| File | Produces |
|---|---|
| `final_bench.py` | Table 1, plus an earlier attention run |
| `extra_bench.py` | Table 2, every column |
| `shape_gen.py` | Table 3 |
| `bench.py`, `compile_bench.py` | earlier measurement passes, kept for the record |

**Generated kernels**

| Directory | Contents |
|---|---|
| `kernels_gemini31propre_0018/` | gemini-3.1-pro-preview, 10/10 accepted |
| `kernels_gemini25pro_2346/` | gemini-2.5-pro, 9/10 accepted |
| `kernels_sigmoid_fix/` | sigmoid attention, corrected reference |
| `kernels_single_tensor/` | the 7 control operators |
| `generated_kernels/` | 2-operator prototype run |

**Data and paper**

| Directory | Contents |
|---|---|
| `results/` | CSVs, JSONs, synthesis logs |

Kernel files are named `<operator>_iter<N>.py` for each attempt and `<operator>_BEST.py`
for the accepted one. **Failed iterations are kept deliberately** — they are the
evidence for the feedback loop described in Sec. 4.4, and you can read the repair
sequence directly (e.g. `softcap_gemma2_iter1.py` calls `tl.math.tanh`, which does not
exist in this Triton version; `iter2` switches to a removed `tl.dot(trans_b=True)`;
`iter3` compiles and verifies).

## Known issues and honest caveats

- **An earlier version of this work reported inflated speedups.** It divided a
  CUDA-event baseline by an Nsight Compute kernel latency. Those instruments measure
  different intervals — events around a single call absorb host dispatch gaps, the
  Nsight counter does not — so the ratio inflated every speedup by dispatch overhead.
  A trivial kernel measures 31.7 us one way and about 2 us the other. Everything in
  the current paper was re-measured with the single loop-based timer in `timer_fix.py`,
  applied identically to all four paths.
- **One kernel is silently wrong at an unseen shape.** The `gemini-3.1-pro-preview`
  kernel for `bias_prefix_lm` passes the correctness gate at S=1024 and S=2048 but
  returns 23% relative error at S=4096. It does not raise, does not return the wrong
  shape, and does not produce NaN. `verify_paper_numbers.py` prints it. Per-shape
  re-verification is required before deploying any LLM-generated kernel.
- **Four `gemini-2.5-pro` kernels hardcoded the synthesis shape** and raise
  `AssertionError` at any S != 2048.
- **One reported failure was our bug, not the model's.** Sigmoid attention initially
  failed for both backends with an identical ~0.8 relative error. Our reference zeroed
  masked scores *before* the sigmoid, and since sigmoid(0) = 0.5 the masked positions
  received half weight instead of none. `sigmoid_fix.py` has the corrected reference.
  An error identical across models and iterations points at the specification, not the
  generated code.
- **Single trial per cell.** LLM sampling is stochastic and we report no variance on
  convergence or latency.
- **Forward pass only**, one GPU, fixed head dimension and batch size.

## Citing

Please cite the A3 @ MICRO 2026 paper. The manuscript is withheld from this repository
during anonymous review and will be added once decisions are released.
