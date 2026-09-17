# 🔬 GraphSynth

**Autonomous Optimization of PyTorch Execution Graphs via Search-Guided Compiler Pass & Triton Kernel Synthesis**

> Submitted to the **CHIA Hackathon @ MICRO 2026**
> Authors: **Joshi Penta**, **Priyesh Shukla**  - International Institute of Information Technology Hyderabad

---

## 📖 Overview

Modern deep learning inference is bottlenecked by **memory bandwidth**, not raw compute — yet static compiler backends like PyTorch Inductor and MLIR only optimize a fixed catalog of known operators. Anything outside that catalog silently falls back to slow, unfused execution. Hand-writing an expert Triton or CUDA kernel to fix this takes an engineer **weeks to months per operator**.

**GraphSynth** replaces that bottleneck with a fully autonomous, closed-loop agent that:

1. **Profiles** a PyTorch subgraph to find genuinely memory-bound operators
2. **Synthesizes** a candidate Triton kernel for it using an LLM (Gemini)
3. **Verifies** the kernel numerically against PyTorch's own reference implementation
4. **Profiles it on real GPU hardware** and decides — accept the kernel, or iterate with hardware feedback

No manual tuning. No fixed kernel template library. Just a search loop grounded in real silicon.

---

## 🏗️ How It Works — The Four-Stage Loop

| Stage | What it does |
|---|---|
| **1. IR Profiler** | Traces the PyTorch FX graph, estimates each operator's theoretical arithmetic intensity (AI) from its real shape/dtype, measures real CUDA time via `torch.cuda.Event`, and ranks every operator by a **bottleneck score** = `CUDA_time / theoretical_AI`. |
| **2. Gemini Synthesis** | Sends the highest-ranked operator's shape, dtype, and hardware constraints to Gemini, which writes a candidate Triton kernel — tiling, fusion, and vectorization included. |
| **3. Numerical Verifier** | Checks the kernel against PyTorch's own output across five stress-test input types (`max|K(x) - R(x)| < 1e-5`). Incorrect kernels are rejected immediately — before ever touching real hardware. |
| **4. CHIA Feedback** | Profiles the verified kernel with **NVIDIA Nsight Compute** on real hardware. If it reaches ≥70% of the roofline bound, it's accepted. Otherwise, the real hardware profile is fed back into Stage 2 for the next attempt. |

The key innovation making Stage 1 reliable: a **shape- and type-aware arithmetic intensity estimator**. A naive estimator that costs every primitive uniformly causes FLOPs and bytes to cancel out, collapsing AI to a near-constant value regardless of an operator's true complexity. GraphSynth instead costs each primitive by its *real traced output shape* and its *specific operation type* (transcendental functions like `exp`/`erf`/`tanh` cost more than basic arithmetic, reflecting real GPU Special Function Unit throughput) — so AI values genuinely differentiate operators of different complexity.

---

## 📊 Results

Evaluated end-to-end on a single **NVIDIA Tesla T4 GPU** (Google Colab), across 9 operators:

| Operator | AI (FLOPs/Byte) | Status | Roofline % | Speedup | Iterations |
|---|---:|---|---:|---:|---:|
| `matmul` | 170.7 | ⏭️ Skipped (compute-bound) | — | — | 0 |
| `conv` | 96.0 | ⏭️ Skipped (compute-bound) | — | — | 0 |
| `layernorm` | 0.063 | ✅ Accepted | 89.7% | 1.62× | 1 |
| `relu` | 0.100 | ✅ Accepted | 90.9% | 1.24× | 1 |
| `leaky_relu` | 0.094 | ✅ Accepted | 90.7% | 1.17× | 1 |
| `softmax` | 0.150 | ✅ Accepted | 89.4% | 1.33× | 1 |
| `silu` | 0.182 | ✅ Accepted | 90.8% | 1.23× | 1 |
| `sigmoid` | 0.219 | ✅ Accepted | 89.8% | 1.12× | 1 |
| `gelu` | 0.227 | ✅ Accepted | 90.0% | 1.15× | 1 |

**Highlights:**
- Stage 1 correctly identified `matmul` and `conv` as compute-bound and **skipped synthesis entirely** — cuBLAS/cuDNN are already near-optimal for these, so no GPU time or LLM calls were wasted on them.
- All **7 memory-bound operators converged in a single Gemini iteration**, each landing in the **89–91% roofline** range.
- Full raw logs backing these numbers are in [`logs/stage1-4_run_output.txt`](logs/stage1-4_run_output.txt).

---

## 🚀 Getting Started

### Requirements
- Google Colab with a **T4 GPU** runtime (or any CUDA GPU with Nsight Compute counter access — see note below)
- A free [Gemini API key](https://aistudio.google.com/apikey)
- No local installation needed — the notebook installs everything itself

### Running it

1. Open **`GraphSynth_MultiOp_Colab.ipynb`** in Google Colab
2. **Runtime → Change runtime type → T4 GPU**
3. Run the config cell and edit `OPS_TO_TEST` to whichever operators you want to test — any name that exists in `torch` or `torch.nn.functional` works automatically, no code changes needed:
```python
   OPS_TO_TEST = ["softmax", "relu", "gelu", "layernorm", "sigmoid"]
```
4. Paste your Gemini API key when prompted
5. Run every remaining cell, top to bottom

Expected runtime: **10–20 minutes**, dominated by Gemini API latency and Nsight profiling passes.

---

## 📁 Repository Structure
    CHIA_proj/
├── README.md <- you are here
├── LICENSE
├── GraphSynth_MultiOp_Colab.ipynb <- the full runnable pipeline
├── generated_kernels/ <- real, accepted Triton kernels
│ ├── layernorm_kernel.py
│ ├── relu_kernel.py
│ ├── leaky_relu_kernel.py
│ ├── softmax_kernel.py
│ ├── silu_kernel.py
│ ├── sigmoid_kernel.py
│ └── gelu_kernel.py
└── logs/
└── stage1-4_run_output.txt <- raw console output backing the results table

## 📄 Citation

If you use this work, please cite the accompanying paper submitted to the CHIA Hackathon @ MICRO 2026.

## 📜 License

MIT — see [`LICENSE`](LICENSE).

## 🤖 Acknowledgment of AI Assistance

Portions of this repository's code implementation, debugging, and documentation were assisted by **Claude (Anthropic)**. All experimental results, design decisions, and conclusions were reviewed and verified by the human authors, who take full responsibility for the entire content, correctness, and quality of this artifact.
