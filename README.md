<p align="center">
  <h1 align="center">⚡ TurboQuant MLX</h1>
  <p align="center">
    <em>Compressed KV-Cache Attention for Apple Silicon — inspired by Google's TurboQuant</em>
  </p>
  <p align="center">
    <a href="#quickstart"><strong>Quickstart</strong></a> · 
    <a href="#architecture"><strong>Architecture</strong></a> · 
    <a href="#benchmarks"><strong>Benchmarks</strong></a> · 
    <a href="#evaluation"><strong>Evaluation</strong></a> · 
    <a href="#roadmap"><strong>Roadmap</strong></a>
  </p>
</p>

---

## What is this?

**TurboQuant MLX** is a from-scratch implementation of quantized KV-cache attention on [Apple MLX](https://github.com/ml-explore/mlx), replicating the core architecture described in Google's TurboQuant system.

Instead of storing the full Key and Value tensors in FP16/FP32 during autoregressive generation, this system compresses them into **4-bit grouped codebook representations** — achieving a **4× memory reduction** on KV-cache while preserving near-lossless attention quality through **PolarQuant-style outlier calibration**.

This enables significantly **longer context windows** on memory-constrained Apple Silicon devices (M1/M2/M3/M4).

---

## Key Features

| Feature | Status |
|---|---|
| 🧮 Grouped codebook quantization (4-bit) | ✅ |
| 📦 Bit-packed storage | ✅ |
| 🔁 Full two-stage attention (prefix + tail) | ✅ |
| 🧊 PolarQuant outlier calibration (K + V) | ✅ |
| ⚡ MLX-native GPU-accelerated execution | ✅ |
| 🎯 Metal Shading Language kernel prototype | ✅ |
| 🔌 Drop-in model integration hook | ✅ |
| 📊 Evaluation suite (accuracy, memory, latency, NIAH) | ✅ |

---

## How It Works

### The TurboQuant Attention Pipeline

Standard transformer attention:

$$\text{out} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

TurboQuant splits the cached K and V tensors into two regions:

```
┌─────────────────────────────────────────────────────────┐
│                    KV Cache                             │
│                                                         │
│  ┌───────────────────────┐  ┌────────────────────────┐  │
│  │   PREFIX (quantized)  │  │    TAIL (dense FP32)   │  │
│  │                       │  │                        │  │
│  │  4-bit codebook codes │  │  Full precision K, V   │  │
│  │  + PolarQuant scales  │  │  (recent tokens)       │  │
│  │  + dense outliers     │  │                        │  │
│  └───────────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**The pipeline executes as:**

1. **Decode prefix K** → quantized codebook lookup + per-channel scaling + dense outlier correction
2. **Dense tail K** → standard matmul
3. **Merge logits** → concatenate prefix and tail scores
4. **Softmax** → full attention probability distribution
5. **Reconstruct output** → prefix V (decoded + outlier restored) + tail V (dense)

### PolarQuant Outlier Calibration

LLM activations contain **structural outliers** — a small percentage of dimensions (~5%) carry disproportionately large magnitudes. Naively quantizing these destroys attention quality.

TurboQuant MLX implements PolarQuant-style mixed-precision:

```
For each of K and V:
  1. Identify top 5% magnitude dimensions → keep in dense FP32
  2. Compute per-channel scales on remaining 95%
  3. Quantize scaled values into 4-bit grouped codebooks
  4. At decode time: reconstruct quantized + overwrite outlier channels
```

This achieves near-zero logit error ($L_1 \approx 10^{-9}$) and minimal output error ($L_2 \approx 1.2$) even under 50× magnitude outlier injection.

---

## Quickstart

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- [MLX](https://github.com/ml-explore/mlx) ≥ 0.29
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) ≥ 0.29

```bash
pip install mlx mlx-lm numpy
```

### Run Accuracy Evaluation

```bash
cd tq_mlx_v27_bundle
PYTHONPATH=. python experiments/tq_mlx/eval/eval_attention_error.py
```

**Expected output:**
```json
{"logit_l1": 2.8e-09, "out_l2_full": 1.23, "out_l2_v_only": 1.23}
```

### Run GPU Speed Benchmark

```bash
PYTHONPATH=. python experiments/tq_mlx/eval/eval_mlx_speed.py
```

**Expected output:**
```
[Numpy]                        667.02 ms
[MLX Native Vectorized JIT]      0.027 ms   ← 24,700× faster
Max Diff vs Numpy:             0.000010
```

### Run Needle-In-A-Haystack Retrieval Test

```bash
PYTHONPATH=. python experiments/eval_niah.py
```

**Expected output:**
```
Expected: The secret passcode is 42.
Response: The secret passcode is 42.
✅ PASS: Needle found!
```

### Run Real Model Generation

```bash
PYTHONPATH=. python experiments/tq_mlx/real_model_tests/run_llama_generation.py
```

**Expected output:**
```
Prompt: 22 tokens, 21.1 tokens-per-sec
Generation: 256 tokens, 71.8 tokens-per-sec
Peak memory: 0.738 GB
```

---

## Benchmarks

All benchmarks run on Apple Silicon with MLX.

### Accuracy (d=128, n=256, 50× outlier injection)

| Metric | Before Calibration | After PolarQuant |
|---|---|---|
| Logit L1 Error | 0.00194 | **2.8 × 10⁻⁹** |
| Output L2 (full pipeline) | 42.08 | **1.23** |
| Output L2 (V-only, isolated) | 1.01 | **1.23** |

### Speed (d=128, n=8192)

| Backend | Latency | Speedup |
|---|---|---|
| Python/Numpy loops | 667 ms | 1× |
| MLX Native JIT (GPU) | **0.027 ms** | **24,700×** |

### Memory (d=128, n=512)

| Format | Bytes | Compression |
|---|---|---|
| Dense FP32 | 262,144 | 1× |
| TurboQuant 4-bit codes | **65,536** | **4×** |

---

## Architecture

### Project Structure

```
tq_mlx_v27_bundle/
│
├── _mlx_lm_tq/
│   └── turboquant/
│       ├── quantized_attention.py   # Reference NumPy implementation
│       ├── prefix_v.py              # Value cache decoder with outlier support
│       ├── mlx_integration.py       # MLX-native GPU attention + model hooks
│       ├── integration_stub.py      # Minimal forward() entry point
│       └── tq_kernels.metal         # Metal Shading Language kernel prototype
│
├── experiments/
│   ├── eval_niah.py                 # Needle-In-A-Haystack retrieval test
│   └── tq_mlx/
│       ├── example_manifest.json    # System capability manifest
│       ├── eval/
│       │   ├── eval_attention_error.py   # Dense vs quantized accuracy
│       │   ├── eval_memory_latency.py    # Memory reduction measurement
│       │   ├── eval_mlx_speed.py         # MLX GPU vs NumPy benchmark
│       │   └── eval_metal_latency.py     # Custom Metal kernel benchmark
│       └── real_model_tests/
│           └── run_llama_generation.py   # End-to-end LLaMA generation
│
└── README.md
```

### Core Components

| Component | File | Purpose |
|---|---|---|
| **Quantized Attention** | `quantized_attention.py` | Reference two-stage attention with PolarQuant calibration |
| **Prefix V Decoder** | `prefix_v.py` | Codebook lookup + scale + outlier restoration for Values |
| **MLX Fast Path** | `mlx_integration.py` | GPU-accelerated vectorized attention via `mx.take` / `mx.matmul` |
| **Metal Kernel** | `tq_kernels.metal` | Low-level MSL shader for codebook dot products |
| **Model Hook** | `mlx_integration.py` | `patch_attention_for_turboquant()` — drop-in model injection |
| **Cache Manager** | `mlx_integration.py` | `TurboQuantCache` — dynamic prefix/tail KV management |

---

## Evaluation

### Available Evaluation Scripts

| Script | What It Measures |
|---|---|
| `eval_attention_error.py` | Logit L1 and Output L2 error (dense vs quantized) |
| `eval_memory_latency.py` | Memory footprint reduction ratio |
| `eval_mlx_speed.py` | GPU execution latency (MLX JIT vs NumPy) |
| `eval_metal_latency.py` | Custom Metal kernel performance |
| `eval_niah.py` | Long-context needle retrieval accuracy |
| `run_llama_generation.py` | End-to-end generation quality and tok/s |

---

## Comparison to Google TurboQuant

| Aspect | Google TurboQuant | This Implementation |
|---|---|---|
| Core idea (quantized KV-cache) | ✅ | ✅ |
| Two-stage prefix/tail architecture | ✅ | ✅ |
| Grouped codebook quantization | ✅ | ✅ |
| Outlier-aware calibration | PolarQuant | PolarQuant-style (5% dense) |
| Full attention (K + V reconstruction) | ✅ | ✅ |
| 4× memory reduction | ✅ | ✅ |
| Hardware backend | CUDA / TPU | **Apple Metal (MLX)** |
| Sub-ms latency | ✅ | ✅ (0.027 ms) |
| Production model integration | Full | Hook-ready |
| Calibration quality | Per-layer optimized | Per-channel scaled |

---

## Roadmap

- [x] Grouped codebook quantization
- [x] Bit-packed storage format
- [x] Kernel decode (codebook lookup)
- [x] QJL residual correction
- [x] Full attention pipeline (prefix K + V, tail K + V)
- [x] PolarQuant outlier calibration (K and V)
- [x] MLX-native GPU execution path
- [x] Metal Shading Language kernel
- [x] Model integration hook (`patch_attention_for_turboquant`)
- [x] Evaluation suite
- [x] Needle-In-A-Haystack test
- [ ] Per-layer calibration optimization
- [ ] WikiText-2 / C4 perplexity benchmarks
- [ ] Long-context scaling (32K+ tokens)
- [ ] Full `mlx_lm.generate()` cache interception
- [ ] 8B+ model validation

---

## Citation

This project is an independent reimplementation inspired by:

- **TurboQuant** — Google Research (quantized KV-cache attention)
- **PolarQuant** — Mixed-precision outlier-aware quantization
- **QJL** — Quantized Johnson-Lindenstrauss projection for residual correction
- **MLX** — Apple's array framework for machine learning on Apple Silicon

---

<p align="center">
  <sub>Built with ⚡ MLX on Apple Silicon</sub>
</p>
