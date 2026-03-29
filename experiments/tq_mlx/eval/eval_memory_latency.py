"""
eval_memory_latency.py — Memory and latency evaluation for TurboQuant.

Measures:
    - Execution latency of the quantized attention pipeline (NumPy reference).
    - Memory footprint: dense FP32 bytes vs quantized 4-bit code bytes.
    - Compression ratio achieved by codebook quantization.

Usage:
    PYTHONPATH=. python experiments/tq_mlx/eval/eval_memory_latency.py
"""

import numpy as np
import time
from _mlx_lm_tq.turboquant.quantized_attention import quantized_attention


def run(d=128, n=512):
    """Run memory and latency benchmark.

    Args:
        d: Head dimension (default 128).
        n: Sequence length / number of cached tokens (default 512).

    Returns:
        Dict with keys: latency_s, dense_bytes, quant_bytes_est.
    """
    rng = np.random.default_rng(0)
    q = rng.standard_normal(d).astype(np.float32)

    cache = {
        "codes_k": rng.integers(0, 16, size=(n, d), dtype=np.uint8),
        "codebooks_k": np.tile(np.linspace(-1,1,16,dtype=np.float32), (16,1)),
        "codes_v": rng.integers(0, 16, size=(n, d), dtype=np.uint8),
        "codebooks_v": np.tile(np.linspace(-1,1,16,dtype=np.float32), (16,1)),
        "scales_v": np.ones(d, dtype=np.float32),
        "v_outlier_idx": np.arange(max(1, int(d * 0.05))),
        "v_outlier_val": rng.standard_normal((n, max(1, int(d * 0.05))), dtype=np.float32),
        "group_map": np.arange(d) % 16,
        "k_tail": np.zeros((0, d), dtype=np.float32),
        "v_tail": np.zeros((0, d), dtype=np.float32)
    }

    t0 = time.time()
    probs, out = quantized_attention(q, cache)
    t = time.time() - t0

    dense_bytes = n * d * 4
    quant_bytes = n * d * 1  # approx (codes only)

    return {
        "latency_s": t,
        "dense_bytes": dense_bytes,
        "quant_bytes_est": quant_bytes
    }

if __name__ == "__main__":
    print(run())
