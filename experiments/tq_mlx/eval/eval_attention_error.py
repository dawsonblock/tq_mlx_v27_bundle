"""
eval_attention_error.py — Accuracy evaluation: dense vs quantized attention.

Compares the full TurboQuant quantized attention pipeline against a dense
FP32 baseline to measure the numerical error introduced by:
    - 4-bit grouped codebook quantization of K and V
    - PolarQuant outlier calibration (top 5% dense, rest scaled + quantized)

Metrics reported:
    - logit_l1:     Mean absolute error of attention probabilities.
    - out_l2_full:  L2 norm of output error (full pipeline, K + V quantized).
    - out_l2_v_only: L2 norm of output error isolating V quantization
                     (using dense-exact attention weights with quantized V).

Usage:
    PYTHONPATH=. python experiments/tq_mlx/eval/eval_attention_error.py
"""

import numpy as np
from _mlx_lm_tq.turboquant.quantized_attention import quantized_attention
from _mlx_lm_tq.turboquant.prefix_v import decode_prefix_v


def dense_attention(q, k, v):
    """Standard dense FP32 attention (ground truth baseline).

    Args:
        q: Query vector, shape [d].
        k: Key matrix, shape [n, d].
        v: Value matrix, shape [n, d].

    Returns:
        Tuple of (probs, output) — attention weights and output vector.
    """
    logits = k @ q
    ex = np.exp(logits - np.max(logits))
    probs = ex / (np.sum(ex) + 1e-9)
    return probs, probs @ v


def run_eval(d=128, n=256):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(d).astype(np.float32)
    k = rng.standard_normal((n, d), dtype=np.float32)
    v = rng.standard_normal((n, d), dtype=np.float32)

    # Inject realistic LLM-style massive outliers in random K and V channels
    outlier_dims_v = rng.choice(d, size=max(1, int(d * 0.05)), replace=False)
    v[:, outlier_dims_v] *= 50.0  # Massive magnitude
    
    outlier_dims_k = rng.choice(d, size=max(1, int(d * 0.05)), replace=False)
    k[:, outlier_dims_k] *= 50.0

    # PolarQuant Calibration for V (Keep Top 5% outliers dense, scale rest)
    num_outliers_v = max(1, int(d * 0.05))
    v_abs_max = np.max(np.abs(v), axis=0)
    v_outlier_idx = np.argsort(v_abs_max)[-num_outliers_v:]
    
    v_clean = np.copy(v)
    v_clean[:, v_outlier_idx] = 0.0
    scales_v = np.max(np.abs(v_clean), axis=0)
    scales_v[scales_v == 0] = 1.0 # avoid div by zero
    
    v_scaled = v_clean / scales_v
    codes_v = np.clip(np.round((v_scaled + 1.0) * 7.5), 0, 15).astype(np.uint8)

    # PolarQuant Calibration for K (Keep Top 5% outliers dense, scale rest)
    num_outliers_k = max(1, int(d * 0.05))
    k_abs_max = np.max(np.abs(k), axis=0)
    k_outlier_idx = np.argsort(k_abs_max)[-num_outliers_k:]
    
    k_clean = np.copy(k)
    k_clean[:, k_outlier_idx] = 0.0
    scales_k = np.max(np.abs(k_clean), axis=0)
    scales_k[scales_k == 0] = 1.0
    
    k_scaled = k_clean / scales_k
    codes_k = np.clip(np.round((k_scaled + 1.0) * 7.5), 0, 15).astype(np.uint8)

    cache = {
        "codes_k": codes_k,
        "codebooks_k": np.tile(np.linspace(-1,1,16,dtype=np.float32), (16,1)),
        "scales_k": scales_k,
        "k_outlier_idx": k_outlier_idx,
        "k_outlier_val": k[:, k_outlier_idx],
        "codes_v": codes_v,
        "codebooks_v": np.tile(np.linspace(-1,1,16,dtype=np.float32), (16,1)),
        "scales_v": scales_v,
        "v_outlier_idx": v_outlier_idx,
        "v_outlier_val": v[:, v_outlier_idx],
        "group_map": np.arange(d) % 16,
        "k_tail": np.zeros((0, d), dtype=np.float32),
        "v_tail": np.zeros((0, d), dtype=np.float32)
    }

    p_q, o_q = quantized_attention(q, cache)
    p_d, o_d = dense_attention(q, k, v)
    
    # Reconstruct with dense probabilities to isolate V quantization error
    v_q = decode_prefix_v(
        cache["codes_v"], 
        cache["codebooks_v"], 
        cache["group_map"],
        scales_v=cache["scales_v"],
        v_outlier_idx=cache["v_outlier_idx"],
        v_outlier_val=cache["v_outlier_val"]
    )
    o_q_isolated_v = p_d @ v_q

    return {
        "logit_l1": float(np.mean(np.abs(p_q - p_d))),
        "out_l2_full": float(np.linalg.norm(o_q - o_d)),
        "out_l2_v_only": float(np.linalg.norm(o_q_isolated_v - o_d))
    }

if __name__ == "__main__":
    print(run_eval())
