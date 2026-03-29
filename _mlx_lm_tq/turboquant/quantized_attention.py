"""
quantized_attention.py — Reference NumPy implementation of TurboQuant attention.

This module implements the full two-stage (prefix + tail) quantized KV-cache
attention pipeline using pure NumPy. It serves as the ground-truth reference
for verifying correctness before deploying the MLX-native GPU path.

Pipeline:
    1. Decode prefix K via grouped codebook lookup + PolarQuant scaling + outlier correction
    2. Compute tail K logits via dense matmul
    3. Merge and softmax
    4. Decode prefix V (same codebook + outlier strategy)
    5. Weighted sum → attention output

See Also:
    mlx_integration.py — GPU-accelerated MLX-native equivalent of this module.
"""

import numpy as np
from _mlx_lm_tq.turboquant.prefix_v import decode_prefix_v


def softmax(x, mask=None):
    """Numerically stable softmax with optional boolean mask.

    Args:
        x: 1-D array of logits.
        mask: Optional boolean array. False positions are masked to -inf.

    Returns:
        Probability distribution (same shape as x).
    """
    if mask is not None:
        x = np.where(mask, x, -1e9)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-9)

def prefix_logits(q, codes_k, codebooks_k, group_map, scales_k=None, k_outlier_idx=None, k_outlier_val=None):
    """Compute attention logits from quantized prefix Key cache.

    Decodes 4-bit codes through grouped codebooks, applies per-channel
    PolarQuant scaling, then computes dot products with the query vector.
    Dense outlier dimensions are added back via a separate matmul.

    Args:
        q: Query vector, shape [d].
        codes_k: Quantized key codes, shape [n_prefix, d], dtype uint8.
        codebooks_k: Codebook table, shape [num_groups, 16] (16 codes per group).
        group_map: Per-dimension group assignment, shape [d].
        scales_k: Optional per-channel scale factors, shape [d].
        k_outlier_idx: Indices of outlier dimensions kept in dense FP32.
        k_outlier_val: Dense values for outlier dimensions, shape [n_prefix, num_outliers].

    Returns:
        Logit scores, shape [n_prefix].
    """
    n, d = codes_k.shape
    logits = np.zeros(n, dtype=np.float32)
    for i in range(n):
        acc = 0.0
        for j in range(d):
            g = group_map[j]
            val = codebooks_k[g][codes_k[i, j]]
            if scales_k is not None:
                val *= scales_k[j]
            acc += q[j] * val
        logits[i] = acc

    # PolarQuant dense outlier correction:
    # The outlier dimensions were zeroed during calibration, so the codebook
    # contribution for those channels is ~0. We add back the exact dot product
    # from the dense outlier values: k_outlier_val @ q[k_outlier_idx].
    if k_outlier_idx is not None and k_outlier_val is not None:
        dense_correction = k_outlier_val @ q[k_outlier_idx]
        logits += dense_correction

    return logits

def tail_logits(q, k_tail):
    """Compute attention logits from the dense tail Key cache.

    The tail stores the most recent tokens in full FP32 precision.

    Args:
        q: Query vector, shape [d].
        k_tail: Dense key matrix, shape [n_tail, d].

    Returns:
        Logit scores, shape [n_tail].
    """
    return (k_tail @ q).astype(np.float32)

def quantized_attention(q, cache, mask=None):
    """Full two-stage quantized attention (reference NumPy implementation).

    Executes the complete TurboQuant attention pipeline:
        1. Quantized prefix K → logits (codebook decode + outlier correction)
        2. Dense tail K → logits
        3. Concatenate and softmax
        4. Quantized prefix V → decoded values (codebook decode + outlier restoration)
        5. Dense tail V → values
        6. Weighted sum → final attention output

    Args:
        q: Query vector, shape [d].
        cache: Dictionary containing all KV-cache state:
            - codes_k: uint8 [n_prefix, d] — quantized key codes
            - codebooks_k: float32 [num_groups, 16] — key codebook table
            - scales_k: float32 [d] — per-channel key scales (PolarQuant)
            - k_outlier_idx: int — indices of dense key outlier dimensions
            - k_outlier_val: float32 [n_prefix, num_outliers] — dense key outlier values
            - codes_v: uint8 [n_prefix, d] — quantized value codes
            - codebooks_v: float32 [num_groups, 16] — value codebook table
            - scales_v: float32 [d] — per-channel value scales (PolarQuant)
            - v_outlier_idx: int — indices of dense value outlier dimensions
            - v_outlier_val: float32 [n_prefix, num_outliers] — dense value outlier values
            - group_map: int [d] — per-dimension codebook group assignment
            - k_tail: float32 [n_tail, d] — dense recent keys
            - v_tail: float32 [n_tail, d] — dense recent values
        mask: Optional boolean mask, shape [n_prefix + n_tail].

    Returns:
        Tuple of (probs, output):
            - probs: Attention probability distribution, shape [n_prefix + n_tail].
            - output: Attention output vector, shape [d].
    """
    # --- Stage 1: Compute Logits ---

    # Prefix K (quantized)
    if cache.get("codes_k") is not None:
        p_logits = prefix_logits(
            q, 
            cache["codes_k"], 
            cache["codebooks_k"], 
            cache["group_map"],
            scales_k=cache.get("scales_k"),
            k_outlier_idx=cache.get("k_outlier_idx"),
            k_outlier_val=cache.get("k_outlier_val")
        )
    else:
        p_logits = np.zeros(0, dtype=np.float32)

    # tail K
    if cache.get("k_tail") is not None:
        t_logits = tail_logits(q, cache["k_tail"])
    else:
        t_logits = np.zeros(0, dtype=np.float32)

    logits = np.concatenate([p_logits, t_logits], axis=0)

    if mask is not None:
        assert mask.shape[0] == logits.shape[0]

    probs = softmax(logits, mask=mask)

    # prefix V
    if cache.get("codes_v") is not None:
        v_prefix = decode_prefix_v(
            cache["codes_v"], 
            cache["codebooks_v"], 
            cache["group_map"],
            scales_v=cache.get("scales_v"),
            v_outlier_idx=cache.get("v_outlier_idx"),
            v_outlier_val=cache.get("v_outlier_val")
        )
    else:
        v_prefix = np.zeros((0, q.shape[0]), dtype=np.float32)

    # tail V
    v_tail = cache.get("v_tail", np.zeros((0, q.shape[0]), dtype=np.float32))

    # split probs
    p_p = probs[:v_prefix.shape[0]]
    p_t = probs[v_prefix.shape[0]:]

    out = None
    if v_prefix.shape[0] > 0:
        out = p_p @ v_prefix
    else:
        out = 0.0

    if v_tail.shape[0] > 0:
        out = out + (p_t @ v_tail)

    return probs, out
