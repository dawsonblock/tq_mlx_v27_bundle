"""
prefix_v.py — Decode quantized Value cache with PolarQuant outlier restoration.

Reconstructs the prefix V tensor from 4-bit grouped codebook codes.
Applies per-channel scaling and overwrites structural outlier dimensions
with their original dense FP32 values to preserve attention quality.
"""

import numpy as np


def decode_prefix_v(codes_v, codebooks_v, group_map, scales_v=None, v_outlier_idx=None, v_outlier_val=None):
    """Decode quantized prefix Value cache back to dense representation.

    For each token and dimension, looks up the codebook entry for the
    corresponding group, applies per-channel scaling, then overwrites
    the top outlier dimensions with their exact dense values.

    Args:
        codes_v: Quantized value codes, shape [n_prefix, d], dtype uint8.
        codebooks_v: Codebook table, shape [num_groups, 16].
        group_map: Per-dimension group assignment, shape [d].
        scales_v: Optional per-channel scale factors, shape [d].
        v_outlier_idx: Indices of outlier dimensions kept in dense FP32.
        v_outlier_val: Dense values for outlier dimensions, shape [n_prefix, num_outliers].

    Returns:
        Reconstructed value matrix, shape [n_prefix, d], dtype float32.
    """
    n, d = codes_v.shape
    out = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        for j in range(d):
            g = group_map[j]
            val = codebooks_v[g][codes_v[i, j]]
            if scales_v is not None:
                val *= scales_v[j]
            out[i, j] = val

    # Overwrite outlier channels with high precision values
    if v_outlier_idx is not None and v_outlier_val is not None:
        for k, j in enumerate(v_outlier_idx):
            out[:, j] = v_outlier_val[:, k]

    return out
