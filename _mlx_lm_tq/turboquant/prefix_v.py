
import numpy as np

def decode_prefix_v(codes_v, codebooks_v, group_map, scales_v=None, v_outlier_idx=None, v_outlier_val=None):
    # codes_v: [n_prefix, d_v]
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
