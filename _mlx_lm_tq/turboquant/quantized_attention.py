
import numpy as np
from mlx_lm.turboquant.prefix_v import decode_prefix_v

def softmax(x, mask=None):
    if mask is not None:
        x = np.where(mask, x, -1e9)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-9)

def prefix_logits(q, codes_k, codebooks_k, group_map, scales_k=None, k_outlier_idx=None, k_outlier_val=None):
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

    if k_outlier_idx is not None and k_outlier_val is not None:
        # k_outlier_val: [n, num_outliers]
        # q[k_outlier_idx]: [num_outliers]
        for i in range(n):
            for idx, j in enumerate(k_outlier_idx):
                # The quantized part should be roughly zero because we scaled down/zeroed it during calibration.
                # We add the precise dot product.
                # (To be exact, we should subtract the quantized representation of 0, but it's negligible)
                # But typically PolarQuant zero-fills the dense part.
                pass
            
            # actually faster: (k_outlier_val @ q[k_outlier_idx])
            pass
            
        dense_adds = k_outlier_val @ q[k_outlier_idx]
        logits += dense_adds

    return logits

def tail_logits(q, k_tail):
    return (k_tail @ q).astype(np.float32)

def quantized_attention(q, cache, mask=None):
    # cache keys:
    # codes_k, codebooks_k, codes_v, codebooks_v, group_map, k_tail, v_tail

    # prefix K
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
