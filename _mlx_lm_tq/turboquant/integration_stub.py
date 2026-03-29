"""
integration_stub.py — Minimal attention forward hook for TurboQuant.

Provides a simple drop-in function signature that can replace a model's
attention forward pass. This stub uses the reference NumPy implementation;
for production use, see mlx_integration.py which runs on Apple Metal GPU.

Usage:
    Replace a model layer's attention call with:
        output = attention_forward(query, kv_cache, mask=causal_mask)
"""

from _mlx_lm_tq.turboquant.quantized_attention import quantized_attention


def attention_forward(q, cache, mask=None):
    """Compute quantized attention output for a single query vector.

    Args:
        q: Query vector, shape [d].
        cache: TurboQuant KV-cache dictionary (see quantized_attention() for schema).
        mask: Optional boolean attention mask.

    Returns:
        Attention output vector, shape [d].
    """
    probs, out = quantized_attention(q, cache, mask=mask)
    return out
