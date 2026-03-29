
# Example hook to replace attention in a model (pseudo-MLX)

from mlx_lm.turboquant.quantized_attention import quantized_attention

def attention_forward(q, cache, mask=None):
    probs, out = quantized_attention(q, cache, mask=mask)
    return out
