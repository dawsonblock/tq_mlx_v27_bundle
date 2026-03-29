import math
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
try:
    from datasets import load_dataset
except ImportError:
    pass
from _mlx_lm_tq.turboquant.mlx_integration import patch_attention_for_turboquant

def evaluate_perplexity_chunked(model, tokenizer, max_tokens=1024, chunk_size=64):
    print(f"Loading wikitext-2-raw-v1, evaluating {max_tokens} tokens in chunks of {chunk_size}...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"][:200])
    
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
    else:
        tokens = tokenizer(text)["input_ids"]
        if isinstance(tokens[0], list):
            tokens = tokens[0]

    tokens = tokens[:max_tokens]
    
    from _mlx_lm_tq.turboquant.mlx_integration import TurboQuantCache
    
    # We will step through the model chunk by chunk
    cache = model.make_cache()
    losses = []
    
    start_time = time.time()
    
    for i in range(0, len(tokens) - 1, chunk_size):
        end_i = min(i + chunk_size, len(tokens) - 1)
        chunk = tokens[i:end_i]
        targets = tokens[i+1:end_i+1]
        
        x = mx.array(chunk)[None] # [1, L]
        y = mx.array(targets)[None] # [1, L]
        
        logits = model(x, cache=cache)
        mx.eval(logits)
        
        loss = nn.losses.cross_entropy(logits, y)
        losses.append((mx.sum(loss).item(), len(chunk)))
        
        if cache and len(cache) > 0 and isinstance(cache[0], TurboQuantCache):
            pass # Just tracking
        
    end_time = time.time()
    
    total_loss = sum(l[0] for l in losses)
    total_tokens = sum(l[1] for l in losses)
    avg_loss = total_loss / total_tokens
    
    ppl = math.exp(avg_loss)
    print(f"PPL: {ppl:.4f}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Time: {end_time - start_time:.4f} s\n")

if __name__ == "__main__":
    model_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    
    print("=== DENSE BASELINE (Chunked) ===")
    model, tokenizer = load(model_path)
    evaluate_perplexity_chunked(model, tokenizer, max_tokens=2048, chunk_size=256)
    
    print("=== TURBOQUANT (Chunked) ===")
    patch_attention_for_turboquant(model)
    evaluate_perplexity_chunked(model, tokenizer, max_tokens=2048, chunk_size=256)
