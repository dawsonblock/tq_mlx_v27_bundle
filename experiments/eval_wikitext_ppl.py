import math
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    exit(1)

from _mlx_lm_tq.turboquant.mlx_integration import patch_attention_for_turboquant

def evaluate_perplexity(model, tokenizer, max_tokens=1024):
    print("Loading wikitext-2-raw-v1...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    text = "\n\n".join(ds["text"][:200])
    
    # Simple tokenization
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
    else:
        tokens = tokenizer(text)["input_ids"]
        if isinstance(tokens[0], list):
            tokens = tokens[0]

    tokens = tokens[:max_tokens]
    print(f"Evaluating over {len(tokens)} tokens.")
    
    if len(tokens) < 2:
        return

    # In a full-sequence evaluation, the entire sequence is put in at once.
    # To test KV-cache effectively, we step through token by token to build cache.
    # Or, with chunking in turboquant_sdpa, passing it all also works!
    
    # We do a single prefill pass here since we patched attention.
    # TurboQuant hooks directly into the MLX SDPA calls.
    x = mx.array(tokens[:-1])[None]  # shape [1, L]
    targets = mx.array(tokens[1:])[None]  # shape [1, L]
    
    start_time = time.time()
    logits = model(x)
    mx.eval(logits)
    end_time = time.time()
    
    # Calculate Cross Entropy Loss
    loss_fn = nn.losses.cross_entropy
    loss = loss_fn(logits, targets)
    avg_loss = mx.mean(loss).item()
    
    ppl = math.exp(avg_loss)
    
    print(f"PPL: {ppl:.4f}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Eval Time: {end_time - start_time:.4f} s")

if __name__ == "__main__":
    model_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    
    print("=== DENSE BASELINE ===")
    model, tokenizer = load(model_path)
    evaluate_perplexity(model, tokenizer, max_tokens=1024)
    
    print("\n=== TURBOQUANT ===")
    # Patch the loaded model
    patch_attention_for_turboquant(model)
    evaluate_perplexity(model, tokenizer, max_tokens=1024)
