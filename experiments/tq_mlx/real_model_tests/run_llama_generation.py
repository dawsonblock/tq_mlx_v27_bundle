import mlx.core as mx
import time
from mlx_lm import load, generate
from _mlx_lm_tq.turboquant.mlx_integration import patch_attention_for_turboquant

def run_real_model_test():
    # Pick a fast, local model for testing (adjust if needed)
    # LLaMA 3 8B or Phi-3 are great. We'll use a tiny model for immediate verification.
    model_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    
    print(f"Loading Model: {model_path}...")
    model, tokenizer = load(model_path)
    
    print("Patching MLX Model with TurboQuant Attention mechanisms...")
    patch_attention_for_turboquant(model, tail_size=128)
    
    prompt = "Explain why optimizing Value cache quantization in large language models is difficult but necessary for long context generation."
    
    print(f"\nPrompt: {prompt}\n")
    print("Generating response with TurboQuant injected KV cache...")
    
    t0 = time.time()
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        verbose=True, 
        max_tokens=256
    )
    t1 = time.time()
    
    print(f"\n[Generation Complete] Setup + Generation Time: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    # Ensure this runs gracefully even if the user hasn't downloaded the weights yet.
    try:
         run_real_model_test()
    except Exception as e:
         print(f"Error during Real Model execution. Note: May need 'huggingface-cli login' or internet access. Details: {e}")
