from mlx_lm import load, generate

def generate_haystack(context_length, needle, insert_depth):
    """Generates dummy context with the needle inserted at a specific depth."""
    filler = "The quick brown fox jumps over the lazy dog. "
    num_filler_tokens = max(1, context_length // 10)
    
    haystack_parts = [filler] * num_filler_tokens
    insert_idx = int(len(haystack_parts) * insert_depth)
    
    haystack_parts.insert(insert_idx, f" [IMPORTANT INFO: {needle}] ")
    return "".join(haystack_parts)

def eval_niah(model_path, context_length, depth):
    # Load model
    model, tokenizer = load(model_path)
    patch_attention_for_turboquant(model)
    
    needle = "The secret passcode is 42."
    question = "Based on the text above, what is the secret passcode?"
    
    print(f"Generating haystack of ~{context_length} tokens with needle at {depth*100}% depth...")
    context = generate_haystack(context_length, needle, depth)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{context}\n\n{question}"}
    ]
    
    # Apply the model's specific chat formatting
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{context}\n\n{question}\nAnswer:"
    
    print("Running generation...")
    response = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
    
    print("\n--- Result ---")
    print(f"Expected: {needle}")
    print(f"Response: {response.strip()}")
    
    if "42" in response:
        print("✅ PASS: Needle found!")
    else:
        print("❌ FAIL: Needle lost in quantized cache.")

if __name__ == "__main__":
    from _mlx_lm_tq.turboquant.mlx_integration import patch_attention_for_turboquant
    
    model_path = "mlx-community/Llama-3.2-1B-Instruct-4bit" # Ensure you use a valid model name/path
    context_length = 512
    depth = 0.5 # Depth must be between 0.0 and 1.0 (50% depth)
    eval_niah(model_path, context_length, depth)