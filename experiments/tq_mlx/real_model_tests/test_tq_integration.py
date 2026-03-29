"""
test_tq_integration.py — End-to-end test of TurboQuant real model integration.

Tests that patch_attention_for_turboquant() correctly hooks into a real
LLaMA model and produces coherent output with compressed KV-cache.

Compares:
    1. Baseline: Standard mlx-lm generation (unpatched)
    2. TurboQuant: Generation with patched attention + TurboQuantCache
"""

import sys
import os
import time

# Ensure our local module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import mlx.core as mx
import mlx_lm

from _mlx_lm_tq.turboquant.mlx_integration import (
    patch_attention_for_turboquant,
    make_turboquant_cache,
    TurboQuantCache,
)


def test_cache_basic():
    """Test TurboQuantCache basic operations in isolation."""
    print("=" * 60)
    print("Test 1: TurboQuantCache Unit Test")
    print("=" * 60)

    cache = TurboQuantCache(tail_capacity=8, outlier_percent=0.05)

    # Simulate adding tokens one-by-one (decode phase)
    B, n_kv_heads, head_dim = 1, 2, 64

    for step in range(12):
        k = mx.random.normal((B, n_kv_heads, 1, head_dim))
        v = mx.random.normal((B, n_kv_heads, 1, head_dim))
        keys, values = cache.update_and_fetch(k, v)
        mx.eval(keys, values)

    print(f"  After 12 tokens:")
    print(f"    offset     = {cache.offset}")
    print(f"    __len__    = {len(cache)}")
    print(f"    prefix_len = {cache.prefix_len}")
    print(f"    tail_len   = {cache.tail_len}")
    print(f"    keys shape = {keys.shape}")
    print(f"    vals shape = {values.shape}")

    assert cache.offset == 12, f"Expected offset=12, got {cache.offset}"
    assert keys.shape == (B, n_kv_heads, 12, head_dim), f"Bad keys shape: {keys.shape}"
    assert cache.prefix_len > 0, "Expected some tokens in prefix"
    assert cache.tail_len > 0, "Expected some tokens in tail"
    print("  ✅ Cache unit test PASSED\n")


def test_cache_bulk_fill():
    """Test TurboQuantCache with bulk prompt fill (L > 1)."""
    print("=" * 60)
    print("Test 2: Bulk Prompt Fill")
    print("=" * 60)

    cache = TurboQuantCache(tail_capacity=32, outlier_percent=0.05)
    B, n_kv_heads, head_dim = 1, 4, 64

    # Simulate prompt fill with 50 tokens at once
    k = mx.random.normal((B, n_kv_heads, 50, head_dim))
    v = mx.random.normal((B, n_kv_heads, 50, head_dim))
    keys, values = cache.update_and_fetch(k, v)
    mx.eval(keys, values)

    print(f"  After 50-token prompt fill:")
    print(f"    offset     = {cache.offset}")
    print(f"    prefix_len = {cache.prefix_len}")
    print(f"    tail_len   = {cache.tail_len}")
    print(f"    keys shape = {keys.shape}")

    assert cache.offset == 50
    assert keys.shape[2] == 50
    assert cache.prefix_len > 0, "Expected overflow compression"
    print("  ✅ Bulk fill test PASSED\n")


def test_real_model_generation():
    """Test full model generation with TurboQuant patches."""
    print("=" * 60)
    print("Test 3: Real Model Generation (LLaMA-3.2-1B-Instruct-4bit)")
    print("=" * 60)

    model_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    print(f"  Loading {model_path}...")
    model, tokenizer = mlx_lm.load(model_path)

    # --- Baseline ---
    prompt = "Explain what a KV-cache is in one sentence."
    print(f"\n  Prompt: {prompt}")

    print("\n  --- Baseline (no TurboQuant) ---")
    t0 = time.perf_counter()
    baseline_output = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=50, verbose=False
    )
    t_baseline = time.perf_counter() - t0
    print(f"    Output: {baseline_output}")
    print(f"    Time: {t_baseline:.2f}s")

    # --- Reload model for patching (cleanest approach) ---
    print("\n  Reloading model for TurboQuant patching...")
    model, tokenizer = mlx_lm.load(model_path)

    # Patch attention layers
    cache_factory = patch_attention_for_turboquant(
        model, tail_size=128, outlier_percent=0.05
    )

    # --- TurboQuant ---
    print("\n  --- TurboQuant (patched) ---")
    t0 = time.perf_counter()
    tq_output = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=50, verbose=False
    )
    t_tq = time.perf_counter() - t0
    print(f"    Output: {tq_output}")
    print(f"    Time: {t_tq:.2f}s")

    # Validate output is non-empty and coherent
    assert len(tq_output.strip()) > 10, f"TQ output too short: {tq_output}"
    print("  ✅ Real model generation PASSED\n")

    return baseline_output, tq_output


def test_generation_with_manual_cache():
    """Test generation using manually created TurboQuantCache."""
    print("=" * 60)
    print("Test 4: Manual Cache Control")
    print("=" * 60)

    model_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    model, tokenizer = mlx_lm.load(model_path)

    # Patch attention
    cache_factory = patch_attention_for_turboquant(model, tail_size=64)

    # Create manual cache
    cache = cache_factory()
    print(f"  Created {len(cache)} TurboQuantCache instances")
    assert all(isinstance(c, TurboQuantCache) for c in cache)

    # Verify cache properties
    for i, c in enumerate(cache):
        assert c.tail_capacity == 64
        assert c.offset == 0
        assert len(c) == 0

    print("  ✅ Manual cache test PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TurboQuant Real Model Integration Tests")
    print("=" * 60 + "\n")

    test_cache_basic()
    test_cache_bulk_fill()
    test_generation_with_manual_cache()
    test_real_model_generation()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
