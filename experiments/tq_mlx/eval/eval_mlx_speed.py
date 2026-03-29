"""
eval_mlx_speed.py — GPU speed benchmark: MLX vectorized JIT vs NumPy.

Benchmarks the core prefix_logits operation (codebook gather + dot product)
using two backends:
    1. Pure NumPy Python loops (reference baseline)
    2. MLX-native vectorized ops compiled via mx.compile() to Apple Metal GPU

Also verifies numerical equivalence between the two implementations.

Typical results on Apple Silicon:
    [Numpy]                        ~667 ms
    [MLX Native Vectorized JIT]    ~0.027 ms   (24,700× speedup)
    Max Diff vs Numpy:             ~0.000010

Usage:
    PYTHONPATH=. python experiments/tq_mlx/eval/eval_mlx_speed.py
"""

import mlx.core as mx
import time
import numpy as np


def run_mlx_vectorized_benchmark(n=8192, d=128):
    """Run the MLX vs NumPy speed benchmark.

    Args:
        n: Sequence length / number of cached tokens (default 8192).
        d: Head dimension (default 128).
    """
    print(f"Benchmarking MLX Vectorized vs Numpy (N={n}, D={d})")
    
    # 1. Setup Data in Numpy
    rng = np.random.default_rng(0)
    q_np = rng.standard_normal(d).astype(np.float32)
    codes_k_np = rng.integers(0, 16, size=(n, d), dtype=np.uint8)
    codebooks_k_np = np.tile(np.linspace(-1,1,16,dtype=np.float32), (16,1))
    group_map_np = (np.arange(d) % 16).astype(np.int32)
    scales_k_np = rng.uniform(0.5, 1.5, size=d).astype(np.float32)

    # 2. Setup Data in MLX
    q_mx = mx.array(q_np)
    codes_k_mx = mx.array(codes_k_np)
    codebooks_k_mx = mx.array(codebooks_k_np)
    group_map_mx = mx.array(group_map_np)
    scales_k_mx = mx.array(scales_k_np)
    
    # Evaluate variables onto GPU
    mx.eval(q_mx, codes_k_mx, codebooks_k_mx, group_map_mx, scales_k_mx)

    def numpy_baseline():
        logits = np.zeros(n, dtype=np.float32)
        for i in range(n):
            acc = 0.0
            for j in range(d):
                g = group_map_np[j]
                val = codebooks_k_np[g][codes_k_np[i, j]]
                val *= scales_k_np[j]
                acc += q_np[j] * val
            logits[i] = acc
        return logits
    
    # Fast Native MLX implementation
    def mlx_vectorized():
        # group_map: [d] -> which group for each channel
        # codes_k_mx: [n, d]
        # codebooks_k_mx: [16, 16] - [group, code]
        
        # We need to map [n, d] codes through the codebooks.
        # Since group_map tells us the group for each of the `d` dimensions,
        # we can build a flat index or vectorized gather.
        
        # groups shape [d], broadcast to [n, d] implicitly during gather?
        # Actually, mx.take or advanced indexing.
        
        # Let's write the gather using MLX slicing:
        # codebooks_k_mx is [G, 16] -> flatten to [G*16]
        flat_codebooks = mx.reshape(codebooks_k_mx, (-1,))
        
        # Construct flat indices for each dimension
        # group_map_mx is [d] ranging 0..15. We multiply by 16.
        # group_offsets is [1, d]
        group_offsets = mx.expand_dims(group_map_mx * 16, 0)
        
        # The codes have values 0..15.
        flat_indices = codes_k_mx + group_offsets # shape [n, d]
        
        # Gather all decoded values
        decoded_k = mx.take(flat_codebooks, flat_indices) # shape [n, d]
        
        # Apply scales
        # decoded_k: [n, d], scales_k_mx: [d]
        scaled_k = decoded_k * scales_k_mx
        
        # Dot product with Q
        # Q is [d], scaled_k is [n, d] -> out [n]
        logits = mx.matmul(scaled_k, q_mx)
        return logits

    # JIT Compile the MLX function for max speed
    mlx_fast = mx.compile(mlx_vectorized)
    
    # Warmup
    _ = mlx_fast()
    mx.eval(_)
    
    # Benchmark Numpy
    t0 = time.time()
    logits_target = numpy_baseline()
    t1 = time.time()
    print(f"[Numpy] Latency: {(t1-t0)*1000:.2f} ms")

    # Benchmark MLX GPU
    num_runs = 100
    t0 = time.time()
    for _ in range(num_runs):
        out_mx = mlx_fast()
    mx.eval(out_mx) # wait for GPU queue
    t1 = time.time()
    
    print(f"[MLX Native Vectorized JIT] Average Latency: {((t1-t0)/num_runs)*1000:.3f} ms")
    
    # Verify correctness
    out_np = np.array(out_mx)
    print(f"Max Diff vs Numpy: {np.max(np.abs(out_np - logits_target)):.6f}")

if __name__ == '__main__':
    run_mlx_vectorized_benchmark()
