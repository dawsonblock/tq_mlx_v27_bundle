import mlx.core as mx
import time
import numpy as np

# A small prototype script showing how to bind the Metal kernel in MLX
# and time it vs. the pure numpy implementation.

def run_metal_benchmark(n=8192, d=128):
    print(f"Benchmarking Quantized Prefix Logits (N={n}, D={d})")
    
    # 1. Setup Data in Numpy (for verification)
    rng = np.random.default_rng(0)
    q_np = rng.standard_normal(d).astype(np.float32)
    codes_k_np = rng.integers(0, 16, size=(n, d), dtype=np.uint8)
    codebooks_k_np = np.tile(np.linspace(-1,1,16,dtype=np.float32), (16,1))
    group_map_np = (np.arange(d) % 16).astype(np.uint32)
    scales_k_np = rng.uniform(0.5, 1.5, size=d).astype(np.float32)

    # 2. Pure Numpy Baseline (what we used in eval.py)
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
    
    t0 = time.time()
    logits_target = numpy_baseline()
    t1 = time.time()
    print(f"[Numpy] Latency: {(t1-t0)*1000:.2f} ms")

    # 3. Setup Data in MLX
    q_mx = mx.array(q_np)
    codes_k_mx = mx.array(codes_k_np)
    codebooks_k_mx = mx.array(codebooks_k_np)
    group_map_mx = mx.array(group_map_np)
    scales_k_mx = mx.array(scales_k_np)
    
    # Pre-allocate output buffer
    logits_mx = mx.zeros((n,), dtype=mx.float32)

    # Load Metal Shader String
    with open("mlx_lm/turboquant/tq_kernels.metal", "r") as f:
        metal_source = f.read()

    # 4. Compile & Run MLX Metal Custom Kernel
    # MLX allows loading raw metal source code and binding function names
    try:
        kernel = mx.fast.metal_kernel(
            name="prefix_logits_kernel",
            input_names=["q", "codes_k", "codebooks_k", "group_map", "scales_k", "D"],
            output_names=["logits"],
            source=metal_source
        )
        # JIT warmup
        out_mx = kernel(
            inputs=[q_mx, codes_k_mx, codebooks_k_mx, group_map_mx, scales_k_mx, mx.array(d, dtype=mx.uint32)],
            grid=(n, 1, 1),
            threadgroup=(1, 1, 1), # One thread per token handles the whole row internally in our basic kernel
            output_shapes=[(n,)],
            output_dtypes=[mx.float32]
        )[0]
        mx.eval(out_mx)
        
        # Time the Metal kernel
        num_runs = 100
        t0 = time.time()
        for _ in range(num_runs):
            out_mx = kernel(
                inputs=[q_mx, codes_k_mx, codebooks_k_mx, group_map_mx, scales_k_mx, mx.array(d, dtype=mx.uint32)],
                grid=(n, 1, 1),
                threadgroup=(1, 1, 1),
                output_shapes=[(n,)],
                output_dtypes=[mx.float32]
            )[0]
        mx.eval(out_mx) # Force async execution
        t1 = time.time()
        
        print(f"[MLX Custom Metal] Average Latency: {((t1-t0)/num_runs)*1000:.3f} ms")
        print(f"Max Diff vs Numpy: {np.max(np.abs(np.array(out_mx) - logits_target)):.6f}")

    except Exception as e:
        print(f"Metal compilation/execution info (API specifics): {str(e)}")

if __name__ == '__main__':
    run_metal_benchmark()
