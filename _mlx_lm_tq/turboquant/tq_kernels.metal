// ============================================================================
// tq_kernels.metal — TurboQuant Metal Shading Language kernels for Apple GPU.
//
// Low-level Metal compute shaders for quantized KV-cache attention.
// These kernels are dispatched by MLX via mx.fast.metal_kernel() for
// maximum throughput on Apple Silicon (M1/M2/M3/M4) unified memory.
//
// Kernel: prefix_logits_kernel
//   Computes: logits[i] = sum_j( q[j] * codebooks[group[j]][codes[i,j]] * scales[j] )
//   Threading: One GPU thread per sequence token (N threads total).
//   Each thread iterates over D dimensions to compute a single dot product.
//
// Performance target: <0.1 ms for N=8192, D=128.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// Computes dot product: q @ decoded_quantized_k for one token per thread.
// Grid: (N, 1, 1) — one thread per sequence position.
// Each thread decodes its row of codes through the grouped codebook,
// applies PolarQuant per-channel scaling, and accumulates the dot product.
kernel void prefix_logits_kernel(
    device const float *q [[buffer(0)]],
    device const uint8_t *codes_k [[buffer(1)]],
    device const float *codebooks_k [[buffer(2)]],
    device const uint32_t *group_map [[buffer(3)]],
    device const float *scales_k [[buffer(4)]],
    device float *logits [[buffer(5)]],
    constant uint32_t &D [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    float acc = 0.0;
    
    // Each thread handles one token's row in codes_k
    uint32_t row_offset = tid * D;
    
    for (uint32_t j = 0; j < D; ++j) {
        // Find which codebook group this dimension uses
        uint32_t group = group_map[j];
        
        // 4-bit/8-bit code (assume uint8_t for this prototype)
        uint8_t code = codes_k[row_offset + j];
        
        // Lookup value from codebook
        // Codebooks shape: [16 groups, 16 codes] -> group * 16 + code
        float val = codebooks_k[group * 16 + code];
        
        // PolarQuant channel scaling
        val *= scales_k[j];
        
        // Dot product with Query
        acc += q[j] * val;
    }
    
    logits[tid] = acc;
}
