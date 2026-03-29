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
    constant uint32_t &stride_n [[buffer(7)]],
    constant uint32_t &stride_d [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    float acc = 0.0;
    
    for (uint32_t j = 0; j < D; ++j) {
        // Find which codebook group this dimension uses
        uint32_t group = group_map[j];
        
        // 4-bit/8-bit code mapping with generic strides
        uint8_t code = codes_k[tid * stride_n + j * stride_d];
        
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

// Computes attention weighted sum for Values: out = probs @ decoded_quantized_v
// Grid: (D, 1, 1) — one thread per output dimension.
// Each thread maps its dimension to a codebook group and computes the dot
// product of the attention probabilities with its quantised value column.
kernel void prefix_v_kernel(
    device const float *probs [[buffer(0)]],
    device const uint8_t *codes_v [[buffer(1)]],
    device const float *codebooks_v [[buffer(2)]],
    device const uint32_t *group_map [[buffer(3)]],
    device const float *scales_v [[buffer(4)]],
    device float *out_v [[buffer(5)]],
    constant uint32_t &N [[buffer(6)]],
    constant uint32_t &stride_n [[buffer(7)]],
    constant uint32_t &stride_d [[buffer(8)]],
    uint j [[thread_position_in_grid]]
) {
    float acc = 0.0;
    uint32_t group = group_map[j];
    float scale = scales_v[j];

    for (uint32_t i = 0; i < N; ++i) {
        uint8_t code = codes_v[i * stride_n + j * stride_d];
        float val = codebooks_v[group * 16 + code] * scale;
        acc += probs[i] * val;
    }

    out_v[j] = acc;
}
