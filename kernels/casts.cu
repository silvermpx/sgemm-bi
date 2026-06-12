// Elementwise dtype conversions for the typed upcast fallback:
// shapes without a native typed bucket run "upcast -> f32 kernel -> RNE
// downcast". Upcasts are exact (16-bit grids embed in f32); downcasts are
// round-to-nearest-even, matching from_f_* in the typed kernels.

extern "C" __global__ void sgb_cast_f32_to_bf16(
    __nv_bfloat16* dst, const float* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16_rn(src[i]);
}

extern "C" __global__ void sgb_cast_bf16_to_f32(
    float* dst, const __nv_bfloat16* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}

extern "C" __global__ void sgb_cast_f32_to_f16(
    __half* dst, const float* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half_rn(src[i]);
}

extern "C" __global__ void sgb_cast_f16_to_f32(
    float* dst, const __half* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}
