// Shared dtype prelude for the typed (bf16/f16) kernel instantiations.
//
// All math happens in f32; dtype conversion is upcast-on-load (`to_f`),
// downcast-on-store (`from_f_*`, round-to-nearest-even) — a single PTX cvt
// instruction each. NVRTC compiles one combined source blob with this file
// inlined first.

#ifndef _SGEMM_BI_PRELUDE_CUH
#define _SGEMM_BI_PRELUDE_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---- Upcast helpers (load) ------------------------------------------------
__device__ __forceinline__ float to_f(float v)          { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v)  { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(__half v)         { return __half2float(v); }

// ---- Downcast helpers (store, round-to-nearest-even) ----------------------
__device__ __forceinline__ float         from_f_f32(float v)  { return v; }
__device__ __forceinline__ __nv_bfloat16 from_f_bf16(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ __half        from_f_f16(float v)  { return __float2half_rn(v); }

#endif // _SGEMM_BI_PRELUDE_CUH
