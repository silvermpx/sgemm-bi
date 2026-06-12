/* sgemm-bi — batch-invariant deterministic GEMM for CUDA.
 *
 * C interface to the Rust engine (build the crate with the `capi`
 * feature: `cargo build --release --features capi`; link against
 * libsgemm_bi.so / .dylib / sgemm_bi.dll, or the static archive).
 *
 * Requirements: NVIDIA Ampere or newer (sm_80+), CUDA driver 12+.
 * Kernels are compiled at engine creation via NVRTC — no CUDA toolkit
 * is needed at build or run time, only the driver.
 *
 * Threading: an engine is NOT thread-safe. Guard it with a mutex or
 * create one engine per thread. Error messages are per-thread.
 *
 * Streams: the engine owns a dedicated NON-BLOCKING stream. It does not
 * implicitly order against the legacy default stream — synchronize the
 * context (or use events / sgb_engine_stream) between default-stream
 * transfers and engine calls. All entry points enqueue asynchronously
 * and return immediately; call sgb_engine_synchronize before reading
 * results.
 *
 * Determinism contracts:
 *   - f32 tier (SGB_F32): bit-identical across runs, batch-invariant
 *     within a dispatch bucket. Full shape coverage.
 *   - typed scalar tier (SGB_BF16/SGB_F16 via sgb_forward etc.):
 *     outputs bit-identical to "upcast to f32, run the f32 tier,
 *     round-to-nearest-even downcast". Full shape coverage.
 *   - tensor-core tier (sgb_*_tc): separate numeric contract (mma.sync,
 *     f32 accumulate). Runs bit-identical to each other; forward is
 *     strictly batch-invariant across all M. Covers 64-tile shapes
 *     (fwd M>=64 && N>=64) — compose with the scalar tier for the rest.
 */

#ifndef SGEMM_BI_H
#define SGEMM_BI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. */
#define SGB_OK 0
#define SGB_ERR_CUDA 1             /* driver / NVRTC / launch failure   */
#define SGB_ERR_UNCOVERED 2        /* shape not covered by this tier    */
#define SGB_ERR_DTYPE 3            /* dtype invalid for this operation  */
#define SGB_ERR_UNSUPPORTED_ARCH 4 /* device below sm_80                */
#define SGB_ERR_INVALID_ARG 5      /* null pointer / bad dimension      */
#define SGB_ERR_PANIC 6            /* internal error (please report)    */

/* Element types for SgbGemm.dtype. */
#define SGB_F32 0
#define SGB_BF16 1
#define SGB_F16 2

/* Opaque engine handle. */
typedef struct SgbEngine SgbEngine;

/* GEMM descriptor. Device pointers are CUdeviceptr values (cuMemAlloc
 * et al.); all matrices are row-major and contiguous. Field roles:
 *
 *   op              | out                  | a         | b         | bias
 *   ----------------|----------------------|-----------|-----------|---------
 *   sgb_forward     | Y [M,N]              | X [M,K]   | W [K,N]   | f32 [N]
 *   sgb_backward_dw | dW [K,N] (f32, +=)   | dY [M,N]  | X [M,K]   | unused
 *   sgb_backward_dx | dX [M,K] (overwrite) | dY [M,N]  | W [K,N]   | unused
 *
 * dtype applies to out/a/b, EXCEPT sgb_backward_dw's out, which is
 * always an f32 master accumulator (gradients accumulate at full
 * precision by design). bias is always f32; pass 0 for no bias.
 */
typedef struct SgbGemm {
  uint64_t out;
  uint64_t a;
  uint64_t b;
  uint64_t bias;
  int64_t m;
  int64_t k;
  int64_t n;
  int32_t dtype;
  int32_t reserved; /* must be 0 */
} SgbGemm;

/* Last error message for the current thread (NUL-terminated, never
 * NULL). Valid until the next failing call on the same thread. */
const char *sgb_last_error(void);

/* Creates an engine on a device (retains the primary context, creates
 * a dedicated stream, compiles kernels — takes a few seconds once).
 * Writes the handle to *out on success. */
int32_t sgb_engine_create(int32_t device_ordinal, SgbEngine **out);

/* Destroys an engine. NULL is a no-op. */
void sgb_engine_destroy(SgbEngine *eng);

/* Blocks until all work enqueued by this engine has completed. */
int32_t sgb_engine_synchronize(const SgbEngine *eng);

/* Raw CUstream handle the engine enqueues on (for event-based ordering
 * against other streams). Returns 0 for a NULL engine. */
uint64_t sgb_engine_stream(const SgbEngine *eng);

/* Scalar tiers — full shape coverage for all three dtypes. */
int32_t sgb_forward(const SgbEngine *eng, const SgbGemm *gemm);
int32_t sgb_backward_dw(const SgbEngine *eng, const SgbGemm *gemm);
int32_t sgb_backward_dx(const SgbEngine *eng, const SgbGemm *gemm);

/* Tensor-core tier — bf16/f16 only; uncovered shapes return
 * SGB_ERR_UNCOVERED (gates: fwd M>=64 && N>=64, dW K>=64 && N>=64,
 * dX M>=64 && K>=64). */
int32_t sgb_forward_tc(const SgbEngine *eng, const SgbGemm *gemm);
int32_t sgb_backward_dw_tc(const SgbEngine *eng, const SgbGemm *gemm);
int32_t sgb_backward_dx_tc(const SgbEngine *eng, const SgbGemm *gemm);

/* Pre-sizes the typed upcast-fallback scratch so later calls up to the
 * given f32 element counts never allocate. REQUIRED before CUDA Graph
 * capture. From your largest GEMM:
 * (max(M*K, M*N), max(K*N, M*K), max(M*N, M*K)). */
int32_t sgb_presize_upcast_scratch(const SgbEngine *eng, int64_t a_elems,
                                   int64_t b_elems, int64_t c_elems);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SGEMM_BI_H */
