/* C ABI smoke test: f32 forward correctness vs a host reference, run
 * twice for bit-identity, plus a typed (bf16) forward and the error
 * path. Exercises the library exactly as an external C caller would.
 *
 * Build (Linux, from the repository root):
 *   cargo build --release --features capi
 *   gcc -O2 -o smoke examples/capi/smoke.c \
 *       -Iinclude -Ltarget/release -lsgemm_bi -lcuda \
 *       -Wl,-rpath,$PWD/target/release
 *   ./smoke
 */

#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgemm_bi.h"

#define CHECK_CU(call)                                                   \
  do {                                                                   \
    CUresult rc_ = (call);                                               \
    if (rc_ != CUDA_SUCCESS) {                                           \
      const char *name_ = NULL;                                          \
      cuGetErrorName(rc_, &name_);                                       \
      fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call,    \
              name_ ? name_ : "?");                                      \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

#define CHECK_SGB(call)                                                  \
  do {                                                                   \
    int32_t rc_ = (call);                                                \
    if (rc_ != SGB_OK) {                                                 \
      fprintf(stderr, "%s:%d: %s -> %d: %s\n", __FILE__, __LINE__,       \
              #call, rc_, sgb_last_error());                             \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

static float frand(unsigned *state) {
  *state = *state * 1664525u + 1013904223u;
  return ((float)(*state >> 8) / (float)(1u << 24)) - 0.5f;
}

static uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u); /* RNE */
  return (uint16_t)(rounded >> 16);
}

int main(void) {
  /* Engine first: it initializes the driver and retains the device's
   * primary context, which this test then joins for its allocations. */
  SgbEngine *eng = NULL;
  CHECK_SGB(sgb_engine_create(0, &eng));

  CHECK_CU(cuInit(0));
  CUdevice dev;
  CHECK_CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK_CU(cuCtxSetCurrent(ctx));

  const int64_t M = 256, K = 192, N = 320;

  float *hx = malloc(M * K * 4), *hw = malloc(K * N * 4);
  float *hb = malloc(N * 4);
  float *hy = malloc(M * N * 4), *hy2 = malloc(M * N * 4);
  unsigned seed = 42;
  for (int64_t i = 0; i < M * K; i++) hx[i] = frand(&seed);
  for (int64_t i = 0; i < K * N; i++) hw[i] = frand(&seed);
  for (int64_t i = 0; i < N; i++) hb[i] = frand(&seed);

  CUdeviceptr dx, dw, db, dy;
  CHECK_CU(cuMemAlloc(&dx, M * K * 4));
  CHECK_CU(cuMemAlloc(&dw, K * N * 4));
  CHECK_CU(cuMemAlloc(&db, N * 4));
  CHECK_CU(cuMemAlloc(&dy, M * N * 4));
  CHECK_CU(cuMemcpyHtoD(dx, hx, M * K * 4));
  CHECK_CU(cuMemcpyHtoD(dw, hw, K * N * 4));
  CHECK_CU(cuMemcpyHtoD(db, hb, N * 4));
  /* The engine stream is non-blocking: order the default-stream copies
   * before enqueueing engine work. */
  CHECK_CU(cuCtxSynchronize());

  /* f32 forward, twice — correctness and bit-identity. */
  SgbGemm g = {.out = dy, .a = dx, .b = dw, .bias = db,
               .m = M, .k = K, .n = N, .dtype = SGB_F32, .reserved = 0};
  CHECK_SGB(sgb_forward(eng, &g));
  CHECK_SGB(sgb_engine_synchronize(eng));
  CHECK_CU(cuMemcpyDtoH(hy, dy, M * N * 4));

  CHECK_CU(cuMemsetD8(dy, 0xAB, M * N * 4));
  CHECK_CU(cuCtxSynchronize());
  CHECK_SGB(sgb_forward(eng, &g));
  CHECK_SGB(sgb_engine_synchronize(eng));
  CHECK_CU(cuMemcpyDtoH(hy2, dy, M * N * 4));

  if (memcmp(hy, hy2, M * N * 4) != 0) {
    fprintf(stderr, "FAIL: f32 forward not bit-identical across runs\n");
    return 1;
  }

  double max_rel = 0.0;
  for (int64_t r = 0; r < M; r += 37) {
    for (int64_t c = 0; c < N; c += 23) {
      double acc = hb[c];
      for (int64_t i = 0; i < K; i++) acc += (double)hx[r * K + i] * hw[i * N + c];
      double rel = fabs(acc - hy[r * N + c]) / (fabs(acc) + 1e-6);
      if (rel > max_rel) max_rel = rel;
    }
  }
  if (max_rel > 1e-4) {
    fprintf(stderr, "FAIL: f32 forward max rel err %.3e\n", max_rel);
    return 1;
  }
  printf("f32 forward: bit-identical across runs, max rel err %.3e\n", max_rel);

  /* bf16 forward through the typed tier (same buffers reinterpreted). */
  uint16_t *hxb = malloc(M * K * 2), *hwb = malloc(K * N * 2);
  for (int64_t i = 0; i < M * K; i++) hxb[i] = f32_to_bf16(hx[i]);
  for (int64_t i = 0; i < K * N; i++) hwb[i] = f32_to_bf16(hw[i]);
  CUdeviceptr dxb, dwb, dyb;
  CHECK_CU(cuMemAlloc(&dxb, M * K * 2));
  CHECK_CU(cuMemAlloc(&dwb, K * N * 2));
  CHECK_CU(cuMemAlloc(&dyb, M * N * 2));
  CHECK_CU(cuMemcpyHtoD(dxb, hxb, M * K * 2));
  CHECK_CU(cuMemcpyHtoD(dwb, hwb, K * N * 2));
  CHECK_CU(cuCtxSynchronize());

  SgbGemm gt = {.out = dyb, .a = dxb, .b = dwb, .bias = db,
                .m = M, .k = K, .n = N, .dtype = SGB_BF16, .reserved = 0};
  CHECK_SGB(sgb_forward(eng, &gt));
  CHECK_SGB(sgb_engine_synchronize(eng));
  printf("bf16 forward: ok\n");

  /* Error paths must report, not crash: TC on an undersized shape, and
   * an invalid dtype code. */
  SgbGemm small = gt;
  small.m = 32; /* below the 64-row TC gate */
  if (sgb_forward_tc(eng, &small) != SGB_ERR_UNCOVERED) {
    fprintf(stderr, "FAIL: expected SGB_ERR_UNCOVERED from TC gate\n");
    return 1;
  }
  printf("tc gate: SGB_ERR_UNCOVERED as expected (%s)\n", sgb_last_error());

  SgbGemm bad = g;
  bad.dtype = 99;
  if (sgb_forward(eng, &bad) != SGB_ERR_INVALID_ARG) {
    fprintf(stderr, "FAIL: expected SGB_ERR_INVALID_ARG for dtype 99\n");
    return 1;
  }

  sgb_engine_destroy(eng);
  printf("smoke: ALL OK\n");
  return 0;
}
