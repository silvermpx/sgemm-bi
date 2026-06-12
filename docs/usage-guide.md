# sgemm-bi usage guide

Practical recipes for all three interfaces. For contract details and
measured performance, see the [README](../README.md) and
[CHANGELOG](../CHANGELOG.md).

## What this library is for

Bitwise-reproducible matrix multiplication on NVIDIA GPUs — the three
GEMMs of a training step (forward, weight gradient, input gradient) with
guarantees cuBLAS does not give you:

- **Run-to-run determinism**: same inputs → bit-identical outputs, every
  launch, on f32, bf16, and f16. No atomics, no data-dependent splits,
  fixed reduction order.
- **Batch invariance**: row 0 of the output does not change when the
  batch grows (strict, across ALL batch sizes, on the tensor-core
  forward). An RL rollout scored at batch 1 stays bit-identical inside a
  batch of 64.
- **A typed bit contract**: bf16/f16 results are exactly "upcast to f32,
  run the f32 kernel, round once at the end" — accumulation never
  happens in 16 bits.

Typical users: RL training where trajectory reproducibility matters,
LLM fine-tuning that must be auditable/replayable, CI pipelines that
diff model weights, research that needs exact ablations.

## Choosing a tier

| you want | use | cost |
|---|---|---|
| exact f32, maximum reproducibility (RL critics) | f32 tier | ~1.3–1.5× of cuBLAS-TF32 (you keep 23 mantissa bits; TF32 keeps 10) |
| mixed-precision training with a bit contract | typed tier (bf16/f16) | ~1.1–1.4× of cuBLAS-PEDANTIC |
| mixed-precision training, fastest deterministic | tensor-core tier | **parity to 0.70×** of cuBLAS-PEDANTIC (faster from d≈768) |

The tensor-core tier has its own numeric contract (a TC reduction tree
cannot bit-match a scalar FMA chain) — runs are still bit-identical to
each other and the forward is strictly batch-invariant. Don't mix tiers
across runs you intend to compare bit-for-bit.

## Rust

```rust
use sgemm_bi::{Dtype, SgemmBi, TypedPtr};

let ctx = cudarc::driver::CudaContext::new(0)?;
let stream = ctx.new_stream()?;
let engine = SgemmBi::new(&ctx, stream.clone())?;   // NVRTC compile, once

// f32 tier — full coverage, exact precision:
engine.forward_f32(y_ptr, x_ptr, w_ptr, Some(bias_ptr), (m, k, n))?;
engine.backward_dw_f32(dw_ptr, dy_ptr, x_ptr, (m, k, n))?;   // dw += (zero it first)
engine.backward_dx_f32(dx_ptr, dy_ptr, w_ptr, (m, k, n))?;

// typed tier — bf16/f16 with the upcast-equivalence bit contract:
let t = |p| TypedPtr::new(p, Dtype::Bf16);
engine.forward(t(y), t(x), t(w), Some(bias_ptr), (m, k, n))?;

// tensor-core tier — try TC, compose with the scalar tier for the rest:
match engine.forward_tc(t(y), t(x), t(w), None, (m, k, n)) {
    Err(sgemm_bi::Error::Uncovered { .. }) =>
        engine.forward(t(y), t(x), t(w), None, (m, k, n))?,
    other => other?,
}
```

Rules of the road:
- Operand buffers must live on (or be event-ordered against) the
  engine's stream; all entry points enqueue and return immediately —
  `engine.stream().synchronize()` before reading results.
- `dW` is **always f32** (master accumulator), even with bf16/f16
  operands, and accumulates (`+=`) — zero it before the first GEMM.
- One engine per stream; the engine is not `Sync`.

### CUDA Graph capture

Every kernel is capture-safe (no allocations or syncs in steady state).
One prerequisite: pre-size the typed-fallback scratch before capture,

```rust
engine.presize_upcast_scratch((max_mk_mn, max_kn_mk, max_mn_mk))?;
```

then wrap your step in `cuStreamBeginCapture`/`EndCapture` as usual.

## C / C++ (and any FFI language)

Build once: `cargo build --release --features capi` → link
`libsgemm_bi.so` (or the static archive) + `include/sgemm_bi.h`.

```c
SgbEngine *eng = NULL;
if (sgb_engine_create(0, &eng) != SGB_OK) {
    fprintf(stderr, "%s\n", sgb_last_error());
    return 1;
}

SgbGemm g = {
    .out = y, .a = x, .b = w, .bias = bias,   /* CUdeviceptr values */
    .m = M, .k = K, .n = N,
    .dtype = SGB_BF16,
};
sgb_forward(eng, &g);          /* scalar tier, full coverage      */
sgb_forward_tc(eng, &g);       /* TC tier, SGB_ERR_UNCOVERED if   */
                               /* a dim is < 64 — fall back to    */
                               /* sgb_forward yourself            */
sgb_engine_synchronize(eng);
sgb_engine_destroy(eng);
```

- Status codes + `sgb_last_error()` (per-thread message).
- The engine stream is non-blocking: synchronize the context between
  default-stream copies and engine calls, or order with events via
  `sgb_engine_stream()`.
- One descriptor struct for all six ops — field roles are documented in
  the header.

## PyTorch

`pip install sgemm-bi` (one abi3 wheel, any torch build, runtime needs
only the NVIDIA driver).

```python
import torch, sgemm_bi

# Drop-in layer. Weight is stored [in, out] (GEMM-natural);
# convert existing layers once:
layer = sgemm_bi.Linear(768, 3072, device="cuda",
                        dtype=torch.bfloat16, tensor_cores=True)
det   = sgemm_bi.Linear.from_torch(my_nn_linear, tensor_cores=True)

y = layer(x)            # deterministic forward
y.sum().backward()      # deterministic dW (f32-accumulated) and dX

# Functional form:
y = sgemm_bi.deterministic_linear(x, weight, bias, tensor_cores=True)
```

Notes:
- `bias` is always float32 (it feeds the f32 accumulator directly).
- Gradients follow torch convention (bf16 params → bf16 grads), but dW
  was accumulated in f32 inside the kernel and rounded once — strictly
  tighter than a 16-bit accumulation.
- NOT `torch.autocast`-aware by design: construct the layer in the
  dtype you train in; the layer raises on dtype mismatch instead of
  silently re-casting.
- Works from the autograd thread; the stream bridge is asynchronous
  (no hidden host syncs).
- The low-level `sgemm_bi.Engine` (typed stub ships with the wheel)
  exposes the raw triad for custom pipelines.

## Verifying determinism yourself

```python
y1 = sgemm_bi.deterministic_linear(x, w, tensor_cores=True)
y2 = sgemm_bi.deterministic_linear(x, w, tensor_cores=True)
assert torch.equal(y1, y2)                       # bit-identical, not allclose

big   = sgemm_bi.deterministic_linear(x4096, w, tensor_cores=True)
small = sgemm_bi.deterministic_linear(x4096[:64].contiguous(), w, tensor_cores=True)
assert torch.equal(small, big[:64])              # strict batch invariance
```

## Requirements and limits

- NVIDIA Ampere or newer (sm_80+) — enforced at engine creation with a
  clear error.
- Row-major contiguous operands.
- The TC tier covers both output dims ≥ 64; compose with the scalar
  tier below that (the high-level APIs do this for you).
- cuBLAS is never linked or called — uncovered shapes are loud errors,
  never silent fallbacks to nondeterminism.
