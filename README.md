# sgemm-bi

Deterministic, batch-invariant CUDA GEMM engine with a full **training
triad** — forward, weight gradient, and input gradient — in **f32, bf16,
and f16**, plus an opt-in **tensor-core tier**.

Existing batch-invariant kernel collections cover inference only and trade
10–40% throughput for determinism. `sgemm-bi` covers the backward pass
too, and on tile-friendly shapes the tensor-core tier makes deterministic
training *faster* than a CUDA-core cuBLAS baseline.

## Guarantees

- **Run-to-run determinism** — fixed reduction order in every kernel: no
  atomics, no data-dependent splits, no vendor-BLAS fallback. Same inputs
  → bit-identical outputs, including through CUDA Graph replay.
- **Batch invariance** — within a dispatch bucket, output row 0 is
  bit-identical regardless of the batch dimension M. The tensor-core
  forward is strictly batch-invariant across *all* M.
- **Typed bit contract** — bf16/f16 results are bit-identical to "upcast
  the inputs to f32, run the f32 tier, round-to-nearest-even downcast the
  output". Accumulation never happens in reduced precision; exactly one
  rounding is applied, at the output store.

## Operations

| op | math | output |
|---|---|---|
| `forward` | `Y[M,N] = X[M,K] @ W[K,N] + bias[N]` | typed / f32 |
| `backward_dw` | `dW[K,N] += X^T[K,M] @ dY[M,N]` | f32 accumulate |
| `backward_dx` | `dX[M,K] = dY[M,N] @ W^T[N,K]` | typed / f32 |

Each op exists in three tiers: `*_f32` (the reference chain), typed
(bf16/f16, bit-equal to the f32 tier on upcast inputs), and `*_tc`
(tensor cores — a separate deterministic contract; mma.sync with f32
accumulators cannot bit-match a scalar FMA chain, but it is deterministic
and strictly batch-invariant).

The f32 and typed tiers cover **every** shape: a bucketed dispatcher
(Big / Slim / narrow / ultra-thin / GEMV / split-K/M/N with fixed-order
tree reduction) handles the common cases natively and the typed tier
falls back to "upcast → f32 kernel → downcast" — same bits by contract —
for the rest. The tensor-core tier covers 128×128-tile shapes and returns
`Error::Uncovered` otherwise.

## Performance (RTX 6000 Ada, bf16)

Tensor-core tier vs the scalar deterministic tier, GEMM level:

| shape (M, K, N) | forward | dW | dX |
|---|---:|---:|---:|
| 2048, 768, 3072 | 3.1× | 3.5× | 6.7× |
| 4096, 1536, 3072 | 2.8× | 3.2× | 6.1× |
| 2048, 768, 512 | 3.4× | 2.1× | 4.3× |

~106 TFLOPS bf16 at M2048 K768 N3072. In an end-to-end mixed-precision
training loop this turns the cost of determinism negative: ~0.8× of a
cuBLAS-PEDANTIC-baseline step on d_model ≥ 768 models.

## Usage

```rust,ignore
use sgemm_bi::{Dtype, SgemmBi, TypedPtr};

let context = cudarc::driver::CudaContext::new(0).unwrap();
let stream = context.new_stream().unwrap();
let engine = SgemmBi::new(&context, stream.clone()).unwrap();

// y/x/w are CUdeviceptr device allocations on `stream` (bf16 storage).
engine
    .forward(
        TypedPtr::new(y, Dtype::Bf16),
        TypedPtr::new(x, Dtype::Bf16),
        TypedPtr::new(w, Dtype::Bf16),
        Some(bias_f32_ptr),
        (m, k, n),
    )
    .unwrap();
```

The engine binds to one stream; all calls enqueue and return. For CUDA
Graph capture, call `presize_upcast_scratch` before capturing so the
typed fallback never allocates inside (or after) a captured graph.

## Requirements

- NVIDIA GPU, `sm_80`+ for the bf16/f16 and tensor-core tiers (`cp.async`,
  `ldmatrix`, bf16 `mma.sync`); the f32 tier runs on older architectures.
- CUDA driver + NVRTC at run time. Kernels compile at engine construction
  for the device's native architecture — no toolkit or `nvcc` needed.
- No cuBLAS: the library never links or calls a vendor BLAS.

## Testing

Contract tests require a CUDA device:

```sh
cargo test --release -- --test-threads=1
```

Covered: f32 run-to-run bit identity; the typed bit contract swept across
~90 dispatch-gate boundary shapes (forward) plus backward shapes;
per-bucket batch invariance; tensor-core determinism, strict all-M
invariance, and accuracy vs the f32 reference. Benchmarks are `#[ignore]`d
(`bench_tc_vs_scalar`).

## Lineage

The Big-tile kernels descend from [siboehm's SGEMM
warptiling](https://github.com/siboehm/SGEMM_CUDA) work; smem padding
follows [salykova's sgemm.cu](https://github.com/salykova/sgemm.cu). The
engine is extracted from the GEMM layer of
[mamba-rs](https://github.com/silvermpx/mamba-rs), where it powers
deterministic SSM training.

## License

Dual-licensed under MIT or Apache-2.0.
