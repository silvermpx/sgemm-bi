# sgemm-bi for PyTorch

Deterministic, batch-invariant CUDA GEMM for PyTorch training:
bit-identical matmuls (forward / dW / dX) in float32, bfloat16, and
float16, with an opt-in tensor-core tier that is faster than cuBLAS
PEDANTIC on transformer-class shapes.

Built on the [sgemm-bi](https://github.com/silvermpx/sgemm-bi) Rust
engine. No libtorch linkage — tensors cross as raw device pointers, so
one wheel works with any PyTorch build. Kernels compile once at engine
creation via NVRTC; the runtime needs only the NVIDIA driver.

**Requirements**: NVIDIA Ampere or newer (sm_80+), PyTorch with CUDA.

## Install

```sh
pip install maturin
cd python && maturin build --release
pip install target/wheels/sgemm_bi-*.whl
```

## Use

```python
import torch, sgemm_bi

# Drop-in layer (weight stored [in, out] — GEMM-natural; convert
# existing layers with Linear.from_torch):
layer = sgemm_bi.Linear(768, 3072, device="cuda", dtype=torch.bfloat16,
                        tensor_cores=True)
y = layer(x)
y.sum().backward()   # deterministic dW (f32-accumulated) and dX

# Functional form:
y = sgemm_bi.deterministic_linear(x, weight, bias, tensor_cores=True)

# Low-level engine (raw pointers, explicit stream):
eng = sgemm_bi.Engine(0)
eng.forward(y.data_ptr(), x.data_ptr(), w.data_ptr(), None,
            m, k, n, "bfloat16",
            torch.cuda.current_stream().cuda_stream, True)
```

## Determinism contracts

| tier | guarantee |
|---|---|
| f32 / typed scalar | bit-identical across runs; bf16/f16 ≡ "upcast → f32 kernel → RNE downcast"; full shape coverage |
| tensor cores (`tensor_cores=True`) | own deterministic contract (mma.sync, f32 accumulate); bit-identical across runs; forward strictly batch-invariant across all M; falls back to the scalar tier when an output dim is < 64 (shape-only dispatch — still deterministic; two bit-identical tile families, 128×128 and 64×64, cover everything above) |

Bias gradient is a plain f32 column sum done in torch (deterministic
run-to-run for a fixed shape; not part of the engine's batch-invariance
contract).

## Tests

```sh
pip install pytest && pytest tests/ -v   # needs a CUDA GPU
```
