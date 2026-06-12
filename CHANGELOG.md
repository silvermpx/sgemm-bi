# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] (0.1.1)

### Added

- **PyTorch binding** (`python/`, PyPI package `sgemm-bi`, import
  `sgemm_bi`): PyO3 0.29 + maturin, abi3 wheel for Python >= 3.9. No
  libtorch linkage — tensors cross as raw device pointers, so one wheel
  works with any PyTorch build; runtime needs only the NVIDIA driver.
  Ships `sgemm_bi.Linear` (deterministic `nn.Linear` replacement with
  GEMM-natural `[in, out]` weight layout and `from_torch` converter),
  the functional `deterministic_linear` autograd op (dW accumulated in
  f32 inside the kernel, one rounding to the parameter dtype), and the
  low-level `Engine`. Engine work is ordered against torch's current
  stream with a CUDA-event bridge (no host syncs); calls release the
  GIL; forward/backward are safe across torch's autograd thread.
  Desk-reviewed against PyTorch/PyO3/maturin/CUDA driver documentation;
  GPU test suite (`python/tests/`) green on RTX 6000 Ada: parity vs
  float64 references, bit-identity across runs in all three dtypes,
  strict all-M batch invariance of the tensor-core forward, end-to-end
  training.
- **CI/release for the binding**: `python-binding` job (fmt, clippy,
  wheel build artifact) and a tag-gated `publish-pypi` job using PyPI
  trusted publishing (OIDC, no token secret).

- **C ABI** behind the `capi` feature (`src/capi.rs`, header
  `include/sgemm_bi.h`, `cdylib`/`staticlib` crate types): engine
  create/destroy/synchronize on a device ordinal, one `SgbGemm`
  descriptor for all six GEMM entry points (scalar forward/dW/dX +
  tensor-core triad) over raw `CUdeviceptr`s, per-thread error strings
  (`sgb_last_error`), raw stream access for event-based ordering, and
  upcast-scratch pre-sizing for CUDA Graph capture. Panics convert to
  `SGB_ERR_PANIC` instead of unwinding across the boundary. Smoke test:
  `examples/capi/smoke.c`.
- **Explicit architecture gate**: `SgemmBi::new` now rejects devices
  below `sm_80` with the new `Error::UnsupportedArch` ("requires Ampere
  or newer") instead of surfacing an opaque NVRTC failure — the kernel
  blob uses `cp.async` and native bf16 in every tier, so pre-Ampere
  devices were never able to run it.

## [0.1.0] - 2026-06-12

### Added

- Initial release: deterministic, batch-invariant CUDA GEMM engine with
  the full training triad — forward `Y = X@W + bias`, weight gradient
  `dW += X^T@dY` (f32 master accumulator), input gradient `dX = dY@W^T`.
- **f32 tier**: full shape coverage via bucketed dispatch (GEMV,
  ultra-thin, narrow, split-K, gap-fill, Big, Slim, split-M/N); fixed
  reduction order, no atomics, no cuBLAS anywhere.
- **Typed tier (bf16/f16)**: native buckets keep f32 shared memory and
  accumulation with the f32 tier's exact FMA chain; uncovered shapes
  take "upcast → f32 kernel → RNE downcast". Both routes are
  bit-identical by contract.
- **Tensor-core tier (bf16/f16)**: `mma.sync.m16n8k16` with f32
  accumulators, 2-stage `cp.async` staging, `ldmatrix` fragment loads.
  Separate numeric contract; bit-identical across runs and strictly
  batch-invariant forward across all M. 3-7x faster than the scalar
  tiers on 128x128-tile shapes.
- GPU contract tests (`tests/contracts.rs`, `tests/tensor_cores.rs`)
  validated on RTX 6000 Ada / CUDA 13.2; CI with fmt, clippy, docs,
  MSRV 1.94, cargo-deny, cross-platform build matrix, and a manual
  tag-gated release pipeline.

[Unreleased]: https://github.com/silvermpx/sgemm-bi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/silvermpx/sgemm-bi/releases/tag/v0.1.0
