//! # sgemm-bi
//!
//! Deterministic, batch-invariant CUDA GEMM engine with a full training
//! triad — forward (`Y = X@W + bias`), weight gradient (`dW += X^T@dY`),
//! and input gradient (`dX = dY@W^T`) — in f32, bf16, and f16, plus an
//! opt-in tensor-core tier.
//!
//! ## Guarantees
//!
//! - **Run-to-run determinism**: every kernel uses a fixed reduction order
//!   (no atomics, no data-dependent splits). Same inputs → bit-identical
//!   outputs, on every call, including through CUDA Graph replay.
//! - **Batch invariance**: within a dispatch bucket, row 0 of the output
//!   is bit-identical regardless of M. The tensor-core forward is
//!   strictly batch-invariant across ALL M.
//! - **Typed bit contract**: bf16/f16 results are bit-identical to
//!   "upcast inputs to f32, run the f32 tier, round-to-nearest-even
//!   downcast" — accumulation never happens in reduced precision, and
//!   exactly one rounding is applied at the output store.
//! - **No vendor BLAS**: cuBLAS is not linked, called, or fallen back to.
//!
//! ## Tiers
//!
//! | tier | entry points | contract |
//! |---|---|---|
//! | f32 | [`SgemmBi::forward_f32`], `backward_dw_f32`, `backward_dx_f32` | reference chain |
//! | typed (bf16/f16) | [`SgemmBi::forward`], `backward_dw`, `backward_dx` | bit-equal to f32 tier on upcast inputs |
//! | tensor cores | [`SgemmBi::forward_tc`], `backward_dw_tc`, `backward_dx_tc` | own deterministic contract (mma.sync, f32 accumulate) |
//!
//! The tensor-core tier is typically 3–7× faster than the scalar tiers on
//! 128x128-tile-friendly shapes and turns the determinism overhead
//! negative against cuBLAS-PEDANTIC-class baselines; it does not
//! bit-match the scalar tiers (a tensor-core reduction tree cannot
//! reproduce a scalar FMA chain).
//!
//! ## Requirements
//!
//! NVIDIA Ampere or newer (`sm_80`+) — the kernel blob uses cp.async,
//! ldmatrix, and native bf16 throughout, so [`SgemmBi::new`] rejects
//! older devices with [`Error::UnsupportedArch`]. Kernels compile at
//! engine construction via NVRTC for the device's native architecture —
//! no CUDA toolkit needed at run time beyond the driver and NVRTC.
//!
//! ## C interface
//!
//! Enable the `capi` feature for a flat `extern "C"` surface (module
//! [`capi`], header `include/sgemm_bi.h`): engine create/destroy, the
//! six GEMM entry points on raw `CUdeviceptr`s, per-thread error
//! strings, and scratch pre-sizing for CUDA Graph capture.
//!
//! ## Example
//!
//! ```no_run
//! use sgemm_bi::{Dtype, SgemmBi, TypedPtr};
//!
//! let context = cudarc::driver::CudaContext::new(0).unwrap();
//! let stream = context.new_stream().unwrap();
//! let engine = SgemmBi::new(&context, stream.clone()).unwrap();
//!
//! let (m, k, n) = (2048, 768, 3072);
//! let x = stream.alloc_zeros::<u16>(m * k).unwrap(); // bf16 storage
//! let w = stream.alloc_zeros::<u16>(k * n).unwrap();
//! let y = stream.alloc_zeros::<u16>(m * n).unwrap();
//! # use cudarc::driver::DevicePtr;
//! let ptr = |b: &cudarc::driver::CudaSlice<u16>| b.device_ptr(&stream).0;
//!
//! engine
//!     .forward(
//!         TypedPtr::new(ptr(&y), Dtype::Bf16),
//!         TypedPtr::new(ptr(&x), Dtype::Bf16),
//!         TypedPtr::new(ptr(&w), Dtype::Bf16),
//!         None,
//!         (m, k, n),
//!     )
//!     .unwrap();
//! stream.synchronize().unwrap();
//! ```

#[cfg(feature = "capi")]
pub mod capi;
mod dispatch;
mod dtype;
mod engine;
mod error;
mod kernels;

pub use dtype::{Dtype, TypedPtr};
pub use engine::SgemmBi;
pub use error::{Error, Result};
