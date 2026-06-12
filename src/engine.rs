//! The public engine: owns the compiled kernels, the stream they run on,
//! and the scratch buffers; exposes the full-coverage GEMM entry points.

use crate::dispatch;
use crate::dtype::{Dtype, TypedPtr};
use crate::error::{Error, Result};
use crate::kernels::{CUptr, Kernels};
use cudarc::driver::{CudaContext, CudaStream, PushKernelArg};
use std::cell::RefCell;
use std::sync::Arc;

/// Deterministic GEMM engine bound to one CUDA stream.
///
/// Construction compiles the kernel blob (NVRTC, native arch) and
/// allocates fixed scratch. All entry points enqueue work on the bound
/// stream and return without synchronizing; operand pointers must be
/// allocated on (or ordered against) the same stream.
///
/// Not `Sync`: the upcast scratch uses interior mutability. Use one engine
/// per stream.
pub struct SgemmBi {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    /// Grow-only f32 scratch triple for the typed upcast fallback: shapes
    /// without a native typed bucket run "upcast → f32 kernel → RNE
    /// downcast" (bit-identical to a native typed kernel by contract).
    /// Lazily grown; steady-state calls do not allocate. For CUDA Graph
    /// capture, call [`SgemmBi::presize_upcast_scratch`] first.
    upcast_scratch: [RefCell<Option<cudarc::driver::CudaSlice<f32>>>; 3],
}

impl SgemmBi {
    /// Build an engine on `stream`. Compiles kernels for the device's
    /// native architecture. Requires Ampere or newer (sm_80+): the kernel
    /// blob uses `cp.async` staging and native bf16 in every tier, so
    /// older devices fail here with [`Error::UnsupportedArch`] rather
    /// than an opaque NVRTC error.
    pub fn new(context: &Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self> {
        let cc = (
            context
                .attribute(cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .map_err(|e| Error::Cuda(format!("query CC major: {e:?}")))? as u32,
            context
                .attribute(cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .map_err(|e| Error::Cuda(format!("query CC minor: {e:?}")))? as u32,
        );
        if cc.0 < 8 {
            return Err(Error::UnsupportedArch {
                major: cc.0,
                minor: cc.1,
            });
        }
        let arch = crate::kernels::nvrtc_arch(cc);
        let kernels = Kernels::compile(context, &stream, arch)?;
        Ok(Self {
            stream,
            kernels,
            upcast_scratch: [RefCell::new(None), RefCell::new(None), RefCell::new(None)],
        })
    }

    /// The stream all work is enqueued on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    // ── f32 tier ────────────────────────────────────────────────────────

    /// `Y[M,N] = X[M,K] @ W[K,N] (+ bias[N])`, f32, full shape coverage.
    /// Bit-identical across runs; batch-invariant within a dispatch bucket.
    pub fn forward_f32(
        &self,
        y: CUptr,
        x: CUptr,
        w: CUptr,
        bias: Option<CUptr>,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        dispatch::sgemm_bi_forward(
            &self.stream,
            &self.kernels,
            y,
            x,
            w,
            bias.unwrap_or(0),
            dims,
        )
    }

    /// `dW[K,N] += X^T[K,M] @ dY[M,N]`, f32 accumulate, full coverage.
    pub fn backward_dw_f32(
        &self,
        dw: CUptr,
        dy: CUptr,
        x_saved: CUptr,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        dispatch::sgemm_bi_backward_dw(&self.stream, &self.kernels, dw, dy, x_saved, dims)
    }

    /// `dX[M,K] = dY[M,N] @ W^T[N,K]`, f32 overwrite, full coverage.
    pub fn backward_dx_f32(
        &self,
        dx: CUptr,
        dy: CUptr,
        w: CUptr,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        dispatch::sgemm_bi_backward_dx(&self.stream, &self.kernels, dx, dy, w, dims)
    }

    // ── typed scalar tier (bf16/f16) ────────────────────────────────────
    //
    // Bit contract: outputs are bit-identical to "upcast the 16-bit inputs
    // to f32, run the f32 tier, RNE-downcast the result". Native typed
    // buckets satisfy it by keeping f32 shared memory + accumulation with
    // the f32 twin's exact FMA chain; all other shapes take the upcast
    // fallback, which satisfies it by construction.

    /// Typed `Y = X @ W (+ bias)`. `y`/`x`/`w` must share one 16-bit dtype;
    /// `bias` stays f32. Full shape coverage.
    pub fn forward(
        &self,
        y: TypedPtr,
        x: TypedPtr,
        w: TypedPtr,
        bias: Option<CUptr>,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        let bias_ptr = bias.unwrap_or(0);
        match dispatch::sgemm_bi_forward_typed(&self.stream, &self.kernels, y, x, w, bias_ptr, dims)
        {
            Err(Error::Uncovered { .. }) => {}
            other => return other,
        }
        let (m, k, n) = dims;
        self.with_upcast_scratch((m * k, k * n, m * n), |xs, ws, ys| {
            self.upcast_to_f32(x, xs, m * k)?;
            self.upcast_to_f32(w, ws, k * n)?;
            dispatch::sgemm_bi_forward(&self.stream, &self.kernels, ys, xs, ws, bias_ptr, dims)?;
            self.downcast_from_f32(y, ys, m * n)
        })
    }

    /// Typed `dW += X^T @ dY` with an f32 master accumulator (`dw` is f32 —
    /// gradients accumulate at full precision by design). Full coverage.
    pub fn backward_dw(
        &self,
        dw: CUptr,
        dy: TypedPtr,
        x_saved: TypedPtr,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        match dispatch::sgemm_bi_backward_dw_typed(
            &self.stream,
            &self.kernels,
            dw,
            dy,
            x_saved,
            dims,
        ) {
            Err(Error::Uncovered { .. }) => {}
            other => return other,
        }
        let (m, k, n) = dims;
        self.with_upcast_scratch((m * n, m * k, 0), |dys, xs, _| {
            self.upcast_to_f32(dy, dys, m * n)?;
            self.upcast_to_f32(x_saved, xs, m * k)?;
            dispatch::sgemm_bi_backward_dw(&self.stream, &self.kernels, dw, dys, xs, dims)
        })
    }

    /// Typed `dX = dY @ W^T` (RNE overwrite of `dx`). Full coverage.
    pub fn backward_dx(
        &self,
        dx: TypedPtr,
        dy: TypedPtr,
        w: TypedPtr,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        match dispatch::sgemm_bi_backward_dx_typed(&self.stream, &self.kernels, dx, dy, w, dims) {
            Err(Error::Uncovered { .. }) => {}
            other => return other,
        }
        let (m, k, n) = dims;
        self.with_upcast_scratch((m * n, k * n, m * k), |dys, ws, dxs| {
            self.upcast_to_f32(dy, dys, m * n)?;
            self.upcast_to_f32(w, ws, k * n)?;
            dispatch::sgemm_bi_backward_dx(&self.stream, &self.kernels, dxs, dys, ws, dims)?;
            self.downcast_from_f32(dx, dxs, m * k)
        })
    }

    // ── tensor-core tier (bf16/f16) ─────────────────────────────────────
    //
    // A SEPARATE numeric contract: mma.sync with f32 accumulators. The TC
    // reduction tree differs from the scalar FMA chain, so outputs do not
    // bit-match the scalar tiers — but runs are bit-identical to each
    // other, and the forward is strictly batch-invariant across ALL M
    // (each output element's K-reduction lives in one warp). Requires
    // sm_80+. Two bit-identical tile families (128x128 and 64x64) are
    // routed by shape. Shapes below the 64x64 tile gates return
    // [`Error::Uncovered`] — compose with the scalar tier as needed.

    /// Tensor-core `Y = X @ W (+ bias)`. Covers `M >= 64 && N >= 64`.
    pub fn forward_tc(
        &self,
        y: TypedPtr,
        x: TypedPtr,
        w: TypedPtr,
        bias: Option<CUptr>,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        dispatch::sgemm_bi_forward_tc(
            &self.stream,
            &self.kernels,
            y,
            x,
            w,
            bias.unwrap_or(0),
            dims,
        )
    }

    /// Tensor-core `dW += X^T @ dY` (f32 master accumulator). Covers
    /// `K >= 64 && N >= 64`.
    pub fn backward_dw_tc(
        &self,
        dw: CUptr,
        dy: TypedPtr,
        x_saved: TypedPtr,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        dispatch::sgemm_bi_backward_dw_tc(&self.stream, &self.kernels, dw, dy, x_saved, dims)
    }

    /// Tensor-core `dX = dY @ W^T` (typed RNE overwrite). Covers
    /// `M >= 64 && K >= 64`.
    pub fn backward_dx_tc(
        &self,
        dx: TypedPtr,
        dy: TypedPtr,
        w: TypedPtr,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        dispatch::sgemm_bi_backward_dx_tc(&self.stream, &self.kernels, dx, dy, w, dims)
    }

    // ── scratch & casts ─────────────────────────────────────────────────

    /// Pre-size the typed upcast-fallback scratch so subsequent calls up
    /// to `elems = (a, b, c)` f32 elements never allocate. Required before
    /// CUDA Graph capture: a lazy grow during capture fails the capture,
    /// and a grow after capture frees pointers a captured graph still
    /// references. Size from your largest GEMM:
    /// `(max(M*K, M*N), max(K*N, M*K), max(M*N, M*K))`.
    pub fn presize_upcast_scratch(&self, elems: (usize, usize, usize)) -> Result<()> {
        self.with_upcast_scratch(elems, |_, _, _| Ok(()))
    }

    fn with_upcast_scratch<R>(
        &self,
        elems: (usize, usize, usize),
        f: impl FnOnce(CUptr, CUptr, CUptr) -> Result<R>,
    ) -> Result<R> {
        let sizes = [elems.0, elems.1, elems.2];
        for (cell, &need) in self.upcast_scratch.iter().zip(&sizes) {
            let mut slot = cell.borrow_mut();
            let have = slot.as_ref().map_or(0, |b| b.len());
            let need = need.max(1);
            if have < need {
                *slot = Some(
                    self.stream
                        .alloc_zeros::<f32>(need)
                        .map_err(|e| Error::Cuda(format!("upcast scratch alloc: {e:?}")))?,
                );
            }
        }
        let ptr = |i: usize| -> CUptr {
            use cudarc::driver::DevicePtr;
            let guard = self.upcast_scratch[i].borrow();
            let (p, _g) = guard
                .as_ref()
                .expect("sized above")
                .device_ptr(&self.stream);
            p
        };
        f(ptr(0), ptr(1), ptr(2))
    }

    fn upcast_to_f32(&self, src: TypedPtr, dst: CUptr, n: usize) -> Result<()> {
        let kernel = match src.dtype {
            Dtype::Bf16 => &self.kernels.cast_bf16_to_f32,
            Dtype::F16 => &self.kernels.cast_f16_to_f32,
            Dtype::F32 => return Err(Error::DtypeMismatch("upcast source is already f32")),
        };
        let n_i = n as i32;
        let src_ptr = src.ptr;
        let mut b = self.stream.launch_builder(kernel);
        b.arg(&dst);
        b.arg(&src_ptr);
        b.arg(&n_i);
        unsafe { b.launch(grid_1d(n)) }
            .map(|_| ())
            .map_err(|e| Error::Cuda(format!("upcast launch: {e:?}")))
    }

    fn downcast_from_f32(&self, dst: TypedPtr, src: CUptr, n: usize) -> Result<()> {
        let kernel = match dst.dtype {
            Dtype::Bf16 => &self.kernels.cast_f32_to_bf16,
            Dtype::F16 => &self.kernels.cast_f32_to_f16,
            Dtype::F32 => return Err(Error::DtypeMismatch("downcast target is already f32")),
        };
        let n_i = n as i32;
        let dst_ptr = dst.ptr;
        let mut b = self.stream.launch_builder(kernel);
        b.arg(&dst_ptr);
        b.arg(&src);
        b.arg(&n_i);
        unsafe { b.launch(grid_1d(n)) }
            .map(|_| ())
            .map_err(|e| Error::Cuda(format!("downcast launch: {e:?}")))
    }
}

fn grid_1d(n: usize) -> cudarc::driver::LaunchConfig {
    let threads = 256u32;
    cudarc::driver::LaunchConfig {
        grid_dim: ((n as u32).div_ceil(threads).max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}
