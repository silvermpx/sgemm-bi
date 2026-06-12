//! Python extension module `sgemm_bi._sgemm_bi`.
//!
//! Exposes one class, [`Engine`], whose methods take raw device
//! pointers (`tensor.data_ptr()`) plus the caller's current CUDA stream
//! handle (`torch.cuda.current_stream().cuda_stream`). The engine
//! enqueues on its own non-blocking stream and bridges ordering with
//! CUDA events: it waits for the caller's stream before the GEMM and
//! makes the caller's stream wait for the GEMM afterwards — fully
//! asynchronous, no host synchronization.
//!
//! Threading: PyTorch runs `backward` on a dedicated autograd thread,
//! so the engine state lives behind a `Mutex` — calls from the main
//! thread (forward) and the autograd thread (backward) serialize, which
//! also keeps the event-bridge reuse and the engine's interior scratch
//! safe.

use cudarc::driver::sys::{CUevent, CUevent_flags, CUevent_wait_flags, CUstream};
use cudarc::driver::{CudaContext, result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use sgemm_bi::{Dtype, SgemmBi, TypedPtr};
use std::sync::{Arc, Mutex};

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("sgemm-bi: {e}"))
}

fn errd<E: std::fmt::Debug>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("sgemm-bi: CUDA error: {e:?}"))
}

fn parse_dtype(s: &str) -> PyResult<Option<Dtype>> {
    match s {
        "f32" | "float32" => Ok(None),
        "bf16" | "bfloat16" => Ok(Some(Dtype::Bf16)),
        "f16" | "float16" | "half" => Ok(Some(Dtype::F16)),
        other => Err(PyRuntimeError::new_err(format!(
            "sgemm-bi: unsupported dtype '{other}' (expected float32, bfloat16, or float16)"
        ))),
    }
}

struct EngineInner {
    engine: SgemmBi,
    context: Arc<CudaContext>,
    /// Bridge events, created once and re-recorded per call (allowed by
    /// the driver; the mutex serializes enqueue order, making reuse safe).
    ev_in: CUevent,
    ev_out: CUevent,
    raw_stream: CUstream,
}

// Raw driver handles (CUevent/CUstream) are valid from any thread; the
// engine's interior scratch is only touched under the outer Mutex.
unsafe impl Send for EngineInner {}

impl EngineInner {
    /// Orders the engine stream after `caller_stream`, runs `f` (which
    /// enqueues on the engine stream), then orders `caller_stream`
    /// after the engine stream.
    fn bridged(
        &self,
        caller_stream: u64,
        f: impl FnOnce(&SgemmBi) -> sgemm_bi::Result<()>,
    ) -> PyResult<()> {
        // PyTorch calls backward from a dedicated autograd thread that has
        // no CUDA context bound; raw driver calls below require one.
        self.context.bind_to_thread().map_err(errd)?;
        let caller = caller_stream as CUstream;
        let same = caller == self.raw_stream;
        unsafe {
            if !same {
                result::event::record(self.ev_in, caller).map_err(errd)?;
                result::stream::wait_event(
                    self.raw_stream,
                    self.ev_in,
                    CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                )
                .map_err(errd)?;
            }
            // Run even when `f` failed: a dispatch error can leave part of
            // the GEMM enqueued, and without the back-wait a later tensor
            // free on the caller stream could race those kernels.
            let outcome = f(&self.engine).map_err(err);
            if !same {
                result::event::record(self.ev_out, self.raw_stream).map_err(errd)?;
                result::stream::wait_event(
                    caller,
                    self.ev_out,
                    CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                )
                .map_err(errd)?;
            }
            outcome
        }
    }
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        unsafe {
            let _ = result::event::destroy(self.ev_in);
            let _ = result::event::destroy(self.ev_out);
        }
    }
}

/// Deterministic GEMM engine bound to one CUDA device.
///
/// Methods enqueue asynchronously and order themselves against the
/// stream handle passed per call. Thread-safe: calls from different
/// threads (e.g. torch's autograd thread) serialize on an internal
/// mutex.
#[pyclass]
struct Engine {
    inner: Mutex<EngineInner>,
}

impl Engine {
    fn locked(&self) -> PyResult<std::sync::MutexGuard<'_, EngineInner>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("sgemm-bi: engine mutex poisoned"))
    }
}

struct Gemm {
    out: u64,
    a: u64,
    b: u64,
    bias: Option<u64>,
    dims: (usize, usize, usize),
    dtype: Option<Dtype>,
}

#[pymethods]
impl Engine {
    /// Builds an engine on `device` (compiles kernels via NVRTC for the
    /// native architecture — takes a few seconds, do it once).
    #[new]
    fn new(py: Python<'_>, device: usize) -> PyResult<Self> {
        // The NVRTC compile takes seconds — don't stall other Python
        // threads on the GIL for it.
        py.detach(|| {
            let context = CudaContext::new(device).map_err(errd)?;
            let stream = context.new_stream().map_err(errd)?;
            let raw_stream = stream.cu_stream();
            let engine = SgemmBi::new(&context, stream).map_err(err)?;
            let flags = CUevent_flags::CU_EVENT_DISABLE_TIMING;
            let ev_in = result::event::create(flags).map_err(errd)?;
            let ev_out = result::event::create(flags).map_err(errd)?;
            Ok(Self {
                inner: Mutex::new(EngineInner {
                    engine,
                    context,
                    ev_in,
                    ev_out,
                    raw_stream,
                }),
            })
        })
    }

    /// `out[M,N] = a[M,K] @ b[K,N] (+ bias[N], f32)`.
    ///
    /// `tensor_cores=True` tries the tensor-core tier first (bf16/f16,
    /// 64-tile shapes: both output dims >= 64) and falls back to the scalar tier — tier
    /// choice depends only on shape and dtype, never on data, so
    /// determinism is preserved.
    #[pyo3(signature = (ptrs, dims, dtype, stream, tensor_cores=false))]
    fn forward(
        &self,
        py: Python<'_>,
        ptrs: (u64, u64, u64, Option<u64>),
        dims: (usize, usize, usize),
        dtype: &str,
        stream: u64,
        tensor_cores: bool,
    ) -> PyResult<()> {
        let g = Gemm {
            out: ptrs.0,
            a: ptrs.1,
            b: ptrs.2,
            bias: ptrs.3,
            dims,
            dtype: parse_dtype(dtype)?,
        };
        // Detached: another thread may hold the mutex through a long
        // synchronize — don't make that wait a GIL stall.
        py.detach(|| {
            self.locked()?.bridged(stream, |e| match g.dtype {
                None => e.forward_f32(g.out, g.a, g.b, g.bias, g.dims),
                Some(dt) => {
                    let (y, x, w) = (
                        TypedPtr::new(g.out, dt),
                        TypedPtr::new(g.a, dt),
                        TypedPtr::new(g.b, dt),
                    );
                    if tensor_cores {
                        match e.forward_tc(y, x, w, g.bias, g.dims) {
                            Err(sgemm_bi::Error::Uncovered { .. }) => {}
                            other => return other,
                        }
                    }
                    e.forward(y, x, w, g.bias, g.dims)
                }
            })
        })
    }

    /// `dw[K,N] (f32) += X^T @ dY`. `dw` is ALWAYS f32 (master
    /// accumulator), zero it before the first accumulation.
    #[pyo3(signature = (ptrs, dims, dtype, stream, tensor_cores=false))]
    fn backward_dw(
        &self,
        py: Python<'_>,
        ptrs: (u64, u64, u64),
        dims: (usize, usize, usize),
        dtype: &str,
        stream: u64,
        tensor_cores: bool,
    ) -> PyResult<()> {
        let (dw, dy, x) = ptrs;
        let (m, k, n) = dims;
        let dtype = parse_dtype(dtype)?;
        py.detach(|| {
            self.locked()?.bridged(stream, |e| match dtype {
                None => e.backward_dw_f32(dw, dy, x, (m, k, n)),
                Some(dt) => {
                    let (dyp, xp) = (TypedPtr::new(dy, dt), TypedPtr::new(x, dt));
                    if tensor_cores {
                        match e.backward_dw_tc(dw, dyp, xp, (m, k, n)) {
                            Err(sgemm_bi::Error::Uncovered { .. }) => {}
                            other => return other,
                        }
                    }
                    e.backward_dw(dw, dyp, xp, (m, k, n))
                }
            })
        })
    }

    /// `dx[M,K] = dy[M,N] @ w[K,N]^T` (overwrite).
    #[pyo3(signature = (ptrs, dims, dtype, stream, tensor_cores=false))]
    fn backward_dx(
        &self,
        py: Python<'_>,
        ptrs: (u64, u64, u64),
        dims: (usize, usize, usize),
        dtype: &str,
        stream: u64,
        tensor_cores: bool,
    ) -> PyResult<()> {
        let (dx, dy, w) = ptrs;
        let (m, k, n) = dims;
        let dtype = parse_dtype(dtype)?;
        py.detach(|| {
            self.locked()?.bridged(stream, |e| match dtype {
                None => e.backward_dx_f32(dx, dy, w, (m, k, n)),
                Some(dt) => {
                    let (dxp, dyp, wp) = (
                        TypedPtr::new(dx, dt),
                        TypedPtr::new(dy, dt),
                        TypedPtr::new(w, dt),
                    );
                    if tensor_cores {
                        match e.backward_dx_tc(dxp, dyp, wp, (m, k, n)) {
                            Err(sgemm_bi::Error::Uncovered { .. }) => {}
                            other => return other,
                        }
                    }
                    e.backward_dx(dxp, dyp, wp, (m, k, n))
                }
            })
        })
    }

    /// Pre-sizes the typed upcast-fallback scratch (f32 element counts).
    /// Required before CUDA Graph capture.
    fn presize_upcast_scratch(
        &self,
        py: Python<'_>,
        a_elems: usize,
        b_elems: usize,
        c_elems: usize,
    ) -> PyResult<()> {
        py.detach(|| {
            let inner = self.locked()?;
            inner.context.bind_to_thread().map_err(errd)?;
            inner
                .engine
                .presize_upcast_scratch((a_elems, b_elems, c_elems))
                .map_err(err)
        })
    }

    /// Blocks until all engine work has completed.
    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.locked()?.engine.stream().synchronize().map_err(errd))
    }
}

#[pymodule]
fn _sgemm_bi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}
