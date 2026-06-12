//! C ABI (`capi` feature). A flat `extern "C"` surface over [`SgemmBi`]
//! for non-Rust callers; the companion header is `include/sgemm_bi.h`.
//!
//! Conventions:
//! - Every fallible call returns an `i32` status (`SGB_OK` = 0). On
//!   failure, a human-readable message is stored thread-locally and
//!   retrievable via [`sgb_last_error`].
//! - Device pointers cross the boundary as `u64` (`CUdeviceptr`).
//! - One GEMM descriptor struct ([`SgbGemm`]) serves all six entry
//!   points; the role of `out`/`a`/`b` per operation is documented in
//!   the header.
//! - The engine is NOT thread-safe (interior scratch mutability): guard
//!   it with a mutex or use one engine per thread.

use crate::dtype::{Dtype, TypedPtr};
use crate::engine::SgemmBi;
use crate::error::{Error, Result};
use cudarc::driver::CudaContext;
use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

pub const SGB_OK: i32 = 0;
pub const SGB_ERR_CUDA: i32 = 1;
pub const SGB_ERR_UNCOVERED: i32 = 2;
pub const SGB_ERR_DTYPE: i32 = 3;
pub const SGB_ERR_UNSUPPORTED_ARCH: i32 = 4;
pub const SGB_ERR_INVALID_ARG: i32 = 5;
pub const SGB_ERR_PANIC: i32 = 6;

pub const SGB_F32: i32 = 0;
pub const SGB_BF16: i32 = 1;
pub const SGB_F16: i32 = 2;

/// Opaque engine handle. Owns the CUDA primary context retain, the
/// stream, and the compiled kernels.
pub struct SgbEngine {
    engine: SgemmBi,
    _context: Arc<CudaContext>,
}

/// GEMM descriptor shared by all six operations. Field roles:
///
/// | op            | `out`        | `a`   | `b`       | `bias`     |
/// |---------------|--------------|-------|-----------|------------|
/// | `forward`     | Y `[M,N]`    | X `[M,K]` | W `[K,N]` | f32 `[N]` or 0 |
/// | `backward_dw` | dW `[K,N]` (f32, +=) | dY `[M,N]` | X `[M,K]` | ignored |
/// | `backward_dx` | dX `[M,K]`   | dY `[M,N]` | W `[K,N]` | ignored |
///
/// `dtype` applies to `out`/`a`/`b` (except `backward_dw`'s `out`, which
/// is always an f32 master accumulator). `bias` is always f32.
#[repr(C)]
pub struct SgbGemm {
    pub out: u64,
    pub a: u64,
    pub b: u64,
    pub bias: u64,
    pub m: i64,
    pub k: i64,
    pub n: i64,
    pub dtype: i32,
    pub reserved: i32,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: &str) {
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("invalid error text").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

fn code_of(e: &Error) -> i32 {
    match e {
        Error::Cuda(_) => SGB_ERR_CUDA,
        Error::Uncovered { .. } => SGB_ERR_UNCOVERED,
        Error::DtypeMismatch(_) => SGB_ERR_DTYPE,
        Error::UnsupportedArch { .. } => SGB_ERR_UNSUPPORTED_ARCH,
    }
}

fn finish(r: Result<()>) -> i32 {
    match r {
        Ok(()) => SGB_OK,
        Err(e) => {
            set_error(&e.to_string());
            code_of(&e)
        }
    }
}

fn invalid(msg: &str) -> i32 {
    set_error(msg);
    SGB_ERR_INVALID_ARG
}

/// Runs `f`, converting a Rust panic into `SGB_ERR_PANIC` instead of
/// unwinding (UB) or aborting across the FFI boundary.
fn guarded(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        set_error("internal panic in sgemm-bi");
        SGB_ERR_PANIC
    })
}

struct GemmArgs {
    out: u64,
    a: u64,
    b: u64,
    bias: Option<u64>,
    dims: (usize, usize, usize),
    dtype: Option<Dtype>,
}

/// Validates the raw descriptor. `None` dtype means f32.
fn parse_gemm(g: &SgbGemm) -> std::result::Result<GemmArgs, String> {
    if g.m <= 0 || g.k <= 0 || g.n <= 0 {
        return Err(format!(
            "dimensions must be positive: M={} K={} N={}",
            g.m, g.k, g.n
        ));
    }
    if g.out == 0 || g.a == 0 || g.b == 0 {
        return Err("out/a/b device pointers must be non-null".into());
    }
    let dtype = match g.dtype {
        SGB_F32 => None,
        SGB_BF16 => Some(Dtype::Bf16),
        SGB_F16 => Some(Dtype::F16),
        other => return Err(format!("unknown dtype code {other}")),
    };
    Ok(GemmArgs {
        out: g.out,
        a: g.a,
        b: g.b,
        bias: (g.bias != 0).then_some(g.bias),
        dims: (g.m as usize, g.k as usize, g.n as usize),
        dtype,
    })
}

/// Dereferences the engine and descriptor, then runs `f` panic-guarded.
///
/// # Safety
/// `eng` must be a live pointer from `sgb_engine_create`; `gemm` must
/// point to a valid descriptor.
unsafe fn with_gemm(
    eng: *const SgbEngine,
    gemm: *const SgbGemm,
    f: impl FnOnce(&SgbEngine, GemmArgs) -> Result<()>,
) -> i32 {
    if eng.is_null() || gemm.is_null() {
        return invalid("null engine or descriptor pointer");
    }
    let (eng, gemm) = unsafe { (&*eng, &*gemm) };
    guarded(|| match parse_gemm(gemm) {
        Ok(args) => finish(f(eng, args)),
        Err(msg) => invalid(&msg),
    })
}

/// Last error message for the current thread, NUL-terminated. Valid
/// until the next failing call on the same thread. Never null.
/// # Safety
/// Always safe to call; the returned pointer must not be written
/// through or freed, and is invalidated by the next failing call on
/// the same thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Creates an engine on `device_ordinal` (retains the device's primary
/// context, creates a dedicated non-blocking stream, compiles kernels).
/// On success writes the handle to `*out` and returns `SGB_OK`.
/// # Safety
/// `out` must be a valid pointer to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_engine_create(device_ordinal: i32, out: *mut *mut SgbEngine) -> i32 {
    if out.is_null() {
        return invalid("null output handle pointer");
    }
    if device_ordinal < 0 {
        return invalid("device ordinal must be non-negative");
    }
    guarded(|| {
        let built = (|| -> Result<Box<SgbEngine>> {
            let context = CudaContext::new(device_ordinal as usize)
                .map_err(|e| Error::Cuda(format!("create context: {e:?}")))?;
            let stream = context
                .new_stream()
                .map_err(|e| Error::Cuda(format!("create stream: {e:?}")))?;
            let engine = SgemmBi::new(&context, stream)?;
            Ok(Box::new(SgbEngine {
                engine,
                _context: context,
            }))
        })();
        match built {
            Ok(handle) => {
                unsafe { *out = Box::into_raw(handle) };
                SGB_OK
            }
            Err(e) => {
                set_error(&e.to_string());
                code_of(&e)
            }
        }
    })
}

/// Destroys an engine. Passing null is a no-op. The handle must not be
/// used afterwards.
/// # Safety
/// `eng` must be null or a handle from [`sgb_engine_create`] that has
/// not been destroyed; no other thread may be using it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_engine_destroy(eng: *mut SgbEngine) {
    if !eng.is_null() {
        drop(unsafe { Box::from_raw(eng) });
    }
}

/// Blocks until all work enqueued by this engine has completed.
/// # Safety
/// `eng` must be null or a live handle from [`sgb_engine_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_engine_synchronize(eng: *const SgbEngine) -> i32 {
    if eng.is_null() {
        return invalid("null engine pointer");
    }
    let eng = unsafe { &*eng };
    guarded(|| {
        finish(
            eng.engine
                .stream()
                .synchronize()
                .map_err(|e| Error::Cuda(format!("synchronize: {e:?}"))),
        )
    })
}

/// Raw `CUstream` handle the engine enqueues on. The stream is created
/// with the non-blocking flag: it does NOT implicitly order against the
/// legacy default stream — order operand transfers with events or run
/// them on this stream.
/// # Safety
/// `eng` must be null or a live handle from [`sgb_engine_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_engine_stream(eng: *const SgbEngine) -> u64 {
    if eng.is_null() {
        return 0;
    }
    let eng = unsafe { &*eng };
    eng.engine.stream().cu_stream() as u64
}

/// `out = a @ b (+ bias)`. f32 and typed dtypes, full shape coverage.
/// # Safety
/// `eng` must be null or a live handle from [`sgb_engine_create`];
/// `gemm` must be null or point to a valid descriptor whose device
/// pointers are allocated on the engine's context with the documented
/// shapes and dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_forward(eng: *const SgbEngine, gemm: *const SgbGemm) -> i32 {
    unsafe {
        with_gemm(eng, gemm, |e, g| match g.dtype {
            None => e.engine.forward_f32(g.out, g.a, g.b, g.bias, g.dims),
            Some(dt) => e.engine.forward(
                TypedPtr::new(g.out, dt),
                TypedPtr::new(g.a, dt),
                TypedPtr::new(g.b, dt),
                g.bias,
                g.dims,
            ),
        })
    }
}

/// `out (f32) += a^T-side weight gradient`: `dW[K,N] += X^T @ dY`.
/// `out` is ALWAYS f32 regardless of `dtype` (master accumulator).
/// # Safety
/// `eng` must be null or a live handle from [`sgb_engine_create`];
/// `gemm` must be null or point to a valid descriptor whose device
/// pointers are allocated on the engine's context with the documented
/// shapes and dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_backward_dw(eng: *const SgbEngine, gemm: *const SgbGemm) -> i32 {
    unsafe {
        with_gemm(eng, gemm, |e, g| match g.dtype {
            None => e.engine.backward_dw_f32(g.out, g.a, g.b, g.dims),
            Some(dt) => e.engine.backward_dw(
                g.out,
                TypedPtr::new(g.a, dt),
                TypedPtr::new(g.b, dt),
                g.dims,
            ),
        })
    }
}

/// `out = dX[M,K] = dY @ W^T` (overwrite; typed dtypes RNE-downcast).
/// # Safety
/// `eng` must be null or a live handle from [`sgb_engine_create`];
/// `gemm` must be null or point to a valid descriptor whose device
/// pointers are allocated on the engine's context with the documented
/// shapes and dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_backward_dx(eng: *const SgbEngine, gemm: *const SgbGemm) -> i32 {
    unsafe {
        with_gemm(eng, gemm, |e, g| match g.dtype {
            None => e.engine.backward_dx_f32(g.out, g.a, g.b, g.dims),
            Some(dt) => e.engine.backward_dx(
                TypedPtr::new(g.out, dt),
                TypedPtr::new(g.a, dt),
                TypedPtr::new(g.b, dt),
                g.dims,
            ),
        })
    }
}

fn require_typed(dtype: Option<Dtype>) -> Result<Dtype> {
    dtype.ok_or(Error::DtypeMismatch(
        "tensor-core tier requires bf16 or f16",
    ))
}

/// Tensor-core forward (bf16/f16 only, `M >= 128 && N >= 128`, separate
/// numeric contract). Uncovered shapes return `SGB_ERR_UNCOVERED`.
/// # Safety
/// `eng` must be null or a live handle from [`sgb_engine_create`];
/// `gemm` must be null or point to a valid descriptor whose device
/// pointers are allocated on the engine's context with the documented
/// shapes and dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_forward_tc(eng: *const SgbEngine, gemm: *const SgbGemm) -> i32 {
    unsafe {
        with_gemm(eng, gemm, |e, g| {
            let dt = require_typed(g.dtype)?;
            e.engine.forward_tc(
                TypedPtr::new(g.out, dt),
                TypedPtr::new(g.a, dt),
                TypedPtr::new(g.b, dt),
                g.bias,
                g.dims,
            )
        })
    }
}

/// Tensor-core `dW += X^T @ dY` (f32 master accumulator; bf16/f16
/// operands, `K >= 128 && N >= 128`).
/// # Safety
/// `eng` must be null or a live handle from [`sgb_engine_create`];
/// `gemm` must be null or point to a valid descriptor whose device
/// pointers are allocated on the engine's context with the documented
/// shapes and dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_backward_dw_tc(eng: *const SgbEngine, gemm: *const SgbGemm) -> i32 {
    unsafe {
        with_gemm(eng, gemm, |e, g| {
            let dt = require_typed(g.dtype)?;
            e.engine.backward_dw_tc(
                g.out,
                TypedPtr::new(g.a, dt),
                TypedPtr::new(g.b, dt),
                g.dims,
            )
        })
    }
}

/// Tensor-core `dX = dY @ W^T` (bf16/f16, `M >= 128 && K >= 128`).
/// # Safety
/// `eng` must be null or a live handle from [`sgb_engine_create`];
/// `gemm` must be null or point to a valid descriptor whose device
/// pointers are allocated on the engine's context with the documented
/// shapes and dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_backward_dx_tc(eng: *const SgbEngine, gemm: *const SgbGemm) -> i32 {
    unsafe {
        with_gemm(eng, gemm, |e, g| {
            let dt = require_typed(g.dtype)?;
            e.engine.backward_dx_tc(
                TypedPtr::new(g.out, dt),
                TypedPtr::new(g.a, dt),
                TypedPtr::new(g.b, dt),
                g.dims,
            )
        })
    }
}

/// Pre-sizes the typed upcast-fallback scratch (required before CUDA
/// Graph capture). Element counts are f32 elements; size from your
/// largest GEMM as `(max(M*K, M*N), max(K*N, M*K), max(M*N, M*K))`.
/// # Safety
/// `eng` must be null or a live handle from [`sgb_engine_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgb_presize_upcast_scratch(
    eng: *const SgbEngine,
    a_elems: i64,
    b_elems: i64,
    c_elems: i64,
) -> i32 {
    if eng.is_null() {
        return invalid("null engine pointer");
    }
    if a_elems < 0 || b_elems < 0 || c_elems < 0 {
        return invalid("scratch element counts must be non-negative");
    }
    let eng = unsafe { &*eng };
    guarded(|| {
        finish(eng.engine.presize_upcast_scratch((
            a_elems as usize,
            b_elems as usize,
            c_elems as usize,
        )))
    })
}
