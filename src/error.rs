//! Error type. NOTE on the determinism contract: unlike an in-app engine,
//! a library must not panic on uncovered shapes — every dispatcher returns
//! [`Error::Uncovered`] instead, and the full-coverage entry points
//! ([`crate::SgemmBi::forward`] etc.) guarantee it never escapes to the
//! caller for valid shapes.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// NVRTC compilation / module load / launch failure.
    Cuda(String),
    /// No deterministic bucket covers this shape in the called tier.
    /// The full-coverage entries never return this; the tier-specific
    /// entries (`*_tc`, native-bucket probes) do.
    Uncovered {
        op: &'static str,
        m: usize,
        k: usize,
        n: usize,
    },
    /// Operand dtypes disagree where the kernel requires them equal.
    DtypeMismatch(&'static str),
    /// Device compute capability below the supported minimum. The kernel
    /// blob uses `cp.async` and native bf16 throughout, so the whole
    /// engine requires Ampere or newer (sm_80+); this is checked once at
    /// construction instead of surfacing as an opaque NVRTC failure.
    UnsupportedArch { major: u32, minor: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cuda(s) => write!(f, "CUDA error: {s}"),
            Error::Uncovered { op, m, k, n } => write!(
                f,
                "{op}: no deterministic bucket for shape M={m} K={k} N={n} in this tier"
            ),
            Error::DtypeMismatch(what) => write!(f, "dtype mismatch: {what}"),
            Error::UnsupportedArch { major, minor } => write!(
                f,
                "unsupported GPU architecture sm_{major}{minor}: sgemm-bi requires \
                 Ampere or newer (sm_80+)"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<cudarc::driver::DriverError> for Error {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Error::Cuda(format!("{e:?}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;
