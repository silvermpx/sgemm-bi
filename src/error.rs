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
