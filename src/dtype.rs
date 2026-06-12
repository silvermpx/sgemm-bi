//! Activation dtypes and the typed device-pointer handle.

/// Element type of a GEMM operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Bf16,
    F16,
}

impl Dtype {
    /// Size of one element in bytes.
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::Bf16 | Dtype::F16 => 2,
        }
    }
}

/// Raw CUDA device pointer tagged with its element dtype.
///
/// The engine never owns memory it computes on — callers pass device
/// pointers allocated on the SAME stream the engine was built with (or
/// properly ordered against it).
#[derive(Debug, Clone, Copy)]
pub struct TypedPtr {
    pub ptr: cudarc::driver::sys::CUdeviceptr,
    pub dtype: Dtype,
}

impl TypedPtr {
    pub fn new(ptr: cudarc::driver::sys::CUdeviceptr, dtype: Dtype) -> Self {
        Self { ptr, dtype }
    }
}
