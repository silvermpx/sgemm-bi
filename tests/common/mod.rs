//! Shared test utilities: deterministic data generation, device buffers,
//! bit comparison.

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use half::{bf16, f16};
use sgemm_bi::{Dtype, SgemmBi};
use std::sync::Arc;

pub struct Harness {
    pub stream: Arc<CudaStream>,
    pub engine: SgemmBi,
}

impl Harness {
    pub fn new() -> Self {
        let context = CudaContext::new(0).expect("CUDA device 0");
        // Single-stream usage: every op (uploads included) is enqueued on
        // this stream, so no cross-stream ordering is needed.
        unsafe { context.disable_event_tracking() };
        let stream = context.new_stream().expect("stream");
        let engine = SgemmBi::new(&context, stream.clone()).expect("engine");
        Self { stream, engine }
    }

    /// Upload f32 values as `dt` storage; returns the buffer and its ptr.
    pub fn upload(&self, data: &[f32], dt: Dtype) -> (CudaSlice<u8>, u64) {
        let bytes: Vec<u8> = match dt {
            Dtype::F32 => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Dtype::Bf16 => data
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
                .collect(),
            Dtype::F16 => data
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                .collect(),
        };
        let mut buf = self.stream.alloc_zeros::<u8>(bytes.len()).expect("alloc");
        self.stream.memcpy_htod(&bytes, &mut buf).expect("upload");
        let ptr = {
            let (p, _g) = buf.device_ptr(&self.stream);
            p
        };
        (buf, ptr)
    }

    /// Allocate a zeroed `dt` buffer of `n` elements.
    pub fn zeros(&self, n: usize, dt: Dtype) -> (CudaSlice<u8>, u64) {
        let buf = self
            .stream
            .alloc_zeros::<u8>(n * dt.size_bytes())
            .expect("alloc");
        let ptr = {
            let (p, _g) = buf.device_ptr(&self.stream);
            p
        };
        (buf, ptr)
    }

    /// Download a `dt` buffer to f32 host values.
    pub fn download(&self, buf: &CudaSlice<u8>, n: usize, dt: Dtype) -> Vec<f32> {
        self.stream.synchronize().expect("sync");
        let mut bytes = vec![0u8; n * dt.size_bytes()];
        self.stream.memcpy_dtoh(buf, &mut bytes).expect("download");
        match dt {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::Bf16 => bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        }
    }
}

/// Deterministic pseudo-random fill (xorshift), range ~[-scale/2, scale/2].
pub fn det(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * scale
        })
        .collect()
}

/// Round f32 host values onto the `dt` representable grid.
pub fn quantize(v: &[f32], dt: Dtype) -> Vec<f32> {
    match dt {
        Dtype::F32 => v.to_vec(),
        Dtype::Bf16 => v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
        Dtype::F16 => v.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
    }
}

/// Bitwise comparison; panics with index context on the first class of
/// mismatch.
pub fn assert_bits(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}: bit mismatch at {i}: got {g:?} want {w:?}"
        );
    }
}
