//! Full deterministic training triad on synthetic data, with the
//! determinism and batch-invariance guarantees checked at runtime.
//!
//! Run on a machine with an NVIDIA GPU (Ampere or newer):
//!
//! ```sh
//! cargo run --release --example deterministic_training
//! ```

use cudarc::driver::{CudaContext, DevicePtr, DeviceRepr, ValidAsZeroBits};
use sgemm_bi::{Dtype, SgemmBi, TypedPtr};
use std::sync::Arc;

fn upload<T: DeviceRepr + ValidAsZeroBits + Clone>(
    stream: &Arc<cudarc::driver::CudaStream>,
    data: &[T],
) -> cudarc::driver::CudaSlice<T> {
    let mut buf = stream.alloc_zeros::<T>(data.len()).expect("alloc");
    stream.memcpy_htod(data, &mut buf).expect("upload");
    buf
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA device");
    let stream = ctx.new_stream().expect("stream");
    let engine = SgemmBi::new(&ctx, stream.clone()).expect("engine"); // NVRTC compile, once

    // A linear layer: Y[M,N] = X[M,K] @ W[K,N], plus its backward pair.
    let (m, k, n) = (512usize, 256, 1024);

    // Synthetic bf16 operands (stored as raw u16 bits).
    let xs: Vec<u16> = (0..m * k)
        .map(|i| half::bf16::from_f32(((i % 97) as f32 - 48.0) * 0.01).to_bits())
        .collect();
    let ws: Vec<u16> = (0..k * n)
        .map(|i| half::bf16::from_f32(((i % 89) as f32 - 44.0) * 0.01).to_bits())
        .collect();
    let dys: Vec<u16> = (0..m * n)
        .map(|i| half::bf16::from_f32(((i % 83) as f32 - 41.0) * 0.001).to_bits())
        .collect();

    let x = upload(&stream, &xs);
    let w = upload(&stream, &ws);
    let dy = upload(&stream, &dys);
    let y = stream.alloc_zeros::<u16>(m * n).expect("alloc");
    let dw = stream.alloc_zeros::<f32>(k * n).expect("alloc"); // f32 master accumulator
    let dx = stream.alloc_zeros::<u16>(m * k).expect("alloc");

    let p = |b: &cudarc::driver::CudaSlice<u16>| b.device_ptr(&stream).0;
    let t = |ptr| TypedPtr::new(ptr, Dtype::Bf16);

    // The full training triad. `forward`/`backward_*` cover EVERY shape;
    // the tensor-core variants cover output dims >= 64 and return
    // Error::Uncovered otherwise — compose them like this:
    let fwd = |engine: &SgemmBi| -> sgemm_bi::Result<()> {
        match engine.forward_tc(t(p(&y)), t(p(&x)), t(p(&w)), None, (m, k, n)) {
            Err(sgemm_bi::Error::Uncovered { .. }) => {
                engine.forward(t(p(&y)), t(p(&x)), t(p(&w)), None, (m, k, n))
            }
            other => other,
        }
    };
    fwd(&engine).expect("fwd");
    engine
        .backward_dw_tc(dw.device_ptr(&stream).0, t(p(&dy)), t(p(&x)), (m, k, n))
        .expect("dW");
    engine
        .backward_dx_tc(t(p(&dx)), t(p(&dy)), t(p(&w)), (m, k, n))
        .expect("dX");
    stream.synchronize().expect("sync");

    let mut y_run1 = vec![0u16; m * n];
    stream.memcpy_dtoh(&y, &mut y_run1).expect("download");

    // Guarantee 1 — run-to-run determinism: bit-identical, not "close".
    fwd(&engine).expect("fwd");
    stream.synchronize().expect("sync");
    let mut y_run2 = vec![0u16; m * n];
    stream.memcpy_dtoh(&y, &mut y_run2).expect("download");
    assert_eq!(y_run1, y_run2, "runs must be bit-identical");

    // Guarantee 2 — strict batch invariance of the TC forward: the first
    // 64 rows do not change when the batch shrinks from 512 to 64.
    let y64 = stream.alloc_zeros::<u16>(64 * n).expect("alloc");
    engine
        .forward_tc(t(p(&y64)), t(p(&x)), t(p(&w)), None, (64, k, n))
        .expect("fwd64");
    stream.synchronize().expect("sync");
    let mut y_small = vec![0u16; 64 * n];
    stream.memcpy_dtoh(&y64, &mut y_small).expect("download");
    assert_eq!(
        &y_run1[..64 * n],
        &y_small[..],
        "row 0..64 must not depend on M"
    );

    println!("forward + dW + dX complete: bit-identical across runs,");
    println!("rows 0..64 bit-identical between M=64 and M=512.");
}
