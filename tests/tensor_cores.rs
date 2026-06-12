//! Tensor-core tier contracts: correctness vs the f32 reference,
//! run-to-run determinism, strict all-M batch invariance, and a
//! speed comparison against the scalar tier (`--ignored`).

mod common;

use common::{Harness, assert_bits, det, quantize};

fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}
use sgemm_bi::{Dtype, TypedPtr};

/// TC forward tracks the f32 reference on quantized inputs. A fragment
/// layout or staging bug shows up as garbage, not noise, so a cosine
/// bound is a sharp check.
#[test]
fn tc_forward_matches_f32_reference() {
    let h = Harness::new();
    for dt in [Dtype::Bf16, Dtype::F16] {
        for (m, k, n) in [
            (256usize, 384usize, 512usize),
            (256, 100, 512),  // K % 32 != 0
            (300, 768, 3072), // tails on every axis
        ] {
            let qx = quantize(&det(m * k, 11, 1.0), dt);
            let qw = quantize(&det(k * n, 22, 0.5), dt);
            let bias = det(n, 33, 0.25);
            let (_bb, bp) = h.upload(&bias, Dtype::F32);

            let (_x32, x32p) = h.upload(&qx, Dtype::F32);
            let (_w32, w32p) = h.upload(&qw, Dtype::F32);
            let (y32, y32p) = h.zeros(m * n, Dtype::F32);
            h.engine
                .forward_f32(y32p, x32p, w32p, Some(bp), (m, k, n))
                .unwrap();
            let reference = h.download(&y32, m * n, Dtype::F32);

            let (_xt, xtp) = h.upload(&qx, dt);
            let (_wt, wtp) = h.upload(&qw, dt);
            let (yt, ytp) = h.zeros(m * n, dt);
            h.engine
                .forward_tc(
                    TypedPtr::new(ytp, dt),
                    TypedPtr::new(xtp, dt),
                    TypedPtr::new(wtp, dt),
                    Some(bp),
                    (m, k, n),
                )
                .unwrap();
            let got = h.download(&yt, m * n, dt);
            let cos = cos_sim(&got, &reference);
            assert!(
                cos > 0.9999,
                "TC fwd {dt:?} M{m} K{k} N{n}: cos {cos} vs f32 reference"
            );
        }
    }
}

/// TC dW accumulates in f32 (tight agreement); TC dX within dtype noise.
#[test]
fn tc_backward_matches_f32_reference() {
    let h = Harness::new();
    for dt in [Dtype::Bf16, Dtype::F16] {
        let (m, k, n) = (300usize, 768usize, 512usize);
        let qx = quantize(&det(m * k, 44, 1.0), dt);
        let qdy = quantize(&det(m * n, 55, 0.5), dt);
        let qw = quantize(&det(k * n, 77, 0.5), dt);

        let (_x32, x32p) = h.upload(&qx, Dtype::F32);
        let (_dy32, dy32p) = h.upload(&qdy, Dtype::F32);
        let (_w32, w32p) = h.upload(&qw, Dtype::F32);
        let (dwr, dwrp) = h.zeros(k * n, Dtype::F32);
        let (dxr, dxrp) = h.zeros(m * k, Dtype::F32);
        h.engine
            .backward_dw_f32(dwrp, dy32p, x32p, (m, k, n))
            .unwrap();
        h.engine
            .backward_dx_f32(dxrp, dy32p, w32p, (m, k, n))
            .unwrap();
        let dw_want = h.download(&dwr, k * n, Dtype::F32);
        let dx_want = h.download(&dxr, m * k, Dtype::F32);

        let (_xt, xtp) = h.upload(&qx, dt);
        let (_dyt, dytp) = h.upload(&qdy, dt);
        let (_wt, wtp) = h.upload(&qw, dt);
        let (dwt, dwtp) = h.zeros(k * n, Dtype::F32);
        let (dxt, dxtp) = h.zeros(m * k, dt);
        h.engine
            .backward_dw_tc(
                dwtp,
                TypedPtr::new(dytp, dt),
                TypedPtr::new(xtp, dt),
                (m, k, n),
            )
            .unwrap();
        h.engine
            .backward_dx_tc(
                TypedPtr::new(dxtp, dt),
                TypedPtr::new(dytp, dt),
                TypedPtr::new(wtp, dt),
                (m, k, n),
            )
            .unwrap();
        let dw_cos = cos_sim(&h.download(&dwt, k * n, Dtype::F32), &dw_want);
        let dx_cos = cos_sim(&h.download(&dxt, m * k, dt), &dx_want);
        assert!(dw_cos > 0.99999, "TC dW {dt:?}: cos {dw_cos}");
        assert!(dx_cos > 0.9999, "TC dX {dt:?}: cos {dx_cos}");
    }
}

/// TC runs are bit-identical to each other, and row 0 of the forward is
/// bit-identical across ALL batch sizes (each output element's entire
/// K-reduction lives in one warp, independent of grid shape).
#[test]
fn tc_forward_is_deterministic_and_strictly_batch_invariant() {
    let h = Harness::new();
    let (k, n) = (768usize, 3072usize);
    for dt in [Dtype::Bf16, Dtype::F16] {
        let row = quantize(&det(k, 42, 1.0), dt);
        let w = quantize(&det(k * n, 43, 0.5), dt);
        let (_wb, wp) = h.upload(&w, dt);

        let run = |m: usize, seed: u32| -> Vec<f32> {
            let mut x = quantize(&det(m * k, seed, 1.0), dt);
            x[..k].copy_from_slice(&row);
            let (_xb, xp) = h.upload(&x, dt);
            let (yb, yp) = h.zeros(m * n, dt);
            h.engine
                .forward_tc(
                    TypedPtr::new(yp, dt),
                    TypedPtr::new(xp, dt),
                    TypedPtr::new(wp, dt),
                    None,
                    (m, k, n),
                )
                .unwrap();
            h.download(&yb, m * n, dt)[..n].to_vec()
        };

        assert_bits(&format!("{dt:?} TC repeat"), &run(256, 100), &run(256, 100));
        assert_bits(
            &format!("{dt:?} TC M128 vs M512"),
            &run(512, 300),
            &run(128, 200),
        );
        assert_bits(
            &format!("{dt:?} TC M128 vs M2048"),
            &run(2048, 400),
            &run(128, 200),
        );
    }
}

/// Wall-clock: TC tier vs the scalar typed tier on tile-friendly shapes.
#[test]
#[ignore] // benchmark — run explicitly on a quiet GPU
fn bench_tc_vs_scalar() {
    use std::time::Instant;
    let h = Harness::new();
    let dt = Dtype::Bf16;
    for (m, k, n) in [
        (2048usize, 768usize, 3072usize),
        (4096, 1536, 3072),
        (2048, 768, 512),
    ] {
        let qx = quantize(&det(m * k, 11, 1.0), dt);
        let qw = quantize(&det(k * n, 22, 0.5), dt);
        let (_xt, xtp) = h.upload(&qx, dt);
        let (_wt, wtp) = h.upload(&qw, dt);
        let (_yt, ytp) = h.zeros(m * n, dt);
        let (y, x, w) = (
            TypedPtr::new(ytp, dt),
            TypedPtr::new(xtp, dt),
            TypedPtr::new(wtp, dt),
        );

        let iters = 50;
        let time = |f: &dyn Fn()| -> f64 {
            for _ in 0..3 {
                f();
            }
            h.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                f();
            }
            h.stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / iters as f64
        };
        let scalar = time(&|| h.engine.forward(y, x, w, None, (m, k, n)).unwrap());
        let tc = time(&|| h.engine.forward_tc(y, x, w, None, (m, k, n)).unwrap());
        eprintln!(
            "[M{m} K{k} N{n}] scalar {scalar:.1} us | TC {tc:.1} us | {:.2}x",
            scalar / tc
        );
    }
}
