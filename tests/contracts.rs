//! Core engine contracts: run-to-run determinism, the typed bit contract
//! (typed output == upcast → f32 tier → RNE downcast), batch invariance,
//! and gate-boundary coverage of the typed dispatch.

mod common;

use common::{Harness, assert_bits, det, quantize};
use sgemm_bi::{Dtype, TypedPtr};

/// f32 forward/dW/dX produce bit-identical results across two launches.
#[test]
fn f32_triad_is_deterministic_across_launches() {
    let h = Harness::new();
    let (m, k, n) = (256usize, 384usize, 512usize);
    let x = det(m * k, 11, 1.0);
    let w = det(k * n, 22, 0.5);
    let dy = det(m * n, 55, 0.5);
    let (_xb, xp) = h.upload(&x, Dtype::F32);
    let (_wb, wp) = h.upload(&w, Dtype::F32);
    let (_dyb, dyp) = h.upload(&dy, Dtype::F32);

    let run = || {
        let (yb, yp) = h.zeros(m * n, Dtype::F32);
        let (dwb, dwp) = h.zeros(k * n, Dtype::F32);
        let (dxb, dxp) = h.zeros(m * k, Dtype::F32);
        h.engine.forward_f32(yp, xp, wp, None, (m, k, n)).unwrap();
        h.engine.backward_dw_f32(dwp, dyp, xp, (m, k, n)).unwrap();
        h.engine.backward_dx_f32(dxp, dyp, wp, (m, k, n)).unwrap();
        (
            h.download(&yb, m * n, Dtype::F32),
            h.download(&dwb, k * n, Dtype::F32),
            h.download(&dxb, m * k, Dtype::F32),
        )
    };
    let (y1, dw1, dx1) = run();
    let (y2, dw2, dx2) = run();
    assert_bits("f32 forward", &y2, &y1);
    assert_bits("f32 dW", &dw2, &dw1);
    assert_bits("f32 dX", &dx2, &dx1);
}

/// The typed bit contract, swept across dispatch-gate boundaries: for every
/// shape, the typed result must equal "quantize inputs, run the f32 tier,
/// RNE-downcast the output" bit for bit — whether the shape lands in a
/// native typed bucket or the upcast fallback.
#[test]
fn typed_tier_bit_matches_f32_reference_across_gates() {
    let h = Harness::new();
    let dt = Dtype::Bf16;
    let mut shapes: Vec<(usize, usize, usize)> = Vec::new();
    // Gate boundaries: slim/big split (512), split-K caps (1024 / 2048),
    // gap-fill edge (128), narrow edge (127/128), plus K tails.
    for m in [8usize, 64, 128, 511, 512, 513, 1024, 1025, 2048] {
        for n in [1usize, 80, 127, 128, 512, 513, 2048, 2052, 3072] {
            shapes.push((m, 384, n));
        }
    }
    for kk in [8usize, 96, 100, 768, 2560] {
        shapes.push((512, kk, 3072));
    }
    for (m, k, n) in shapes {
        // GEMV gate needs K >= 32 on N=1; skip the one uncoverable combo.
        if n == 1 && k < 32 {
            continue;
        }
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
        let reference = quantize(&h.download(&y32, m * n, Dtype::F32), dt);

        let (_xt, xtp) = h.upload(&qx, dt);
        let (_wt, wtp) = h.upload(&qw, dt);
        let (yt, ytp) = h.zeros(m * n, dt);
        h.engine
            .forward(
                TypedPtr::new(ytp, dt),
                TypedPtr::new(xtp, dt),
                TypedPtr::new(wtp, dt),
                Some(bp),
                (m, k, n),
            )
            .unwrap_or_else(|e| panic!("typed fwd M{m} K{k} N{n}: {e}"));
        let got = h.download(&yt, m * n, dt);
        assert_bits(&format!("typed fwd M{m} K{k} N{n}"), &got, &reference);
    }
}

/// Typed dW accumulates into f32 and must bit-match the f32 tier run on
/// quantized inputs; typed dX must match its RNE-downcast.
#[test]
fn typed_backward_bit_matches_f32_reference() {
    let h = Harness::new();
    for dt in [Dtype::Bf16, Dtype::F16] {
        for (m, k, n) in [
            (256usize, 96usize, 80usize),
            (256, 384, 512),
            (2048, 768, 512),
            (250, 100, 512),
            (64, 768, 3072),
        ] {
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
            let dx_want = quantize(&h.download(&dxr, m * k, Dtype::F32), dt);

            let (_xt, xtp) = h.upload(&qx, dt);
            let (_dyt, dytp) = h.upload(&qdy, dt);
            let (_wt, wtp) = h.upload(&qw, dt);
            let (dwt, dwtp) = h.zeros(k * n, Dtype::F32);
            let (dxt, dxtp) = h.zeros(m * k, dt);
            h.engine
                .backward_dw(
                    dwtp,
                    TypedPtr::new(dytp, dt),
                    TypedPtr::new(xtp, dt),
                    (m, k, n),
                )
                .unwrap();
            h.engine
                .backward_dx(
                    TypedPtr::new(dxtp, dt),
                    TypedPtr::new(dytp, dt),
                    TypedPtr::new(wtp, dt),
                    (m, k, n),
                )
                .unwrap();
            assert_bits(
                &format!("typed dW {dt:?} M{m} K{k} N{n}"),
                &h.download(&dwt, k * n, Dtype::F32),
                &dw_want,
            );
            assert_bits(
                &format!("typed dX {dt:?} M{m} K{k} N{n}"),
                &h.download(&dxt, m * k, dt),
                &dx_want,
            );
        }
    }
}

/// Row 0 of the typed forward is bit-identical across batch sizes that
/// share a dispatch bucket (per-bucket batch invariance).
#[test]
fn typed_forward_is_batch_invariant_within_bucket() {
    let h = Harness::new();
    let (k, n) = (384usize, 512usize);
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
                .forward(
                    TypedPtr::new(yp, dt),
                    TypedPtr::new(xp, dt),
                    TypedPtr::new(wp, dt),
                    None,
                    (m, k, n),
                )
                .unwrap();
            h.download(&yb, m * n, dt)[..n].to_vec()
        };

        // Ultra-thin bucket: M in [1, 32).
        assert_bits(
            &format!("{dt:?} ultra-thin M1 vs M16"),
            &run(16, 200),
            &run(1, 100),
        );
        // Split-K bucket: M in [32, 1024], K % 32 == 0.
        assert_bits(
            &format!("{dt:?} split-K M64 vs M256"),
            &run(256, 400),
            &run(64, 300),
        );
    }
}
