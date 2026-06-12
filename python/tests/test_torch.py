"""GPU contract tests for the PyTorch binding (run on real hardware).

Covers: forward/backward parity against torch references, bit-identity
across runs in all three dtypes, tensor-core tier batch invariance, and
the nn.Linear conversion path.
"""

import pytest
import torch

import sgemm_bi

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

DEV = "cuda:0"


def _seeded(*shape, dtype=torch.float32, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.rand(*shape, generator=g, device=DEV, dtype=torch.float32) - 0.5).to(dtype)


def _max_norm_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max abs error normalized by the reference magnitude (robust to
    near-zero entries, unlike elementwise relative error)."""
    return ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()


def test_forward_f32_parity_and_bits():
    m, k, n = 513, 384, 769  # deliberately non-tile-aligned
    x, w = _seeded(m, k, seed=1), _seeded(k, n, seed=2)
    b = _seeded(n, seed=3)

    y1 = sgemm_bi.deterministic_linear(x, w, b)
    y2 = sgemm_bi.deterministic_linear(x, w, b)
    assert torch.equal(y1, y2), "f32 forward must be bit-identical across runs"

    # Reference in float64: immune to TF32 matmul settings and exact
    # enough to certify the engine's f32 chain.
    ref = (x.double() @ w.double() + b.double()).float()
    err = _max_norm_err(y1, ref)
    assert err < 1e-6, f"f32 forward max-norm err {err:.3e}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tc", [False, True])
def test_forward_typed_parity_and_bits(dtype, tc):
    m, k, n = 256, 192, 384
    x, w = _seeded(m, k, dtype=dtype, seed=4), _seeded(k, n, dtype=dtype, seed=5)

    y1 = sgemm_bi.deterministic_linear(x, w, tensor_cores=tc)
    y2 = sgemm_bi.deterministic_linear(x, w, tensor_cores=tc)
    assert torch.equal(y1, y2), f"{dtype} forward (tc={tc}) must be bit-identical"

    ref = (x.float() @ w.float()).to(dtype)
    err = _max_norm_err(y1, ref)
    assert err < 0.02, f"{dtype} forward max-norm err {err:.3e}"


def test_tc_forward_strict_batch_invariance():
    k, n = 256, 512
    big = 2048
    x_big = _seeded(big, k, dtype=torch.bfloat16, seed=6)
    w = _seeded(k, n, dtype=torch.bfloat16, seed=7)

    y_big = sgemm_bi.deterministic_linear(x_big, w, tensor_cores=True)
    for m in (128, 384, 1024):
        y_m = sgemm_bi.deterministic_linear(x_big[:m].contiguous(), w, tensor_cores=True)
        assert torch.equal(y_m[:128], y_big[:128]), (
            f"TC forward rows 0..128 must be bit-identical at M={m} vs M={big}"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_parity_and_bits(dtype):
    m, k, n = 320, 256, 192
    x = _seeded(m, k, dtype=dtype, seed=8).requires_grad_(True)
    w = _seeded(k, n, dtype=dtype, seed=9).requires_grad_(True)
    b = _seeded(n, seed=10).requires_grad_(True)
    dy = _seeded(m, n, dtype=dtype, seed=11)

    def run():
        if x.grad is not None:
            x.grad = None
            w.grad = None
            b.grad = None
        y = sgemm_bi.deterministic_linear(x, w, b)
        y.backward(dy)
        return x.grad.clone(), w.grad.clone(), b.grad.clone()

    dx1, dw1, db1 = run()
    dx2, dw2, db2 = run()
    assert torch.equal(dx1, dx2) and torch.equal(dw1, dw2) and torch.equal(db1, db2), (
        "backward must be bit-identical across runs"
    )

    xf = x.detach().double().requires_grad_(True)
    wf = w.detach().double().requires_grad_(True)
    bf = b.detach().double().requires_grad_(True)
    (xf @ wf + bf).backward(dy.double())
    tol = 1e-6 if dtype == torch.float32 else 0.02
    for got, ref, name in ((dx1, xf.grad, "dx"), (dw1, wf.grad, "dw"), (db1, bf.grad, "db")):
        err = _max_norm_err(got, ref)
        assert err < tol, f"{name} max-norm err {err:.3e} (dtype={dtype})"


def test_linear_from_torch_matches():
    torch.manual_seed(0)
    ref = torch.nn.Linear(192, 320, device=DEV)
    det = sgemm_bi.Linear.from_torch(ref)
    x = _seeded(64, 192, seed=12)
    # nn.Linear may run its f32 matmul through TF32 depending on global
    # settings; certify against a float64 reference instead.
    ref64 = (x.double() @ ref.weight.double().t() + ref.bias.double()).float()
    err = _max_norm_err(det(x), ref64)
    assert err < 1e-6, f"Linear.from_torch max-norm err {err:.3e}"


def test_linear_trains():
    layer = sgemm_bi.Linear(128, 64, device=DEV, dtype=torch.bfloat16, tensor_cores=True)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    x = _seeded(256, 128, dtype=torch.bfloat16, seed=13)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = layer(x).float().pow(2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, f"loss did not drop: {losses[0]} -> {losses[-1]}"


def test_input_validation():
    x = _seeded(8, 16, seed=14)
    w = _seeded(16, 8, seed=15)
    with pytest.raises(ValueError, match="contiguous"):
        sgemm_bi.deterministic_linear(x.t(), w)
    with pytest.raises(ValueError, match="dtype"):
        sgemm_bi.deterministic_linear(x.to(torch.bfloat16), w)
    with pytest.raises(ValueError, match="CUDA"):
        sgemm_bi.deterministic_linear(x.cpu(), w.cpu())
