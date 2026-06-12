"""PyTorch integration: autograd function and a Linear drop-in.

Layout note: weights here are stored ``[in_features, out_features]``
(K, N) — the GEMM-natural layout — unlike ``torch.nn.Linear``'s
``[out_features, in_features]``. Use :meth:`Linear.from_torch` to
convert an existing layer (one transpose at conversion time, none per
step).

Gradient dtypes follow PyTorch convention (grads match parameter
dtype). For bf16/f16 weights, dW is still ACCUMULATED in f32 inside the
kernel and rounded once at the end — strictly tighter than a 16-bit
accumulation.
"""

from __future__ import annotations

import threading

import torch

from ._sgemm_bi import Engine

_DTYPE_NAMES = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}

_engines: dict[int, Engine] = {}
_engines_lock = threading.Lock()


def _engine(device_index: int) -> Engine:
    """One engine per device, built lazily (NVRTC compile happens once)."""
    with _engines_lock:
        eng = _engines.get(device_index)
        if eng is None:
            eng = Engine(device_index)
            _engines[device_index] = eng
        return eng


def _check_operand(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise ValueError(f"sgemm-bi: {name} must be a CUDA tensor")
    if not t.is_contiguous():
        raise ValueError(f"sgemm-bi: {name} must be contiguous")
    if t.dtype not in _DTYPE_NAMES:
        raise ValueError(
            f"sgemm-bi: {name} has unsupported dtype {t.dtype} "
            "(float32, bfloat16, float16)"
        )


class _DeterministicLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None, tensor_cores: bool) -> torch.Tensor:
        # x: [*, K] -> [M, K]; weight: [K, N]; bias: f32 [N] or None.
        k, n = weight.shape
        x2 = x.reshape(-1, k)
        m = x2.shape[0]
        y = torch.empty(m, n, device=x.device, dtype=x.dtype)

        eng = _engine(x.device.index)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if m > 0:
            eng.forward(
                (y.data_ptr(), x2.data_ptr(), weight.data_ptr(),
                 bias.data_ptr() if bias is not None else None),
                (m, k, n), _DTYPE_NAMES[x.dtype], stream, tensor_cores,
            )

        ctx.save_for_backward(x2, weight)
        ctx.has_bias = bias is not None
        ctx.tensor_cores = tensor_cores
        ctx.x_shape = x.shape
        return y.reshape(*x.shape[:-1], n)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy: torch.Tensor):
        x2, weight = ctx.saved_tensors
        k, n = weight.shape
        dy2 = dy.reshape(-1, n).contiguous()
        m = dy2.shape[0]
        dtype = _DTYPE_NAMES[dy2.dtype]

        eng = _engine(dy2.device.index)
        stream = torch.cuda.current_stream(dy2.device).cuda_stream

        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty(m, k, device=dy2.device, dtype=dy2.dtype)
            if m > 0:
                eng.backward_dx(
                    (dx.data_ptr(), dy2.data_ptr(), weight.data_ptr()),
                    (m, k, n), dtype, stream, ctx.tensor_cores,
                )
            dx = dx.reshape(ctx.x_shape)
        if ctx.needs_input_grad[1]:
            # f32 master accumulation inside the kernel; one rounding to
            # the parameter dtype at the end (torch grad convention).
            dw32 = torch.zeros(k, n, device=dy2.device, dtype=torch.float32)
            if m > 0:
                eng.backward_dw(
                    (dw32.data_ptr(), dy2.data_ptr(), x2.data_ptr()),
                    (m, k, n), dtype, stream, ctx.tensor_cores,
                )
            dw = dw32 if weight.dtype == torch.float32 else dw32.to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            # Bias gradient is a plain column sum; done in f32 in torch.
            # Deterministic run-to-run for a fixed shape, but not part of
            # the engine's batch-invariance contract.
            db = dy2.float().sum(dim=0)

        return dx, dw, db, None


def deterministic_linear(x: torch.Tensor, weight: torch.Tensor,
                         bias: torch.Tensor | None = None,
                         tensor_cores: bool = False) -> torch.Tensor:
    """Functional deterministic linear: ``y = x @ weight (+ bias)``.

    ``weight`` is ``[in_features, out_features]`` (K, N). ``bias`` must
    be float32 (it is added to the f32 accumulator before the output
    rounding). All tensors must be contiguous CUDA tensors on the same
    device; x/weight must share one dtype of float32/bfloat16/float16.
    """
    _check_operand(x, "x")
    _check_operand(weight, "weight")
    if x.dtype != weight.dtype:
        raise ValueError(
            f"sgemm-bi: x dtype {x.dtype} != weight dtype {weight.dtype}"
        )
    if x.device != weight.device or (bias is not None and bias.device != x.device):
        raise ValueError(
            f"sgemm-bi: operands must share one device, got x={x.device}, "
            f"weight={weight.device}"
            + (f", bias={bias.device}" if bias is not None else "")
        )
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"sgemm-bi: x last dim {x.shape[-1]} != weight rows {weight.shape[0]}"
        )
    if bias is not None:
        if bias.dtype != torch.float32:
            raise ValueError("sgemm-bi: bias must be float32")
        if not bias.is_cuda or not bias.is_contiguous():
            raise ValueError("sgemm-bi: bias must be a contiguous CUDA tensor")
        if bias.shape != (weight.shape[1],):
            raise ValueError(
                f"sgemm-bi: bias shape {tuple(bias.shape)} != ({weight.shape[1]},)"
            )
    return _DeterministicLinear.apply(x, weight, bias, tensor_cores)


class Linear(torch.nn.Module):
    """Drop-in deterministic replacement for ``torch.nn.Linear``.

    Differences from ``torch.nn.Linear``:

    - ``weight`` is stored ``[in_features, out_features]`` (transposed
      relative to torch) — the GEMM-natural layout, no per-step
      transpose. Convert existing layers with :meth:`from_torch`.
    - ``bias`` is always float32, regardless of the weight dtype (it
      feeds the f32 accumulator directly).
    - ``tensor_cores=True`` opts into the tensor-core tier (bf16/f16,
      own deterministic numeric contract, strictly batch-invariant
      forward, 3.5-6.3x faster GEMMs than the scalar tier; covers
      both output dims >= 64).
    - NOT ``torch.autocast``-aware: autocast casts activations while the
      stored weight keeps its dtype, which this layer rejects (dtype
      mismatch) instead of silently re-casting per step. Construct the
      layer in the dtype you train in (mixed precision here means bf16/
      f16 weights with the built-in f32 gradient accumulation), or cast
      inputs explicitly.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype: torch.dtype = torch.float32,
                 tensor_cores: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tensor_cores = tensor_cores
        self.weight = torch.nn.Parameter(
            torch.empty(in_features, out_features, device=device, dtype=dtype)
        )
        self.bias = (
            torch.nn.Parameter(
                torch.empty(out_features, device=device, dtype=torch.float32)
            )
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Matches torch.nn.Linear's kaiming-uniform on the [N, K] view.
        torch.nn.init.kaiming_uniform_(self.weight.t(), a=5**0.5)
        if self.bias is not None:
            bound = 1.0 / (self.in_features**0.5)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_torch(cls, linear: torch.nn.Linear,
                   tensor_cores: bool = False) -> "Linear":
        """Converts a ``torch.nn.Linear`` (weight transposed once)."""
        mod = cls.__new__(cls)
        torch.nn.Module.__init__(mod)
        mod.in_features = linear.in_features
        mod.out_features = linear.out_features
        mod.tensor_cores = tensor_cores
        mod.weight = torch.nn.Parameter(linear.weight.detach().t().contiguous())
        mod.bias = (
            torch.nn.Parameter(linear.bias.detach().float().clone())
            if linear.bias is not None
            else None
        )
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return deterministic_linear(x, self.weight, self.bias, self.tensor_cores)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, tensor_cores={self.tensor_cores}"
        )
