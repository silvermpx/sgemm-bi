"""Type stubs for the native extension module.

The implementation lives in Rust (PyO3); this stub gives IDEs and type
checkers full signatures and docs for the low-level engine. Most users
want the high-level API instead: :class:`sgemm_bi.Linear` and
:func:`sgemm_bi.deterministic_linear`.
"""

from typing import Optional, Tuple

class Engine:
    """Deterministic GEMM engine bound to one CUDA device.

    Construction compiles the kernels via NVRTC for the device's native
    architecture (a few seconds, once) and creates a dedicated CUDA
    stream. Requires an NVIDIA Ampere or newer GPU (sm_80+).

    All methods take raw device pointers (``tensor.data_ptr()``) plus the
    caller's current CUDA stream handle
    (``torch.cuda.current_stream().cuda_stream``); work is ordered
    against that stream with CUDA events — no host synchronization.
    Thread-safe: calls serialize on an internal mutex.

    Example::

        eng = Engine(0)
        eng.forward(
            (y.data_ptr(), x.data_ptr(), w.data_ptr(), None),
            (m, k, n), "bfloat16",
            torch.cuda.current_stream().cuda_stream,
            True,  # tensor cores
        )
    """

    def __init__(self, device: int) -> None: ...

    def forward(
        self,
        ptrs: Tuple[int, int, int, Optional[int]],
        dims: Tuple[int, int, int],
        dtype: str,
        stream: int,
        tensor_cores: bool = False,
    ) -> None:
        """``out[M,N] = a[M,K] @ b[K,N] (+ bias[N])``.

        ``ptrs`` is ``(out, a, b, bias)`` — device pointers; ``bias`` is
        float32 or ``None``. ``dims`` is ``(M, K, N)``. ``dtype`` is one
        of ``"float32"``, ``"bfloat16"``, ``"float16"``.

        ``tensor_cores=True`` tries the tensor-core tier first (bf16/
        f16, both output dims >= 64) and falls back to the scalar tier —
        the choice depends only on shape and dtype, never on data, so
        determinism is preserved.
        """

    def backward_dw(
        self,
        ptrs: Tuple[int, int, int],
        dims: Tuple[int, int, int],
        dtype: str,
        stream: int,
        tensor_cores: bool = False,
    ) -> None:
        """``dw[K,N] += X^T @ dY`` — weight gradient, f32 master accumulator.

        ``ptrs`` is ``(dw, dy, x)``. ``dw`` is ALWAYS float32 regardless
        of ``dtype`` (gradients accumulate at full precision); zero it
        before the first accumulation.
        """

    def backward_dx(
        self,
        ptrs: Tuple[int, int, int],
        dims: Tuple[int, int, int],
        dtype: str,
        stream: int,
        tensor_cores: bool = False,
    ) -> None:
        """``dx[M,K] = dY @ W^T`` — input gradient (overwrite).

        ``ptrs`` is ``(dx, dy, w)``.
        """

    def presize_upcast_scratch(
        self, a_elems: int, b_elems: int, c_elems: int
    ) -> None:
        """Pre-size the typed upcast-fallback scratch (f32 element counts).

        Required before CUDA Graph capture: a lazy allocation during
        capture fails it. From your largest GEMM:
        ``(max(M*K, M*N), max(K*N, M*K), max(M*N, M*K))``.
        """

    def synchronize(self) -> None:
        """Block until all engine work has completed."""
