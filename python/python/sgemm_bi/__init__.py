"""Deterministic, batch-invariant CUDA GEMM for PyTorch.

Bit-identical training matmuls (forward / dW / dX) in f32, bf16, and
f16, with an opt-in tensor-core tier. No libtorch linkage: tensors are
passed as raw device pointers, so this package works with any PyTorch
build that has CUDA tensors.

Quick start::

    import torch
    import sgemm_bi

    layer = sgemm_bi.Linear(768, 3072, device="cuda", dtype=torch.bfloat16,
                            tensor_cores=True)
    y = layer(x)            # deterministic forward
    y.sum().backward()      # deterministic dW / dX

Requirements: NVIDIA Ampere or newer (sm_80+).
"""

import platform as _platform

if _platform.system() != "Linux":  # pragma: no cover
    raise ImportError(
        "sgemm-bi requires Linux x86_64 with an NVIDIA GPU (Ampere or "
        f"newer); this platform is {_platform.system()}/{_platform.machine()}. "
        "There is no CUDA on macOS or non-NVIDIA hardware, so the package "
        "cannot work here by design."
    )

# PEP 484 re-export form ("import X as X"): in a py.typed package a plain
# `from .torch import Linear` is treated as a private import by type
# checkers/IDEs - PyCharm then reports "Cannot find reference 'Linear'".
from ._sgemm_bi import Engine as Engine
from .torch import Linear as Linear
from .torch import deterministic_linear as deterministic_linear

__all__ = ["Engine", "Linear", "deterministic_linear"]
__version__ = "0.1.1.post2"
