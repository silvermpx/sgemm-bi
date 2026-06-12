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

from ._sgemm_bi import Engine
from .torch import Linear, deterministic_linear

__all__ = ["Engine", "Linear", "deterministic_linear"]
__version__ = "0.1.1"
