"""Deterministic training with sgemm_bi.Linear — a complete, runnable demo.

Trains the same tiny model twice from the same seed and asserts the
final weights are BIT-IDENTICAL (torch.equal, not allclose) — the
property cuBLAS does not give you.

Requires: an NVIDIA GPU (Ampere+), torch with CUDA, `pip install sgemm-bi`.
"""

import torch

import sgemm_bi


def train_once(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        sgemm_bi.Linear(256, 512, device="cuda", dtype=torch.bfloat16,
                        tensor_cores=True),
        torch.nn.GELU(),
        sgemm_bi.Linear(512, 64, device="cuda", dtype=torch.bfloat16,
                        tensor_cores=True),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    g = torch.Generator(device="cuda").manual_seed(seed)
    for _ in range(50):
        x = torch.rand(128, 256, generator=g, device="cuda").bfloat16() - 0.5
        opt.zero_grad()
        loss = model(x).float().pow(2).sum()
        loss.backward()
        opt.step()

    return torch.cat([p.detach().float().flatten() for p in model.parameters()])


def main() -> None:
    w1 = train_once(seed=7)
    w2 = train_once(seed=7)
    assert torch.equal(w1, w2), "two runs must produce bit-identical weights"
    print(f"50 steps x 2 runs: {w1.numel()} weights bit-identical. ✓")

    # Batch invariance: the same row scores identically alone or in a batch.
    torch.manual_seed(0)
    layer = sgemm_bi.Linear(256, 512, device="cuda", dtype=torch.bfloat16,
                            tensor_cores=True)
    x = (torch.rand(512, 256, device="cuda").bfloat16() - 0.5)
    big = layer(x)
    small = layer(x[:64].contiguous())
    assert torch.equal(small, big[:64]), "rows must not depend on batch size"
    print("rows 0..64 bit-identical between batch=64 and batch=512. ✓")


if __name__ == "__main__":
    main()
