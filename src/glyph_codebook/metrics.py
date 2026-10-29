from __future__ import annotations

import torch


def frechet_distance(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """FID in an arbitrary frozen glyph feature space."""
    real = real.double(); fake = fake.double(); mu1 = real.mean(0); mu2 = fake.mean(0)
    cov1 = torch.cov(real.T); cov2 = torch.cov(fake.T)
    values, vectors = torch.linalg.eigh(cov1); sqrt1 = (vectors * values.clamp_min(0).sqrt()) @ vectors.T
    middle = sqrt1 @ cov2 @ sqrt1
    trace_sqrt = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum()
    return ((mu1 - mu2).square().sum() + torch.trace(cov1) + torch.trace(cov2) - 2 * trace_sqrt).float()


def polynomial_kid(real: torch.Tensor, fake: torch.Tensor, subset_size: int = 1000, subsets: int = 10, seed: int = 42) -> torch.Tensor:
    """Unbiased polynomial-kernel MMD squared (KID)."""
    x = real.float(); y = fake.float(); n = min(subset_size, len(x), len(y)); generator = torch.Generator().manual_seed(seed)
    if n < 2: raise ValueError("KID needs at least two real and fake samples")
    estimates = []
    for _ in range(subsets):
        xs = x[torch.randperm(len(x), generator=generator)[:n]]; ys = y[torch.randperm(len(y), generator=generator)[:n]]; dim = xs.shape[1]
        kxx = (xs @ xs.T / dim + 1).pow(3); kyy = (ys @ ys.T / dim + 1).pow(3); kxy = (xs @ ys.T / dim + 1).pow(3)
        xx = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1)); yy = (kyy.sum() - kyy.diag().sum()) / (n * (n - 1))
        estimates.append(xx + yy - 2 * kxy.mean())
    return torch.stack(estimates).mean()
