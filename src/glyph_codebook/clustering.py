from __future__ import annotations

import torch
import torch.nn.functional as F


def spherical_kmeans(features: torch.Tensor, clusters: int, iterations: int = 50, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2: raise ValueError("features must be [N,D]")
    if not 1 < clusters <= len(features): raise ValueError("clusters must be in [2, N]")
    x = F.normalize(features.float(), dim=-1)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    first = torch.randint(len(x), (1,), generator=generator, device=x.device)
    selected = [int(first)]
    for _ in range(1, clusters):
        similarity = x @ x[selected].T
        distance = (1 - similarity.max(1).values).clamp_min(1e-8)
        selected.append(int(torch.multinomial(distance, 1, generator=generator)))
    centroids = x[selected]
    assignments = torch.zeros(len(x), dtype=torch.long, device=x.device)
    for _ in range(iterations):
        updated_assignments = (x @ centroids.T).argmax(1)
        new_centroids = []
        for index in range(clusters):
            members = x[updated_assignments == index]
            new_centroids.append(F.normalize(members.mean(0), dim=0) if len(members) else centroids[index])
        new_centroids = torch.stack(new_centroids)
        if torch.equal(updated_assignments, assignments) and torch.allclose(new_centroids, centroids, atol=1e-5):
            assignments = updated_assignments; centroids = new_centroids; break
        assignments = updated_assignments; centroids = new_centroids
    return centroids, assignments


def medoid_indices(features: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features.float(), dim=-1)
    return (normalized @ F.normalize(centroids.float(), dim=-1).T).argmax(0)
