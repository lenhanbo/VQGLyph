from __future__ import annotations

import torch
import torch.nn.functional as F


def edge_map(image: torch.Tensor) -> torch.Tensor:
    dx = image[..., :, 1:] - image[..., :, :-1]; dy = image[..., 1:, :] - image[..., :-1, :]
    return F.pad(dx.abs(), (0, 1, 0, 0)) + F.pad(dy.abs(), (0, 0, 0, 1))


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ink_weight = 1 + 3 * ((target + 1) * .5 < .8).float()
    return (ink_weight * (prediction - target).abs()).mean(), F.l1_loss(edge_map(prediction), edge_map(target))
