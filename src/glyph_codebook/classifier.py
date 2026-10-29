from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                                                     nn.GroupNorm(min(8, out_channels), out_channels), nn.SiLU())
    def forward(self, x): return self.net(x)


class CharacterClassifier(nn.Module):
    """Checkpoint-compatible copy of CharWriteViT's frozen classifier architecture."""
    def __init__(self, classes: int, width=64):
        super().__init__(); self.features = nn.Sequential(_ConvBlock(1, width, 2), _ConvBlock(width, width * 2, 2),
                                                          _ConvBlock(width * 2, width * 2, 2), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(width * 2, classes)
    def forward(self, image): return self.head(self.features(image).flatten(1))


def load_frozen_classifier(path: str, device: torch.device) -> tuple[CharacterClassifier, dict[str, int]]:
    state = torch.load(path, map_location=device, weights_only=False); alphabet = state.get("config", {}).get("model", {}).get("alphabet")
    if not alphabet: raise ValueError("CharWriteViT classifier checkpoint must contain config.model.alphabet")
    tokens = ["<pad>", "<unk>", "<blank>", *list(dict.fromkeys(alphabet))]; stoi = {token: i for i, token in enumerate(tokens)}
    classifier = CharacterClassifier(len(tokens)).to(device); classifier.load_state_dict(state["model"])
    return classifier.eval().requires_grad_(False), stoi
