from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn



class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                                                     nn.GroupNorm(min(8, out_channels), out_channels), nn.SiLU())
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)


class GlyphEmbeddingEncoder(nn.Module):
    def __init__(self, classes: int = 62, width: int = 64, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(ConvBlock(1, width, 2), ConvBlock(width, width * 2, 2),
                                      ConvBlock(width * 2, width * 2, 2), ConvBlock(width * 2, embedding_dim, 1))
        self.content_projection = nn.Linear(embedding_dim, embedding_dim)
        self.style_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial = self.features(images)
        pooled = spatial.mean((-2, -1))
        content = F.normalize(self.content_projection(pooled), dim=-1)
        style = F.normalize(self.style_projection(pooled), dim=-1)
        return {"spatial": spatial, "content": content, "style": style}


class GlyphCodebook(nn.Module):
    def __init__(self, keys: torch.Tensor, medoid_images: torch.Tensor, alphabet: str):
        super().__init__()
        if keys.ndim != 3 or medoid_images.ndim != 5 or keys.shape[:2] != medoid_images.shape[:2]:
            raise ValueError("Expected keys [C,K,D] and medoid_images [C,K,1,H,W]")
        self.register_buffer("keys", F.normalize(keys.float(), dim=-1))
        self.register_buffer("medoid_images", medoid_images.float())
        self.alphabet = alphabet

    @property
    def clusters(self) -> int: return self.keys.shape[1]

    def select(self, character_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys[character_ids], self.medoid_images[character_ids]


class AttentionGlyphGenerator(nn.Module):
    """Q=reference style, K=medoid feature, V=medoid pixels or spatial features."""
    def __init__(self, codebook: GlyphCodebook, embedding_dim: int = 128, hidden_dim: int = 128,
                 value_mode: str = "feature", temperature: float = 0.1):
        super().__init__()
        if value_mode not in {"pixel", "feature"}: raise ValueError("value_mode must be 'pixel' or 'feature'")
        self.codebook = codebook; self.value_mode = value_mode; self.temperature = temperature
        self.reference_encoder = GlyphEmbeddingEncoder(len(codebook.alphabet), hidden_dim // 2, embedding_dim)
        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_projection = nn.Linear(codebook.keys.shape[-1], embedding_dim, bias=False)
        self.character_embedding = nn.Embedding(len(codebook.alphabet), hidden_dim)
        self.value_encoder = nn.Sequential(ConvBlock(1, hidden_dim // 2, 2), ConvBlock(hidden_dim // 2, hidden_dim, 2))
        self.condition = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            ConvBlock(hidden_dim * 2, hidden_dim), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(hidden_dim, hidden_dim // 2), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(hidden_dim // 2, hidden_dim // 4), nn.Conv2d(hidden_dim // 4, 1, 3, 1, 1), nn.Tanh())

    def encode_references(self, references: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = references.shape
        style = self.reference_encoder(references.reshape(b * n, c, h, w))["style"].reshape(b, n, -1)
        return F.normalize(style.mean(1), dim=-1)

    def forward(self, references: torch.Tensor, character_ids: torch.Tensor,
                oracle_clusters: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        query = self.encode_references(references)
        keys, values = self.codebook.select(character_ids)
        q = F.normalize(self.query_projection(query), dim=-1)
        k = F.normalize(self.key_projection(keys), dim=-1)
        logits = torch.einsum("bd,bkd->bk", q, k) / self.temperature
        attention = F.one_hot(oracle_clusters, keys.shape[1]).float() if oracle_clusters is not None else logits.softmax(-1)
        if self.value_mode == "pixel":
            selected_pixels = torch.einsum("bk,bkchw->bchw", attention, values)
            selected = self.value_encoder(selected_pixels)
        else:
            b, clusters, channels, height, width = values.shape
            features = self.value_encoder(values.reshape(b * clusters, channels, height, width))
            features = features.reshape(b, clusters, *features.shape[1:])
            selected = torch.einsum("bk,bkchw->bchw", attention, features)
            selected_pixels = torch.einsum("bk,bkchw->bchw", attention, values)
        condition = self.condition(torch.cat((query, self.character_embedding(character_ids)), dim=-1))
        condition = condition[:, :, None, None].expand(-1, -1, selected.shape[-2], selected.shape[-1])
        image = self.decoder(torch.cat((selected, condition), dim=1))
        return {"image": image, "logits": logits, "attention": attention, "query": query,
                "selected_pixels": selected_pixels}


def load_codebook(path: str, device: torch.device | str = "cpu") -> GlyphCodebook:
    artifact = torch.load(path, map_location=device, weights_only=False)
    return GlyphCodebook(artifact["keys"], artifact["medoid_images"], artifact["alphabet"])
