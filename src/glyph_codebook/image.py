from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def canonicalize_polarity(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L"); array = np.asarray(grayscale, dtype=np.uint8)
    if not array.size: return grayscale
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    return ImageOps.invert(grayscale) if float(np.median(border)) < 127.5 else grayscale


def canonical_glyph(image: Image.Image, size: int = 32, margin: int = 2) -> torch.Tensor:
    grayscale = canonicalize_polarity(image); array = np.asarray(grayscale, dtype=np.uint8); foreground = array < 245
    if foreground.any():
        ys, xs = np.where(foreground); grayscale = grayscale.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    inner = max(1, size - 2 * margin); fitted = ImageOps.contain(grayscale, (inner, inner), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255); canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.detach().cpu().squeeze().clamp(-1, 1) + 1) * 127.5).byte().numpy()
    return Image.fromarray(array, mode="L")
