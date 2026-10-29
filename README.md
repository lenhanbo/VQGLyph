# GlyphCodebook

Standalone NIST SD19 pipeline for fixed per-character clustering and
writer-conditioned Q-K-V glyph selection.

```text
same-writer reference glyphs -> Q
character-specific medoid features -> K
medoid pixels or spatial features -> V
attention(Q,K,V) -> single-glyph decoder -> 32x32 glyph
```

The repository deliberately stops at one glyph. Word layout, kerning and the
CharWriteViT word generator remain in the separate `CharWriteViT` repository.
The only interoperability point is an optional frozen CharWriteViT
`CharacterClassifier` checkpoint used for character-consistency loss. Its
architecture is reproduced locally, so this package never imports CharWriteViT.

## Kaggle

Edit input paths in `configs/kaggle.yaml`, then run:

```bash
pip install -e ".[dev]"
python kaggle/run.py --smoke
python kaggle/run.py --clusters 4 --value-mode feature
```

See [docs/KAGGLE.md](docs/KAGGLE.md) for raw SD19 indexing, individual stages,
resume commands and the pixel-value ablation.

## Tests

```bash
pytest
```
"# VQGLyph" 
