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

## Training stages

### Stage 0 — Prepare NIST SD19

- Cross-index `by_write` and `by_class`, or validate an existing normalized manifest.
- Normalize isolated glyphs and create writer-disjoint `train/val/test` manifests.
- No model is trained in this stage.
- Output: `train.json`, `val.json`, `test.json`, `summary.json`.

```bash
python -m glyph_codebook.cli.prepare --config configs/kaggle.yaml
```

### Stage 1 — Pretrain content/style encoder

- Content positive: same character, different writer.
- Style positive: same writer, different character.
- Train two normalized embedding heads with multi-positive supervised contrastive loss.
- Select `best.pt` using the mean of validation content-retrieval and style-retrieval.
- Output: `latest.pt`, `best.pt`, `metrics.jsonl`.

```bash
python kaggle/run.py --stage encoder
```

### Stage 2 — Build the fixed glyph codebook

- Freeze the Stage-1 encoder.
- Run spherical k-means independently for each of the 62 characters.
- Assign every glyph a fixed pseudo-label; labels never change in later training.
- Store centroid assignments separately from medoid-image keys.
- Output: `codebook.pt`, clustered manifests, `medoids.png`, `metadata.json`.

```bash
python kaggle/run.py --stage codebook --clusters 4
```

### Stage 3 — Train Q-K-V selector and single-glyph decoder

- Q: pooled style embedding from same-writer references that exclude the target character.
- K: frozen feature of each medoid image in the query character's codebook.
- V: medoid pixels or medoid spatial features.
- Warm up the decoder with oracle one-hot values, then train with predicted soft attention.
- Use fixed cluster cross-entropy plus foreground, edge and frozen-character-classifier losses.
- Output: selector/decoder `latest.pt` and training metrics.

```bash
python kaggle/run.py --clusters 4 --value-mode feature
```

Run `--value-mode pixel` as the main value-representation ablation.

### Stage 4 — Evaluate the glyph module

- Cluster top-1 and macro top-1 measure whether the correct fixed code was selected.
- Foreground L1, edge L1 and character accuracy measure per-glyph reconstruction.
- Glyph-FID/KID are secondary distribution-quality metrics, not selection metrics.
- Compare oracle, hard and soft selection when running experiments.

The all-stage runner invokes this evaluation automatically after Stage 3.

### Stage 5 — Integrate into CharWriteViT (future)

- Generate one styled glyph for each character in the input text.
- Replace or fuse the GNU Unifont content path with encoded generated-glyph features.
- Feed the glyph sequence into the CharWriteViT contextualizer and word-level generator.
- Add kerning, connections and adversarial word training only in the CharWriteViT repository.

Stage 5 is intentionally not implemented in this standalone repository.

## Kaggle

Edit input paths in `configs/kaggle.yaml`, then run:

```bash
pip install -e ".[dev]"
python kaggle/run.py --smoke
python kaggle/run.py --stage encoder
python kaggle/run.py --clusters 4 --value-mode feature
```

See [docs/KAGGLE.md](docs/KAGGLE.md) for raw SD19 indexing, individual stages,
resume commands and the pixel-value ablation.

## Tests

```bash
pytest
```
"# VQGLyph" 
