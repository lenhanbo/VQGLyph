# NIST SD19 glyph codebook on Kaggle

This stage trains a single-glyph Q-K-V selector. It does not generate words.
Cluster assignments are written once and never changed during selector training.

## Kaggle inputs

Attach the repository and an SD19 dataset with a normalized JSON or JSONL
manifest. Each row must contain:

```json
{"path":"/kaggle/input/nist19-png/...png","character":"a","writer_id":"0123","partition":"hsf_0"}
```

`path` may be relative to the manifest. `character` must be one of the 62 SD19
classes. `sample_id` is optional. The official second-edition `by_write` tree
does not put the checked character class in each PNG pathname, so the preparer
intentionally refuses to infer labels from filenames.

Attach the existing CharWriteViT character-classifier `best.pt` as well and
set `data.classifier_checkpoint` in the YAML. It remains the single frozen
character evaluator used by the decoder loss; the NIST encoder is responsible
for content/style embeddings and clustering, not a replacement classifier.

If both official `by_write` and `by_class` trees are attached, the preparer can
cross-index their duplicate PNGs and build the manifest without guessed labels:

```bash
python -m glyph_codebook.cli.prepare \
  --config configs/kaggle.yaml \
  --by-write /kaggle/input/nist-sd19/by_write \
  --by-class /kaggle/input/nist-sd19/by_class
```

This first pass hashes every PNG in both trees and can take time. Persist the
resulting prepared directory as a Kaggle Dataset so later runs can use its
`manifest.json` directly.

For a reusable Kaggle input, prepare it locally from the official archives.
Extract `by_class` into the output directory, then run:

```bash
python -m glyph_codebook.cli.prepare \
  --config configs/kaggle.yaml \
  --by-write-archive /path/to/by_write.zip \
  --by-class-archive /path/to/by_class.zip \
  --by-class /path/to/nist19-prepared/by_class \
  --output /path/to/nist19-prepared \
  --portable
```

Upload only `nist19-prepared` (`by_class` plus the JSON files); `by_write` is
not needed after writer IDs have been recorded. Set `data.nist19_prepared_dir`
to the Kaggle input mount, for example `/kaggle/input/nist19-prepared`.

Install and validate:

```bash
cd /kaggle/working/GlyphCodebook
pip install -e ".[dev]"
python -m glyph_codebook.cli.prepare --config configs/kaggle.yaml
```

After editing the two input paths in the YAML, the whole pipeline can also be
run (and safely resumed from `latest.pt`) with:

```bash
python kaggle/run.py --smoke
python kaggle/run.py --stage encoder
python kaggle/run.py --clusters 4 --value-mode feature
```

Run a short encoder smoke test and then the full job:

```bash
python -m glyph_codebook.cli.train_encoder \
  --config configs/kaggle.yaml --output /kaggle/working/outputs/nist-encoder-smoke \
  --epochs 1 --max-steps 20

python -m glyph_codebook.cli.train_encoder \
  --config configs/kaggle.yaml --output /kaggle/working/outputs/nist-encoder
```

Build the immutable `k=4` codebook and fixed manifests:

```bash
python -m glyph_codebook.cli.build_codebook \
  --config configs/kaggle.yaml \
  --encoder /kaggle/working/outputs/nist-encoder/latest.pt \
  --output /kaggle/working/outputs/nist-codebook --clusters 4
```

Train both value modes as separate experiments:

```bash
python -m glyph_codebook.cli.train_selector \
  --config configs/kaggle.yaml \
  --encoder /kaggle/working/outputs/nist-encoder/latest.pt \
  --classifier /kaggle/input/charwritevit-classifier/best.pt \
  --codebook /kaggle/working/outputs/nist-codebook/codebook.pt \
  --value-mode pixel --output /kaggle/working/outputs/selector-pixel

python -m glyph_codebook.cli.train_selector \
  --config configs/kaggle.yaml \
  --encoder /kaggle/working/outputs/nist-encoder/latest.pt \
  --classifier /kaggle/input/charwritevit-classifier/best.pt \
  --codebook /kaggle/working/outputs/nist-codebook/codebook.pt \
  --value-mode feature --output /kaggle/working/outputs/selector-feature
```

Resume with `--resume /kaggle/working/outputs/selector-feature/latest.pt`.

Evaluate selection accuracy and glyph FID/KID:

```bash
python -m glyph_codebook.cli.evaluate \
  --config configs/kaggle.yaml \
  --encoder /kaggle/working/outputs/nist-encoder/latest.pt \
  --classifier /kaggle/input/charwritevit-classifier/best.pt \
  --codebook /kaggle/working/outputs/nist-codebook/codebook.pt \
  --checkpoint /kaggle/working/outputs/selector-feature/latest.pt \
  --output /kaggle/working/outputs/selector-feature/eval
```

Persist `/kaggle/working/outputs` as notebook output before the session ends.
