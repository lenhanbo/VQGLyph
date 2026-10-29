from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from glyph_codebook.classifier import load_frozen_classifier
from glyph_codebook.config import load_config
from glyph_codebook.data import NIST19EpisodeDataset
from glyph_codebook.losses import reconstruction_loss
from glyph_codebook.models import AttentionGlyphGenerator, GlyphEmbeddingEncoder, load_codebook
from glyph_codebook.metrics import frechet_distance, polynomial_kid
from glyph_codebook.utils import load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed-label glyph selection and generation")
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--encoder", required=True)
    parser.add_argument("--classifier")
    parser.add_argument("--codebook"); parser.add_argument("--split", default="test", choices=("val", "test")); parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output", required=True); args = parser.parse_args(); cfg = load_config(args.config); mc = cfg["glyph_codebook"]; device = resolve_device()
    state = load_checkpoint(args.checkpoint, device); codebook_path = args.codebook or state["codebook"]; codebook = load_codebook(codebook_path)
    model = AttentionGlyphGenerator(codebook, int(mc.get("embedding_dim", 128)), int(mc.get("hidden_dim", 128)), state["value_mode"], float(mc.get("attention_temperature", 0.1))).to(device)
    model.load_state_dict(state["model"]); model.eval()
    encoder_state = load_checkpoint(args.encoder, device); evaluator = GlyphEmbeddingEncoder(62, int(encoder_state["encoder_width"]), int(encoder_state["embedding_dim"])).to(device)
    evaluator.load_state_dict(encoder_state["model"]); evaluator.eval()
    classifier, classifier_stoi = load_frozen_classifier(args.classifier or state["classifier"], device)
    classifier_ids = torch.tensor([classifier_stoi[character] for character in codebook.alphabet], device=device)
    dataset = NIST19EpisodeDataset(Path(codebook_path).parent / f"{args.split}.json", int(mc.get("references", 8)), int(mc.get("image_size", 32)))
    loader = DataLoader(dataset, batch_size=int(mc.get("evaluate_batch_size", 128)), shuffle=False, num_workers=int(mc.get("num_workers", 2)))
    correct = total = 0; pixel_sum = edge_sum = char_correct = 0.0; real_features = []; fake_features = []; per_class = defaultdict(lambda: [0, 0])
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate"):
            remaining = args.max_samples - total
            if remaining <= 0: break
            if len(batch["image"]) > remaining:
                for key, value in list(batch.items()):
                    if torch.is_tensor(value): batch[key] = value[:remaining]
            refs = batch["references"].to(device); target = batch["image"].to(device); chars = batch["character_id"].to(device); labels = batch["cluster_id"].to(device)
            result = model(refs, chars); prediction = result["logits"].argmax(-1); pixel, edge = reconstruction_loss(result["image"], target)
            real = evaluator(target); fake = evaluator(result["image"]); size = len(target)
            classifier_input = F.interpolate(result["image"], size=(32, 16), mode="bilinear", align_corners=False)
            correct += int((prediction == labels).sum()); char_correct += float((classifier(classifier_input).argmax(-1) == classifier_ids[chars]).sum())
            pixel_sum += float(pixel) * size; edge_sum += float(edge) * size; total += size
            real_features.append(real["content"].cpu()); fake_features.append(fake["content"].cpu())
            for c, hit in zip(chars.cpu().tolist(), (prediction == labels).cpu().tolist()): per_class[c][0] += int(hit); per_class[c][1] += 1
            if total >= args.max_samples: break
    real_feature = torch.cat(real_features); fake_feature = torch.cat(fake_features)
    metrics = {"samples": total, "cluster_top1": correct / max(1, total), "macro_cluster_top1": sum(a / b for a, b in per_class.values()) / max(1, len(per_class)),
               "foreground_l1": pixel_sum / max(1, total), "edge_l1": edge_sum / max(1, total), "character_accuracy": char_correct / max(1, total),
               "glyph_fid": float(frechet_distance(real_feature, fake_feature)), "glyph_kid": float(polynomial_kid(real_feature, fake_feature))}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
