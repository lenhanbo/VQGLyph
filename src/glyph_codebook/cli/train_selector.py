from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from glyph_codebook.classifier import load_frozen_classifier
from glyph_codebook.cli.common import training_parser
from glyph_codebook.config import load_config
from glyph_codebook.data import NIST19EpisodeDataset
from glyph_codebook.losses import reconstruction_loss
from glyph_codebook.models import AttentionGlyphGenerator, load_codebook
from glyph_codebook.utils import amp_context, load_checkpoint, make_scaler, resolve_device, save_checkpoint, seed_everything


def main() -> None:
    parser = training_parser("Train fixed-label Q-K-V glyph selector and decoder")
    parser.add_argument("--codebook", required=True); parser.add_argument("--encoder", required=True); parser.add_argument("--classifier", required=True)
    parser.add_argument("--value-mode", choices=("pixel", "feature")); parser.add_argument("--oracle-warmup-epochs", type=int)
    args = parser.parse_args(); cfg = load_config(args.config); seed_everything(int(cfg.get("seed", 42))); device = resolve_device()
    mc = cfg["glyph_codebook"]; tc = mc["selector_train"]; codebook = load_codebook(args.codebook)
    value_mode = args.value_mode or str(mc.get("value_mode", "feature"))
    model = AttentionGlyphGenerator(codebook, int(mc.get("embedding_dim", 128)), int(mc.get("hidden_dim", 128)),
                                    value_mode, float(mc.get("attention_temperature", 0.1))).to(device)
    encoder_state = load_checkpoint(args.encoder, device)
    try: model.reference_encoder.load_state_dict(encoder_state["model"])
    except RuntimeError as error: raise ValueError("Selector dimensions must match the clustering encoder") from error
    classifier, classifier_stoi = load_frozen_classifier(args.classifier, device)
    classifier_ids = torch.tensor([classifier_stoi[character] for character in codebook.alphabet], device=device)
    artifact_root = Path(args.codebook).parent
    dataset = NIST19EpisodeDataset(artifact_root / "train.json", int(mc.get("references", 8)), int(mc.get("image_size", 32)), int(cfg.get("seed", 42)))
    loader = DataLoader(dataset, batch_size=int(tc.get("batch_size", 64)), shuffle=True, num_workers=int(tc.get("num_workers", 2)), pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc.get("learning_rate", 2e-4)), weight_decay=float(tc.get("weight_decay", 1e-4)))
    scaler = make_scaler(device, tc.get("amp", True)); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    start_epoch = 0; step = 0
    if args.resume:
        state = load_checkpoint(args.resume, device); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {})); start_epoch = int(state["epoch"]) + 1; step = int(state.get("step", 0))
    epochs = args.epochs or int(tc.get("epochs", 30)); oracle_epochs = args.oracle_warmup_epochs if args.oracle_warmup_epochs is not None else int(tc.get("oracle_warmup_epochs", 3))
    for epoch in range(start_epoch, epochs):
        model.train(); totals = {"loss": 0.0, "selection": 0.0, "pixel": 0.0, "edge": 0.0, "character": 0.0, "accuracy": 0.0}; seen = 0
        for batch in tqdm(loader, desc=f"selector {epoch + 1}/{epochs}"):
            references = batch["references"].to(device, non_blocking=True); target = batch["image"].to(device, non_blocking=True)
            chars = batch["character_id"].to(device); clusters = batch["cluster_id"].to(device)
            oracle = clusters if epoch < oracle_epochs else None
            with amp_context(device, tc.get("amp", True)):
                result = model(references, chars, oracle_clusters=oracle)
                selection = F.cross_entropy(result["logits"], clusters); pixel, edge = reconstruction_loss(result["image"], target)
                classifier_input = F.interpolate(result["image"], size=(32, 16), mode="bilinear", align_corners=False)
                character = F.cross_entropy(classifier(classifier_input), classifier_ids[chars])
                loss = float(tc.get("selection_weight", 1.0)) * selection + float(tc.get("pixel_weight", 10.0)) * pixel + float(tc.get("edge_weight", 1.0)) * edge + float(tc.get("character_weight", 1.0)) * character
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            size = len(target); seen += size; step += 1
            values = {"loss": loss, "selection": selection, "pixel": pixel, "edge": edge, "character": character,
                      "accuracy": (result["logits"].argmax(-1) == clusters).float().mean()}
            for name, value in values.items(): totals[name] += float(value.detach()) * size
            if args.max_steps and step >= args.max_steps: break
        metrics = {name: value / max(1, seen) for name, value in totals.items()}; metrics.update(epoch=epoch, step=step, oracle=epoch < oracle_epochs)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(metrics) + "\n")
        save_checkpoint(output / "latest.pt", model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, step=step, value_mode=value_mode, codebook=str(Path(args.codebook).resolve()),
                        classifier=str(Path(args.classifier).resolve()), config=cfg)
        print(metrics)
        if args.max_steps and step >= args.max_steps: break


if __name__ == "__main__": main()
