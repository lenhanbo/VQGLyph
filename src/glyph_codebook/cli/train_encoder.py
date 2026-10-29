from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from glyph_codebook.cli.common import training_parser
from glyph_codebook.config import load_config
from glyph_codebook.data import NIST19PairDataset
from glyph_codebook.models import GlyphEmbeddingEncoder
from glyph_codebook.utils import amp_context, load_checkpoint, make_scaler, resolve_device, save_checkpoint, seed_everything


def supervised_contrastive(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Multi-positive SupCon without treating equal labels as false negatives."""
    logits = features @ features.T / temperature
    eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    logits = logits.masked_fill(eye, -torch.inf)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    valid = positive.any(1)
    return -(log_probability.masked_fill(~positive, 0).sum(1)[valid] / positive.sum(1)[valid]).mean()


@torch.inference_mode()
def validate(model, loader, device, temperature: float, max_batches: int) -> dict[str, float]:
    model.eval(); content_hits = style_hits = total = 0; content_loss = style_loss = 0.0
    for batch_index, batch in enumerate(loader):
        anchor, content_positive, style_positive = [batch[name].to(device, non_blocking=True) for name in ("anchor", "content_positive", "style_positive")]
        labels = batch["character_id"].to(device); writer_names = list(batch["writer_id"])
        writer_map = {name: i for i, name in enumerate(dict.fromkeys(writer_names))}
        writers = torch.tensor([writer_map[name] for name in writer_names], device=device)
        a = model(anchor); cp = model(content_positive); sp = model(style_positive)
        content_loss += float(supervised_contrastive(torch.cat((a["content"], cp["content"])), labels.repeat(2), temperature)) * len(anchor)
        style_loss += float(supervised_contrastive(torch.cat((a["style"], sp["style"])), writers.repeat(2), temperature)) * len(anchor)
        content_match = (a["content"] @ cp["content"].T).argmax(1); style_match = (a["style"] @ sp["style"].T).argmax(1)
        content_hits += int((labels[content_match] == labels).sum()); style_hits += int((writers[style_match] == writers).sum()); total += len(anchor)
        if batch_index + 1 >= max_batches: break
    return {"val_content_loss": content_loss / max(1, total), "val_style_loss": style_loss / max(1, total),
            "val_content_retrieval": content_hits / max(1, total), "val_style_retrieval": style_hits / max(1, total)}


def main() -> None:
    parser = training_parser("Pretrain NIST19 content/style glyph encoder"); args = parser.parse_args()
    cfg = load_config(args.config); seed_everything(int(cfg.get("seed", 42))); device = resolve_device()
    dc = cfg["data"]; mc = cfg["glyph_codebook"]; tc = mc["encoder_train"]
    manifest = Path(dc["nist19_prepared_dir"]) / "train.json"
    dataset = NIST19PairDataset(manifest, int(mc.get("image_size", 32)), int(cfg.get("seed", 42)))
    loader = DataLoader(dataset, batch_size=int(tc.get("batch_size", 128)), shuffle=True,
                        num_workers=int(tc.get("num_workers", 2)), pin_memory=device.type == "cuda", drop_last=True)
    validation = NIST19PairDataset(Path(dc["nist19_prepared_dir"]) / "val.json", int(mc.get("image_size", 32)), int(cfg.get("seed", 42)) + 1)
    validation_loader = DataLoader(validation, batch_size=int(tc.get("batch_size", 128)), shuffle=False,
                                   num_workers=int(tc.get("num_workers", 2)), pin_memory=device.type == "cuda")
    model = GlyphEmbeddingEncoder(62, int(mc.get("encoder_width", 64)), int(mc.get("embedding_dim", 128))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc.get("learning_rate", 3e-4)), weight_decay=float(tc.get("weight_decay", 1e-4)))
    scaler = make_scaler(device, tc.get("amp", True)); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or int(tc.get("epochs", 20)); step = 0; start_epoch = 0; best_score = -1.0
    if args.resume:
        state = load_checkpoint(args.resume, device); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {})); start_epoch = int(state["epoch"]) + 1; step = int(state.get("step", 0)); best_score = float(state.get("best_score", -1.0))
    for epoch in range(start_epoch, epochs):
        model.train(); totals = {"loss": 0.0, "content": 0.0, "style": 0.0, "decorrelation": 0.0}; seen = 0
        for batch in tqdm(loader, desc=f"encoder {epoch + 1}/{epochs}"):
            anchor, content_positive, style_positive = [batch[name].to(device, non_blocking=True) for name in ("anchor", "content_positive", "style_positive")]
            labels = batch["character_id"].to(device)
            writer_names = list(batch["writer_id"]); writer_map = {name: i for i, name in enumerate(dict.fromkeys(writer_names))}
            writer_labels = torch.tensor([writer_map[name] for name in writer_names], device=device)
            with amp_context(device, tc.get("amp", True)):
                a = model(anchor); cp = model(content_positive); sp = model(style_positive)
                content = supervised_contrastive(torch.cat((a["content"], cp["content"])), labels.repeat(2), float(tc.get("temperature", 0.1)))
                style = supervised_contrastive(torch.cat((a["style"], sp["style"])), writer_labels.repeat(2), float(tc.get("temperature", 0.1)))
                cross = a["content"].T @ a["style"] / len(anchor); decorrelation = cross.square().mean()
                loss = float(tc.get("content_weight", 1.0)) * content + float(tc.get("style_weight", 1.0)) * style + float(tc.get("decorrelation_weight", 0.1)) * decorrelation
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            size = len(anchor); seen += size; step += 1
            for name, value in {"loss": loss, "content": content, "style": style, "decorrelation": decorrelation}.items(): totals[name] += float(value.detach()) * size
            if args.max_steps and step >= args.max_steps: break
        metrics = {name: value / max(1, seen) for name, value in totals.items()}
        metrics.update(validate(model, validation_loader, device, float(tc.get("temperature", .1)), int(tc.get("validation_batches", 50))))
        metrics.update(epoch=epoch, step=step); score = (metrics["val_content_retrieval"] + metrics["val_style_retrieval"]) * .5
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(metrics) + "\n")
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch, step=step,
                     best_score=max(best_score, score), encoder_width=int(mc.get("encoder_width", 64)),
                     embedding_dim=int(mc.get("embedding_dim", 128)), config=cfg)
        save_checkpoint(output / "latest.pt", **state)
        if score >= best_score: save_checkpoint(output / "best.pt", **state)
        best_score = max(best_score, score)
        print(metrics)
        if args.max_steps and step >= args.max_steps: break


if __name__ == "__main__": main()
