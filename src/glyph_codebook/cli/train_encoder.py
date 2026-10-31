from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # The JSONL/stdout logger remains usable without TensorBoard.
    SummaryWriter = None

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
    progress = tqdm(loader, desc="validation", total=min(len(loader), max_batches), leave=False, dynamic_ncols=True)
    for batch_index, batch in enumerate(progress):
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
    data_started = time.perf_counter()
    print(f"[data] loading train manifest: {manifest}", flush=True)
    dataset = NIST19PairDataset(manifest, int(mc.get("image_size", 32)), int(cfg.get("seed", 42)))
    train_writers = len({str(row["writer_id"]) for row in dataset.records})
    print(f"[data] train ready: raw={len(dataset.records):,}, eligible={len(dataset):,}, "
          f"writers={train_writers:,}, elapsed={time.perf_counter() - data_started:.1f}s", flush=True)
    loader = DataLoader(dataset, batch_size=int(tc.get("batch_size", 128)), shuffle=True,
                        num_workers=int(tc.get("num_workers", 2)), pin_memory=device.type == "cuda", drop_last=True)
    validation_manifest = Path(dc["nist19_prepared_dir"]) / "val.json"
    validation_started = time.perf_counter()
    print(f"[data] loading validation manifest: {validation_manifest}", flush=True)
    validation = NIST19PairDataset(validation_manifest, int(mc.get("image_size", 32)), int(cfg.get("seed", 42)) + 1)
    validation_writers = len({str(row["writer_id"]) for row in validation.records})
    print(f"[data] validation ready: raw={len(validation.records):,}, eligible={len(validation):,}, "
          f"writers={validation_writers:,}, elapsed={time.perf_counter() - validation_started:.1f}s", flush=True)
    validation_loader = DataLoader(validation, batch_size=int(tc.get("batch_size", 128)), shuffle=False,
                                   num_workers=int(tc.get("num_workers", 2)), pin_memory=device.type == "cuda")
    model = GlyphEmbeddingEncoder(62, int(mc.get("encoder_width", 64)), int(mc.get("embedding_dim", 128))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc.get("learning_rate", 3e-4)), weight_decay=float(tc.get("weight_decay", 1e-4)))
    scaler = make_scaler(device, tc.get("amp", True)); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    tensorboard_enabled = bool(tc.get("tensorboard", True)) and SummaryWriter is not None
    writer = SummaryWriter(log_dir=str(output / "tensorboard")) if tensorboard_enabled else None
    if tc.get("tensorboard", True) and SummaryWriter is None:
        print("[logging] TensorBoard is unavailable; continuing with stdout and metrics.jsonl", flush=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"[run] device={device}, parameters={parameters:,}, train_batches={len(loader):,}, "
          f"val_batches={len(validation_loader):,}, tensorboard={tensorboard_enabled}", flush=True)
    epochs = args.epochs or int(tc.get("epochs", 20)); step = 0; start_epoch = 0; best_score = -1.0
    if args.resume:
        state = load_checkpoint(args.resume, device); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {})); start_epoch = int(state["epoch"]) + 1; step = int(state.get("step", 0)); best_score = float(state.get("best_score", -1.0))
        print(f"[resume] checkpoint={args.resume}, start_epoch={start_epoch + 1}, step={step}, best_score={best_score:.4f}", flush=True)
    log_every = max(1, int(tc.get("log_every_steps", 50)))
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        model.train(); totals = {"loss": 0.0, "content": 0.0, "style": 0.0, "decorrelation": 0.0}; seen = 0
        progress = tqdm(loader, desc=f"encoder {epoch + 1}/{epochs}", dynamic_ncols=True)
        for batch_index, batch in enumerate(progress, 1):
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
            batch_metrics = {"loss": loss, "content": content, "style": style, "decorrelation": decorrelation}
            for name, value in batch_metrics.items(): totals[name] += float(value.detach()) * size
            if batch_index == 1 or batch_index % log_every == 0:
                averages = {name: value / max(1, seen) for name, value in totals.items()}
                postfix = {"loss": f"{averages['loss']:.4f}", "content": f"{averages['content']:.4f}",
                           "style": f"{averages['style']:.4f}", "decor": f"{averages['decorrelation']:.5f}"}
                if device.type == "cuda": postfix["gpu_mb"] = f"{torch.cuda.max_memory_allocated(device) / 2**20:.0f}"
                progress.set_postfix(postfix)
                if writer is not None:
                    for name, value in batch_metrics.items(): writer.add_scalar(f"train_batch/{name}", float(value.detach()), step)
                    writer.add_scalar("train_batch/learning_rate", optimizer.param_groups[0]["lr"], step)
                    if device.type == "cuda": writer.add_scalar("system/gpu_memory_mb", torch.cuda.max_memory_allocated(device) / 2**20, step)
            if args.max_steps and step >= args.max_steps: break
        metrics = {name: value / max(1, seen) for name, value in totals.items()}
        print(f"[validation] epoch={epoch + 1}, max_batches={int(tc.get('validation_batches', 50))}", flush=True)
        metrics.update(validate(model, validation_loader, device, float(tc.get("temperature", .1)), int(tc.get("validation_batches", 50))))
        metrics.update(epoch=epoch, step=step, epoch_seconds=time.perf_counter() - epoch_started,
                       learning_rate=optimizer.param_groups[0]["lr"])
        score = (metrics["val_content_retrieval"] + metrics["val_style_retrieval"]) * .5
        metrics["val_mean_retrieval"] = score
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(metrics) + "\n")
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch, step=step,
                     best_score=max(best_score, score), encoder_width=int(mc.get("encoder_width", 64)),
                     embedding_dim=int(mc.get("embedding_dim", 128)), config=cfg)
        save_checkpoint(output / "latest.pt", **state)
        is_best = score >= best_score
        if is_best: save_checkpoint(output / "best.pt", **state)
        best_score = max(best_score, score)
        if writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)): writer.add_scalar(f"epoch/{name}", value, epoch + 1)
            writer.flush()
        print(f"[epoch] {json.dumps(metrics, sort_keys=True)}", flush=True)
        print(f"[checkpoint] latest={output / 'latest.pt'}, best_updated={is_best}, best_score={best_score:.4f}", flush=True)
        if args.max_steps and step >= args.max_steps: break
    if writer is not None: writer.close()


if __name__ == "__main__": main()
