from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from glyph_codebook.config import load_config
from glyph_codebook.data import NIST19_ALPHABET, NIST19GlyphDataset, load_glyph_manifest, save_glyph_manifest
from glyph_codebook.image import tensor_to_pil
from glyph_codebook.models import GlyphEmbeddingEncoder
from glyph_codebook.clustering import medoid_indices, spherical_kmeans
from glyph_codebook.utils import load_checkpoint, resolve_device


@torch.inference_mode()
def extract(model, dataset, device, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    features, classes = [], []
    for batch in tqdm(loader, desc="extract"):
        features.append(model(batch["image"].to(device))["style"].cpu()); classes.append(batch["character_id"])
    return torch.cat(features), torch.cat(classes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one fixed per-character SD19 glyph codebook")
    parser.add_argument("--config", required=True); parser.add_argument("--encoder", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", type=int); parser.add_argument("--max-samples-per-class", type=int)
    args = parser.parse_args(); cfg = load_config(args.config); dc = cfg["data"]; mc = cfg["glyph_codebook"]; device = resolve_device()
    checkpoint = load_checkpoint(args.encoder, device)
    model = GlyphEmbeddingEncoder(62, int(checkpoint["encoder_width"]), int(checkpoint["embedding_dim"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    prepared = Path(dc["nist19_prepared_dir"]); train_set = NIST19GlyphDataset(prepared / "train.json", int(mc.get("image_size", 32)))
    features, classes = extract(model, train_set, device, int(mc.get("extract_batch_size", 512)), int(mc.get("num_workers", 2)))
    clusters = args.clusters or int(mc.get("clusters", 4)); keys = []; centroids_by_class = []; medoid_images = []; seed = int(cfg.get("seed", 42))
    for class_id, character in enumerate(NIST19_ALPHABET):
        indices = torch.where(classes == class_id)[0]
        if args.max_samples_per_class and len(indices) > args.max_samples_per_class:
            generator = torch.Generator().manual_seed(seed + class_id)
            indices = indices[torch.randperm(len(indices), generator=generator)[:args.max_samples_per_class]]
        class_features = features[indices]
        centroids, _ = spherical_kmeans(class_features, clusters, int(mc.get("kmeans_iterations", 50)), seed + class_id)
        local_medoids = medoid_indices(class_features, centroids); global_medoids = indices[local_medoids]
        centroids_by_class.append(centroids)
        keys.append(F.normalize(class_features[local_medoids], dim=-1))
        medoid_images.append(torch.stack([train_set[int(i)]["image"] for i in global_medoids]))
        print(f"{character}: {len(indices)} samples")
    artifact = {"keys": torch.stack(keys), "cluster_centroids": torch.stack(centroids_by_class),
                "medoid_images": torch.stack(medoid_images), "alphabet": NIST19_ALPHABET,
                "clusters": clusters, "encoder_checkpoint": str(Path(args.encoder).resolve())}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); torch.save(artifact, output / "codebook.pt")
    grid = Image.new("L", (clusters * int(mc.get("image_size", 32)), len(NIST19_ALPHABET) * int(mc.get("image_size", 32))), 255)
    for class_id in range(len(NIST19_ALPHABET)):
        for cluster_id in range(clusters):
            grid.paste(tensor_to_pil(artifact["medoid_images"][class_id, cluster_id]),
                       (cluster_id * int(mc.get("image_size", 32)), class_id * int(mc.get("image_size", 32))))
    grid.save(output / "medoids.png")
    normalized_centroids = F.normalize(artifact["cluster_centroids"], dim=-1)
    split_statistics = {}
    for split in ("train", "val", "test"):
        source = prepared / f"{split}.json"
        if not source.is_file(): continue
        dataset = NIST19GlyphDataset(source, int(mc.get("image_size", 32)))
        split_features, split_classes = extract(model, dataset, device, int(mc.get("extract_batch_size", 512)), int(mc.get("num_workers", 2)))
        rows = load_glyph_manifest(source)
        normalized_features = F.normalize(split_features, dim=-1); assignments = torch.empty(len(dataset), dtype=torch.long)
        for class_id in range(len(NIST19_ALPHABET)):
            mask = split_classes == class_id
            assignments[mask] = (normalized_features[mask] @ normalized_centroids[class_id].T).argmax(1)
        for row, cluster_id in zip(rows, assignments.tolist()): row["cluster_id"] = cluster_id
        save_glyph_manifest(rows, output / f"{split}.json")
        counts = torch.zeros(len(NIST19_ALPHABET), clusters, dtype=torch.long)
        for class_id in range(len(NIST19_ALPHABET)): counts[class_id] = torch.bincount(assignments[split_classes == class_id], minlength=clusters)
        split_statistics[split] = {character: counts[i].tolist() for i, character in enumerate(NIST19_ALPHABET)}
    (output / "metadata.json").write_text(json.dumps({"clusters": clusters, "alphabet": NIST19_ALPHABET,
                                                       "cluster_counts": split_statistics}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
