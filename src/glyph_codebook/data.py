from __future__ import annotations

import json
import hashlib
import os
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torch.utils.data import Dataset

from .image import canonical_glyph


NIST19_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _digest(path: Path) -> tuple[int, bytes]:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return path.stat().st_size, digest.digest()


def _zip_digest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, bytes]:
    digest = hashlib.sha1()
    with archive.open(info) as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return info.file_size, digest.digest()


def build_manifest_from_sd19_archives(
    by_write_archive: str | Path, by_class_archive: str | Path, extracted_by_class: str | Path
) -> tuple[list[dict], dict[str, int]]:
    """Cross-index SD19 inside ZIPs while pointing records at extracted class PNGs."""
    origins: dict[tuple[int, bytes], list[tuple[str, str, str]]] = defaultdict(list)
    writer_pattern = re.compile(r"^f(\d{4})_(\d{2})$")
    partition_pattern = re.compile(r"^hsf_\d+$")
    with zipfile.ZipFile(by_write_archive) as archive:
        images = [info for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith(".png")]
        for index, info in enumerate(images, 1):
            parts = PurePosixPath(info.filename).parts
            writer_part = next((part for part in parts if writer_pattern.match(part)), None)
            if writer_part is None:
                continue
            match = writer_pattern.match(writer_part)
            partition = next((part for part in parts if partition_pattern.match(part)), "unknown")
            origins[_zip_digest(archive, info)].append((match.group(1), match.group(2), partition))
            if index % 100_000 == 0:
                print(f"indexed by_write archive: {index}/{len(images)}", flush=True)

    class_root = Path(extracted_by_class).resolve()
    records = []
    ambiguous = missing = 0
    with zipfile.ZipFile(by_class_archive) as archive:
        images = [info for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith(".png")]
        for index, info in enumerate(images, 1):
            parts = PurePosixPath(info.filename).parts
            class_part = next((part for part in parts if len(part) == 2 and all(c in "0123456789abcdefABCDEF" for c in part)), None)
            if class_part is None:
                continue
            character = chr(int(class_part, 16))
            if character not in NIST19_ALPHABET:
                continue
            matches = origins.get(_zip_digest(archive, info), [])
            unique = {(writer, form, partition) for writer, form, partition in matches}
            if not unique:
                missing += 1
                continue
            if len(unique) != 1:
                ambiguous += 1
                continue
            writer, form, partition = next(iter(unique))
            relative_parts = parts[parts.index("by_class") + 1:] if "by_class" in parts else parts
            image = class_root.joinpath(*relative_parts)
            records.append({"path": str(image), "character": character, "writer_id": writer,
                            "form_id": form, "partition": partition, "sample_id": image.stem})
            if index % 100_000 == 0:
                print(f"indexed by_class archive: {index}/{len(images)}", flush=True)
    return records, {"indexed": len(records), "missing_hash": missing, "ambiguous_hash": ambiguous}


def build_manifest_from_sd19_trees(by_write: str | Path, by_class: str | Path) -> tuple[list[dict], dict[str, int]]:
    """Cross-index official second-edition trees using their duplicate PNGs.

    The by_write tree carries writer identity but not a pathname class label;
    by_class carries the checked hexadecimal class but not writer identity.
    """
    writer_root = Path(by_write); class_root = Path(by_class); origins: dict[tuple[int, bytes], list[tuple[str, str, str]]] = defaultdict(list)
    writer_pattern = re.compile(r"^f(\d{4})_(\d{2})$"); partition_pattern = re.compile(r"^hsf_\d+$")
    for image in writer_root.rglob("*.png"):
        writer_dir = next((part for part in image.parents if writer_pattern.match(part.name)), None)
        if writer_dir is None: continue
        match = writer_pattern.match(writer_dir.name); partition = next((part.name for part in image.parents if partition_pattern.match(part.name)), "unknown")
        origins[_digest(image)].append((match.group(1), match.group(2), partition))
    records = []; ambiguous = missing = 0
    for image in class_root.rglob("*.png"):
        class_dir = next((part for part in image.parents if len(part.name) == 2 and all(c in "0123456789abcdefABCDEF" for c in part.name)), None)
        if class_dir is None: continue
        character = chr(int(class_dir.name, 16))
        if character not in NIST19_ALPHABET: continue
        matches = origins.get(_digest(image), [])
        unique = {(writer, form, partition) for writer, form, partition in matches}
        if not unique: missing += 1; continue
        if len(unique) != 1: ambiguous += 1; continue
        writer, form, partition = next(iter(unique))
        records.append({"path": str(image.resolve()), "character": character, "writer_id": writer,
                        "form_id": form, "partition": partition, "sample_id": image.stem})
    return records, {"indexed": len(records), "missing_hash": missing, "ambiguous_hash": ambiguous}


def load_glyph_manifest(path: str | Path) -> list[dict]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    records = json.loads(text) if stripped.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    required = {"path", "character", "writer_id"}
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Manifest row {index} is missing {sorted(missing)}")
        if record["character"] not in NIST19_ALPHABET:
            raise ValueError(f"Manifest row {index} has unsupported character {record['character']!r}")
        image = Path(record["path"])
        if not image.is_absolute():
            record["path"] = str((source.parent / image).resolve())
        record.setdefault("sample_id", Path(record["path"]).stem)
    return records


def save_glyph_manifest(records: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")


def make_manifest_portable(records: list[dict], root: str | Path) -> list[dict]:
    """Return rows with image paths relative to a portable dataset root."""
    base = Path(root).resolve()
    portable = []
    for row in records:
        image = Path(os.path.abspath(row["path"]))
        try:
            relative = image.relative_to(base)
        except ValueError as error:
            raise ValueError(f"Image is outside portable root {base}: {image}") from error
        portable.append({**row, "path": relative.as_posix()})
    return portable


def writer_disjoint_splits(records: list[dict], ratios=(0.8, 0.1, 0.1), seed: int = 42) -> dict[str, list[dict]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to one")
    by_partition: dict[str, set[str]] = defaultdict(set)
    by_writer: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        writer = str(row["writer_id"])
        by_writer[writer].append(row)
        by_partition[str(row.get("partition", "all"))].add(writer)
    groups = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for partition in sorted(by_partition):
        writers = sorted(by_partition[partition]); rng.shuffle(writers)
        train_end = round(len(writers) * ratios[0]); val_end = train_end + round(len(writers) * ratios[1])
        groups["train"].extend(writers[:train_end]); groups["val"].extend(writers[train_end:val_end]); groups["test"].extend(writers[val_end:])
    return {name: [row for writer in writers for row in by_writer[writer]] for name, writers in groups.items()}


def _load_image(path: str, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        return canonical_glyph(image.convert("L"), size)


class NIST19GlyphDataset(Dataset):
    def __init__(self, manifest: str | Path, image_size: int = 32, require_cluster: bool = False):
        self.records = [row for row in load_glyph_manifest(manifest) if Path(row["path"]).is_file()]
        if require_cluster and any("cluster_id" not in row for row in self.records):
            raise ValueError("Every row must contain a fixed cluster_id")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        item = {
            "image": _load_image(row["path"], self.image_size),
            "character_id": NIST19_ALPHABET.index(row["character"]),
            "character": row["character"], "writer_id": str(row["writer_id"]),
            "sample_id": row["sample_id"], "path": row["path"],
        }
        if "cluster_id" in row: item["cluster_id"] = int(row["cluster_id"])
        return item


class NIST19PairDataset(NIST19GlyphDataset):
    """Return content-positive and style-positive pairs for encoder pretraining."""
    def __init__(self, manifest: str | Path, image_size: int = 32, seed: int = 42):
        super().__init__(manifest, image_size)
        self.seed = seed; self.by_character = defaultdict(list); self.by_writer = defaultdict(list)
        for i, row in enumerate(self.records):
            self.by_character[row["character"]].append(i); self.by_writer[str(row["writer_id"])].append(i)
        self.indices = [i for i, row in enumerate(self.records)
                        if any(self.records[j]["writer_id"] != row["writer_id"] for j in self.by_character[row["character"]])
                        and any(self.records[j]["character"] != row["character"] for j in self.by_writer[str(row["writer_id"])])]

    def __len__(self): return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        anchor_index = self.indices[index]; row = self.records[anchor_index]
        rng = random.Random(self.seed + index + random.randrange(1 << 20))
        content_pool = [i for i in self.by_character[row["character"]] if self.records[i]["writer_id"] != row["writer_id"]]
        style_pool = [i for i in self.by_writer[str(row["writer_id"])] if self.records[i]["character"] != row["character"]]
        return {
            "anchor": _load_image(row["path"], self.image_size),
            "content_positive": _load_image(self.records[rng.choice(content_pool)]["path"], self.image_size),
            "style_positive": _load_image(self.records[rng.choice(style_pool)]["path"], self.image_size),
            "character_id": NIST19_ALPHABET.index(row["character"]),
            "writer_id": str(row["writer_id"]),
        }


class NIST19EpisodeDataset(NIST19GlyphDataset):
    """Single-target episodes whose references exclude the target character."""
    def __init__(self, manifest: str | Path, references: int = 8, image_size: int = 32, seed: int = 42):
        super().__init__(manifest, image_size, require_cluster=True)
        self.references = references; self.seed = seed; self.by_writer = defaultdict(list)
        for i, row in enumerate(self.records): self.by_writer[str(row["writer_id"])].append(i)
        self.indices = [i for i, row in enumerate(self.records)
                        if any(self.records[j]["character"] != row["character"] for j in self.by_writer[str(row["writer_id"])])]

    def __len__(self): return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        target_index = self.indices[index]; target = super().__getitem__(target_index); row = self.records[target_index]
        pool = [i for i in self.by_writer[str(row["writer_id"])] if self.records[i]["character"] != row["character"]]
        rng = random.Random(self.seed + index + random.randrange(1 << 20))
        chosen = [rng.choice(pool) for _ in range(self.references)]
        target["references"] = torch.stack([_load_image(self.records[i]["path"], self.image_size) for i in chosen])
        return target
