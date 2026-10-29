from __future__ import annotations

import argparse
import json
from pathlib import Path

from glyph_codebook.config import load_config
from glyph_codebook.data import build_manifest_from_sd19_trees, load_glyph_manifest, save_glyph_manifest, writer_disjoint_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an SD19 glyph manifest and create writer-disjoint splits")
    parser.add_argument("--config", required=True); parser.add_argument("--manifest"); parser.add_argument("--output")
    parser.add_argument("--by-write"); parser.add_argument("--by-class")
    args = parser.parse_args(); cfg = load_config(args.config); data = cfg["data"]
    output = Path(args.output or data["nist19_prepared_dir"])
    if args.by_write or args.by_class:
        if not args.by_write or not args.by_class: parser.error("--by-write and --by-class must be provided together")
        records, index_summary = build_manifest_from_sd19_trees(args.by_write, args.by_class)
        output.mkdir(parents=True, exist_ok=True); save_glyph_manifest(records, output / "manifest.json")
        print(json.dumps(index_summary, indent=2))
    else:
        source = Path(args.manifest or data["nist19_manifest"]); records = load_glyph_manifest(source)
    missing = [row["path"] for row in records if not Path(row["path"]).is_file()]
    if missing: raise FileNotFoundError(f"{len(missing)} manifest images do not exist; first missing path: {missing[0]}")
    ratios = tuple(data.get("nist19_split", [0.8, 0.1, 0.1]))
    splits = writer_disjoint_splits(records, ratios, int(cfg.get("seed", 42)))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items(): save_glyph_manifest(rows, output / f"{name}.json")
    summary = {name: {"samples": len(rows), "writers": len({row["writer_id"] for row in rows})} for name, rows in splits.items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
