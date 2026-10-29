"""Notebook-friendly runner for the fixed NIST19 glyph-codebook experiment."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run(*arguments: str) -> None:
    command = [sys.executable, "-m", *map(str, arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--output-root", default="/kaggle/working/outputs")
    parser.add_argument("--value-mode", choices=("pixel", "feature"), default="feature")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--classifier")
    parser.add_argument("--stage", choices=("encoder", "codebook", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); config = Path(args.config); cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    prepared = Path(cfg["data"]["nist19_prepared_dir"]); root = Path(args.output_root)
    encoder = root / "nist-encoder" / "latest.pt"; codebook = root / f"nist-codebook-k{args.clusters}" / "codebook.pt"
    selector_root = root / f"selector-k{args.clusters}-{args.value_mode}"; selector = selector_root / "latest.pt"
    if not (prepared / "train.json").is_file():
        manifest = Path(cfg["data"]["nist19_manifest"])
        by_write = Path(cfg["data"].get("nist19_by_write", "")); by_class = Path(cfg["data"].get("nist19_by_class", ""))
        if manifest.is_file(): run("glyph_codebook.cli.prepare", "--config", config)
        elif by_write.is_dir() and by_class.is_dir():
            run("glyph_codebook.cli.prepare", "--config", config, "--by-write", by_write, "--by-class", by_class)
        else:
            raise FileNotFoundError(f"Attach either the normalized manifest ({manifest}) or both SD19 trees ({by_write}, {by_class})")
    encoder_args = ["glyph_codebook.cli.train_encoder", "--config", config, "--output", encoder.parent]
    if encoder.is_file(): encoder_args += ["--resume", encoder]
    if args.smoke: encoder_args += ["--epochs", "1", "--max-steps", "20"]
    run(*encoder_args)
    if args.stage == "encoder": return
    if not codebook.is_file():
        run("glyph_codebook.cli.build_codebook", "--config", config, "--encoder", encoder,
            "--output", codebook.parent, "--clusters", str(args.clusters))
    if args.stage == "codebook": return
    classifier = Path(args.classifier or cfg["data"]["classifier_checkpoint"])
    if not classifier.is_file(): raise FileNotFoundError(f"Attach the existing character classifier checkpoint: {classifier}")
    selector_args = ["glyph_codebook.cli.train_selector", "--config", config, "--encoder", encoder,
                     "--classifier", classifier, "--codebook", codebook, "--value-mode", args.value_mode, "--output", selector_root]
    if selector.is_file(): selector_args += ["--resume", selector]
    if args.smoke: selector_args += ["--epochs", "1", "--max-steps", "20", "--oracle-warmup-epochs", "1"]
    run(*selector_args)
    run("glyph_codebook.cli.evaluate", "--config", config, "--encoder", encoder,
        "--classifier", classifier, "--codebook", codebook, "--checkpoint", selector, "--split", "test", "--max-samples", "1000" if args.smoke else "10000",
        "--output", selector_root / "eval")


if __name__ == "__main__": main()
